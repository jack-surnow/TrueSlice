# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging

import numpy as np
import torch
from tqdm import tqdm

from .config import config
from .model_adapter import LayerAdapter, ModelAdapter
from .model_utils import get_layer0_inputs, get_signals_sequential, get_signals_parallel
from .slicing_scheduler import SlicingScheduler
from .utils import cleanup_memory


def rotate_attention_inputs(layer_adapter: LayerAdapter, Q: torch.Tensor) -> None:
    # Rotate the WQ, WK and WV matrices of the self-attention layer.
    for W in layer_adapter.get_attention_inputs():
        dtype = W.weight.dtype
        W_ = W.weight.to(device=config.device, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)


def slice_attention_inputs(layer_adapter: LayerAdapter, new_embedding_dimension: int) -> None:
    # Slice the WQ, WK and WV matrices of the self-attention layer.
    for W in layer_adapter.get_attention_inputs():
        W.weight.data = W.weight.data[:, :new_embedding_dimension]
        W.in_features = new_embedding_dimension


def rotate_attention_output(layer_adapter: LayerAdapter, Q: torch.Tensor) -> None:
    # Rotate output matrix of the self-attention layer.
    W = layer_adapter.get_attention_output()

    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=config.device, dtype=torch.float64)
    W.weight.data = torch.matmul(Q.T, W_).to(device="cpu", dtype=dtype)
    if W.bias is not None:
        b = W.bias.data.to(device=config.device, dtype=torch.float64)
        W.bias.data = torch.matmul(Q.T, b).to(device="cpu", dtype=dtype)


def slice_attention_output(layer_adapter: LayerAdapter, new_embedding_dimension: int) -> None:
    # Slice output matrix of the self-attention layer.
    W = layer_adapter.get_attention_output()
    W.weight.data = W.weight.data[:new_embedding_dimension, :]
    if W.bias is not None:
        W.bias.data = W.bias.data[:new_embedding_dimension]
    W.out_features = new_embedding_dimension


def rotate_mlp_input(layer_adapter: LayerAdapter, Q: torch.Tensor) -> None:
    # Rotate the MLP input weights.
    for W in layer_adapter.get_mlp_inputs():
        dtype = W.weight.dtype
        W_ = W.weight.data.to(device=config.device, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)


def slice_mlp_input(layer_adapter: LayerAdapter, new_embedding_dimension: int) -> None:
    # Slice the MLP input weights.
    for W in layer_adapter.get_mlp_inputs():
        W.weight.data = W.weight.data[:, :new_embedding_dimension]
        W.in_features = new_embedding_dimension


def rotate_mlp_output(layer_adapter: LayerAdapter, Q: torch.Tensor) -> None:
    # Rotate the MLP output weights and bias.
    W = layer_adapter.get_mlp_output()
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=config.device, dtype=torch.float64)
    W.weight.data = torch.matmul(Q.T, W_).to(device="cpu", dtype=dtype)
    if W.bias is not None:
        b = W.bias.data.to(device=config.device, dtype=torch.float64)
        W.bias.data = torch.matmul(Q.T, b).to(device="cpu", dtype=dtype)


def slice_mlp_output(layer_adapter: LayerAdapter, new_embedding_dimension: int) -> None:
    # Slice the MLP output weights and bias.
    W = layer_adapter.get_mlp_output()
    W.weight.data = W.weight.data[:new_embedding_dimension, :]
    if W.bias is not None:
        W.bias.data = W.bias.data[:new_embedding_dimension]
    W.out_features = new_embedding_dimension


def rotate_embeddings(model_adapter: ModelAdapter, Q: torch.Tensor) -> None:
    # Rotate the embeddings.
    for W in model_adapter.get_embeddings():
        dtype = W.weight.data.dtype
        W_ = W.weight.data.to(device=config.device, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)


def slice_embeddings(model_adapter: ModelAdapter, new_embedding_dimensions: dict[int, int]) -> None:
    # Slice the embeddings.
    for i, W in enumerate(model_adapter.get_embeddings()):
        W.weight.data = W.weight.data[:, : new_embedding_dimensions[i]]
        W.embedding_dim = new_embedding_dimensions[i]


def rotate_head(model_adapter: ModelAdapter, Q: torch.Tensor) -> None:
    # Rotate the head.
    W = model_adapter.get_lm_head()
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=config.device, dtype=torch.float64)
    W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)


def slice_head(model_adapter: ModelAdapter, new_embedding_dimension: int) -> None:
    # Slice the head.
    lm_head = model_adapter.get_lm_head()
    lm_head.weight.data = lm_head.weight.data[:, :new_embedding_dimension]
    lm_head.in_features = new_embedding_dimension


def rotate_and_slice(
    model_adapter: ModelAdapter,
    dataloader: torch.utils.data.DataLoader[torch.Tensor],
    slicing_scheduler: SlicingScheduler,
    apply_mask: bool = True,
    final_orientation: str = 'pca',
) -> None:
    """
    Rotate and slice a model, with interleaved slicing and PCA calculations
    """
    if model_adapter.parallel_blocks:
        rotate_and_slice_parallel(model_adapter, dataloader, slicing_scheduler, apply_mask, final_orientation)
    else:
        rotate_and_slice_sequential(model_adapter, dataloader, slicing_scheduler, apply_mask, final_orientation)


@torch.no_grad()
def rotate_and_slice_sequential(
    model_adapter: ModelAdapter,
    dataloader: torch.utils.data.DataLoader[torch.Tensor],
    slicing_scheduler: SlicingScheduler,
    apply_mask: bool = True,
    final_orientation: str = 'pca',
) -> None:
    """
    Rotate and slice the provided model, with interleaved slicing and PCA calculations.
    This method works for models where the MLP block is computed after the attention block.
    """

    # Set up calibration data
    logging.info("rotate_and_slice_sequential: Setting up calibration data")

    model_adapter.model.eval()

    layers = model_adapter.get_layers()

    for layer_adapter in layers:
        layer_adapter.layer.to("cpu")

    model_adapter.get_lm_head().to("cpu")

    for W in model_adapter.get_embeddings():
        W.to(device=config.device)

    base_model = getattr(model_adapter.model, "model", model_adapter.model)
    if hasattr(base_model, "rotary_emb"):
        base_model.rotary_emb.to(config.device)

    inps, args, kwargs, ignore_masks = [], [], [], []
    for batch in dataloader:
        inp_batch, args_batch, kwargs_batch = get_layer0_inputs(model_adapter, batch)
        inps.append(inp_batch)
        args.append(args_batch)
        kwargs.append(kwargs_batch)
        if apply_mask:
            ignore_masks.append(batch["attention_mask"])

    for W in model_adapter.get_embeddings():
        W.to("cpu")

    if hasattr(base_model, "rotary_emb"):
        base_model.rotary_emb.to("cpu")

    slicing_scheduler.setup(hidden_size=model_adapter.hidden_size, layers_num=len(layers), parallel_blocks=False)

    # Profiling model layers
    logging.info("rotate_and_slice_sequential: Profiling model layers")

    # Accumulate with normalization
    C_inps_raw = get_scatter_mat(inps, ignore_masks)
    C_global = C_inps_raw / torch.trace(C_inps_raw)
    del inps, C_inps_raw

    cleanup_memory()

    for layer_idx, layer_adapter in enumerate(tqdm(layers, unit="layer", desc="Accumulating Covariance")):
        layer_adapter.layer.to(device=config.device)

        layer_kwargs = [
            batch_kwargs["layer_kwargs"][layer_idx]
            for batch_kwargs in kwargs
        ]

        mlp_ln_inputs, layer_outputs = get_signals_sequential(
            layer_adapter, args, layer_kwargs
        )

        # Acculumate with normalization
        C_mlp = get_scatter_mat(mlp_ln_inputs, ignore_masks)
        C_out = get_scatter_mat(layer_outputs, ignore_masks)
        C_global += C_mlp / torch.trace(C_mlp)
        C_global += C_out / torch.trace(C_out)
        del mlp_ln_inputs, C_mlp, C_out

        for i, out in enumerate(layer_outputs):
            clean_out = out.detach().to("cpu").clone()
            args[i] = layer_adapter.get_updated_args(clean_out, args[i])
            del clean_out

        del layer_outputs

        layer_adapter.layer.to("cpu")

        cleanup_memory()

    # Calculating the global PCA
    logging.info("rotate_and_slice_sequential: Calculating the global PCA")

    damp = 0.01 * torch.mean(torch.diag(C_global))
    diag = torch.arange(C_global.shape[-1]).to(device=config.device)
    C_global[diag, diag] = C_global[diag, diag] + damp
    X_eig = torch.linalg.eigh(C_global)
    index = torch.argsort(X_eig[0], descending=True)
    Q_global = X_eig[1][:, index]
    Q_global = Q_global.to(device=config.device, dtype=torch.float64)

    del args, kwargs, ignore_masks, C_global
    cleanup_memory()

    if final_orientation == 'random':
        R = random_orthogonal_upper_left(Q_global.shape[0], slicing_scheduler.get_embedding_dimensions()[0])
        Q_global = Q_global @ R.to(Q_global.device)

    # Rotating and slicing layers
    logging.info("rotate_and_slice_sequential: Rotating and slicing layers")

    for W in model_adapter.get_embeddings():
        W.to(device=config.device)

    rotate_embeddings(model_adapter, Q_global)
    slice_embeddings(model_adapter, slicing_scheduler.get_embedding_dimensions())

    for W in model_adapter.get_embeddings():
        W.to("cpu")

    global_dim = slicing_scheduler.get_embedding_dimensions()[0]

    for layer_adapter in tqdm(layers, unit="layer", desc="Slicing weights"):
        layer = layer_adapter.layer
        layer.to(device=config.device)

        rotate_attention_inputs(layer_adapter, Q_global)
        slice_attention_inputs(layer_adapter, global_dim)

        rotate_attention_output(layer_adapter, Q_global)
        slice_attention_output(layer_adapter, global_dim)

        rotate_mlp_input(layer_adapter, Q_global)
        slice_mlp_input(layer_adapter, global_dim)

        rotate_mlp_output(layer_adapter, Q_global)
        slice_mlp_output(layer_adapter, global_dim)

        layer.to("cpu")
        cleanup_memory()

    model_adapter.get_lm_head().to(device=config.device)

    rotate_head(model_adapter, Q_global)
    slice_head(model_adapter, global_dim)

    model_adapter.get_lm_head().to("cpu")

    model_adapter.slicing_conf = slicing_scheduler.slicing_conf.clone()

    logging.info("rotate_and_slice_sequential: Complete")


@torch.no_grad()
def rotate_and_slice_parallel(
    model_adapter: ModelAdapter,
    dataloader: torch.utils.data.DataLoader[torch.Tensor],
    slicing_scheduler: SlicingScheduler,
    apply_mask: bool = True,
    final_orientation: str = 'pca',
) -> None:
    """
    """

    # Set up calibration data
    logging.info("rotate_and_slice_parallel: Setting up calibration data")

    model_adapter.model.eval()

    layers = model_adapter.get_layers()

    for layer_adapter in layers:
        layer_adapter.layer.to("cpu")

    model_adapter.get_lm_head().to("cpu")

    for W in model_adapter.get_embeddings():
        W.to(device=config.device)

    inps, args, kwargs, ignore_masks = [], [], [], []
    for batch in dataloader:
        inp_batch, args_batch, kwargs_batch = get_layer0_inputs(model_adapter, batch)
        inps.append(inp_batch)
        args.append(args_batch)
        kwargs.append(kwargs_batch)
        if apply_mask:
            ignore_masks.append(batch["attention_mask"])

    for W in model_adapter.get_embeddings():
        W.to("cpu")
    
    slicing_scheduler.setup(hidden_size=model_adapter.hidden_size, layers_num=len(layers), parallel_blocks=True)

    # Profiling model layers
    logging.info("rotate_and_slice_parallel: Profiling model layers")

    # Accumulate with normalization
    C_inps_raw = get_scatter_mat(inps, ignore_masks)
    C_global = C_inps_raw / torch.trace(C_inps_raw)
    del inps, C_inps_raw

    cleanup_memory()

    for layer_adapter in tqdm(layers, unit="layer", desc="Accumulating Covariance"):
        layer = layer_adapter.layer.to(device=config.device)

        outputs = get_signals_parallel(layer_adapter, args, kwargs)

        # Accumulate with normalization
        C_layer = get_scatter_mat(outputs, ignore_masks)
        C_global += C_layer / torch.trace(C_layer)
        del C_layer

        for i, out in enumerate(outputs):
            clean_out = out.detach().to("cpu").clone()
            args[i] = layer_adapter.get_updated_args(clean_out, args[i])
            del clean_out

        del outputs

        layer_adapter.layer.to("cpu")

        cleanup_memory()

    # Calculating the global PCA
    logging.info("rotate_and_slice_parallel: Calculating the global PCA")

    damp = 0.01 * torch.mean(torch.diag(C_global))
    diag = torch.arange(C_global.shape[-1]).to(device=config.device)
    C_global[diag, diag] = C_global[diag, diag] + damp
    X_eig = torch.linalg.eigh(C_global)
    index = torch.argsort(X_eig[0], descending=True)
    Q_global = X_eig[1][:, index]
    Q_global = Q_global.to(device=config.device, dtype=torch.float64)

    del args, kwargs, ignore_masks, C_global
    cleanup_memory()

    if final_orientation == 'random':
        R = random_orthogonal_upper_left(Q_global.shape[0], slicing_scheduler.get_embedding_dimensions()[0])
        Q_global = Q_global @ R.to(Q_global.device)

    # Rotating and slicing layers
    logging.info("rotate_and_slice_parallel: Rotating and slicing layers")

    for W in model_adapter.get_embeddings():
        W.to(device=config.device)

    rotate_embeddings(model_adapter, Q_global)
    slice_embeddings(model_adapter, slicing_scheduler.get_embedding_dimensions())

    for W in model_adapter.get_embeddings():
        W.to("cpu")

    global_dim = slicing_scheduler.get_embedding_dimensions()[0]

    for layer_adapter in tqdm(layers, unit="layer", desc="Slicing weights"):
        layer = layer_adapter.layer
        layer.to(device=config.device)

        rotate_attention_inputs(layer_adapter, Q_global)
        slice_attention_inputs(layer_adapter, global_dim)

        rotate_attention_output(layer_adapter, Q_global)
        slice_attention_output(layer_adapter, global_dim)

        rotate_mlp_input(layer_adapter, Q_global)
        slice_mlp_input(layer_adapter, global_dim)

        rotate_mlp_output(layer_adapter, Q_global)
        slice_mlp_output(layer_adapter, global_dim)

        layer.to("cpu")
        cleanup_memory()

    model_adapter.get_lm_head().to(device=config.device)

    rotate_head(model_adapter, Q_global)
    slice_head(model_adapter, global_dim)

    model_adapter.get_lm_head().to("cpu")

    model_adapter.slicing_conf = slicing_scheduler.slicing_conf.clone()

    logging.info("rotate_and_slice_parallel: Complete")


def random_orthogonal_upper_left(total_dim, upper_block_dim):
    """
    Create a square matrix where the upper left block is a random orthogonal matrix, and the remainder is the identity.
    """
    A = np.random.rand(upper_block_dim, upper_block_dim)
    Q, _ = np.linalg.qr(A)
    R = np.eye(total_dim)
    R[:upper_block_dim, :upper_block_dim] = Q
    return torch.from_numpy(R)


@torch.no_grad()
def get_scatter_mat(
    X: list[torch.Tensor], ignore_masks: list[torch.Tensor] | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    """
    cleanup_memory()

    H = None
    for idx, X_batch in enumerate(X):
        if ignore_masks:
            X_batch[ignore_masks[idx] == 0] = 0

        X_batch = X_batch.double().to(device=config.device)
        H_batch = torch.sum(X_batch.mT @ X_batch, dim=0)  # sum over the batch dimension.
        H = H_batch if H is None else H + H_batch
        del X_batch, H_batch

    return H