import logging
from contextlib import contextmanager
from typing import Generator, List, Dict, Any

import torch.nn as nn
import torch
from tqdm import tqdm

from .config import config
from .model_adapter import LayerAdapter, ModelAdapter
from .model_utils import get_layer0_inputs, get_signals
from .slicing_scheduler import SlicingScheduler
from .utils import cleanup_memory, map_tensors

from slicegpt import rotate


def global_slice_attention_inputs(layer_adapter: LayerAdapter, new_embedding_dimension: int) -> None:
    # Slice the WQ, WK and WV matrices of the self-attention layer.
    for W in layer_adapter.get_attention_inputs():
        W.weight.data = W.weight.data[:, :new_embedding_dimension]
        W.in_features = new_embedding_dimension


def global_rotate_and_slice(
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
        global_rotate_and_slice_parallel(model_adapter, dataloader, slicing_scheduler, apply_mask, final_orientation)
    else:
        global_rotate_and_slice_sequential(model_adapter, dataloader, slicing_scheduler, apply_mask, final_orientation)


@torch.no_grad()
def global_rotate_and_slice_sequential(
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
    logging.info("global_rotate_and_slice_sequential: Setting up calibration data")

    model_adapter.model.eval()

    inps, args, kwargs, ignore_masks = [], [], [], []
    for batch in dataloader:
        inp_batch, args_batch, kwargs_batch = get_layer0_inputs(model_adapter, batch)
        inps.append(inp_batch)
        args.append(args_batch)
        kwargs.append(kwargs_batch)
        if apply_mask:
            ignore_masks.append(batch["attention_mask"])

    layers = model_adapter.get_layers()
    slicing_scheduler.setup(hidden_size=model_adapter.hidden_size, layers_num=len(layers), parallel_blocks=False)

    # Profiling model layers
    logging.info("global_rotate_and_slice_sequential: Profiling model layers")

    C_global = get_covariance(inps, ignore_masks)
    del inps
    cleanup_memory()

    for layer_adapter in tqdm(layers, unit="layer", desc="Accumulating Covariance"):
        mlp_ln_inputs, layer_outputs = get_signals(layer_adapter, args, kwargs)

        # Acculumate without normalization
        C_global += get_covariance(mlp_ln_inputs, ignore_masks)
        C_global += get_covariance(layer_outputs, ignore_masks)

        del mlp_ln_inputs

        for i, out in enumerate(layer_outputs):
            clean_out = out.detach().cpu().clone()
            args[i] = layer_adapter.get_updated_args(clean_out, args[i])
            del clean_out

        del layer_outputs

        layer_adapter.layer.to("cpu")

        cleanup_memory()

    # Calculating the global PCA
    logging.info("global_rotate_and_slice_sequential: Calculating the global PCA")

    damp = 0.01 * torch.mean(torch.diag(C_global))
    diag = torch.arange(C_global.shape[-1]).to(device=config.device)
    C_global[diag, diag] = C_global[diag, diag] + damp
    X_eig = torch.linalg.eigh(C_global)
    index = torch.argsort(X_eig[0], descending=True)
    Q_global = X_eig[1][:, index]
    Q_global = Q_global.to(device=config.device, dtype=torch.float64)

    if final_orientation == 'random':
        R = rotate.random_orthogonal_upper_left(Q_global.shape[0], slicing_scheduler.get_embedding_dimensions()[0])
        Q_global = Q_global @ R.to(Q_global.device)

    del C_global
    cleanup_memory()

    # Rotating and slicing layers
    logging.info("global_rotate_and_slice_sequential: Rotating and slicing layers")

    rotate.rotate_embeddings(model_adapter, Q_global)
    rotate.slice_embeddings(model_adapter, slicing_scheduler.get_embedding_dimensions())

    global_dim = slicing_scheduler.get_embedding_dimensions()[0]

    for layer_adapter in tqdm(layers, unit="layer", desc="Slicing weights"):
        layer = layer_adapter.layer

        rotate.rotate_attention_inputs(layer_adapter, Q_global)
        global_slice_attention_inputs(layer_adapter, global_dim)

        rotate.rotate_attention_output(layer_adapter, Q_global)
        rotate.slice_attention_output(layer_adapter, global_dim)

        rotate.rotate_mlp_input(layer_adapter, Q_global)
        rotate.slice_mlp_input(layer_adapter, global_dim)

        rotate.rotate_mlp_output(layer_adapter, Q_global)
        rotate.slice_mlp_output(layer_adapter, global_dim)

        layer.to('cpu')
        cleanup_memory()

    rotate.rotate_head(model_adapter, Q_global)
    rotate.slice_head(model_adapter, global_dim)

    model_adapter.Q_global = Q_global.detach().clone()
    model_adapter.slicing_conf = slicing_scheduler.slicing_conf.clone()

    logging.info("global_rotate_and_slice_sequential: Complete")


@torch.no_grad()
def global_rotate_and_slice_parallel(
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
    logging.info("global_rotate_and_slice_parallel: Setting up calibration data")

    model_adapter.model.eval()

    inps, args, kwargs, ignore_masks = [], [], [], []
    for batch in dataloader:
        inp_batch, args_batch, kwargs_batch = get_layer0_inputs(model_adapter, batch)
        inps.append(inp_batch)
        args.append(args_batch)
        kwargs.append(kwargs_batch)
        if apply_mask:
            ignore_masks.append(batch["attention_mask"])

    layers = model_adapter.get_layers()
    slicing_scheduler.setup(hidden_size=model_adapter.hidden_size, layers_num=len(layers), parallel_blocks=True)

    # Profiling model layers
    logging.info("global_rotate_and_slice_parallel: Profiling model layers")

    C_global = get_covariance(inps, ignore_masks)
    del inps
    cleanup_memory()

    for layer_adapter in tqdm(layers, unit="layer", desc="Accumulating Covariance"):
        layer = layer_adapter.layer.to(config.device)
        outputs = []

        # Run forward pass for this layer to get outputs
        for layer_args_batch, layer_kwargs_batch in zip(args, kwargs):
            layer_args_batch, layer_kwargs_batch = map_tensors(
                [layer_args_batch, layer_kwargs_batch], device=config.device
            )
            out = layer(*layer_args_batch, **layer_kwargs_batch)

            if isinstance(out, tuple):
                out = out[layer_adapter.hidden_states_output_position]

            out = out.cpu()
            outputs.append(out)

        # Accumulate without normalization
        C_global += get_covariance(outputs, ignore_masks)

        for i, out in enumerate(outputs):
            clean_out = out.detach().cpu().clone()
            args[i] = layer_adapter.get_updated_args(clean_out, args[i])
            del clean_out

        del outputs

        layer_adapter.layer.to('cpu')

        cleanup_memory()

    # Calculating the global PCA
    logging.info("global_rotate_and_slice_parallel: Calculating the global PCA")

    damp = 0.01 * torch.mean(torch.diag(C_global))
    diag = torch.arange(C_global.shape[-1]).to(device=config.device)
    C_global[diag, diag] = C_global[diag, diag] + damp
    X_eig = torch.linalg.eigh(C_global)
    index = torch.argsort(X_eig[0], descending=True)
    Q_global = X_eig[1][:, index]
    Q_global = Q_global.to(device=config.device, dtype=torch.float64)

    if final_orientation == 'random':
        R = rotate.random_orthogonal_upper_left(Q_global.shape[0], slicing_scheduler.get_embedding_dimensions()[0])
        Q_global = Q_global @ R.to(Q_global.device)

    del C_global
    cleanup_memory()

    # Rotating and slicing layers
    logging.info("global_rotate_and_slice_parallel: Rotating and slicing layers")

    rotate.rotate_embeddings(model_adapter, Q_global)
    rotate.slice_embeddings(model_adapter, slicing_scheduler.get_embedding_dimensions())

    global_dim = slicing_scheduler.get_embedding_dimensions()[0]

    for layer_adapter in tqdm(layers, unit="layer", desc="Slicing weights"):
        layer = layer_adapter.layer

        rotate.rotate_attention_inputs(layer_adapter, Q_global)
        global_slice_attention_inputs(layer_adapter, global_dim)

        rotate.rotate_attention_output(layer_adapter, Q_global)
        rotate.slice_attention_output(layer_adapter, global_dim)

        rotate.rotate_mlp_input(layer_adapter, Q_global)
        rotate.slice_mlp_input(layer_adapter, global_dim)

        rotate.rotate_mlp_output(layer_adapter, Q_global)
        rotate.slice_mlp_output(layer_adapter, global_dim)

        layer.to('cpu')
        cleanup_memory()

    rotate.rotate_head(model_adapter, Q_global)
    rotate.slice_head(model_adapter, global_dim)

    model_adapter.slicing_conf = slicing_scheduler.slicing_conf.clone()

    logging.info("global_rotate_and_slice_parallel: Complete")


@torch.no_grad()
def get_covariance(
    X: list[torch.Tensor], ignore_masks: list[torch.Tensor] | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Run PCA on a list of batched data. Returns the eigenvalues and eigenvectors.
    """
    # Run GC and cleanup GPU memory
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
