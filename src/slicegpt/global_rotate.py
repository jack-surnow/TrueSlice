import logging

import numpy as np
import torch
import torch.nn as nn
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

    return H



@torch.no_grad()
def global_calibrate_norms_ols(
    dense_adapter: ModelAdapter,
    sliced_adapter: ModelAdapter,
    dataloader: torch.utils.data.DataLoader, 
    d_new: int, 
    device: torch.device,
    apply_mask: bool = True,
    eps: float = 1e-8
) -> None:

    dense_adapter.model.eval()
    sliced_adapter.model.eval()

    Q_global = sliced_adapter.Q_global

    inps_dense, args_dense, kwargs_dense, ignore_masks_dense = [], [], [], []
    for batch in dataloader:
        inp_batch_dense, args_batch_dense, kwargs_batch_dense = get_layer0_inputs(dense_adapter, batch)
        inps_dense.append(inp_batch_dense)
        args_dense.append(args_batch_dense)
        kwargs_dense.append(kwargs_batch_dense)
        if apply_mask:
            ignore_masks_dense.append(batch["attention_mask"])

    inps_sliced, args_sliced, kwargs_sliced, ignore_masks_sliced = [], [], [], []
    for batch in dataloader:
        inp_batch_sliced, args_batch_sliced, kwargs_batch_sliced = get_layer0_inputs(sliced_adapter, batch)
        inps_sliced.append(inp_batch_sliced)
        args_sliced.append(args_batch_sliced)
        kwargs_sliced.append(kwargs_batch_sliced)
        if apply_mask:
            ignore_masks_sliced.append(batch["attention_mask"])

    optimal_norm_params = {}

    layers = zip(dense_adapter.get_layers(), sliced_adapter.get_layers())
    for idx, (dense_layer, sliced_layer) in enumerate(tqdm(layers, unit="layer", desc="Performing Chanel-wise OLS")):

        dense_layer = dense_layer.to(config.device)
        sliced_layer = sliced_layer.to(config.device)

        acts = {}
        def get_hook(name):
            return lambda m, i, o: acts.update({name: o.detach().to(torch.float32)})

        # Pass 1
        h_d1 = dense_layer.get_first_layernorm().register_forward_hook(get_hook("d1"))
        h_s1 = sliced_layer.get_first_layernorm().register_forward_hook(get_hook("s1"))

        sum_uu1 = torch.zeros(d_new, dtype=torch.float64, device=device)
        sum_uy1 = torch.zeros(d_new, dtype=torch.float64, device=device)

        for i in range(len(inps_dense)):
            _ = dense_layer.layer(*args_dense[i], **kwargs_dense[i])
            _ = sliced_layer.layer(*args_sliced[i], **kwargs_sliced[i])

            Y1 = torch.matmul(acts["d1"], Q_global).reshape(-1, d_new).to(torch.float64)
            U1 = acts["s1"].reshape(-1, d_new).to(torch.float64)
            sum_uu1 += (U1 * U1).sum(dim=0)
            sum_uy1 += (U1 * Y1).sum(dim=0)

        gamma1 = sum_uy1 / (sum_uu1 + eps)
        optimal_norm_params[f"layer_{idx}_norm1_gamma"] = gamma1.clone()

        h_d1.remove()
        h_s1.remove()

        # Pass 2
        def modifier_hook1(m, i, o):
            return o * gamma1.to(torch.float32)

        h_mod1 = sliced_layer.get_first_layernorm().register_forward_hook(modifier_hook1)

        h_d2 = dense_layer.get_second_layernorm().register_forward_hook(get_hook("d2"))
        h_s2 = sliced_layer.get_second_layernorm().register_forward_hook(get_hook("s2"))

        sum_uu2 = torch.zeros(d_new, dtype=torch.float64, device=device)
        sum_uy2 = torch.zeros(d_new, dtype=torch.float64, device=device)

        for i in range(len(inps_dense)):
            _ = dense_layer.layer(*args_dense[i], **kwargs_dense[i])
            _ = sliced_layer.layer(*args_sliced[i], **kwargs_sliced[i])

            Y2 = torch.matmul(acts["d2"], Q_global).reshape(-1, d_new).to(torch.float64)
            U2 = acts["s2"].reshape(-1, d_new).to(torch.float64)
            sum_uu2 += (U2 * U2).sum(dim=0)
            sum_uy2 += (U2 * Y2).sum(dim=0)

        gamma2 = sum_uy2 / (sum_uu2 + eps)
        optimal_norm_params[f"layer_{idx}_norm2_gamma"] = gamma2.clone()

        h_d2.remove()
        h_s2.remove()

        # Pass 3
        def modifier_hook2(m, i, o):
            return o * gamma2.to(torch.float32)

        h_mod2 = sliced_layer.get_second_layernorm().register_forward_hook(modifier_hook2)

        for i in range(len(inps_dense)):
            out_d = dense_layer.layer(*args_dense[i], **kwargs_dense[i])
            out_s = sliced_layer.layer(*args_sliced[i], **kwargs_sliced[i])

            if isinstance(out_d, tuple):
                args_dense[i] = dense_layer.get_updated_args(out_d[0].detach(), args_dense[i])
                args_sliced[i] = sliced_layer.get_updated_args(out_s[0].detach(), args_sliced[i])
            else:
                args_dense[i] = dense_layer.get_updated_args(out_d.detach(), args_dense[i])
                args_sliced[i] = sliced_layer.get_updated_args(out_s.detach(), args_sliced[i])

        h_mod1.remove()
        h_mod2.remove()

        dense_layer.layer.to("cpu")
        sliced_layer.layer.to("cpu")
        torch.cuda.empty_cache()

    sliced_adapter.optimal_norm_params = optimal_norm_params