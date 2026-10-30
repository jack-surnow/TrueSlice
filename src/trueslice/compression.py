# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
#
# Modified for TrueSlice.

import copy
import logging

import numpy as np
import torch
from tqdm import tqdm

from .model_adapter import LayerAdapter, ModelAdapter
from .calibration import prepare_calibration_inputs, get_scatter_mat
from .utils import cleanup_memory, map_tensors
from .mapping import Mapping
from .workers import ComputeWorkers


def rotate_attention_inputs(layer_adapter: LayerAdapter, Q: torch.Tensor) -> None:
    # Rotate the input projections exposed by the attention or mixer adapter.
    for W in layer_adapter.get_attention_inputs():
        dtype = W.weight.dtype
        W_ = W.weight.to(device=Q.device, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device=W.weight.device, dtype=dtype)


def slice_attention_inputs(layer_adapter: LayerAdapter, new_embedding_dimension: int) -> None:
    # Slice the residual-input columns of the attention or mixer projections.
    for W in layer_adapter.get_attention_inputs():
        W.weight.data = W.weight.data[:, :new_embedding_dimension].clone()
        W.in_features = new_embedding_dimension


def rotate_attention_output(layer_adapter: LayerAdapter, Q: torch.Tensor) -> None:
    # Rotate output matrix of the self-attention layer.
    W = layer_adapter.get_attention_output()

    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=Q.device, dtype=torch.float64)
    W.weight.data = torch.matmul(Q.T, W_).to(device=W.weight.device, dtype=dtype)
    if W.bias is not None:
        b = W.bias.data.to(device=Q.device, dtype=torch.float64)
        W.bias.data = torch.matmul(Q.T, b).to(device=W.weight.device, dtype=dtype)


def slice_attention_output(layer_adapter: LayerAdapter, new_embedding_dimension: int) -> None:
    # Slice output matrix of the self-attention layer.
    W = layer_adapter.get_attention_output()
    W.weight.data = W.weight.data[:new_embedding_dimension, :].clone()
    if W.bias is not None:
        W.bias.data = W.bias.data[:new_embedding_dimension].clone()
    W.out_features = new_embedding_dimension


def rotate_mlp_input(layer_adapter: LayerAdapter, Q: torch.Tensor) -> None:
    # Rotate the MLP input weights.
    for W in layer_adapter.get_mlp_inputs():
        dtype = W.weight.dtype
        W_ = W.weight.data.to(device=Q.device, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device=W.weight.device, dtype=dtype)


def slice_mlp_input(layer_adapter: LayerAdapter, new_embedding_dimension: int) -> None:
    # Slice the MLP input weights.
    for W in layer_adapter.get_mlp_inputs():
        W.weight.data = W.weight.data[:, :new_embedding_dimension].clone()
        W.in_features = new_embedding_dimension


def rotate_mlp_output(layer_adapter: LayerAdapter, Q: torch.Tensor) -> None:
    # Rotate the MLP output weights and bias.
    W = layer_adapter.get_mlp_output()
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=Q.device, dtype=torch.float64)
    W.weight.data = torch.matmul(Q.T, W_).to(device=W.weight.device, dtype=dtype)
    if W.bias is not None:
        b = W.bias.data.to(device=Q.device, dtype=torch.float64)
        W.bias.data = torch.matmul(Q.T, b).to(device=W.weight.device, dtype=dtype)


def slice_mlp_output(layer_adapter: LayerAdapter, new_embedding_dimension: int) -> None:
    # Slice the MLP output weights and bias.
    W = layer_adapter.get_mlp_output()
    W.weight.data = W.weight.data[:new_embedding_dimension, :].clone()
    if W.bias is not None:
        W.bias.data = W.bias.data[:new_embedding_dimension].clone()
    W.out_features = new_embedding_dimension


def _rotate_rows(weight, basis, row_chunk_size=1024):
    """Bound vocabulary rotation scratch space to a chunk, preserving storage placement."""
    output = torch.empty((weight.shape[0], basis.shape[1]), dtype=weight.dtype, device=weight.device)
    for start in range(0, weight.shape[0], row_chunk_size):
        rows = weight[start:start + row_chunk_size].to(device=basis.device, dtype=torch.float64)
        output[start:start + row_chunk_size] = (rows @ basis).to(device=weight.device, dtype=weight.dtype)
    return output

def rotate_embeddings(model_adapter: ModelAdapter, Q: torch.Tensor, new_embedding_dimension=None) -> None:
    # Rotate the embeddings.
    basis = Q if new_embedding_dimension is None else Q[:, :new_embedding_dimension]
    for embedding in model_adapter.get_embeddings():
        embedding.weight.data = _rotate_rows(embedding.weight.data, basis)


def slice_embeddings(model_adapter: ModelAdapter, new_embedding_dimension: int) -> None:
    # Slice the embeddings.
    for W in model_adapter.get_embeddings():
        W.weight.data = W.weight.data[:, :new_embedding_dimension].clone()
        W.embedding_dim = new_embedding_dimension


def rotate_head(model_adapter: ModelAdapter, Q: torch.Tensor, new_embedding_dimension=None) -> None:
    # Rotate the head.
    head = model_adapter.get_lm_head()
    basis = Q if new_embedding_dimension is None else Q[:, :new_embedding_dimension]
    head.weight.data = _rotate_rows(head.weight.data, basis)


def slice_head(model_adapter: ModelAdapter, new_embedding_dimension: int) -> None:
    # Slice the head.
    lm_head = model_adapter.get_lm_head()
    lm_head.weight.data = lm_head.weight.data[:, :new_embedding_dimension].clone()
    lm_head.in_features = new_embedding_dimension


@torch.no_grad()
def rotate_and_slice(
    model_adapter: ModelAdapter,
    dataloader: torch.utils.data.DataLoader[torch.Tensor],
    new_embedding_dimension: int,
    apply_mask: bool = True,
    final_orientation: str = 'pca',
    *,
    mapping=None,
) -> None:
    """Compress a fused model using one shared basis and retained residual width.

    Profile batches across devices and reduce raw moments, then rotate independent
    layers. Embeddings, attention/MLP residual dimensions, and the head all retain
    new_embedding_dimension features. The adapter determines the decoder architecture.
    Multiple devices use persistent spawned processes, each retaining its own
    calibration batches and only the current decoder layer. One device runs inline.
    """
    if isinstance(new_embedding_dimension, bool) or not isinstance(new_embedding_dimension, int):
        raise ValueError('new_embedding_dimension must be an integer')
    if not 1 <= new_embedding_dimension <= model_adapter.hidden_size:
        raise ValueError('new_embedding_dimension must be between 1 and hidden_size')
    if final_orientation not in ('pca', 'random'):
        raise ValueError('final_orientation must be pca or random')

    mapping = mapping or Mapping()
    devices = mapping.compute_devices
    adapter = model_adapter
    adapter.model.eval()
    mapping.to_calibration(adapter)
    layers = adapter.get_layers()
    args, kwargs, masks, inps = [], [], [], []
    with mapping.on_compute_device(adapter.get_pre_layer_modules()):
        for batch in dataloader:
            hidden, batch_args, batch_kwargs = prepare_calibration_inputs(adapter, batch)
            inps.append(hidden)
            args.append(batch_args)
            kwargs.append(batch_kwargs)
            if apply_mask:
                masks.append(batch.get('attention_mask', torch.ones(hidden.shape[:-1], dtype=torch.bool)))

    if not args:
        raise ValueError('Calibration requires at least one batch')

    def normalized(moment):
        trace = torch.trace(moment)
        if not torch.isfinite(trace) or trace <= 0:
            raise ValueError('Calibration scatter has nonpositive or nonfinite trace')
        return moment / trace

    # Keep accumulated moments in system RAM, independently of compute placement.
    moment = get_scatter_mat(inps, masks, device=devices[0]).cpu()
    global_moment = normalized(moment)
    del inps, moment

    shards = [list(range(i, len(args), len(devices))) for i in range(min(len(args), len(devices)))]
    with ComputeWorkers(devices) as workers:
        # Transport CPU tensors once; each worker retains and updates its own
        # activations on its compute device. CUDA tensors never cross processes.
        workers.call('initialize', [map_tensors((
            [args[i] for i in indices], [kwargs[i] for i in indices],
            [masks[i] for i in indices] if masks else [],
        ), device='cpu') for indices in shards])
        del args, kwargs, masks

        for index, layer in enumerate(tqdm(layers, desc='Accumulating covariance', unit='layer', dynamic_ncols=True)):
            # IPC moves CPU storage into shared memory. Send a temporary copy
            # so profiling does not gradually move the entire model there.
            replica = copy.deepcopy(layer) if len(devices) > 1 else layer
            results = workers.call('profile', [(replica, index, adapter.parallel_blocks)] * len(shards))
            del replica
            for group in zip(*results):
                # Normalize only after summing all shards, including uneven ones.
                global_moment += normalized(sum(group))
            del results, group
        logging.info('Accumulating covariance done')
        workers.call('finish_calibration', [()] * len(shards))
        cleanup_memory()

        global_moment = global_moment.to(devices[0])
        global_moment.diagonal().add_(0.01 * global_moment.diagonal().mean())
        _, basis = torch.linalg.eigh(global_moment)
        basis = basis.flip(-1)
        del global_moment
        if final_orientation == 'random':
            basis = basis @ random_orthogonal_upper_left(basis.shape[0], new_embedding_dimension).to(basis.device)

        rotate_embeddings(adapter, basis, new_embedding_dimension)
        slice_embeddings(adapter, new_embedding_dimension)
        workers.call('set_basis', [(basis.cpu(),)] * len(devices))
        for start in range(0, len(layers), len(devices)):
            chunk = layers[start:start + len(devices)]
            jobs = [(copy.deepcopy(layer) if len(devices) > 1 else layer, new_embedding_dimension)
                    for layer in chunk]
            results = workers.call('rotate', jobs)
            del jobs
            # Child processes replace parameter storage when slicing. Install their
            # returned modules explicitly; those replacements cannot update the parent.
            for index, layer in enumerate(results, start):
                if len(devices) > 1:
                    # Returned IPC tensors belong in ordinary RAM once installed.
                    layer.layer._apply(lambda tensor: tensor.clone())
                adapter.set_raw_layer_at(index, layer.layer)
                layers[index] = layer
        rotate_head(adapter, basis, new_embedding_dimension)
        slice_head(adapter, new_embedding_dimension)
    mapping.to_calibration(adapter)


def random_orthogonal_upper_left(total_dim, upper_block_dim):
    """
    Create a square matrix where the upper left block is a random orthogonal matrix, and the remainder is the identity.
    """
    A = np.random.rand(upper_block_dim, upper_block_dim)
    Q, _ = np.linalg.qr(A)
    R = np.eye(total_dim)
    R[:upper_block_dim, :upper_block_dim] = Q
    return torch.from_numpy(R)
