# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
#
# Modified for TrueSlice.

from functools import partial
from typing import Any

import torch
from torch import Tensor

from . import utils
from .config import config
from .model_adapter import LayerAdapter, ModelAdapter


class _InputsCaptured(Exception):
    """Stop the model before final normalization and the language-model head."""


@torch.no_grad()
def prepare_calibration_inputs(
    model_adapter: ModelAdapter,
    batch: dict[str, Tensor],
    *,
    device: torch.device | str | None = None,
) -> tuple[Tensor, tuple, dict[str, Any]]:
    """Capture embeddings and per-decoder kwargs, preserving hybrid-model masks.

    Run the model's native input preparation with temporary decoder forwards that
    record inputs and pass hidden states through unchanged. Keeping the decoder
    modules intact preserves attributes used to select masks in hybrid models.
    The caller places preparation modules for the entire calibration dataset.
    Return CPU tensors for the initial handoff to workers, which then retain
    activations on their compute devices. By default use the embedding device.
    """
    device = model_adapter.get_embeddings()[0].weight.device if device is None else device
    layers = model_adapter.get_layers()
    if not layers:
        raise ValueError('Input capture requires at least one decoder layer')
    captured = [None] * len(layers)
    missing = object()
    original_forwards = [layer.layer.__dict__.get("forward", missing) for layer in layers]

    def capture(index, *args, **kwargs):
        captured[index] = (args, kwargs)
        if index == len(layers) - 1:
            raise _InputsCaptured
        hidden = args[layers[index].hidden_states_args_position]
        return (hidden,) if model_adapter.decoder_returns_tuple else hidden

    try:
        for i, layer in enumerate(layers):
            layer.layer.forward = partial(capture, i)
        batch = utils.map_tensors(batch, device=device)
        model_adapter.model(**{**batch, "use_cache": False, "past_key_values": None})
    except _InputsCaptured:
        pass
    finally:
        for layer, forward in zip(layers, original_forwards):
            if forward is missing:
                # Restore lookup on the class, including after partial setup.
                layer.layer.__dict__.pop("forward", None)
            else:
                layer.layer.forward = forward

    if any(inputs is None for inputs in captured):
        raise RuntimeError('The model did not call every decoder during input capture')
    args, layer_kwargs = utils.map_tensors(
        (captured[0][0], [inputs[1] for inputs in captured]), device='cpu')
    kwargs = {"layer_kwargs": layer_kwargs}
    return args[layers[0].hidden_states_args_position], args, kwargs


@torch.no_grad()
def get_signals(
    layer_adapter: LayerAdapter,
    layer_args: list[tuple],
    layer_kwargs: list[dict[str, Any]],
    *,
    device=None,
    storage_device="cpu",
    capture_mlp_input: bool = False,
) -> list[list[torch.Tensor]]:
    """Run a decoder with optional observation of its intermediate MLP input.

    Return signal groups in moment-accumulation order: MLP inputs (when requested),
    then decoder outputs. The caller must place the layer on the compute device.
    """
    device = config.device if device is None else device
    mlp_ln_inputs = []
    outputs = []

    def hook_fn(_, args: tuple, _output: Any) -> None:
        inp = args[0]  # Position in RMSN.forward args
        mlp_ln_inputs.append(inp.detach().to(device=storage_device, copy=True))

    hook = None
    if capture_mlp_input:
        hook = layer_adapter.get_second_layernorm().register_forward_hook(hook_fn)

    try:
        for i, (layer_args_batch, layer_kwargs_batch) in enumerate(zip(layer_args, layer_kwargs, strict=True)):
            gpu_args_batch, gpu_kwargs_batch = utils.map_tensors(
                [layer_args_batch, layer_kwargs_batch], device=device
            )
            out = layer_adapter.layer(*gpu_args_batch, **gpu_kwargs_batch)

            if isinstance(out, tuple):
                out = out[layer_adapter.hidden_states_output_position]

            outputs.append(out.detach().to(device=storage_device, copy=True))

            if capture_mlp_input:
                if len(mlp_ln_inputs) != i + 1:
                    raise RuntimeError('Expected exactly one MLP normalization call per decoder batch')
                if mlp_ln_inputs[i].ndim == 2:
                    batch_size, seqlen, _ = out.shape
                    mlp_ln_inputs[i] = mlp_ln_inputs[i].reshape(batch_size, seqlen, -1)

            del gpu_args_batch, gpu_kwargs_batch, out

    finally:
        if hook is not None:
            hook.remove()
    return [mlp_ln_inputs, outputs] if capture_mlp_input else [outputs]
