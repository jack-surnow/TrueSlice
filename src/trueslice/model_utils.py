# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Any

import torch
from torch import Tensor

from trueslice.modules import RMSN

from . import utils
from .config import config
from .model_adapter import LayerAdapter, ModelAdapter


@torch.no_grad()
def get_layer0_inputs(model_adapter: ModelAdapter, batch: dict[str, Tensor]) -> tuple[Tensor, tuple, dict[str, Any]]:
    """Capture embeddings and each decoder's kwargs without running decoders.

    Unlike first-layer-only capture, this preserves hybrid models' per-layer masks.
    Catchers pass hidden states through unchanged until the last layer is reached.
    """
    layers = model_adapter.get_layers()
    pre_layer_modules = model_adapter.get_pre_layer_modules()
    saved_kwargs = []

    class InputsCaptured(Exception):
        pass

    class Catcher(torch.nn.Module):
        def __init__(self, layer_adapter, last):
            super().__init__()
            self.last = last
            self.hidden_states_position = layer_adapter.hidden_states_args_position
            for name in ("attention_type", "layer_type", "block_type"):
                if hasattr(layer_adapter.layer, name):
                    setattr(self, name, getattr(layer_adapter.layer, name))

        def forward(self, *args, **kwargs):
            if not saved_kwargs:
                self.saved_args = args
            saved_kwargs.append(kwargs)
            if self.last:
                raise InputsCaptured
            hidden = args[self.hidden_states_position]
            return (hidden,) if model_adapter.decoder_returns_tuple else hidden

    catchers = [Catcher(layer, i == len(layers) - 1) for i, layer in enumerate(layers)]
    try:
        for module in pre_layer_modules:
            module.to(config.device)
        for i, catcher in enumerate(catchers):
            model_adapter.set_raw_layer_at(i, catcher)
        batch = utils.map_tensors(batch, device=config.device)
        model_adapter.model(**{**batch, "use_cache": False, "past_key_values": None})
    except InputsCaptured:
        pass
    finally:
        for i, layer in enumerate(layers):
            model_adapter.set_raw_layer_at(i, layer.layer)
        for module in pre_layer_modules:
            module.to("cpu")
        utils.cleanup_memory()

    args = utils.map_tensors(catchers[0].saved_args, device="cpu")
    kwargs = {"layer_kwargs": utils.map_tensors(saved_kwargs, device="cpu")}
    return args[layers[0].hidden_states_args_position], args, kwargs


@torch.no_grad()
def get_signals_sequential(
    layer_adapter: LayerAdapter, layer_args: list[tuple], layer_kwargs: list[dict[str, Any]]
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """
    Take the input signals ("activations") for a layer, run the layer forward.
    Return the output of the layer (not layernormed) and the input to the MLP (pre-layernorm).
    """
    mlp_ln_inputs = []
    outputs = []

    def hook_fn(_, args: tuple, _output: Any) -> None:
        inp = args[0]  # Position in RMSN.forward args
        mlp_ln_inputs.append(inp.detach().to("cpu").clone())

    second_layernorm = layer_adapter.get_second_layernorm()
    hook = second_layernorm.register_forward_hook(hook_fn)

    try:
        for i, (layer_args_batch, layer_kwargs_batch) in enumerate(zip(layer_args, layer_kwargs)):
            gpu_args_batch, gpu_kwargs_batch = utils.map_tensors(
                [layer_args_batch, layer_kwargs_batch], device=config.device
            )
            out = layer_adapter.layer(*gpu_args_batch, **gpu_kwargs_batch)

            if isinstance(out, tuple):
                out = out[layer_adapter.hidden_states_output_position]

            cpu_out = out.detach().to("cpu").clone()
            outputs.append(cpu_out)

            if mlp_ln_inputs[i].ndim == 2:
                batch_size, seqlen, _ = out.shape  # both batch_size and seqlen are can vary from batch to batch
                mlp_ln_inputs[i] = mlp_ln_inputs[i].reshape(batch_size, seqlen, -1)

            del gpu_args_batch, gpu_kwargs_batch, out

    finally:
        hook.remove()
    return mlp_ln_inputs, outputs


@torch.no_grad()
def get_signals_parallel(
    layer_adapter: LayerAdapter, layer_args: list[tuple], layer_kwargs: list[dict[str, Any]]
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """
    Take the input signals ("activations") for a layer, run the layer forward.
    Return the output of the layer (not layernormed) and the input to the MLP (pre-layernorm).
    """
    outputs = []

    for layer_args_batch, layer_kwargs_batch in zip(layer_args, layer_kwargs):
        gpu_args_batch, gpu_kwargs_batch = utils.map_tensors(
            [layer_args_batch, layer_kwargs_batch], device=config.device
        )
        
        out = layer_adapter.layer(*gpu_args_batch, **gpu_kwargs_batch)

        if isinstance(out, tuple):
            out = out[layer_adapter.hidden_states_output_position]

        cpu_out = out.detach().to("cpu").clone()
        outputs.append(cpu_out)

        # Force Python to release the GPU tensors immediately
        del gpu_args_batch, gpu_kwargs_batch, out

    return outputs
