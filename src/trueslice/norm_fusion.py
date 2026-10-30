# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
#
# Modified for TrueSlice.

import logging
from typing import Callable, Iterable, Optional, TypeVar

import math
import torch
import torch.nn as nn
from torch.nn import Linear, Module, Parameter

from .model_adapter import ModelAdapter


AnyModule = TypeVar("AnyModule", bound=Module)


class RMSN(torch.nn.Module):
    """
    This class implements the Root Mean Square Normalization (RMSN) layer.
    We use the implementation from LLAMARMSNorm here:
    https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L75
    """

    def __init__(self, mean_dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.mean_dim = mean_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        if x.dtype in (torch.float16, torch.bfloat16):
            x = x.to(torch.float32)
        variance = x.pow(2).sum(-1, keepdim=True) / self.mean_dim
        x = x * torch.rsqrt(variance + self.eps)
        return x.to(input_dtype)


def fuse_modules(model_adapter: ModelAdapter) -> None:
    """
    This function fuses the linear and layernorm into each other inplace.
    After this function is called, the model should outputs the same results as before.

    args:
        model_adapter: A ModelAdapter for the model to be fused
    """

    logging.info("Fusing layernorm modules")

    # make a copy of the weights in the lm head, which are shared with embeddings...
    head = model_adapter.get_lm_head()
    head.weight = Parameter(head.weight.clone().detach())
    model_adapter.model.config.tie_word_embeddings = False

    # We add the mean subtraction to the first embeddings
    if model_adapter.should_bake_mean_into_linear:
        for W in model_adapter.get_embeddings():
            W_ = W.weight.data.double()
            W.weight.data = (W_ - W_.mean(dim=-1, keepdim=True)).to(W.weight.data.dtype)

    layer_adapters = model_adapter.get_layers()
    weight_offset = model_adapter.layernorm_weight_offset

    # First we modify the layernorms to fold their weights
    for layer_adapter in layer_adapters:
        if model_adapter.parallel_blocks:
            fuse_ln_linear(
                layer_adapter.get_first_layernorm(),
                layer_adapter.get_attention_inputs() + layer_adapter.get_mlp_inputs(),
                weight_offset=weight_offset,
            )
        else:
            fuse_ln_linear(layer_adapter.get_first_layernorm(), layer_adapter.get_attention_inputs(), weight_offset=weight_offset)
            fuse_ln_linear(layer_adapter.get_second_layernorm(), layer_adapter.get_mlp_inputs(), weight_offset=weight_offset)

        if model_adapter.should_bake_mean_into_linear:
            # Then we bake the mean substitution into the previous linear layers
            bake_mean_into_linear(layer_adapter.get_attention_output())
            bake_mean_into_linear(layer_adapter.get_mlp_output())

    fuse_ln_linear(model_adapter.get_pre_head_layernorm(), [model_adapter.get_lm_head()], weight_offset=weight_offset)

    config = model_adapter.model.config
    eps = getattr(config, "rms_norm_eps", 
        getattr(config, "layer_norm_eps", 
        getattr(config, "layer_norm_epsilon", 1e-5)))

    # Replace only residual-stream norms, preserving head/internal mixer norms.
    for layer_adapter in layer_adapters:
        layer_adapter.set_first_layernorm(RMSN(model_adapter.hidden_size, eps))
        if not model_adapter.parallel_blocks:
            layer_adapter.set_second_layernorm(RMSN(model_adapter.hidden_size, eps))
    model_adapter.set_pre_head_layernorm(RMSN(model_adapter.hidden_size, eps))
    
    logging.info("Fusing layernorm modules done")


def bake_mean_into_linear(linear: Linear) -> None:
    """
    This function takes a linear layer and subtracts the means from the
    weights and biases. This will result in the linear layer performing
    the mean substitution which is usually done inside layernorm.
    """
    linear_dtype = linear.weight.dtype
    W_ = linear.weight.data.double()
    linear.weight.data = W_ - W_.mean(dim=-2, keepdim=True)
    linear.weight.data = linear.weight.data.to(linear_dtype)
    if linear.bias is not None:
        b_ = linear.bias.data.double()
        linear.bias.data = b_ - b_.mean()
        linear.bias.data = linear.bias.data.to(linear_dtype)


def fuse_ln_linear(layernorm: Module, linear_layers: Iterable[Linear], weight_offset: float = 0.0) -> None:
    """
    fuse the linear operations in Layernorm into the adjacent linear blocks.
    """
    for linear in linear_layers:
        linear_dtype = linear.weight.dtype

        # Calculating new weight and bias
        W_ = linear.weight.data.double()
        linear.weight.data = (W_ * (layernorm.weight.to(device=W_.device, dtype=torch.float64) + weight_offset)).to(linear_dtype)

        if hasattr(layernorm, 'bias'):
            if linear.bias is None:
                linear.bias = Parameter(torch.zeros(linear.out_features, dtype=torch.float64, device=W_.device))
            linear.bias.data = linear.bias.data.double() + torch.matmul(W_, layernorm.bias.to(device=W_.device, dtype=torch.float64))
            linear.bias.data = linear.bias.data.to(linear_dtype)


@torch.no_grad()
def restore_norms(model_adapter: ModelAdapter, d_new) -> None:
    logging.info("Restoring original norm layers with identity parameters")
    
    model_param = next(model_adapter.model.parameters())
    target_device = model_param.device
    target_dtype = model_param.dtype

    config = model_adapter.model.config
    eps = getattr(config, "rms_norm_eps", 
          getattr(config, "layer_norm_eps", 
          getattr(config, "layer_norm_epsilon", 1e-5)))

    d_original = model_adapter.hidden_size
    gain_scale = math.sqrt(d_original / d_new)
    restored_eps = eps * gain_scale ** 2

    if not issubclass(model_adapter.original_layer_norm_type, nn.LayerNorm):
        # Scale the residual stream by sqrt(d_new / d_original), so native
        # RMSNorm's mean uses the same denominator as the fused RMSN. Its gain
        # cancels this scale. Keeping the original epsilon makes standard HF
        # reloads exact without changing the epsilon of internal Q/K norms.
        residual_scale = 1 / gain_scale
        if residual_scale != 1:
            for embedding in model_adapter.get_embeddings():
                embedding.weight.mul_(residual_scale)
            for layer_adapter in model_adapter.get_layers():
                for output in (layer_adapter.get_attention_output(), layer_adapter.get_mlp_output()):
                    output.weight.mul_(residual_scale)
                    if output.bias is not None:
                        output.bias.mul_(residual_scale)
        restored_eps = eps

    def create_norm() -> nn.Module:
        norm_layer = model_adapter.original_layer_norm_type(d_new, eps=restored_eps).to(device=target_device, dtype=target_dtype)

        if hasattr(norm_layer, 'weight') and norm_layer.weight is not None:
            # Qwen3.5 applies (1 + weight); ordinary RMSNorm applies weight.
            nn.init.constant_(norm_layer.weight, gain_scale - model_adapter.layernorm_weight_offset)

        if hasattr(norm_layer, 'bias') and norm_layer.bias is not None:
            nn.init.zeros_(norm_layer.bias)

        return norm_layer

    for layer_adapter in model_adapter.get_layers():
        layer_adapter.set_first_layernorm(create_norm()) 

        if not model_adapter.parallel_blocks:
            layer_adapter.set_second_layernorm(create_norm())

    model_adapter.set_pre_head_layernorm(create_norm())

    logging.info("Restoration of original norm layers complete")
