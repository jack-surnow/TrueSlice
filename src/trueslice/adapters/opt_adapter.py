# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
#
# Modified for TrueSlice.

import torch
from torch import FloatTensor, Tensor
from torch.nn import LayerNorm, Linear, Module
from transformers import PretrainedConfig
from transformers.models.opt.modeling_opt import OPTConfig, OPTDecoderLayer, OPTForCausalLM

from trueslice.model_adapter import LayerAdapter, ModelAdapter


class OPTLayerAdapter(LayerAdapter):
    def __init__(self, layer: OPTDecoderLayer) -> None:
        super().__init__()
        self._layer: OPTDecoderLayer = layer

    @property
    def layer(self) -> Module:
        return self._layer

    @property
    def hidden_states_args_position(self) -> int:
        return 0

    @property
    def hidden_states_output_position(self) -> int:
        return 0

    def get_first_layernorm(self) -> Module:
        return self.layer.self_attn_layer_norm

    def set_first_layernorm(self, layernorm: Module) -> None:
        self.layer.self_attn_layer_norm = layernorm

    def get_second_layernorm(self) -> Module:
        return self.layer.final_layer_norm

    def set_second_layernorm(self, layernorm: Module) -> None:
            self.layer.final_layer_norm = layernorm

    def get_attention_inputs(self) -> list[Linear]:
        return [self.layer.self_attn.q_proj, self.layer.self_attn.k_proj, self.layer.self_attn.v_proj]

    def get_attention_output(self) -> Linear:
        return self.layer.self_attn.out_proj

    def get_mlp_inputs(self) -> list[Linear]:
        return [self.layer.fc1]

    def get_mlp_output(self) -> Linear:
        return self.layer.fc2


class OPTModelAdapter(ModelAdapter):
    decoder_returns_tuple = True
    should_bake_mean_into_linear = True
    
    def __init__(self, model: OPTForCausalLM) -> None:
        super().__init__()
        self._model: OPTForCausalLM = model

    @property
    def model(self) -> Module:
        return self._model

    @property
    def config(self) -> PretrainedConfig:
        return self._model.config

    @property
    def config_type(self) -> type:
        return OPTConfig

    @property
    def seqlen(self) -> int:
        return self.config.max_position_embeddings

    @property
    def hidden_size(self) -> int:
        return self.config.hidden_size

    @property
    def original_layer_type(self) -> type:
        return OPTDecoderLayer

    @property
    def original_layer_norm_type(self) -> type:
        return LayerNorm

    @property
    def layer_adapter_type(self) -> type:
        return OPTLayerAdapter


    @property
    def use_cache(self) -> bool:
        return self.config.use_cache

    @use_cache.setter
    def use_cache(self, value: bool) -> None:
        self.config.use_cache = value

    def compute_output_logits(self, input_ids: Tensor) -> FloatTensor:
        return self.model(input_ids=input_ids).logits


    def get_layers(self) -> list[LayerAdapter]:
        return [self.layer_adapter_type(layer) for layer in self.model.model.decoder.layers]

    def get_raw_layer_at(self, index: int) -> Module:
        return self.model.model.decoder.layers[index]

    def set_raw_layer_at(self, index: int, new_layer: Module) -> None:
        self.model.model.decoder.layers[index] = new_layer

    def get_embeddings(self) -> list[Module]:
        return [self.model.model.decoder.embed_tokens, self.model.model.decoder.embed_positions]

    def get_pre_head_layernorm(self) -> Module:
        # pre_head_layernorm = self.model.model.decoder.final_layer_norm
        # assert pre_head_layernorm is not None
        # return pre_head_layernorm
        return self.model.model.decoder.final_layer_norm

    def set_pre_head_layernorm(self, layernorm: Module) -> None:
        self.model.model.decoder.final_layer_norm = layernorm

    def get_lm_head(self) -> Linear:
        return self.model.lm_head

    @classmethod
    def _from_pretrained(
        cls,
        model_name: str,
        model_path: str,
        *,
        device_map: str | dict[str, int] | None = "cpu",
        max_memory: dict[str, int] | None = None,
        dtype: torch.dtype = torch.float16,
        local_files_only: bool = False,
        token: str | bool | None = None,
    ) -> ModelAdapter | None:
        if not model_name.startswith("facebook/opt"):
            return None

        model = OPTForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype, local_files_only=local_files_only, device_map=device_map, max_memory=max_memory
        )
        model.config.torch_dtype = dtype

        return OPTModelAdapter(model)

