# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
#
# This file contains derivations from
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/modeling_qwen3.py
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
# https://www.apache.org/licenses/LICENSE-2.0

import torch
from torch import FloatTensor, Tensor
from torch.nn import Linear, Module
from transformers import PretrainedConfig, PreTrainedTokenizerBase
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextConfig, Qwen3_5DecoderLayer, Qwen3_5ForCausalLM, Qwen3_5RMSNorm

from trueslice.model_adapter import LayerAdapter, ModelAdapter


class Qwen3_5LayerAdapter(LayerAdapter):
    def __init__(self, layer: Qwen3_5DecoderLayer) -> None:
        super().__init__()
        self._layer: Qwen3_5DecoderLayer = layer

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
        return self.layer.input_layernorm

    def set_first_layernorm(self, layernorm: Module) -> None:
        self.layer.input_layernorm = layernorm

    def get_second_layernorm(self) -> Module:
        return self.layer.post_attention_layernorm

    def set_second_layernorm(self, layernorm: Module) -> None:
        self.layer.post_attention_layernorm = layernorm

    def get_attention_inputs(self) -> list[Linear]:
        if self.layer.layer_type == "linear_attention":
            mixer = self.layer.linear_attn
            return [mixer.in_proj_qkv, mixer.in_proj_z, mixer.in_proj_b, mixer.in_proj_a]
        return [self.layer.self_attn.q_proj, self.layer.self_attn.k_proj, self.layer.self_attn.v_proj]

    def get_attention_output(self) -> Linear:
        if self.layer.layer_type == "linear_attention":
            return self.layer.linear_attn.out_proj
        return self.layer.self_attn.o_proj

    def get_mlp_inputs(self) -> list[Linear]:
        return [self.layer.mlp.gate_proj, self.layer.mlp.up_proj]

    def get_mlp_output(self) -> Linear:
        return self.layer.mlp.down_proj


class Qwen3_5ModelAdapter(ModelAdapter):
    layernorm_weight_offset = 1.0

    def __init__(self, model: Qwen3_5ForCausalLM) -> None:
        super().__init__()
        self._model: Qwen3_5ForCausalLM = model

    @property
    def model(self) -> Module:
        return self._model

    @property
    def config(self) -> PretrainedConfig:
        return self._model.config

    @property
    def config_type(self) -> type:
        return Qwen3_5TextConfig

    @property
    def seqlen(self) -> int:
        return self.config.max_position_embeddings

    @property
    def hidden_size(self) -> int:
        return self.config.hidden_size

    @property
    def original_layer_type(self) -> type:
        return Qwen3_5DecoderLayer

    @property
    def original_layer_norm_type(self) -> type:
        return Qwen3_5RMSNorm

    @property
    def layer_adapter_type(self) -> type:
        return Qwen3_5LayerAdapter

    @property
    def use_cache(self) -> bool:
        return self.config.use_cache

    @use_cache.setter
    def use_cache(self, value: bool) -> None:
        self.config.use_cache = value

    def compute_output_logits(self, input_ids: Tensor) -> FloatTensor:
        return self.model(input_ids=input_ids).logits

    def get_layers(self) -> list[LayerAdapter]:
        return [self.layer_adapter_type(layer) for layer in self.model.model.layers]

    def get_raw_layer_at(self, index: int) -> Module:
        return self.model.model.layers[index]

    def set_raw_layer_at(self, index: int, new_layer: Module) -> None:
        self.model.model.layers[index] = new_layer

    def get_embeddings(self) -> list[Module]:
        return [self.model.model.embed_tokens]

    def get_pre_head_layernorm(self) -> Module:
        return self.model.model.norm

    def set_pre_head_layernorm(self, layernorm: Module) -> None:
        self.model.model.norm = layernorm

    def get_lm_head(self) -> Linear:
        return self.model.lm_head

    def get_pre_layer_modules(self) -> list[Module]:
        return self.get_embeddings() + [self.model.model.rotary_emb]

    def post_init(self, tokenizer: PreTrainedTokenizerBase) -> None:
        # Prevent token ID errors for missing pads
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        self.config.pad_token_id = tokenizer.pad_token_id

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
        if not (model_name.startswith("Qwen/Qwen3.5-")):
            return None

        model = Qwen3_5ForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype, token=token, local_files_only=local_files_only, device_map=device_map, max_memory=max_memory
        )
        model.config.torch_dtype = dtype

        return cls(model)

