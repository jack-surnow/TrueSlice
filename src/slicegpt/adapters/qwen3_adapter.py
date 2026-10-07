# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
#
# Modified for TrueSlice.
# Adapted to support Qwen3 and residual rotation shortcuts.
#
# Additional notice retained from the inherited adapter code:
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This file also contains derivations from Hugging Face Transformers' Qwen3 implementation:
# https://github.com/huggingface/transformers/blob/v5.2.0/src/transformers/models/qwen3/modeling_qwen3.py
# Copyright 2025 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
#
# The Transformers-derived portions are licensed under the Apache License, Version 2.0:
# https://www.apache.org/licenses/LICENSE-2.0

import torch
from torch import FloatTensor, LongTensor, Tensor, matmul
from torch.nn import Linear, Module
from transformers import PretrainedConfig, PreTrainedTokenizerBase
from transformers.cache_utils import Cache
from transformers.models.qwen3.modeling_qwen3 import Qwen3Config, Qwen3DecoderLayer, Qwen3ForCausalLM, Qwen3RMSNorm

from slicegpt.model_adapter import LayerAdapter, ModelAdapter


class CompressedQwen3DecoderLayer(Qwen3DecoderLayer):
    """
    This class simulates the Qwen3DecoderLayer class from transformers
    (https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/modeling_qwen3.py#L376)
    but with the addition of a shortcut_Q attribute. This attribute is used to rotate the residual tensors.
    """

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        position_ids: LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        cache_position: LongTensor | None = None,
        position_embeddings: tuple[Tensor, Tensor] | None = None,
        **kwargs,
    ) -> Tensor:
        """
        """
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )

        if self.attn_shortcut_Q is not None:
            residual = matmul(residual, self.attn_shortcut_Q)
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)

        if self.mlp_shortcut_Q is not None:
            residual = matmul(residual, self.mlp_shortcut_Q)
        return residual + hidden_states


class Qwen3LayerAdapter(LayerAdapter):
    def __init__(self, layer: Qwen3DecoderLayer) -> None:
        super().__init__()
        self._layer: Qwen3DecoderLayer = layer

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

    def get_q_norm(self) -> Module | None:
        return getattr(self.layer.self_attn, "q_norm", None)

    def get_k_norm(self) -> Module | None:
        return getattr(self.layer.self_attn, "k_norm", None)

    def get_attention_inputs(self) -> list[Linear]:
        return [self.layer.self_attn.q_proj, self.layer.self_attn.k_proj, self.layer.self_attn.v_proj]

    def get_attention_output(self) -> Linear:
        return self.layer.self_attn.o_proj

    def get_mlp_inputs(self) -> list[Linear]:
        return [self.layer.mlp.gate_proj, self.layer.mlp.up_proj]

    def get_mlp_output(self) -> Linear:
        return self.layer.mlp.down_proj


class Qwen3ModelAdapter(ModelAdapter):
    def __init__(self, model: Qwen3ForCausalLM) -> None:
        super().__init__()
        self._model: Qwen3ForCausalLM = model

    @property
    def model(self) -> Module:
        return self._model

    @property
    def config(self) -> PretrainedConfig:
        return self._model.config

    @property
    def config_type(self) -> type:
        return Qwen3Config

    @property
    def parallel_blocks(self) -> bool:
        return False

    @property
    def seqlen(self) -> int:
        return self.config.max_position_embeddings

    @property
    def hidden_size(self) -> int:
        return self.config.hidden_size

    @property
    def should_bake_mean_into_linear(self) -> bool:
        return False

    @property
    def original_layer_type(self) -> type:
        return Qwen3DecoderLayer

    @property
    def original_layer_norm_type(self) -> type:
        return Qwen3RMSNorm

    @property
    def layer_adapter_type(self) -> type:
        return Qwen3LayerAdapter

    @property
    def compressed_layer_type(self) -> type:
        return CompressedQwen3DecoderLayer

    @property
    def use_cache(self) -> bool:
        return self.config.use_cache

    @use_cache.setter
    def use_cache(self, value: bool) -> None:
        self.config.use_cache = value

    def compute_output_logits(self, input_ids: Tensor) -> FloatTensor:
        return self.model(input_ids=input_ids).logits

    def convert_layer_to_compressed(self, layer: Module, layer_idx: int | None) -> Module:
        compressed_layer = self.compressed_layer_type(self.config, layer_idx).to(self.config.torch_dtype)
        compressed_layer.load_state_dict(layer.state_dict(), strict=True)
        return compressed_layer

    def get_layers(self) -> list[LayerAdapter]:
        return [self.layer_adapter_type(layer) for layer in self.model.model.layers]

    def get_raw_layer_at(self, index: int) -> Module:
        return self.model.model.layers[index]

    def set_raw_layer_at(self, index: int, new_layer: Module) -> None:
        self.model.model.layers[index] = new_layer

    def get_embeddings(self) -> list[Module]:
        return [self.model.model.embed_tokens]

    def get_pre_head_layernorm(self) -> Module:
        # pre_head_layernorm = self.model.model.norm
        # assert isinstance(pre_head_layernorm, self.original_layer_norm_type)
        # return pre_head_layernorm
        return self.model.model.norm

    def set_pre_head_layernorm(self, layernorm: Module) -> None:
        self.model.model.norm = layernorm

    def get_lm_head(self) -> Linear:
        return self.model.lm_head

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
        dtype: torch.dtype = torch.float16,
        local_files_only: bool = False,
        token: str | bool | None = None,
    ) -> ModelAdapter | None:
        if not ("qwen" in model_name.lower()):
            return None

        model = Qwen3ForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype, token=token, local_files_only=local_files_only
        )
        model.config.torch_dtype = dtype

        return Qwen3ModelAdapter(model)

    @classmethod
    def _from_uninitialized(
        cls,
        model_name: str,
        model_path: str,
        *,
        dtype: torch.dtype = torch.float16,
        local_files_only: bool = False,
        token: str | bool | None = None,
    ) -> ModelAdapter | None:
        if not ("qwen" in model_name.lower()):
            return None

        class UninitializedQwen3ForCausalLM(Qwen3ForCausalLM):
            def _init_weights(self, _) -> None:
                # Prevent weight initialization
                pass

        config = Qwen3Config.from_pretrained(
            model_path, torch_dtype=dtype, token=token, local_files_only=local_files_only
        )
        model = UninitializedQwen3ForCausalLM(config)
        model = model.to(dtype=dtype)

        return Qwen3ModelAdapter(model)
