# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""
"""

import torch
from torch import FloatTensor, Tensor
from torch.nn import Linear, Module
from transformers import PretrainedConfig, PreTrainedTokenizerBase
from transformers.models.llama.modeling_llama import LlamaConfig, LlamaDecoderLayer, LlamaForCausalLM, LlamaRMSNorm

from trueslice.model_adapter import LayerAdapter, ModelAdapter


class LlamaLayerAdapter(LayerAdapter):
    def __init__(self, layer: LlamaDecoderLayer) -> None:
        super().__init__()
        self._layer: LlamaDecoderLayer = layer

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
        return [self.layer.self_attn.q_proj, self.layer.self_attn.k_proj, self.layer.self_attn.v_proj]

    def get_attention_output(self) -> Linear:
        return self.layer.self_attn.o_proj

    def get_mlp_inputs(self) -> list[Linear]:
        return [self.layer.mlp.gate_proj, self.layer.mlp.up_proj]

    def get_mlp_output(self) -> Linear:
        return self.layer.mlp.down_proj


class LlamaModelAdapter(ModelAdapter):
    def __init__(self, model: LlamaForCausalLM) -> None:
        super().__init__()
        self._model: LlamaForCausalLM = model

    @property
    def model(self) -> Module:
        return self._model

    @property
    def config(self) -> PretrainedConfig:
        return self._model.config

    @property
    def config_type(self) -> type:
        return LlamaConfig

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
        return LlamaDecoderLayer

    @property
    def original_layer_norm_type(self) -> type:
        return LlamaRMSNorm

    @property
    def layer_adapter_type(self) -> type:
        return LlamaLayerAdapter

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
        if not (model_name.startswith("meta-llama/Llama-2") or model_name.startswith("meta-llama/Meta-Llama-3") or
                model_name.startswith("meta-llama/Llama-3")):
            return None

        model = LlamaForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype, token=token, local_files_only=local_files_only
        )
        model.config.torch_dtype = dtype

        return cls(model)

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
        if not (model_name.startswith("meta-llama/Llama-2") or model_name.startswith("meta-llama/Meta-Llama-3") or
                model_name.startswith("meta-llama/Llama-3")):
            return None

        class UninitializedLlamaForCausalLM(LlamaForCausalLM):
            def _init_weights(self, _) -> None:
                # Prevent weight initialization
                pass

        config = LlamaConfig.from_pretrained(
            model_path, torch_dtype=dtype, token=token, local_files_only=local_files_only
        )
        model = UninitializedLlamaForCausalLM(config)
        model = model.to(dtype=dtype)

        return cls(model)
