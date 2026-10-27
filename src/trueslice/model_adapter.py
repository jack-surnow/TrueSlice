# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
#
# Modified for TrueSlice.

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

import torch
from torch import FloatTensor, Tensor
from torch.nn import Linear, Module
from transformers import PreTrainedTokenizerBase

"""
Model integration interfaces for TrueSlice.

Add a ModelAdapter subclass for loading and model-level access, and a LayerAdapter
subclass for decoder components. Import the model adapter before loading a model.
See src/trueslice/adapters/llama_adapter.py for an example.
"""


class LayerAdapter(ABC):
    """
    To implement a new layer adapter, implement the interface defined in this class
    """

    @property
    @abstractmethod
    def layer(self) -> Module:
        """
        Instance of the transformer layer to be wrapped. This contains the forward() method of the original model
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def hidden_states_args_position(self) -> int:
        """
        Returns the position of the hidden_states argument in the layer's forward method.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def hidden_states_output_position(self) -> int:
        """
        Returns the position of the hidden_states in the output of the layer's forward method.
        """
        raise NotImplementedError

    @abstractmethod
    def get_first_layernorm(self) -> Module:
        """
        Returns the first layer norm in the layer, usually the one before the attention component.
        """
        raise NotImplementedError

    @abstractmethod
    def set_first_layernorm(self, layernorm: Module) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_second_layernorm(self) -> Module:
        """Return the normalization module before the MLP, or None for a shared parallel-block norm."""
        raise NotImplementedError

    @abstractmethod
    def set_second_layernorm(self, layernorm: Module) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_attention_inputs(self) -> Sequence[Linear]:
        """
        Returns a list of the Linear layers (nn.modules) that are inputs to the attention component.
        """
        raise NotImplementedError

    @abstractmethod
    def get_attention_output(self) -> Linear:
        """
        Returns the Linear layer (nn.module) that is the output of the attention component.
        """
        raise NotImplementedError

    @abstractmethod
    def get_mlp_inputs(self) -> Sequence[Linear]:
        """Return the MLP input projections, including any separate gate projection."""
        raise NotImplementedError

    @abstractmethod
    def get_mlp_output(self) -> Linear:
        """
        Returns the Linear layer (nn.module) that is the output of the MLP component (usually fc2 or down_proj)
        """
        raise NotImplementedError

    def get_updated_args(self, hidden_states: Any, args: tuple) -> tuple:
        """
        `args` is a tuple of the arguments to the layer's forward method. hidden_states is the new value for the
        hidden_states argument. This method returns a new tuple of arguments with the hidden_states argument updated.
        """
        return (
            args[: self.hidden_states_args_position] + (hidden_states,) + args[self.hidden_states_args_position + 1 :]
        )


class ModelAdapter(ABC):
    """
    To implement a new model adapter, implement the interface defined in this class
    """
    decoder_returns_tuple = False
    parallel_blocks = False
    should_bake_mean_into_linear = False
    layernorm_weight_offset = 0.0

    @property
    @abstractmethod
    def model(self) -> Module:
        """
        The original model that trueslice interacts with.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def config(self) -> object:
        """
        The model config
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def config_type(self) -> type:
        """
        Type of the config class
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def seqlen(self) -> int:
        """
        The (maximum) sequence length of the model
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def hidden_size(self) -> int:
        """
        The hidden size of the model
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def original_layer_type(self) -> type:
        """
        Type of the transformer layer containing forward() method of the original model
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def original_layer_norm_type(self) -> type:
        """
        The class of the LayerNorm (or equivalent) in the original model, so that we can replace it with RMSNorm
        (needed for computational invariance).
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def layer_adapter_type(self) -> type:
        """
        Type of the class implementing the sliceGPT.LayerAdapter interface
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def use_cache(self) -> bool:
        """Must define a setter"""
        raise NotImplementedError

    @use_cache.setter
    @abstractmethod
    def use_cache(self, value: bool) -> None:
        raise NotImplementedError

    @abstractmethod
    def compute_output_logits(self, input_ids: Tensor) -> FloatTensor:
        """
        Returns the logits for the model on the given input_ids. For example, this might look like:
            `self._model(input_ids=input_ids).logits`
        """
        raise NotImplementedError

    @abstractmethod
    def get_layers(self) -> Sequence[LayerAdapter]:
        """
        Returns a list of LayerAdapters, one for each layer in the model.
        """
        raise NotImplementedError

    @abstractmethod
    def get_raw_layer_at(self, index: int) -> Module:
        """
        Returns the raw layer (no adapter) at the given index.
        """
        raise NotImplementedError

    @abstractmethod
    def set_raw_layer_at(self, index: int, new_layer: Module) -> None:
        """
        Assigns the given layer to the model at the given index.
        """
        raise NotImplementedError

    @abstractmethod
    def get_embeddings(self) -> list[Module]:
        """
        Returns a list of the embedding modules in the model.
        """
        raise NotImplementedError

    @abstractmethod
    def get_pre_head_layernorm(self) -> Module:
        """
        Returns the layer norm (or equivalent) before the head of the model.
        """
        raise NotImplementedError

    @abstractmethod
    def set_pre_head_layernorm(self, layernorm: Module) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_lm_head(self) -> Linear:
        """
        Returns the linear layer at the head of the model (usually of size hidden-size x vocab-size)
        """
        raise NotImplementedError

    @property
    def no_split_module_classes(self) -> list[str] | None:
        """
        A list of strings specifying the class names of modules that should not be split.
        See https://huggingface.co/docs/accelerate/concept_guides/big_model_inference for more details.
        """
        return [self.original_layer_type.__name__]

    def get_pre_layer_modules(self) -> list[Module]:
        """Modules executed before the first decoder during calibration."""
        return self.get_embeddings()

    def post_init(self, tokenizer: PreTrainedTokenizerBase) -> None:
        """
        This method is called after the model is initialized and all the properties are set.
        Override in subclasses to perform any additional setup.
        """
        pass

    @classmethod
    def from_model(
        cls,
        model_name: str,
        model_path: str,
        *,
        device_map: str | dict[str, str | int | torch.device] | None = "cpu",
        max_memory: dict[str | int, str | int] | None = None,
        offload_folder: str | None = None,
        dtype: torch.dtype = torch.float16,
        local_files_only: bool = False,
        token: str | bool | None = None,
    ) -> ModelAdapter:
        """
        Create the model based on the given name path and return the corresponding ModelAdapter instance.
        Raise NotImplementedError if the model is not supported.
        Note: for this method to work the corresponding ModelAdapter subclass must be imported.

        Args:
            model_name: The name of the model, e.g. 'microsoft/phi-2'.
            model_path: The path to the model.
            device_map: The device map to use for the model.
            dtype: The torch dtype to create the model with.
            local_files_only: Whether to only load local files (no attempt to download).
            token: The token to use for authentication.

        Returns:
            The corresponding ModelAdapter instance.
        """

        def find_recursively(adapter_cls: type[ModelAdapter]) -> ModelAdapter | None:
            """
            Recursively search for a subclass that can handle the model.
            """
            # depth first search to find the most specific subclass that can handle the model
            for subclass in adapter_cls.__subclasses__():
                candidate = find_recursively(subclass)
                if candidate is not None:
                    return candidate

            if inspect.isabstract(adapter_cls):
                return None

            return adapter_cls._from_pretrained(
                model_name,
                model_path=model_path,
                device_map=device_map,
                max_memory=max_memory,
                offload_folder=offload_folder,
                dtype=dtype,
                local_files_only=local_files_only,
                token=token,
            )

        adapter = find_recursively(cls)
        if adapter is not None:
            return adapter

        raise NotImplementedError(f"{model_path} is neither a Hugging Face model nor a supported local model.")


    @classmethod
    @abstractmethod
    def _from_pretrained(
        cls,
        model_name: str,
        model_path: str,
        *,
        device_map: str | dict[str, str | int | torch.device] | None = "cpu",
        max_memory: dict[str | int, str | int] | None = None,
        offload_folder: str | None = None,
        dtype: torch.dtype = torch.float16,
        local_files_only: bool = False,
        token: str | bool | None = None,
    ) -> ModelAdapter | None:
        """
        Load the pretrained model from the given path and return a ModelAdapter instance.
        Return None if the model_name is not supported.
        See `from_model` for more details.
        """
        raise NotImplementedError
