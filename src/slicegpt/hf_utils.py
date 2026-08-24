# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import pathlib
from typing import Any

import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from .layernorm_fusion import fuse_modules, replace_layers
from .model_adapter import ModelAdapter, SlicingConfig
from .rotate import slice_rotated_model


def do_not_initialize(func):
    """
    A decorator that prevents initialization of torch.nn modules.
    """

    def skip(*args, **kwargs) -> None:
        pass

    def wrapper(*args, **kwargs):
        kaiming_fn = torch.nn.init.kaiming_uniform_
        uniform_fn = torch.nn.init.uniform_
        normal_fn = torch.nn.init.normal_

        torch.nn.init.kaiming_uniform_ = skip
        torch.nn.init.uniform_ = skip
        torch.nn.init.normal_ = skip

        result = func(*args, **kwargs)

        torch.nn.init.kaiming_uniform_ = kaiming_fn
        torch.nn.init.uniform_ = uniform_fn
        torch.nn.init.normal_ = normal_fn

        return result

    return wrapper


@do_not_initialize
def get_model_and_tokenizer(
    model_name: str,
    model_path: str | None = None,
    *,
    uninitialized: bool = False,
    dtype: torch.dtype = torch.float16,
    token: str | bool | None = None,
) -> tuple[ModelAdapter, PreTrainedTokenizerBase]:
    """
    Load the model and the tokenizer from the given path.
    Set uninitialized to True when loading a pre-rotated and sliced model; in this case no weights are loaded
    in this method.
    The corresponding model adapter class must be imported before calling this method.
    Scenarios:
    - Rotate & slice HF model: model_name = name, model_path = empty, uninitialized = False
        -> Obtain the model config and weights from HF through path = name.
        -> Ignore model_path if provided.
    - Slice pre-rotated HF model: model_name = name, model_path = empty or local path, uninitialized = True
        -> Obtain the model config from HF via path = name and create uninitialized model.
        -> If the model_path is provided, confirm this use case by checking that config.json does not exist.
        -> There are no other uses of model_path in this case.
    - Rotate & slice local model: model_name = name, model_path = local path, uninitialized = False
        -> Obtain the model config through path, and the pretrained weights from the local path.
        -> Use the model name only to determine the correct model adapter to use.
    - Slice pre-rotated local model: model_name = name, model_path = local path, uninitialized = True
        -> Obtain the model config from the local path and create an uninitialized model.
        -> Use the model name only to determine the correct model adapter to use.
        -> Confirm this case by checking that config.json exists.
    """
    model_type = "uninitialized" if uninitialized else "pretrained"
    local_model = model_path is not None

    if local_model and uninitialized:
        local_model = (pathlib.Path(model_path) / "config.json").exists()

    # for HF models the path to use is the model name
    if not local_model:
        model_path = model_name

    logging.info(
        f"Loading %s config %s from %s",
        model_name,
        "and model weights" if not uninitialized else "",
        model_path if local_model else 'Hugging Face',
    )

    model_adapter = ModelAdapter.from_model(
        model_name,
        model_path=model_path,
        model_type=model_type,
        dtype=dtype,
        local_files_only=local_model,
        token=token,
    )

    model = model_adapter.model
    model.seqlen = model.config.max_position_embeddings
    model.eval()  # This switches off dropout.
    model_adapter.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, token=token, local_files_only=local_model)

    model_adapter.post_init(tokenizer)
    logging.info("Loading model done")

    return model_adapter, tokenizer


@do_not_initialize
def load_sliced_model(
    model_name: str,
    sliced_model_path: str,
    *,
    token: str | None = None,
    lora_config: Any = None,
    sparsity: float | None = None,
    round_interval: int | None = 1,
) -> tuple[ModelAdapter, PreTrainedTokenizerBase]:
    """
    Load the sliced model and the tokenizer from the given path. If lora_config: peft.LoraConfig is supplied
    as an arg then this function will return a PEFT model (post-slicing finetuned model). Despite being declared as
    "Any", lora_config is supposed to have the type peft.LoraConfig. It has type "Any" in the function's signature,
    so that it would be possible to use it without taking a dependency on peft, when one is not required.
    The corresponding model adapter class must be imported before calling this method.
    """
    my_model_suffix = pathlib.Path(model_name).name
    my_sliced_model_name = f"{my_model_suffix}_{sparsity}.pt"
    my_sliced_model_config = f"{my_model_suffix}_{sparsity}.json"

    model_adapter, tokenizer = get_model_and_tokenizer(
        model_name,
        model_path=sliced_model_path,
        uninitialized=True,
        token=token,
    )
    replace_layers(model_adapter)
    fuse_modules(model_adapter)

    hidden_size = model_adapter.hidden_size
    for layer_adapter in model_adapter.get_layers():
        if not model_adapter.parallel_blocks:
            layer_adapter.layer.mlp_shortcut_Q = torch.nn.Parameter(
                torch.zeros(hidden_size, hidden_size).to(dtype=torch.float16)
            )
        layer_adapter.layer.attn_shortcut_Q = torch.nn.Parameter(
            torch.zeros(hidden_size, hidden_size).to(dtype=torch.float16)
        )

    config_path = pathlib.Path(sliced_model_path) / my_sliced_model_config

    if config_path.exists():
        model_adapter.slicing_conf = SlicingConfig.from_json_string(config_path.read_text())

    if model_adapter.slicing_conf is None:
        # assume the model was sliced with the const sparsity specified in the arguments to this method
        new_embedding_dimension = int((1 - sparsity) * hidden_size)
        new_embedding_dimension -= new_embedding_dimension % round_interval
        config = SlicingConfig()
        config.const_dimension = new_embedding_dimension
        model_adapter.slicing_conf = config

    slice_rotated_model(model_adapter)

    if lora_config:
        from peft import get_peft_model

        model_adapter.model = get_peft_model(model_adapter.model, lora_config)

    logging.info(f"Loading sliced model weights from {sliced_model_path}")
    model_adapter.model.load_state_dict(
        torch.load(str(pathlib.Path(sliced_model_path) / my_sliced_model_name), map_location="cpu")
    )
    model_adapter.model.eval()

    return model_adapter, tokenizer


def global_save_model(model_adapter, model, args):
    sliced_model_dir = pathlib.Path(args.save_dir)
    sliced_model_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Exporting Hugging Face compatible model to {args.save_dir}...")

    # Extract and update the model config
    hf_config = model.config
    
    new_hidden_size = list(model_adapter.slicing_conf.embedding_dimensions.values())[0]
    hf_config.hidden_size = new_hidden_size
    hf_config.tie_word_embeddings = False
    
    # Extract the current PyTorch state dict
    state_dict = model.state_dict()
    dtype = next(iter(state_dict.values())).dtype

    # Restore parameterless Norms
    for name, module in model.named_modules():
        if "norm" in name.lower() or "ln" in name.lower():
            weight_key = f"{name}.weight"
            bias_key = f"{name}.bias"
            
            # If SliceGPT made the weight parameterless, inject an all-ones tensor
            if weight_key not in state_dict:
                state_dict[weight_key] = torch.ones(new_hidden_size, dtype=dtype, device="cpu")
            
            # If the original norm required a bias (like in OPT), inject an all-zeros tensor
            if hasattr(module, "bias") and module.bias is not None and bias_key not in state_dict:
                state_dict[bias_key] = torch.zeros(new_hidden_size, dtype=dtype, device="cpu")

    # Cannot be done using this rotation method as the final norm is fused into the lm_head
    
    # Handle tied word embeddings by deleting the lm head
    # if getattr(hf_config, "tie_word_embeddings", False):
    #     lm_head_key = "lm_head.weight"
    #     if lm_head_key in state_dict:
    #         del state_dict[lm_head_key]
    #         logging.info("Tied embeddings detected: Re-tying lm_head to embeddings.")

    # Save using standard .safetensors
    model.save_pretrained(
        save_directory=sliced_model_dir,
        state_dict=state_dict,
        safe_serialization=True,
    )
    
    # Save the tokenizer and other essential config files
    source_path = args.model_path if args.model_path else args.model
    try:
        tokenizer = AutoTokenizer.from_pretrained(source_path, trust_remote_code=True)
        tokenizer.save_pretrained(sliced_model_dir)
    except Exception as e:
        logging.info(f'Failed to copy tokenizer: {e}')

    logging.info(f"Successfully saved standard Hugging Face model to {args.save_dir}")