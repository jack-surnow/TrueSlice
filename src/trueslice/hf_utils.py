# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
#
# Modified for TrueSlice by the anonymous submission authors.

import logging
import pathlib

import torch
from transformers import AutoConfig, AutoTokenizer, PreTrainedTokenizerBase

from .model_adapter import ModelAdapter


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


def resolve_model_dtype(
    model_path: str,
    dtype: torch.dtype | None = None,
    token: str | bool | None = None,
    local_files_only: bool = False,
) -> torch.dtype:
    """
    Resolve the torch.dtype from Hugging Face config if dtype is 'auto' or None.
    """
    if isinstance(dtype, torch.dtype):
        return dtype

    config = AutoConfig.from_pretrained(
        model_path,
        token=token,
        local_files_only=local_files_only,
    )
    config_dtype = getattr(config, "torch_dtype", None)

    if isinstance(config_dtype, torch.dtype):
        return config_dtype
    
    if isinstance(config_dtype, str) and hasattr(torch, config_dtype):
        return getattr(torch, config_dtype)

    return dtype


@do_not_initialize
def get_model_and_tokenizer(
    model_name: str,
    model_path: str | None = None,
    *,
    device_map: str | dict[str, int] | None = "cpu",
    max_memory: dict[str, int] | None = None,
    dtype: torch.dtype | None = None,
    token: str | bool | None = None,
) -> tuple[ModelAdapter, PreTrainedTokenizerBase]:
    """
    Load pretrained weights and a tokenizer, returning a TrueSlice adapter and tokenizer.

    Use model_path for a local model, otherwise load model_name from Hugging Face.
    Set evaluation mode, disable caching, and run the adapter setup hook. The model
    family adapter must already be imported.
    """
    local_model = model_path is not None

    # for HF models the path to use is the model name
    if not local_model:
        model_path = model_name

    resolved_dtype = resolve_model_dtype(
        model_path=model_path,
        dtype=dtype,
        token=token,
        local_files_only=local_model,
    )

    logging.info(
        f"Loading %s config %s from %s with dtype %s",
        model_name,
        "and model weights",
        model_path if local_model else 'Hugging Face',
        resolved_dtype,
    )

    model_adapter = ModelAdapter.from_model(
        model_name,
        model_path=model_path,
        device_map=device_map,
        max_memory=max_memory,
        dtype=resolved_dtype,
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


def save_model(model_adapter, model, args):
    sliced_model_dir = pathlib.Path(args.save_dir)
    sliced_model_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Exporting Hugging Face compatible model to {args.save_dir}...")

    # Extract and update the model config
    hf_config = model.config
    
    new_hidden_size = list(model_adapter.slicing_conf.embedding_dimensions.values())[0]
    hf_config.hidden_size = new_hidden_size
    hf_config.tie_word_embeddings = False
    
    state_dict = model.state_dict()

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
