# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
#
# Modified for TrueSlice.

import logging
import pathlib

import torch
from transformers import AutoConfig, AutoTokenizer, PreTrainedTokenizerBase
from transformers.utils import logging as transformers_logging

from .model_adapter import ModelAdapter


class _LoadingProgress:
    """Silence Transformers' per-parameter bar while preserving other bars/locks."""

    def __init__(self, progress):
        self.progress = progress

    def __call__(self, *args, **kwargs):
        if kwargs.get('desc') == 'Loading weights':
            kwargs['disable'] = True
        return self.progress(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.progress, name)


def resolve_model_dtype(
    model_path: str,
    dtype: torch.dtype | None = None,
    token: str | bool | None = None,
    local_files_only: bool = False,
) -> torch.dtype | None:
    """
    Resolve the text model's dtype from Hugging Face config if dtype is 'auto' or None.
    """
    if isinstance(dtype, torch.dtype):
        return dtype

    config = AutoConfig.from_pretrained(
        model_path,
        token=token,
        local_files_only=local_files_only,
    )
    # Composite checkpoints such as Qwen3.5 declare the language model's dtype
    # in text_config, while the outer multimodal config may have no dtype.
    config_dtype = getattr(config.get_text_config(), "dtype", None) or getattr(config, "dtype", None)

    if isinstance(config_dtype, torch.dtype):
        return config_dtype
    
    if isinstance(config_dtype, str) and hasattr(torch, config_dtype):
        return getattr(torch, config_dtype)

    return dtype


def get_model_and_tokenizer(
    model_name: str,
    model_path: str | None = None,
    *,
    device_map: str | dict[str, str | int | torch.device] | None = "cpu",
    max_memory: dict[str | int, str | int] | None = None,
    offload_folder: str | None = None,
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

    progress = transformers_logging.tqdm
    transformers_logging.tqdm = _LoadingProgress(progress)
    try:
        model_adapter = ModelAdapter.from_model(
            model_name,
            model_path=model_path,
            device_map=device_map,
            max_memory=max_memory,
            offload_folder=offload_folder,
            dtype=resolved_dtype,
            local_files_only=local_model,
            token=token,
        )
    finally:
        transformers_logging.tqdm = progress

    model = model_adapter.model
    model.seqlen = model.config.max_position_embeddings
    model.eval()  # This switches off dropout.
    model_adapter.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, token=token, local_files_only=local_model)

    model_adapter.post_init(tokenizer)
    logging.info("Loading model done")

    return model_adapter, tokenizer


def count_parameters(model) -> int:
    """Count learned parameters, excluding tied embeddings even if the checkpoitn contains identical embeddings and lm head."""
    count = sum(parameter.numel() for parameter in model.parameters())
    if getattr(model.config, 'tie_word_embeddings', False):
        embedding = model.get_input_embeddings()
        head = model.get_output_embeddings()
        if embedding is not None and head is not None and embedding.weight is not head.weight:
            if embedding.weight.shape != head.weight.shape:
                raise ValueError('Tied input/output embeddings must have matching shapes')
            count -= head.weight.numel()
    return count


def save_model(model_adapter, model, args):
    sliced_model_dir = pathlib.Path(args.save_dir)
    sliced_model_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Exporting Hugging Face compatible model to {args.save_dir}...")

    # Extract and update the model config
    hf_config = model.config
    
    hf_config.hidden_size = model_adapter.get_lm_head().weight.shape[1]
    hf_config.tie_word_embeddings = False
    
    state_dict = model.state_dict()

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
