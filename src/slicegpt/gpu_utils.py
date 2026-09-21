# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import logging
import inspect
import time

import numpy as np
import torch
from accelerate import dispatch_model, infer_auto_device_map
from accelerate.utils import get_balanced_memory
from torch.utils.data import DataLoader
from tqdm import tqdm

from . import utils
from .config import config
from .model_adapter import ModelAdapter


@torch.no_grad()
def evaluate_ppl(
    model: torch.nn.Module, pad_token_id: int | None, testloader: DataLoader[dict[str, torch.Tensor]]
) -> float:
    """
    Evaluate the model's perplexity on the test set using batch processing.
    It is expected that model is already on the correct device.
    """
    sync_gpus()

    start_time = time.time()

    model.eval()

    if pad_token_id:
        loss_fn = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=pad_token_id)
    else:
        loss_fn = torch.nn.CrossEntropyLoss(reduction="none")

    nlls = []

    logging.info("Evaluating perplexity...")
    for batch in testloader:
        logging.debug(f"Evaluating batch {len(nlls)}")
        batch = utils.map_tensors(batch, config.device)
        logits = model(**batch).logits

        # shift outputs and labels autoregressively.
        logits = logits[:, :-1, :]
        shift_labels = batch["input_ids"][:, 1:]

        # CrossEntropyLoss demands data dimension is dimension 1.
        nll = loss_fn(logits.permute(0, 2, 1), shift_labels).float()

        mask = shift_labels != loss_fn.ignore_index
        nll_means = (nll * mask).sum(dim=1) / mask.sum(dim=1)
        nlls.append(nll_means)

    nlls_tensor = torch.cat(nlls)
    ppl = torch.exp(nlls_tensor.mean())

    sync_gpus()

    elapsed = time.time() - start_time
    logging.info(
        "Time spent on evaluation: %s",
        time.strftime("%H:%M:%S.{}".format(str(elapsed % 1)[2:])[:13], time.gmtime(elapsed)),
    )

    return ppl.item()


def distribute_model(model_adapter: ModelAdapter) -> None:
    """Distribute the model across available GPUs."""
    model = model_adapter.model
    max_memory = get_balanced_memory(
        model,
        no_split_module_classes=model_adapter.no_split_module_classes,
    )

    device_map = infer_auto_device_map(
        model, max_memory=max_memory, no_split_module_classes=model_adapter.no_split_module_classes
    )

    dispatch_model(
        model,
        device_map=device_map,
        offload_buffers=True,
        offload_dir="offload",
        state_dict=model.state_dict(),
    )

    # Run GC and cleanup GPU memory
    utils.cleanup_memory()


def sync_gpus() -> None:
    """Sync all GPUs to make sure all operations are finished, needed for correct benchmarking of latency/throughput."""
    for i in range(torch.cuda.device_count()):
        torch.cuda.synchronize(device=i)


@torch.no_grad()
def benchmark(model_adapter: ModelAdapter, input_batch: dict[str, torch.Tensor], warmup_steps: int = 5, prompt_tokens: int = 0) -> dict:
    """Time cached, single-token forwards over a supplied sequence, starting with an empty cache.

    Inputs are teacher-forced (no sampling). Warmup uses a separate cache and is
    excluded, as are input transfers. Latency is seconds per batch decode step;
    throughput counts all decode tokens in the batch. With prompt_tokens > 0,
    separately time prompt prefill and greedy first-token selection (on device).
    """
    input_ids = input_batch["input_ids"]
    attention_mask = input_batch["attention_mask"]
    if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape or input_ids.numel() == 0:
        raise ValueError("Expected nonempty input_ids and attention_mask with matching [batch, sequence] shapes")
    if not 0 <= prompt_tokens < input_ids.shape[1]:
        raise ValueError("prompt_tokens must leave at least one decode token")
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be nonnegative")
    if not bool(attention_mask.all()):
        raise ValueError("Benchmark inputs must be unpadded so every timed step represents a real token")

    input_ids = input_ids.to(device=config.device)
    attention_mask = attention_mask.to(device=config.device)
    batch_size, input_seq_len = input_ids.shape
    decode_tokens = input_seq_len - prompt_tokens
    model = model_adapter.model
    prefill_kwargs = {"logits_to_keep": 1} if "logits_to_keep" in inspect.signature(model.forward).parameters else {}
    was_training = model.training
    original_use_cache = model_adapter.use_cache
    try:
        model.eval()
        model_adapter.use_cache = True

        def forward_token(index, past):
            return model(
                input_ids[:, index:index + 1],
                past_key_values=past,
                attention_mask=attention_mask[:, :index + 1],
                use_cache=True,
            )

        def prefill():
            output = model(
                input_ids[:, :prompt_tokens], attention_mask=attention_mask[:, :prompt_tokens],
                past_key_values=None, use_cache=True, **prefill_kwargs,
            )
            first_token = output.logits[:, -1, :].argmax(dim=-1)
            if output.past_key_values is None:
                raise RuntimeError("The model did not return a cache with use_cache=True")
            return output.past_key_values, first_token

        past = None
        for i in range(warmup_steps):
            offset = i % decode_tokens
            if offset == 0:
                past = prefill()[0] if prompt_tokens else None
            output = forward_token(prompt_tokens + offset, past)
            past = output.past_key_values
            del output
        sync_gpus()
        past = None
        ttft_seconds = None
        if prompt_tokens:
            sync_gpus()
            start_time = time.perf_counter()
            past, first_token = prefill()
            sync_gpus()
            ttft_seconds = time.perf_counter() - start_time
            del first_token
        time_measurements = []

        for i in tqdm(range(prompt_tokens, input_seq_len), desc="Benchmarking"):
            sync_gpus()
            start_time = time.perf_counter()
            output = forward_token(i, past)
            sync_gpus()
            time_measurements.append(time.perf_counter() - start_time)
            # Preserve the native Cache object; never convert it to a list or
            # clear individual layers, which destroys the decoding context.
            past = output.past_key_values
            if past is None:
                raise RuntimeError("The model did not return a cache with use_cache=True")
            del output

        median_time = float(np.median(time_measurements))
        return {
            "median_time": median_time,
            "latency": median_time,
            "throughput": batch_size / median_time,
            "batch_size": batch_size,
            "ntokens": decode_tokens,
            "prompt_tokens": prompt_tokens,
            "ttft_seconds": ttft_seconds,
            "step_times_seconds": time_measurements,
            "total_time_seconds": float(sum(time_measurements)),
        }
    finally:
        model_adapter.use_cache = original_use_cache
        model.train(was_training)


def benchmark_batches(
    model_adapter: ModelAdapter,
    batches: list[dict[str, torch.Tensor]],
    repetitions: int = 5,
    warmup_steps: int = 5,
    prompt_tokens: int = 0,
) -> dict:
    """Repeat identical, full batches; summarize variability between complete runs.

    Each call to benchmark starts a fresh cache and performs untimed warmup.
    All input preparation, transfers, and warmup are excluded from measured time.
    """
    if repetitions <= 0 or warmup_steps < 0 or not batches:
        raise ValueError("Provide batches, positive repetitions, and nonnegative warmup_steps")
    shape = batches[0]["input_ids"].shape
    if len(shape) != 2 or min(shape) <= 0:
        raise ValueError("Benchmark batches must be nonempty [batch, sequence] tensors")
    for batch in batches:
        if batch["input_ids"].shape != shape or batch["attention_mask"].shape != shape:
            raise ValueError("All benchmark batches must have the same full shape")
        if not bool(batch["attention_mask"].all()):
            raise ValueError("Benchmark batches must not contain padding")
    if not 0 <= prompt_tokens < shape[1]:
        raise ValueError("prompt_tokens must leave at least one decode token")
    batch_size, sequence_length = shape
    ntokens = sequence_length - prompt_tokens
    steps_per_run = len(batches) * ntokens
    tokens_per_run = steps_per_run * batch_size
    runs = []
    for repetition in range(repetitions):
        logging.info(f"Benchmark repetition {repetition + 1}/{repetitions}: {len(batches)} batches")
        batch_results = []
        for batch in batches:
            batch_results.append(benchmark(model_adapter, batch, warmup_steps=warmup_steps, prompt_tokens=prompt_tokens))
        times = [duration for result in batch_results for duration in result["step_times_seconds"]]
        total_seconds = float(sum(times))
        runs.append({
            "repetition": repetition + 1,
            "total_time_seconds": total_seconds,
            "total_tokens": tokens_per_run,
            "mean_step_latency_seconds": total_seconds / steps_per_run,
            "median_step_latency_seconds": float(np.median(times)),
            "throughput_tokens_per_second": tokens_per_run / total_seconds,
            "batches": batch_results,
        })
        if prompt_tokens:
            runs[-1]["ttft_seconds"] = float(np.mean([result["ttft_seconds"] for result in batch_results]))

    def stats(key):
        values = [run[key] for run in runs]
        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if repetitions > 1 else None,
            "median": float(np.median(values)),
            "min": float(min(values)),
            "max": float(max(values)),
        }

    results = {
        "batch_size": batch_size,
        "prompt_tokens": prompt_tokens,
        "nsamples": batch_size * len(batches),
        "ntokens": ntokens,
        "batches_per_repetition": len(batches),
        "repetitions": repetitions,
        "warmup_steps_per_batch": warmup_steps,
        "tokens_per_repetition": tokens_per_run,
        "total_measured_tokens": tokens_per_run * repetitions,
        "throughput_tokens_per_second": stats("throughput_tokens_per_second"),
        "mean_step_latency_seconds": stats("mean_step_latency_seconds"),
        "total_time_seconds": stats("total_time_seconds"),
        "runs": runs,
    }

    if prompt_tokens:
        results["ttft_seconds"] = stats("ttft_seconds")
    return results
