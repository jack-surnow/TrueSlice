# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
#
# Modified for TrueSlice.

import logging
import inspect
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from . import utils
from .model_adapter import ModelAdapter
from .mapping import Mapping


@torch.no_grad()
def evaluate_ppl(
    model: torch.nn.Module,
    pad_token_id: int | None,
    testloader: DataLoader[dict[str, torch.Tensor]],
) -> float:
    """
    Evaluate token-weighted perplexity, excluding padding using attention masks.
    The model is already placed; inputs follow its embedding execution device.
    """
    utils.sync_gpus()

    start_time = time.time()

    model.eval()

    device = Mapping.input_device(model)

    total_nll = torch.zeros((), dtype=torch.float64)
    total_tokens = 0

    logging.info("Evaluating perplexity...")
    for index, batch in enumerate(testloader):
        logging.debug(f"Evaluating batch {index}")
        batch = utils.map_tensors(batch, device)
        logits = model(**{k: v for k, v in batch.items() if k != "labels"}, use_cache=False).logits

        # A real EOS must still count when the tokenizer also uses it for padding.
        valid = batch.get("attention_mask")
        if valid is None:
            valid = (batch["input_ids"] != pad_token_id) if pad_token_id is not None else torch.ones_like(batch["input_ids"])
        valid = valid.to(device=logits.device, dtype=torch.bool)
        mask = valid[:, :-1] & valid[:, 1:]
        shift_labels = batch["input_ids"][:, 1:].to(logits.device)
        shift_labels = shift_labels.masked_fill(~mask, -100)

        # Upcast BEFORE cross entropy. Chunk by token to avoid a full FP32 copy
        # of [batch, sequence, vocabulary] logits for large-vocabulary models.
        batch_nll = torch.zeros((), dtype=torch.float64, device=logits.device)
        for row in range(logits.shape[0]):
            for start in range(0, shift_labels.shape[1], 256):
                stop = min(start + 256, shift_labels.shape[1])
                losses = torch.nn.functional.cross_entropy(
                    logits[row, start:stop].float(), shift_labels[row, start:stop], reduction="none",
                )
                batch_nll += losses.double().sum()
        total_nll += batch_nll.cpu()
        total_tokens += mask.sum().item()
        del logits

    if total_tokens == 0:
        raise ValueError("Perplexity evaluation requires at least one valid next-token target; no tokens were scored")
    ppl = torch.exp(total_nll / total_tokens)

    utils.sync_gpus()

    elapsed = time.time() - start_time
    logging.info(
        "Time spent on evaluation: %s",
        time.strftime("%H:%M:%S.{}".format(str(elapsed % 1)[2:])[:13], time.gmtime(elapsed)),
    )

    return ppl.item()


@torch.no_grad()
def benchmark(
    model_adapter: ModelAdapter,
    input_batch: dict[str, torch.Tensor],
    warmup_steps: int = 5,
    prompt_tokens: int = 0,
) -> dict:
    """Time cached, single-token forwards over a supplied sequence, starting with an empty cache.

    Inputs are teacher-forced (no sampling). Warmup uses a separate cache and is
    excluded, as are input transfers. Latency is seconds per batch decode step;
    throughput counts all decode tokens in the batch. With prompt_tokens > 0,
    separately time prompt prefill and greedy first-token selection (on device).
    The model is already placed by Mapping; inputs follow its embedding device.
    Sequence length includes both the prompt and the measured decode steps.
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

    device = Mapping.input_device(model_adapter.model)
    input_ids = input_ids.to(device=device)
    attention_mask = attention_mask.to(device=device)
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
        utils.sync_gpus()
        past = None
        ttft_seconds = None
        if prompt_tokens:
            utils.sync_gpus()
            start_time = time.perf_counter()
            past, first_token = prefill()
            utils.sync_gpus()
            ttft_seconds = time.perf_counter() - start_time
            del first_token
        time_measurements = []

        for i in tqdm(range(prompt_tokens, input_seq_len), desc="Benchmarking"):
            utils.sync_gpus()
            start_time = time.perf_counter()
            output = forward_token(i, past)
            utils.sync_gpus()
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
            "seqlen": input_seq_len,
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
        "seqlen": sequence_length,
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
