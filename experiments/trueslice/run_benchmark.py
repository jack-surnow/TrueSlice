# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
#
# Modified for TrueSlice.

import argparse
import hashlib
import json
import logging
import os
import pathlib
import random

# Set terminal sizing before Transformers imports tqdm, as in run_trueslice.
if __name__ == "__main__":
    os.environ.setdefault("TQDM_DYNAMIC_NCOLS", "1")

import numpy as np
import torch
import wandb

from trueslice import data_loading, evaluate, utils
from trueslice.mapping import Mapping


def benchmarking_arg_parser(interactive: bool = True) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B", help="Model to load")
    parser.add_argument("--model-path", type=str, default=None, help="Path to load the model and tokenizer from (required for local models, not required for HF models).")
    parser.add_argument("--dtype", type=str, help="Data type to use.", choices=["fp32", "fp16", "bf16", "auto"], default="auto")
    parser.add_argument("--dataset", type=str, help="Dataset to benchmark on.", choices=["wikitext2", "ptb", "c4", "alpaca"], default="wikitext2")

    parser.add_argument("--seed", type=int, default=42, help="Set random seed for reproducibility.")

    parser.add_argument("--eval-seqlen", type=int, default=2048, help="Total sequence length, including the prompt and cached decode steps. Inputs are packed without padding.")
    parser.add_argument("--eval-batch-size", type=int, default=8, help="Number of sequences benchmarked simultaneously.")
    parser.add_argument("--eval-nsamples", type=int, default=128, help="Number of evaluation sequences per repetition; must be a multiple of --eval-batch-size.")

    parser.add_argument("--eval-device-map", default="auto", help="Evaluation placement: auto, cpu, cuda:0, or a JSON Hugging Face module map.")
    parser.add_argument("--eval-max-memory", default=None, help='JSON weight memory budgets, e.g. {"cuda:0":"9GB","cpu":"100GiB"}.')

    parser.add_argument("--prompt-tokens", type=int, default=128, help="Initial prompt length for time to first token; remaining tokens are cached decode steps. Use 0 to skip prompt prefill.")
    parser.add_argument("--repetitions", type=int, default=5, help="Repeated runs over the same prepared inputs.")
    parser.add_argument("--warmup-steps", type=int, default=5, help="Untimed decode steps before each batch in every repetition.")
    parser.add_argument("--require-gpu-resident-model", action="store_true", help="Reject CPU/disk offload for GPU-only benchmark comparisons.")

    parser.add_argument("--results-file", type=str, default=None, help="Save settings, summary, and raw timing measurements as JSON.")

    parser.add_argument('--hf-token', type=str, default=os.getenv('HF_TOKEN', None))
    parser.add_argument('--wandb-project', type=str, default="slicegpt-bench", help="wandb project name.")
    parser.add_argument('--no-wandb', action="store_true", help="Disable wandb.")

    return parser.parse_args() if interactive else parser.parse_args([])


def process_benchmarking_args(args: argparse.Namespace):
    for arg, argv in vars(args).items():
        logging.debug('%s = %s', arg, '[redacted]' if arg == 'hf_token' else argv)

    for name in ("eval_seqlen", "eval_batch_size", "eval_nsamples", "repetitions"):
        if getattr(args, name) <= 0:
            raise argparse.ArgumentTypeError(f"--{name.replace('_', '-')} must be positive")
    if args.eval_nsamples % args.eval_batch_size:
        raise argparse.ArgumentTypeError("--eval-nsamples must be a multiple of --eval-batch-size so all timed batches have equal size")
    if not 0 <= args.prompt_tokens < args.eval_seqlen:
        raise argparse.ArgumentTypeError("--prompt-tokens must be nonnegative and less than --eval-seqlen, leaving at least one decode step")
    if args.warmup_steps < 0:
        raise argparse.ArgumentTypeError("--warmup-steps must be nonnegative")
    if not 0 <= args.seed < 2**32:
        raise argparse.ArgumentTypeError("--seed must be in the range [0, 2**32)")

    if args.dtype not in {"fp32", "fp16", "bf16", "auto"}:
        raise argparse.ArgumentTypeError("Data type should be one of fp32, fp16, bf16, auto")


def benchmarking_main(args: argparse.Namespace) -> None:
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16, "auto": None}[args.dtype]
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    logging.info("Running model benchmark.")
    logging.info(f"Number of available cuda devices: {torch.cuda.device_count()}")

    mapping = Mapping(
        eval_device_map=args.eval_device_map,
        eval_max_memory=args.eval_max_memory,
    )

    try:
        tracking_args = {name: value for name, value in vars(args).items() if name != 'hf_token'}
        wandb.init(project=args.wandb_project, config=tracking_args, mode='disabled' if args.no_wandb else None)
    except wandb.UsageError as e:
        logging.info(f'Failed to initialize wandb: {e}, continuing without wandb')
        wandb.init(project=args.wandb_project, mode='disabled')

    model_adapter, tokenizer = mapping.load(
        args.model, args.model_path, token=args.hf_token, dtype=dtype, evaluate=True,
    )
    model = model_adapter.model

    if args.require_gpu_resident_model:
        placement = getattr(model, "hf_device_map", {})
        offloaded = any(str(device) in {"cpu", "disk"} for device in placement.values())
        if offloaded or any(p.device.type != "cuda" for p in model.parameters()):
            raise RuntimeError("GPU-only benchmark requested, but model parameters are on CPU/disk/meta")

    if args.eval_seqlen > model_adapter.seqlen:
        raise ValueError(f"--eval-seqlen exceeds the model context length ({model_adapter.seqlen})")

    dataset = data_loading.get_dataset(args.dataset, seed=args.seed)
    eval_split = "test" if "test" in dataset else "validation"
    eval_loader = data_loading.prepare_eval_dataloader(
        dataset=dataset[eval_split],
        tokenizer=tokenizer,
        seqlen=args.eval_seqlen,
        batch_size=args.eval_batch_size,
        nsamples=args.eval_nsamples,
        seed=args.seed,
    )

    # Prepare once on CPU so repetitions use identical sequences and batches.
    # Loading, transfers, and warmup are excluded from measured forward times.
    batches = list(eval_loader)
    if (sum(batch["input_ids"].shape[0] for batch in batches) != args.eval_nsamples
            or any(batch["input_ids"].shape != (args.eval_batch_size, args.eval_seqlen) for batch in batches)):
        raise ValueError("Dataset did not produce the requested full batches; reduce --eval-nsamples or --eval-seqlen, or choose another dataset")
    logging.info(
        f"Cached token-forward benchmark: {args.eval_nsamples} sequences, batch size {args.eval_batch_size}, "
        f"{args.prompt_tokens} prompt tokens, {args.eval_seqlen - args.prompt_tokens} decode steps per sequence, "
        f"{args.repetitions} repetitions, {args.warmup_steps} untimed warmup steps per batch."
    )
    results = evaluate.benchmark_batches(
        model_adapter, batches, repetitions=args.repetitions,
        warmup_steps=args.warmup_steps, prompt_tokens=args.prompt_tokens,
    )
    results["dataset_split"] = eval_split
    fingerprint = hashlib.sha256()
    for batch in batches:
        for key in ("input_ids", "attention_mask"):
            fingerprint.update(batch[key].to("cpu").contiguous().numpy().tobytes())
    results["input_sha256"] = fingerprint.hexdigest()
    results["settings"] = {key: value for key, value in vars(args).items() if key != "hf_token"}
    results["torch_version"] = str(torch.__version__)
    summary = {}
    metrics = [
        ("throughput_tokens_per_second", "Throughput (tokens/s)"),
        ("mean_step_latency_seconds", "Mean decode-step latency (s/step)"),
        ("total_time_seconds", "Measured decode forward time per repetition (s)"),
    ]
    if args.prompt_tokens:
        metrics.append(("ttft_seconds", "Model-only time to first token (s)"))
    for metric, label in metrics:
        values = results[metric]
        spread = f"{values['std']:.6g}" if values['std'] is not None else "unavailable (one repetition)"
        logging.info(f"{label}: mean {values['mean']:.6g}, std {spread}")
        summary.update({f"{metric}/{stat}": value for stat, value in values.items() if value is not None})
    wandb.log(summary)
    if args.results_file:
        output = pathlib.Path(args.results_file)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
        logging.info(f"Saved benchmark results to {output}")



if __name__ == "__main__":
    utils.configure_logging(log_to_console=True, log_to_file=False, level=logging.INFO)
    os.environ["WANDB__SERVICE_WAIT"] = "300"

    benchmarking_args = benchmarking_arg_parser()
    process_benchmarking_args(benchmarking_args)
    benchmarking_main(benchmarking_args)
