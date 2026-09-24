# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
#
# Modified for TrueSlice by the anonymous submission authors.

import argparse
import logging
import hashlib
import json
import pathlib
import os

import torch
import wandb

from slicegpt import data_utils, gpu_utils, hf_utils, utils
from slicegpt.config import config


def benchmarking_arg_parser(interactive: bool = True) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="facebook/opt-125m",
        help="Model to load",
    )
    path_group = parser.add_mutually_exclusive_group()
    path_group.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to load the model and tokenizer from (required for local models, not required for HF models)",
    )
    path_group.add_argument(
        "--sliced-model-path",
        type=str,
        help="Path to load the model to fine-tune (sliced) and tokenizer from",
        default=None,
    )

    parser.add_argument("--dtype", type=str, help="Data type to use.", choices=["fp32", "fp16", "bf16", "auto"], default="fp16")
    parser.add_argument(
        "--eval-dataset",
        type=str,
        help="Dataset to evaluate on.",
        choices=["wikitext2", "ptb", "c4"],
        default="wikitext2",
    )
    parser.add_argument(
        "--ntokens",
        type=int,
        help="Number of cached single-token forward steps per sequence (not dataset samples).",
        default=128,
    )
    parser.add_argument("--prompt-tokens", type=int, default=128, help="Prompt length for model-only TTFT; 0 disables prefill and preserves the legacy benchmark.")
    parser.add_argument("--require-gpu-resident-model", action="store_true", help="Reject CPU/disk offload for GPU-only benchmark comparisons.")
    parser.add_argument("--batch-size", type=int, default=1, help="Number of sequences processed simultaneously.")
    parser.add_argument("--nsamples", type=int, default=128, help="Total sequences per repetition; must be a multiple of batch-size.")
    parser.add_argument("--repetitions", type=int, default=5, help="Repeated runs over the same prepared inputs.")
    parser.add_argument("--warmup-steps", type=int, default=5, help="Untimed token steps before each batch in every repetition.")
    parser.add_argument("--results-file", type=str, default=None, help="Save settings, summary, and raw timing measurements as JSON.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for sampling the calibration data.")
    parser.add_argument(
        "--sparsity", type=float, default=0.0, help="A measure of how much slicing is applied (in the range [0, 1))"
    )
    parser.add_argument(
        "--distribute-model",
        action="store_true",
        help="Use accelerate to put the model on multiple GPUs for evaluation. It is recommended to use it for models with 30B parameters and above.",
    )

    parser.add_argument('--hf-token', type=str, default=os.getenv('HF_TOKEN', None))

    parser.add_argument('--wandb-project', type=str, default="slicegpt-bench", help="wandb project name.")
    parser.add_argument('--no-wandb', action="store_true", help="Disable wandb.")
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help="PyTorch device to use. Example values are 'cpu', 'cuda', 'cuda:0'. If not specified it will be defaulted to 'cuda' if available and 'cpu' otherwise.",
    )

    return parser.parse_args() if interactive else parser.parse_args('')


def process_benchmarking_args(args: argparse.Namespace):
    logging.debug(f'Parsed arguments:')
    for arg, argv in vars(args).items():
        logging.debug('%s = %s', arg, '[redacted]' if arg == 'hf_token' else argv)

    if not 0 <= args.sparsity < 1:
        raise argparse.ArgumentTypeError(f"Sparsity should be in the range [0, 1)")

    if min(args.ntokens, args.batch_size, args.nsamples, args.repetitions) <= 0:
        raise argparse.ArgumentTypeError("ntokens, batch-size, nsamples, and repetitions must be positive")
    if args.nsamples % args.batch_size:
        raise argparse.ArgumentTypeError("nsamples must be a multiple of batch-size so all timed batches have equal size")
    if args.prompt_tokens < 0:
        raise argparse.ArgumentTypeError("prompt-tokens must be nonnegative")
    if args.warmup_steps < 0:
        raise argparse.ArgumentTypeError("warmup-steps must be nonnegative")

    if args.device:
        config.device = torch.device(args.device)

    if args.dtype == "fp16":
        config.dtype = torch.float16
    elif args.dtype == "fp32":
        config.dtype = torch.float32
    elif args.dtype == "bf16":
        config.dtype = torch.bfloat16
    elif args.dtype == "auto":
        config.dtype = None
    else:
        raise argparse.ArgumentTypeError(f"Data type should be one of 'fp16', 'fp32'")


def benchmarking_main(args: argparse.Namespace) -> None:
    logging.info("Running benchmarking of a sliced model.")
    logging.info(f"PyTorch device: {config.device}")
    logging.info(f"Number of available cuda devices: {torch.cuda.device_count()}")

    try:
        wandb.init(project=args.wandb_project, config=args, mode='disabled' if args.no_wandb else None)
    except wandb.UsageError as e:
        # wandb.init will throw an error if the user is not logged in and the process is running in a non-shell
        # environment, e.g. notebook, IDE, no-shell process, etc. In this case, we want to continue without wandb.
        logging.info(f'Failed to initialize wandb: {e}, continuing without wandb')
        wandb.init(project=args.wandb_project, mode='disabled')

    if args.sliced_model_path:
        # load the model from sliced_model_path to compute perplexity and skip rotation and slicing
        logging.info(f"Loading sliced {args.model} model from {args.sliced_model_path} with sparsity {args.sparsity}")
        model_adapter, tokenizer = hf_utils.load_sliced_model(
            args.model, args.sliced_model_path, sparsity=args.sparsity, token=args.hf_token, dtype=config.dtype
        )
    else:
        # load one of the pre-trained models
        model_adapter, tokenizer = hf_utils.get_model_and_tokenizer(
            args.model, args.model_path, token=args.hf_token, dtype=config.dtype
        )

    if args.distribute_model:
        # distribute model across available GPUs
        gpu_utils.distribute_model(model_adapter)
    else:
        model_adapter.model.to(device=config.device)

    if args.require_gpu_resident_model:
        placement = getattr(model_adapter.model, "hf_device_map", {})
        offloaded = any(str(device) in {"cpu", "disk"} for device in placement.values())
        if offloaded or any(p.device.type != "cuda" for p in model_adapter.model.parameters()):
            raise RuntimeError("GPU-only benchmark requested, but model parameters are on CPU/disk/meta")

    sequence_length = args.prompt_tokens + args.ntokens
    if sequence_length > model_adapter.seqlen:
        raise ValueError(f"prompt-tokens + ntokens exceeds the model context length ({model_adapter.seqlen})")

    dataset = data_utils.get_dataset(args.eval_dataset)
    train_loader = data_utils.prepare_dataloader(
        dataset=dataset["train"],
        tokenizer=tokenizer,
        max_seqlen=sequence_length,
        batch_size=args.batch_size,
        nsamples=args.nsamples,
        exact_seqlen=True,
        seed=args.seed,
    )

    # Materialize once on CPU: keep sequence order and batch membership identical
    # across repetitions, and keep data preparation outside the measurements.
    batches = [{key: batch[key] for key in ("input_ids", "attention_mask")} for batch in train_loader]
    if (sum(batch["input_ids"].shape[0] for batch in batches) != args.nsamples
            or any(batch["input_ids"].shape != (args.batch_size, sequence_length) for batch in batches)):
        raise ValueError("Dataset did not produce the requested full batches and token count; choose another seed or dataset")
    logging.info(
        f"Cached token-forward benchmark: {args.nsamples} sequences, batch size {args.batch_size}, "
        f"{args.prompt_tokens} prompt tokens, {args.ntokens} decode steps per sequence, {args.repetitions} repetitions, "
        f"{args.warmup_steps} untimed warmup steps per batch."
    )
    results = gpu_utils.benchmark_batches(model_adapter, batches, args.repetitions, args.warmup_steps, prompt_tokens=args.prompt_tokens)
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
    utils.configure_logging()
    os.environ["WANDB__SERVICE_WAIT"] = "300"
    benchmarking_args = benchmarking_arg_parser()
    process_benchmarking_args(benchmarking_args)
    benchmarking_main(benchmarking_args)
