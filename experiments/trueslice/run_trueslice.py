# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
#
# Modified for TrueSlice.

import argparse
import logging
import json
import os
import pathlib
import random
import time

# tqdm reads its environment defaults during import, including when imported by
# Transformers. Size its loading bars to the terminal before importing either.
if __name__ == "__main__":
    os.environ.setdefault("TQDM_DYNAMIC_NCOLS", "1")

import math
import numpy as np
import torch
import wandb

from trueslice import data_utils, gpu_utils, hf_utils, layernorm_fusion, rotate, utils
from trueslice.config import config
from trueslice.mapping import Mapping


def computation_timestamp() -> float:
    """Read wall time after all pending work on the compression device finishes."""
    gpu_utils.sync_gpus()
    return time.perf_counter()


def slicing_arg_parser(interactive: bool = True) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B", help="Model to load")
    parser.add_argument("--model-path", type=str, default=None, help="Path to load the model and tokenizer from (required for local models, not required for HF models).")
    parser.add_argument("--dtype", type=str, help="Data type to use.", choices=["fp32", "fp16", "bf16", "auto"], default="auto")
    parser.add_argument("--sparsity", type=float, default=0.0, help="A measure of how much slicing is applied (in the range [0, 1)).")
    parser.add_argument("--dataset", type=str, help="Dataset to calibrate and calculate perplexity on.", choices=["wikitext2", "ptb", "c4", "alpaca"], default="wikitext2")

    parser.add_argument("--seed", type=int, default=42, help="Set random seed for reproducibility.")
    parser.add_argument("--round-interval", type=int, default=8, help="Interval for rounding the weights (the best value may depend on your hardware).")
    parser.add_argument("--final-orientation", type=str, default="pca", choices=["random", "pca"], help="Final orientation of the sliced weights.")

    parser.add_argument("--cal-seqlen", type=int, default=2048, help="Maximum sequence length for the calibration data.")
    parser.add_argument("--cal-batch-size", type=int, default=16, help="Batch size for loading the calibration data.")
    parser.add_argument("--cal-nsamples", type=int, help="Number of samples of the calibration data to load.", default=128)

    parser.add_argument("--eval-seqlen", type=int, default=2048, help="Sequence length for evaluating the perplexity.")
    parser.add_argument("--eval-batch-size", type=int, default=8, help="Batch size for evaluating the perplexity.")
    parser.add_argument("--eval-nsamples", type=int, default=128, help="Maximum number of full sequences to evaluate perplexity on; sampled using --seed.")

    parser.add_argument("--cal-device-map", default="auto", help="Calibration compute devices: auto (all CUDA GPUs, otherwise CPU), cpu, or comma-separated CUDA devices. Model weights are stored on CPU.")
    parser.add_argument("--eval-device-map", default="auto", help="Evaluation placement: auto, cpu, cuda:0, or a JSON Hugging Face module map.")
    parser.add_argument("--eval-max-memory", default=None, help="JSON weight memory budgets, e.g. {\"cuda:0\":\"9GB\",\"cpu\":\"100GiB\"}.")

    parser.add_argument("--eval-only", action="store_true", help="Only evaluate the baseline model.")
    parser.add_argument("--eval-baseline", action="store_true", help="Evaluate the baseline model.")
    parser.add_argument("--eval-fused-model", action="store_true", help="Evaluate the fused model.")
    parser.add_argument("--eval-resulting", action="store_true", help="Evaluate the resulting model.")

    parser.add_argument("--results-file", type=str, default=None, help="Save compression-only timing and parameter counts as JSON.")
    parser.add_argument("--save-dir", type=str, default=None, help="Path to save the model.")

    parser.add_argument('--hf-token', type=str, default=os.getenv('HF_TOKEN', None))
    parser.add_argument('--wandb-project', type=str, default="slicegpt", help="wandb project name.")
    parser.add_argument('--no-wandb', action="store_true", help="Disable wandb.")

    return parser.parse_args() if interactive else parser.parse_args([])


def process_slicing_args(args):
    for arg, argv in vars(args).items():
        logging.debug('%s = %s', arg, '[redacted]' if arg == 'hf_token' else argv)

    if not 0 <= args.sparsity < 1:
        raise argparse.ArgumentTypeError("Sparsity should be in the range [0, 1)")
    for name in ("round_interval", "cal_nsamples", "cal_batch_size", "cal_seqlen",
                 "eval_nsamples", "eval_batch_size", "eval_seqlen"):
        if getattr(args, name) <= 0:
            raise argparse.ArgumentTypeError(f"--{name.replace('_', '-')} must be positive")
    if args.eval_seqlen < 2:
        raise argparse.ArgumentTypeError("--eval-seqlen must be at least 2 to predict a next token")
    if not 0 <= args.seed < 2**32:
        raise argparse.ArgumentTypeError("--seed must be in the range [0, 2**32)")

    if args.dtype == "fp16":
        config.dtype = torch.float16
    elif args.dtype == "fp32":
        config.dtype = torch.float32
    elif args.dtype == "bf16":
        config.dtype = torch.bfloat16
    elif args.dtype == "auto":
        config.dtype = None
    else:
        raise argparse.ArgumentTypeError("Data type should be one of 'fp32', 'fp16', 'bf16', 'auto'")


def slicing_main(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    logging.info("Running TrueSlice experiment.")
    logging.info(f"Number of available cuda devices: {torch.cuda.device_count()}")

    mapping = Mapping(
        eval_device_map=args.eval_device_map,
        eval_max_memory=args.eval_max_memory,
        cal_device_map=args.cal_device_map,
    )

    try:
        tracking_args = {name: value for name, value in vars(args).items() if name != 'hf_token'}
        wandb.init(project=args.wandb_project, config=tracking_args, mode='disabled' if args.no_wandb else None)
    except wandb.UsageError as e:
        # wandb.init will throw an error if the user is not logged in and the process is running in a non-shell
        # environment, e.g. notebook, IDE, no-shell process, etc. In this case, we want to continue without wandb.
        logging.info(f'Failed to initialize wandb: {e}, continuing without wandb')
        wandb.init(project=args.wandb_project, mode='disabled')

    model_adapter, tokenizer = mapping.load(
        args.model, args.model_path, token=args.hf_token, dtype=config.dtype,
        evaluate=args.eval_only or args.eval_baseline,
    )
    model = model_adapter.model

    dataset = data_utils.get_dataset(args.dataset, seed=args.seed)
    train_loader = None
    if not args.eval_only:
        train_loader = data_utils.prepare_train_dataloader(
            dataset=dataset["train"],
            tokenizer=tokenizer,
            max_seqlen=args.cal_seqlen,
            batch_size=args.cal_batch_size,
            nsamples=args.cal_nsamples,
            seed=args.seed,
        )
    test_loader = None
    if args.eval_only or args.eval_baseline or args.eval_fused_model or args.eval_resulting:
        test_dataset = dataset["test"] if "test" in dataset else dataset["validation"]
        test_loader = data_utils.prepare_test_dataloader(
            dataset=test_dataset,
            tokenizer=tokenizer,
            batch_size=args.eval_batch_size,
            seqlen=args.eval_seqlen,
            nsamples=args.eval_nsamples,
            seed=args.seed,
        )

    dense_param_count = hf_utils.count_parameters(model)
    logging.info(f'Dense model parameters: {dense_param_count:,d}')
    parameter_instances = sum(p.numel() for p in model.parameters())
    if parameter_instances != dense_param_count:
        logging.info('Dense parameter instances including tied-weight dispatch copies: %s',
                    f'{parameter_instances:,d}')

    # Evaluate the loaded checkpoint once and skip all compression stages.
    if args.eval_only:
        dataset_ppl = gpu_utils.evaluate_ppl(model, model.config.pad_token_id, test_loader)
        logging.info(f'Loaded model perplexity: {dataset_ppl}')
        wandb.log({"original_ppl": dataset_ppl})
        return

    # original ppl
    if args.eval_baseline:
        dataset_ppl = gpu_utils.evaluate_ppl(model, model.config.pad_token_id, test_loader)
        logging.info(f'Original ppl: {dataset_ppl:.4f}')
        wandb.log({"original_ppl": dataset_ppl})

    mapping.to_calibration(model_adapter)

    # Time compression only: exclude loading, data preparation, evaluation, and saving.
    computation_start = computation_timestamp()

    # fuse norms
    layernorm_fusion.fuse_modules(model_adapter)

    fusion_seconds = computation_timestamp() - computation_start
    if args.eval_fused_model:
        mapping.to_evaluation(model_adapter)
        fused_ppl = gpu_utils.evaluate_ppl(model, model.config.pad_token_id, test_loader)
        logging.info(f'Fused model perplexity: {fused_ppl:.4f}')
        wandb.log({"fused_ppl": fused_ppl})
        mapping.to_calibration(model_adapter)
    computation_start = computation_timestamp()

    fused_param_count = hf_utils.count_parameters(model)
    logging.info(f'Fused model parameters: {fused_param_count:,d}')

    # Round the retained residual width up to a multiple of round_interval.
    new_embedding_dimension = math.ceil(model_adapter.hidden_size * (1 - args.sparsity) / args.round_interval) * args.round_interval

    logging.info(
        f"New embedding dimension: {new_embedding_dimension} (sparsity {100*(1 - new_embedding_dimension / model_adapter.hidden_size):.4f} %)"
    )

    rotate.rotate_and_slice(
        model_adapter, train_loader, new_embedding_dimension=new_embedding_dimension,
        final_orientation=args.final_orientation, mapping=mapping,
    )

    layernorm_fusion.restore_norms(model_adapter, new_embedding_dimension)

    mapping.to_calibration(model_adapter)
    computation_seconds = fusion_seconds + computation_timestamp() - computation_start
    logging.info(f'Total compression computation time: {computation_seconds:.3f} seconds')
    wandb.log({"compression_time_seconds": computation_seconds})

    if args.results_file:
        result_path = pathlib.Path(args.results_file)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        compressed_params = hf_utils.count_parameters(model)
        result_path.write_text(json.dumps({
            "method": "trueslice",
            "model": args.model,
            "input_sparsity": args.sparsity,
            "compression_time_seconds": computation_seconds,
            "dense_parameters": dense_param_count,
            "compressed_parameters": compressed_params,
            "parameter_reduction_fraction": 1 - compressed_params / dense_param_count,
            "embedding_dimension": new_embedding_dimension,
            "excludes": ["model_loading", "model_saving", "dataset_preparation", "perplexity_evaluation"],
        }, indent=2) + "\n")

    if args.save_dir:
        hf_utils.save_model(model_adapter, model, args)

    sliced_param_count = hf_utils.count_parameters(model)
    sliced_fraction = 1.0 - sliced_param_count / dense_param_count
    logging.info(f'Sliced model parameters: {sliced_param_count:,d} (sliced fraction {sliced_fraction:.4f})')

    utils.cleanup_memory()

    if not args.eval_resulting:
        return

    mapping.to_evaluation(model_adapter)
    dataset_ppl = gpu_utils.evaluate_ppl(model, model.config.pad_token_id, test_loader)
    logging.info(f'After rotating and slicing {dataset_ppl:.4f}')
    wandb.log({"sliced_ppl": dataset_ppl})


if __name__ == "__main__":
    utils.configure_logging(log_to_console=True, log_to_file=False, level=logging.INFO)
    os.environ["WANDB__SERVICE_WAIT"] = "300"

    slicing_args = slicing_arg_parser()
    process_slicing_args(slicing_args)
    slicing_main(slicing_args)
