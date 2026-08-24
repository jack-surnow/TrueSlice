# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import logging
import os
import pathlib
import shutil

import torch
import torch.nn as nn
from tqdm import tqdm
import wandb

from slicegpt import data_utils, gpu_utils, hf_utils, layernorm_fusion, rotate, utils
from slicegpt.config import config
from slicegpt.slicing_scheduler import ConstSlicingScheduler


def slicing_arg_parser(interactive: bool = True) -> argparse.Namespace:
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
    parser.add_argument("--dtype", type=str, help="Data type to use.", choices=["fp32", "fp16", "bf16"], default="fp16")
    parser.add_argument(
        "--cal-dataset",
        type=str,
        help="Dataset to calibrate and calculate perplexity on.",
        choices=["wikitext2", "ptb", "c4", "alpaca"],
        default="wikitext2",
    )
    parser.add_argument(
        "--cal-nsamples",
        type=int,
        help="Number of samples of the calibration data to load.",
        default=128,
    )
    parser.add_argument("--cal-batch-size", type=int, default=16, help="Batch size for loading the calibration data.")
    parser.add_argument(
        "--cal-max-seqlen", type=int, default=2048, help="Maximum sequence length for the calibration data."
    )
    parser.add_argument("--varied-seqlen", action="store_true", help="Varied sequence lengths in the calibration data.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for sampling the calibration data.")
    parser.add_argument(
        "--sparsity", type=float, default=0.0, help="A measure of how much slicing is applied (in the range [0, 1))"
    )
    parser.add_argument(
        "--round-interval",
        type=int,
        default=8,
        help="Interval for rounding the weights (the best value may depend on your hardware)",
    )
    parser.add_argument(
        "--final-orientation",
        type=str,
        default="random",
        choices=["random", "pca"],
        help="Final orientation of the sliced weights.",
    )
    parser.add_argument(
        "--ppl-eval-seqlen", type=int, default=2048, help="Sequence length for evaluating the perplexity."
    )
    parser.add_argument("--ppl-eval-batch-size", type=int, default=8, help="Batch size for evaluating the perplexity.")
    parser.add_argument(
        "--ppl-eval-nsamples", type=int, default=128, help="Number of samples to evaluate the perplexity on."
    )
    parser.add_argument("--eval-baseline", action="store_true", help="Evaluate the baseline model.")
    parser.add_argument("--eval-fused-model", action="store_true", help="Evaluate the fused model.")
    parser.add_argument("--ppl-only", action="store_true", help="Evaluate the loaded model without doing compression.")
    parser.add_argument(
        "--distribute-model",
        action="store_true",
        help="Use accelerate to put the model on multiple GPUs for evaluation. It is recommended to use it for models with 30B parameters and above.",
    )

    parser.add_argument("--save-dir", type=str, default=None, help="Path to save the model.")

    parser.add_argument('--hf-token', type=str, default=os.getenv('HF_TOKEN', None))

    parser.add_argument('--wandb-project', type=str, default="slicegpt", help="wandb project name.")
    parser.add_argument('--no-wandb', action="store_true", help="Disable wandb.")
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help="PyTorch device to use. Example values are 'cpu', 'cuda', 'cuda:0'. If not specified it will be defaulted to 'cuda' if available and 'cpu' otherwise.",
    )

    return parser.parse_args() if interactive else parser.parse_args('')


def process_slicing_args(args):
    for arg, argv in vars(args).items():
        logging.debug(f'{arg} = {argv}')

    if not 0 <= args.sparsity < 1:
        raise argparse.ArgumentTypeError(f"Sparsity should be in the range [0, 1)")

    if args.device:
        config.device = torch.device(args.device)

    if args.dtype == "fp16":
        config.dtype = torch.float16
    elif args.dtype == "fp32":
        config.dtype = torch.float32
    elif args.dtype == "bf16":
        config.dtype = torch.bfloat16
    else:
        raise argparse.ArgumentTypeError(f"Data type should be one of 'fp16', 'fp32'")


def slicing_main(args: argparse.Namespace) -> None:
    logging.info("Running SliceGPT experiment.")
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
        model_adapter, tokenizer = hf_utils.load_sliced_model(
            args.model,
            args.sliced_model_path,
            sparsity=args.sparsity,
            round_interval=args.round_interval,
            token=args.hf_token,
        )
    else:
        # load one of the pre-trained models
        model_adapter, tokenizer = hf_utils.get_model_and_tokenizer(
            args.model, args.model_path, token=args.hf_token, dtype=config.dtype
        )

    model = model_adapter.model

    def reset_model_device() -> None:
        if args.distribute_model:
            # distribute model across available GPUs
            gpu_utils.distribute_model(model_adapter)
        else:
            model.to(config.device)

    dataset = data_utils.get_dataset(args.cal_dataset)
    train_dataset = dataset["train"]
    test_dataset = dataset["test"] if "test" in dataset else dataset["validation"]
    if len(train_dataset) > 5000:
        logging.info(f"truncating training set of len: {len(train_dataset)}")
        train_dataset = train_dataset.select(range(5000))
    if len(test_dataset) > 1000:
        logging.info(f"truncating testing set of len: {len(test_dataset)}")
        test_dataset = test_dataset.select(range(1000))

    train_loader = data_utils.prepare_dataloader(
        dataset=train_dataset,
        tokenizer=tokenizer,
        max_seqlen=args.cal_max_seqlen,
        batch_size=args.cal_batch_size,
        nsamples=args.cal_nsamples,
        varied_seqlen=args.varied_seqlen,
        seed=args.seed,
    )
    test_loader = data_utils.prepare_test_dataloader(
        dataset=test_dataset, tokenizer=tokenizer, batch_size=args.ppl_eval_batch_size
    )

    # evaluate perplexity and exit if sliced model is loaded or if ppl_only is set
    if args.sliced_model_path or args.ppl_only:
        reset_model_device()
        dataset_ppl = gpu_utils.evaluate_ppl(model, model.config.pad_token_id, test_loader)
        logging.info(f'Loaded model perplexity: {dataset_ppl}')
        wandb.log({"original_ppl": dataset_ppl})
        sliced_param_count = sum(int(p.nelement()) for p in model.parameters())
        logging.info(f'Sliced model parameters: {sliced_param_count:,d}')
        return

    # original ppl
    if args.eval_baseline:
        reset_model_device()
        dataset_ppl = gpu_utils.evaluate_ppl(model, model.config.pad_token_id, test_loader)
        logging.info(f'Original ppl: {dataset_ppl:.4f}')
        wandb.log({"original_ppl": dataset_ppl})
        model.cpu()
        utils.cleanup_memory()

    # replace modules with compressible equivalents
    layernorm_fusion.replace_layers(model_adapter)

    # fuse layernorms and add rotations to skip connections
    layernorm_fusion.fuse_modules(model_adapter)

    # don't run this on large and/or distributed models
    if args.eval_fused_model and not args.distribute_model:
        model.to(config.device)

        dataset_ppl = gpu_utils.evaluate_ppl(model, model.config.pad_token_id, test_loader)
        logging.info(f'Post-fusion: {dataset_ppl:.4f}')
        wandb.log({"post_fusion_ppl": dataset_ppl})

        model.cpu()

        # run GC and cleanup GPU memory
        utils.cleanup_memory()

    original_param_count = sum(int(p.nelement()) for p in model.parameters())
    logging.info(f'Original model parameters: {original_param_count:,d}')

    # compute new embedding dimension given the desired sparsity level
    new_embedding_dimension = int((1 - args.sparsity) * model_adapter.hidden_size)
    # round (down) to the nearest multiple of round_interval
    new_embedding_dimension -= new_embedding_dimension % args.round_interval

    logging.info(
        f"New embedding dimension: {new_embedding_dimension} (sparsity {100*(1 - new_embedding_dimension / model_adapter.hidden_size):.4f} %)"
    )

    scheduler = ConstSlicingScheduler(new_embedding_dimension)
    rotate.rotate_and_slice(model_adapter, train_loader, scheduler, final_orientation=args.final_orientation)

    if args.save_dir:
        sliced_model_dir = pathlib.Path(args.save_dir)
        sliced_model_dir.mkdir(parents=True, exist_ok=True)

        sliced_model_name = sliced_model_dir / f'{pathlib.Path(args.model).name}_{args.sparsity}.pt'

        # Save the sliced model
        torch.save(model.state_dict(), sliced_model_name)

        # Save the slicing config
        config_path = sliced_model_name.with_suffix('.json')
        config_path.write_text(model_adapter.slicing_conf.to_json_string())

        # If slicing a local model, also save HF config files in sliced model dir
        if args.model_path:
            try:
                # copy all config files (tokenizer, model and slicing configs)
                for file in pathlib.Path(args.model_path).glob("*.json"):
                    if 'safetensors' not in str(file):
                        shutil.copy(str(file), sliced_model_dir)
                # copy all tokenizer models
                for file in pathlib.Path(args.model_path).glob("*token*.model"):
                    shutil.copy(str(file), sliced_model_dir)
                # copy vocab merges if any
                for file in pathlib.Path(args.model_path).glob("merges.txt"):
                    shutil.copy(str(file), sliced_model_dir)
            except OSError as e:
                logging.info(f'Failed to copy configs and tokenizer files: {e}')

        logging.info(f"Saved sliced model to {args.save_dir}")

    reset_model_device()
    dataset_ppl = gpu_utils.evaluate_ppl(model, model.config.pad_token_id, test_loader)
    logging.info(f'After rotating and slicing {dataset_ppl:.4f}')
    wandb.log({"sliced_ppl": dataset_ppl})

    sliced_param_count = sum(int(p.nelement()) for p in model.parameters())
    sliced_fraction = 1.0 - sliced_param_count / original_param_count
    logging.info(f'Sliced model parameters: {sliced_param_count:,d} (sliced fraction {sliced_fraction:.4f})')


@torch.no_grad()
def calibrate_final_norm_ols(
    dense_model: nn.Module, 
    sliced_model: nn.Module, 
    dataloader: torch.utils.data.DataLoader, 
    Q_global: torch.Tensor, 
    d_new: int, 
    device: torch.device, 
    eps: float = 1e-8
) -> None:
    """
    Calibrates the final norm of the sliced model using Channel-Wise OLS.
    """
    dense_model.eval()
    sliced_model.eval()
    Q_s = Q_global[:, :d_new].to(device=device, dtype=torch.float32)

    def get_final_norm(m):
        if hasattr(m, "model") and hasattr(m.model, "norm"): return m.model.norm
        if hasattr(m, "transformer") and hasattr(m.transformer, "ln_f"): return m.transformer.ln_f
        if hasattr(m, "norm"): return m.norm
        raise ValueError("Final norm not found.")

    dense_norm = get_final_norm(dense_model)
    sliced_norm = get_final_norm(sliced_model)
    has_bias = sliced_norm.bias is not None

    # Tensors to accumulate sufficient statistics per channel (using float64)
    sum_uu = torch.zeros(d_new, dtype=torch.float64, device=device)
    sum_uy = torch.zeros(d_new, dtype=torch.float64, device=device)
    if has_bias:
        sum_u = torch.zeros(d_new, dtype=torch.float64, device=device)
        sum_y = torch.zeros(d_new, dtype=torch.float64, device=device)

    total_tokens = 0
    activations = {}

    # Register hooks: Dense target gets output, Sliced gets input
    t_handle = dense_norm.register_forward_hook(lambda m, i, o: activations.update({"dense_out": o.detach()}))
    s_handle = sliced_norm.register_forward_hook(lambda m, i, o: activations.update({"sliced_in": i[0].detach()}))

    logging.info("Accumulating statistics for Channel-Wise OLS Calibration...")
    for batch in tqdm(dataloader, desc="OLS Calibration"):
        input_ids = batch["input_ids"].to(device) if isinstance(batch, dict) else batch[0].to(device)
        
        _ = dense_model(input_ids)
        _ = sliced_model(input_ids)

        # Target: Fully scaled output projected to global basis
        Y_dense = activations["dense_out"].to(torch.float32)
        Y_target = torch.matmul(Y_dense, Q_s)
        Y_flat = Y_target.reshape(-1, d_new).to(torch.float64)
        
        # Source: Parameter-free normalized input
        X_sliced_in = activations["sliced_in"].to(torch.float32)
        if has_bias:
            mean = X_sliced_in.mean(-1, keepdim=True)
            var = (X_sliced_in - mean).pow(2).mean(-1, keepdim=True)
            U_pruned = (X_sliced_in - mean) * torch.rsqrt(var + getattr(sliced_norm, "eps", 1e-5))
        else:
            var = X_sliced_in.pow(2).mean(-1, keepdim=True)
            U_pruned = X_sliced_in * torch.rsqrt(var + getattr(sliced_norm, "variance_epsilon", 1e-6))
        
        U_flat = U_pruned.reshape(-1, d_new).to(torch.float64)
        
        # Accumulate statistics
        sum_uu += (U_flat * U_flat).sum(dim=0)
        sum_uy += (U_flat * Y_flat).sum(dim=0)
        if has_bias:
            sum_u += U_flat.sum(dim=0)
            sum_y += Y_flat.sum(dim=0)
        
        total_tokens += U_flat.shape[0]

    t_handle.remove()
    s_handle.remove()

    # Calculate optimal parameters and inject them back into the model
    if has_bias:
        mean_u = sum_u / total_tokens
        mean_y = sum_y / total_tokens
        cov_uy = (sum_uy / total_tokens) - (mean_u * mean_y)
        var_u = (sum_uu / total_tokens) - (mean_u ** 2)
        
        gamma = cov_uy / (var_u + eps)
        beta = mean_y - (gamma * mean_u)
        
        sliced_norm.weight.data.copy_(gamma.to(sliced_norm.weight.dtype))
        sliced_norm.bias.data.copy_(beta.to(sliced_norm.bias.dtype))
        logging.info("Applied Channel-Wise OLS (gamma and beta).")
    else:
        gamma = sum_uy / (sum_uu + eps)
        sliced_norm.weight.data.copy_(gamma.to(sliced_norm.weight.dtype))
        logging.info("Applied Channel-Wise OLS (gamma only).")


if __name__ == "__main__":
    utils.configure_logging(log_to_console=True, log_to_file=False, level=logging.INFO)
    os.environ["WANDB__SERVICE_WAIT"] = "300"

    slicing_args = slicing_arg_parser()
    process_slicing_args(slicing_args)
    slicing_main(slicing_args)
