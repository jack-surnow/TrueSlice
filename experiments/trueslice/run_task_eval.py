# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
# Modified for TrueSlice.

"""Run lm-evaluation-harness benchmarks with the trueslice model loader."""

import argparse
import json
import logging
import os
from pathlib import Path

import torch

from trueslice import model_loading

DEFAULT_TASKS = ["piqa", "winogrande", "hellaswag", "arc_easy", "arc_challenge", "ifeval", "mmlu", "gsm8k"]


def eval_arg_parser(argv=None):
    parser = argparse.ArgumentParser(description=f"Run trueslice task benchmarks with lm-evaluation-harness.")
    parser.add_argument("--model", required=True, help="Original Hugging Face model ID (selects the adapter).")
    paths = parser.add_mutually_exclusive_group()
    paths.add_argument("--model-path", help="Local Hugging Face model directory, including a saved TrueSlice model.")
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS, help="Harness task/group names or patterns.")
    parser.add_argument("--num-fewshot", type=int, default=None, help="Override all tasks; default uses each task's settings.")
    parser.add_argument("--batch-size", default="1", help="Positive integer or 'auto'.")
    parser.add_argument("--dtype", choices=["auto", "fp16", "bf16", "fp32"], default="auto")
    parser.add_argument("--device", default=None)
    parser.add_argument("--distribute-model", action="store_true")
    parser.add_argument("--apply-chat-template", action="store_true", help="Use the tokenizer's chat template.")
    parser.add_argument("--limit", type=float, help="Smoke test: examples per task, or fraction in (0, 1).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", default=f"results/task_eval/trueslice")
    parser.add_argument("--hf-token", default=os.getenv("HF_TOKEN"))
    parser.add_argument("--wandb-project", default=f"trueslice-task-eval")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args(argv)
    if args.batch_size != "auto":
        try:
            args.batch_size = int(args.batch_size)
            if args.batch_size < 1:
                raise ValueError
        except ValueError:
            parser.error("--batch-size must be a positive integer or 'auto'")
    if args.num_fewshot is not None and args.num_fewshot < 0:
        parser.error("--num-fewshot must be nonnegative")
    if args.limit is not None and (not 0 < args.limit or (args.limit >= 1 and not args.limit.is_integer())):
        parser.error("--limit must be a positive integer or a fraction in (0, 1)")
    if args.limit is not None and args.limit >= 1:
        args.limit = int(args.limit)
    return args


def load_model(args):
    dtype = {"auto": None, "fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    adapter, tokenizer = model_loading.get_model_and_tokenizer(
        args.model, args.model_path, token=args.hf_token, dtype=dtype,
        device_map="auto" if args.distribute_model else str(device),
    )
    # Sliced embeddings and output heads must not be re-tied by the harness.
    adapter.model.tie_weights = lambda *a, **kw: None
    adapter.model.eval()
    return adapter.model, tokenizer


def eval_main(args):
    import lm_eval
    from lm_eval.models.huggingface import HFLM
    from lm_eval.tasks import TaskManager
    from lm_eval.utils import handle_non_serializable

    manager = TaskManager()
    task_names = manager.match_tasks(args.tasks)
    unmatched = [name for name in args.tasks if not manager.match_tasks([name])]
    if unmatched:
        raise ValueError(f"Unknown task names or patterns: {unmatched}")
    logging.info("Selected tasks/groups: %s", task_names)
    model, tokenizer = load_model(args)
    hflm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=args.batch_size, backend="causal")
    results = lm_eval.simple_evaluate(
        model=hflm, tasks=task_names, task_manager=manager,
        num_fewshot=args.num_fewshot, batch_size=args.batch_size, limit=args.limit,
        apply_chat_template=args.apply_chat_template, log_samples=False,
        random_seed=args.seed, numpy_random_seed=args.seed, torch_random_seed=args.seed,
        fewshot_random_seed=args.seed,
    )
    if results is None:
        raise RuntimeError("The harness returned no results")
    # Preserve native metrics and harness-weighted groups (including all of MMLU).
    # Accuracy, exact match, and instruction-following scores are not averaged together.
    summary = {"results": results["results"], "groups": results.get("groups", {})}
    output = Path(args.save_dir)
    output.mkdir(parents=True, exist_ok=True)
    settings = {key: value for key, value in vars(args).items() if key != "hf_token"}
    for filename, payload in [("full_results.json", results), ("task_results.json", summary),
                              ("run_config.json", {"method": "trueslice", **settings})]:
        (output / filename).write_text(json.dumps(payload, indent=2, default=handle_non_serializable) + "\n")
    logging.info("%s", json.dumps(summary, indent=2, default=handle_non_serializable))
    logging.info("Saved task results to %s", output)
    if not args.no_wandb:
        import wandb

        with wandb.init(project=args.wandb_project, config=settings):
            wandb.log(summary)


def main():
    args = eval_arg_parser()
    logging.basicConfig(level=logging.INFO)
    eval_main(args)


if __name__ == "__main__":
    main()
