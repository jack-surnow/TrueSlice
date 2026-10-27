#!/bin/bash

set -e

BASE_PATH="/mnt/ML_Storage/TrueSlice"

EVAL_DENSE=1
EVAL_TRUESLICE=0
EVAL_SLICEGPT=0

MODELS=(
    # "meta-llama/Llama-3.2-1B|Llama-3.2-1B"
    # "meta-llama/Llama-3.2-3B|Llama-3.2-3B"
    # "meta-llama/Llama-3.1-8B|Llama-3.1-8B"
    # "meta-llama/Llama-3.1-70B|Llama-3.1-70B"

    # "Qwen/Qwen3-1.7B|Qwen3-1.7B"
    # "Qwen/Qwen3-4B|Qwen3-4B"
    # "Qwen/Qwen3-8B|Qwen3-8B"
    # "Qwen/Qwen3-14B|Qwen3-14B"
    # "Qwen/Qwen3-32B|Qwen3-32B"

    "Qwen/Qwen3.5-0.8B|Qwen3.5-0.8B"
    # "Qwen/Qwen3.5-2B|Qwen3.5-2B"
    # "Qwen/Qwen3.5-4B|Qwen3.5-4B"
    # "Qwen/Qwen3.5-9B|Qwen3.5-9B"
)

declare -A TRUESLICE_PATHS=(
    ["meta-llama/Llama-3.2-1B"]="Llama-3.2-1B_TrueSlice_0.422364_wikitext2_128"
    ["meta-llama/Llama-3.2-3B"]=""
    ["meta-llama/Llama-3.1-8B"]=""
    ["meta-llama/Llama-3.1-70B"]=""

    ["Qwen/Qwen3-4B"]=""
    ["Qwen/Qwen3-8B"]=""
    ["Qwen/Qwen3-14B"]=""
    ["Qwen/Qwen3-32B"]=""

    ["Qwen/Qwen3.5-4B"]=""
    ["Qwen/Qwen3.5-9B"]=""
)

declare -A SLICEGPT_PATHS=(
    ["meta-llama/Llama-3.2-1B"]="Llama-3.2-1B_SliceGPT_0.542969_wikitext2_128"
    ["meta-llama/Llama-3.2-3B"]=""
    ["meta-llama/Llama-3.1-8B"]=""
    ["meta-llama/Llama-3.1-70B"]=""

    ["Qwen/Qwen3-4B"]=""
    ["Qwen/Qwen3-8B"]=""
    ["Qwen/Qwen3-14B"]=""
    ["Qwen/Qwen3-32B"]=""

    ["Qwen/Qwen3.5-4B"]=""
    ["Qwen/Qwen3.5-9B"]=""
)

# Match each SliceGPT checkpoint to the sparsity used when saving it.
declare -A SLICEGPT_SPARSITIES=(
    ["Llama-3.2-1B_SliceGPT_0.542969_wikitext2_128"]="0.542969"
)

TASKS=(
    "piqa"
    "winogrande"
    "hellaswag"
    "arc_easy"
    "arc_challenge"
    "ifeval"
    "mmlu"
    "gsm8k"
)

EVAL_BATCH_SIZE=4
DTYPE="bf16"
SEED=42
ROUND_INTERVAL=1
RESULTS_DIR="task_eval_results"

# Leave empty to use each task's default few-shot count and full dataset.
NUM_FEWSHOT=""
EVAL_LIMIT=""
APPLY_CHAT_TEMPLATE=0
DISTRIBUTE_MODEL=0

EXTRA_ARGS=()
if [ -n "${NUM_FEWSHOT}" ]; then
    EXTRA_ARGS+=(--num-fewshot "${NUM_FEWSHOT}")
fi
if [ -n "${EVAL_LIMIT}" ]; then
    EXTRA_ARGS+=(--limit "${EVAL_LIMIT}")
fi
if [ "${APPLY_CHAT_TEMPLATE}" -eq 1 ]; then
    EXTRA_ARGS+=(--apply-chat-template)
fi
if [ "${DISTRIBUTE_MODEL}" -eq 1 ]; then
    EXTRA_ARGS+=(--distribute-model)
fi

for MODEL_ENTRY in "${MODELS[@]}"; do
    # Obtain Model Name
    IFS="|" read -r MODEL_ID MODEL_NAME <<< "${MODEL_ENTRY}"

    # Ensure checkpoint paths and slicing settings exist before evaluating this model.
    if [ "${EVAL_TRUESLICE}" -eq 1 ]; then
        IFS=' ' read -r -a TRUESLICE_PATHS_LIST <<< "${TRUESLICE_PATHS[$MODEL_ID]-}"
        if [ "${#TRUESLICE_PATHS_LIST[@]}" -eq 0 ]; then
            echo "Error: missing TrueSlice paths for ${MODEL_ID}" >&2
            exit 1
        fi
    fi

    if [ "${EVAL_SLICEGPT}" -eq 1 ]; then
        IFS=' ' read -r -a SLICEGPT_PATHS_LIST <<< "${SLICEGPT_PATHS[$MODEL_ID]-}"
        if [ "${#SLICEGPT_PATHS_LIST[@]}" -eq 0 ]; then
            echo "Error: missing SliceGPT paths for ${MODEL_ID}" >&2
            exit 1
        fi
        for SLICEGPT_PATH in "${SLICEGPT_PATHS_LIST[@]}"; do
            if [ -z "${SLICEGPT_SPARSITIES[$SLICEGPT_PATH]-}" ]; then
                echo "Error: missing SliceGPT sparsity for ${SLICEGPT_PATH}" >&2
                exit 1
            fi
        done
    fi

    # Run Dense Task Eval
    if [ "${EVAL_DENSE}" -eq 1 ]; then
        echo "======================================================================"
        echo "Running Dense Task Eval for ${MODEL_NAME}: ${MODEL_ID}"
        echo "Tasks: ${TASKS[*]}"
        echo "======================================================================"

        python experiments_trueslice/run_task_eval.py \
            --model "${MODEL_ID}" \
            --tasks "${TASKS[@]}" \
            --batch-size "${EVAL_BATCH_SIZE}" \
            --seed "${SEED}" \
            --dtype "${DTYPE}" \
            --save-dir "${RESULTS_DIR}/${MODEL_NAME}_dense" \
            "${EXTRA_ARGS[@]}" \
            --no-wandb
    fi

    # Run TrueSlice Task Eval
    if [ "${EVAL_TRUESLICE}" -eq 1 ]; then
        for TRUESLICE_PATH in "${TRUESLICE_PATHS_LIST[@]}"; do
            echo "======================================================================"
            echo "Running TrueSlice Task Eval for ${MODEL_NAME}: ${TRUESLICE_PATH}"
            echo "Tasks: ${TASKS[*]}"
            echo "======================================================================"

            python experiments_trueslice/run_task_eval.py \
                --model "${MODEL_ID}" \
                --model-path "${BASE_PATH}/${TRUESLICE_PATH}" \
                --tasks "${TASKS[@]}" \
                --batch-size "${EVAL_BATCH_SIZE}" \
                --seed "${SEED}" \
                --dtype "${DTYPE}" \
                --save-dir "${RESULTS_DIR}/${TRUESLICE_PATH}" \
                "${EXTRA_ARGS[@]}" \
                --no-wandb
        done
    fi

    # Run SliceGPT Task Eval
    if [ "${EVAL_SLICEGPT}" -eq 1 ]; then
        for SLICEGPT_PATH in "${SLICEGPT_PATHS_LIST[@]}"; do
            echo "======================================================================"
            echo "Running SliceGPT Task Eval for ${MODEL_NAME}: ${SLICEGPT_PATH}"
            echo "Tasks: ${TASKS[*]}"
            echo "======================================================================"

            python experiments_slicegpt/run_task_eval.py \
                --model "${MODEL_ID}" \
                --sliced-model-path "${BASE_PATH}/${SLICEGPT_PATH}" \
                --sparsity "${SLICEGPT_SPARSITIES[$SLICEGPT_PATH]}" \
                --round-interval "${ROUND_INTERVAL}" \
                --tasks "${TASKS[@]}" \
                --batch-size "${EVAL_BATCH_SIZE}" \
                --seed "${SEED}" \
                --dtype "${DTYPE}" \
                --save-dir "${RESULTS_DIR}/${SLICEGPT_PATH}" \
                "${EXTRA_ARGS[@]}" \
                --no-wandb
        done
    fi
done
