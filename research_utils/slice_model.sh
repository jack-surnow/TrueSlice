#!/bin/bash

# set -e

BASE_PATH="/mnt/ML_Storage/TrueSlice"

RUN_DENSE_PPL=0
RUN_TRUESLICE=0
RUN_SLICEGPT=1

ROTATION="pca"

MODELS=(
    # "meta-llama/Llama-3.2-1B|Llama-3.2-1B"
    # "meta-llama/Llama-3.2-3B|Llama-3.2-3B"
    "meta-llama/Llama-3.1-8B|Llama-3.1-8B"

    # "Qwen/Qwen3-1.7B|Qwen3-1.7B"
    # "Qwen/Qwen3-4B|Qwen3-4B"
    # "Qwen/Qwen3-8B|Qwen3-8B"
    # "Qwen/Qwen3-14B|Qwen3-14B"

    # "Qwen/Qwen3.5-4B|Qwen3.5-4B"
    # "Qwen/Qwen3.5-9B|Qwen3.5-9B"
)

declare -A TRUESLICE_SPARSITIES=(
    ["meta-llama/Llama-3.2-1B"]="0.257325 0.339844 0.422364"
    ["meta-llama/Llama-3.2-3B"]="0.198243 0.287110 0.376303"
    ["meta-llama/Llama-3.1-8B"]="0.099854 0.199952 0.299805"

    ["Qwen/Qwen3-4B"]="0.179297 0.270313 0.361719"
    ["Qwen/Qwen3-8B"]="0.099854 0.199952 0.299805"
    ["Qwen/Qwen3-14B"]="0.100000 0.200000 0.300001"

    ["Qwen/Qwen3.5-4B"]="0.217969 0.305079 0.391797"
    ["Qwen/Qwen3.5-9B"]="0.099854 0.199952 0.299805"
)

declare -A SLICEGPT_SPARSITIES=(
    ["meta-llama/Llama-3.2-1B"]="0.361817 0.451661 0.542969"
    ["meta-llama/Llama-3.2-3B"]="0.304688 0.387370 0.472331"
    ["meta-llama/Llama-3.1-8B"]="0.209473 0.297364 0.387208"

    ["Qwen/Qwen3-4B"]="0.262110 0.348438 0.436719"
    ["Qwen/Qwen3-8B"]="0.209473 0.297364 0.387208"
    ["Qwen/Qwen3-14B"]="0.202344 0.289063 0.377540"

    ["Qwen/Qwen3.5-4B"]="0.301563 0.390625 0.480860"
    ["Qwen/Qwen3.5-9B"]="0.200440 0.294922 0.391114"
)

DATASETS=(
    "wikitext2"
    # "alpaca"
)

CAL_SET_SIZE=128
EVAL_SET_SIZE=128

CAL_BATCH_SIZE=2
EVAL_BATCH_SIZE=1
DTYPE="auto"

for MODEL_ENTRY in "${MODELS[@]}"; do
    # Obtain Model Name
    IFS="|" read -r MODEL_ID MODEL_NAME <<< "${MODEL_ENTRY}"

    # Ensure existing sparsity values
    if [ "${RUN_TRUESLICE}" -eq 1 ]; then
        IFS=' ' read -r -a TRUESLICE_SPARSITIES_LIST <<< "${TRUESLICE_SPARSITIES[$MODEL_ID]-}"

        if [ "${#TRUESLICE_SPARSITIES_LIST[@]}" -eq 0 ]; then
            echo "Error: missing TrueSlice sparsities for ${MODEL_ID}" >&2
            exit 1
        fi
    fi

    if [ "${RUN_SLICEGPT}" -eq 1 ]; then
        IFS=' ' read -r -a SLICEGPT_SPARSITIES_LIST <<< "${SLICEGPT_SPARSITIES[$MODEL_ID]-}"

        if [ "${#SLICEGPT_SPARSITIES_LIST[@]}" -eq 0 ]; then
            echo "Error: missing SliceGPT sparsities for ${MODEL_ID}" >&2
            exit 1
        fi
    fi

    # Run Dense Eval
    if [ "${RUN_DENSE_PPL}" -eq 1 ]; then
        for DATASET in "${DATASETS[@]}"; do
            echo "======================================================================"
            echo "Running Perplexity Eval for ${MODEL_NAME}"
            echo "Calibration Dataset: ${DATASET}"
            echo "======================================================================"

            python experiments_trueslice/run_eval.py \
                --model "${MODEL_ID}" \
                --ppl-only \
                --cal-dataset "${DATASET}" \
                --cal-nsamples "${CAL_SET_SIZE}" \
                --cal-batch-size "${CAL_BATCH_SIZE}" \
                --cal-max-seqlen 2048 \
                --seed 42 \
                --ppl-eval-seqlen 2048 \
                --ppl-eval-batch-size "${EVAL_BATCH_SIZE}" \
                --ppl-eval-nsamples "${EVAL_SET_SIZE}" \
                --dtype "${DTYPE}" \
                --distribute-model \
                --no-wandb
        done
    fi

    # Run TrueSlice
    if [ "${RUN_TRUESLICE}" -eq 1 ]; then
        for DATASET in "${DATASETS[@]}"; do
            for SPARSITY in "${TRUESLICE_SPARSITIES_LIST[@]}"; do
            echo "======================================================================"
            echo "Running TrueSlice for ${MODEL_NAME}"
            echo "Sparsity: ${SPARSITY} | Calibration Dataset: ${DATASET} (${CAL_SET_SIZE} samples)"
            echo "======================================================================"

            python experiments_trueslice/run_trueslice.py \
                --model "${MODEL_ID}" \
                --save-dir "${BASE_PATH}/${MODEL_NAME}_TrueSlice_${SPARSITY}_${DATASET}_${CAL_SET_SIZE}" \
                --cal-dataset "${DATASET}" \
                --cal-nsamples "${CAL_SET_SIZE}" \
                --cal-batch-size "${CAL_BATCH_SIZE}" \
                --cal-max-seqlen 2048 \
                --seed 42 \
                --sparsity "${SPARSITY}" \
                --round-interval 1 \
                --final-orientation ${ROTATION} \
                --ppl-eval-seqlen 2048 \
                --ppl-eval-batch-size "${EVAL_BATCH_SIZE}" \
                --ppl-eval-nsamples "${EVAL_SET_SIZE}" \
                --dtype "${DTYPE}" \
                --distribute-model \
                --no-wandb
            done
        done
    fi

    # Run SliceGPT
    if [ "${RUN_SLICEGPT}" -eq 1 ]; then
        for DATASET in "${DATASETS[@]}"; do
            for SPARSITY in "${SLICEGPT_SPARSITIES_LIST[@]}"; do
            echo "======================================================================"
            echo "Running SliceGPT for ${MODEL_NAME}"
            echo "Sparsity: ${SPARSITY} | Calibration Dataset: ${DATASET} (${CAL_SET_SIZE} samples)"
            echo "======================================================================"

            python experiments_slicegpt/run_slicegpt.py \
                --model "${MODEL_ID}" \
                --save-dir "${BASE_PATH}/${MODEL_NAME}_SliceGPT_${SPARSITY}_${DATASET}_${CAL_SET_SIZE}" \
                --cal-dataset "${DATASET}" \
                --cal-nsamples "${CAL_SET_SIZE}" \
                --cal-batch-size "${CAL_BATCH_SIZE}" \
                --cal-max-seqlen 2048 \
                --seed 42 \
                --sparsity "${SPARSITY}" \
                --round-interval 1 \
                --final-orientation ${ROTATION} \
                --ppl-eval-seqlen 2048 \
                --ppl-eval-batch-size "${EVAL_BATCH_SIZE}" \
                --ppl-eval-nsamples "${EVAL_SET_SIZE}" \
                --dtype "${DTYPE}" \
                --distribute-model \
                --no-wandb
            done
        done
    fi
done