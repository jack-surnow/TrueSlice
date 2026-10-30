#!/bin/bash

# set -e

BASE_PATH="/mnt/ML_Storage/TrueSlice"

BENCHMARK_DENSE=1
BENCHMARK_SLICEGPT=1
BENCHMARK_TRUESLICE=1

MODELS=(
    "meta-llama/Llama-3.2-1B|Llama-3.2-1B"
    # "meta-llama/Llama-3.2-3B|Llama-3.2-3B"
    # "meta-llama/Llama-3.1-8B|Llama-3.1-8B"
    # "meta-llama/Llama-3.1-70B|Llama-3.1-70B"

    # "Qwen/Qwen3-1.7B|Qwen3-1.7B"
    # "Qwen/Qwen3-4B|Qwen3-4B"
    # "Qwen/Qwen3-8B|Qwen3-8B"
    # "Qwen/Qwen3-14B|Qwen3-14B"
    # "Qwen/Qwen3-32B|Qwen3-32B"

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

DATASETS=(
    "wikitext2"
    # "alpaca"
)

SPARSITY_OF_SLICEGPT_MODEL=0.542969

EVAL_PROMPT_TOKENS=128
EVAL_NTOKENS=128
EVAL_NSAMPLES=8
EVAL_REPETITIONS=1
EVAL_WARMUP_STEPS=5
EVAL_BATCH_SIZE=1
DTYPE="auto"
RESULTS_DIR="benchmark_results"

for MODEL_ENTRY in "${MODELS[@]}"; do
    # Obtain Model Name
    IFS="|" read -r MODEL_ID MODEL_NAME <<< "${MODEL_ENTRY}"

    # Ensure existing sparsity values
    if [ "${BENCHMARK_TRUESLICE}" -eq 1 ]; then
        IFS=' ' read -r -a TRUESLICE_PATHS_LIST <<< "${TRUESLICE_PATHS[$MODEL_ID]-}"

        if [ "${#TRUESLICE_PATHS_LIST[@]}" -eq 0 ]; then
            echo "Error: missing TrueSlice paths for ${MODEL_ID}" >&2
            exit 1
        fi
    fi

    if [ "${BENCHMARK_SLICEGPT}" -eq 1 ]; then
        IFS=' ' read -r -a SLICEGPT_PATHS_LIST <<< "${SLICEGPT_PATHS[$MODEL_ID]-}"

        if [ "${#SLICEGPT_PATHS_LIST[@]}" -eq 0 ]; then
            echo "Error: missing TrueSlice paths for ${MODEL_ID}" >&2
            exit 1
        fi
    fi

    # Run Dense Benchmark
    if [ "${BENCHMARK_DENSE}" -eq 1 ]; then
        for DATASET in "${DATASETS[@]}"; do
            echo "======================================================================"
            echo "Running Dense Benchmark for ${MODEL_NAME}: ${MODEL_ID}"
            echo "Calibration Dataset: ${DATASET}"
            echo "======================================================================"

            python experiments_slicegpt/run_benchmark.py \
                --model "${MODEL_ID}" \
                --eval-dataset "${DATASET}" \
                --prompt-tokens "${EVAL_PROMPT_TOKENS}" --ntokens "${EVAL_NTOKENS}" \
                --nsamples "${EVAL_NSAMPLES}" \
                --repetitions "${EVAL_REPETITIONS}" \
                --warmup-steps "${EVAL_WARMUP_STEPS}" \
                --batch-size "${EVAL_BATCH_SIZE}" \
                --seed 42 \
                --dtype "${DTYPE}" \
                --results-file "${RESULTS_DIR}/${MODEL_NAME}_dense_${DATASET}.json" \
                --no-wandb
        done
    fi

    # Run TrueSlice Benchmark
    if [ "${BENCHMARK_TRUESLICE}" -eq 1 ]; then
        for DATASET in "${DATASETS[@]}"; do
            for TRUESLICE_PATH in "${TRUESLICE_PATHS_LIST[@]}"; do
                echo "======================================================================"
                echo "Running TrueSlice Benchmark for ${MODEL_NAME}: ${TRUESLICE_PATH}"
                echo "Calibration Dataset: ${DATASET}"
                echo "======================================================================"

                python experiments/trueslice/run_benchmark.py \
                    --model "${MODEL_ID}" \
                    --model-path "${BASE_PATH}/${TRUESLICE_PATH}" \
                    --dataset "${DATASET}" \
                    --prompt-tokens "${EVAL_PROMPT_TOKENS}" --eval-seqlen "$((EVAL_PROMPT_TOKENS + EVAL_NTOKENS))" \
                    --eval-nsamples "${EVAL_NSAMPLES}" \
                    --repetitions "${EVAL_REPETITIONS}" \
                    --warmup-steps "${EVAL_WARMUP_STEPS}" \
                    --eval-batch-size "${EVAL_BATCH_SIZE}" \
                    --seed 42 \
                    --dtype "${DTYPE}" \
                    --results-file "${RESULTS_DIR}/${TRUESLICE_PATH}_${DATASET}.json" \
                    --no-wandb
            done
        done
    fi

    # Run SliceGPT Benchmark
    if [ "${BENCHMARK_SLICEGPT}" -eq 1 ]; then
        for DATASET in "${DATASETS[@]}"; do
            for SLICEGPT_PATH in "${SLICEGPT_PATHS_LIST[@]}"; do
                echo "======================================================================"
                echo "Running SliceGPT Benchmark for ${MODEL_NAME}: ${SLICEGPT_PATH}"
                echo "Calibration Dataset: ${DATASET}"
                echo "======================================================================"

                python experiments_slicegpt/run_benchmark.py \
                    --model "${MODEL_ID}" \
                    --sliced-model-path "${BASE_PATH}/${SLICEGPT_PATH}" \
                    --sparsity "${SPARSITY_OF_SLICEGPT_MODEL}" \
                    --eval-dataset "${DATASET}" \
                    --prompt-tokens "${EVAL_PROMPT_TOKENS}" --ntokens "${EVAL_NTOKENS}" \
                    --nsamples "${EVAL_NSAMPLES}" \
                    --repetitions "${EVAL_REPETITIONS}" \
                    --warmup-steps "${EVAL_WARMUP_STEPS}" \
                    --batch-size "${EVAL_BATCH_SIZE}" \
                    --seed 42 \
                    --dtype "${DTYPE}" \
                    --results-file "${RESULTS_DIR}/${SLICEGPT_PATH}_${DATASET}.json" \
                    --no-wandb
            done
        done
    fi


done
