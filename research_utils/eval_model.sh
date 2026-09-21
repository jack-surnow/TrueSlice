#!/bin/bash

# set -e

MODELS=(
  "meta-llama/Llama-3.2-1B|Llama-3.2-1B"
  # "meta-llama/Llama-3.2-3B|Llama-3.2-3B"
  # "meta-llama/Llama-3.1-8B|Llama-3.1-8B"

  # "Qwen/Qwen3-1.7B|Qwen3-1.7B"
  # "Qwen/Qwen3-4B|Qwen3-4B"
  # "Qwen/Qwen3-8B|Qwen3-8B"
  # "Qwen/Qwen3-14B|Qwen3-14B"

  # "Qwen/Qwen3.5-4B|Qwen3.5-4B"
)

declare -A LOCAL_PATHS=(
  ["meta-llama/Llama-3.2-1B"]="Llama-3.2-1B_TrueSlice_0.257325_wikitext2"
  ["meta-llama/Llama-3.2-3B"]=""
  ["meta-llama/Llama-3.1-8B"]=""

  ["Qwen/Qwen3-4B"]=""
  ["Qwen/Qwen3-8B"]=""
  ["Qwen/Qwen3-14B"]="Qwen3-14B_TrueSlice_0.100000_alpaca_128 Qwen3-14B_U_TrueSlice_0.100000_alpaca_128"
# "Qwen3-14B_TrueSlice_0.100000_alpaca_128 \
# Qwen3-14B_TrueSlice_0.200000_alpaca_128 \
# Qwen3-14B_TrueSlice_0.300001_alpaca_128 \
# Qwen3-14B_U_TrueSlice_0.100000_alpaca_128 \
# Qwen3-14B_U_TrueSlice_0.200000_alpaca_128 \
# Qwen3-14B_U_TrueSlice_0.300001_alpaca_128"

  ["Qwen/Qwen3.5-4B"]=""
)

CAL_SETS=(
  "wikitext2"
  # "alpaca"
)

CAL_SET_SIZE=128
EVAL_SET_SIZE=128

CAL_BATCH_SIZE=1
EVAL_BATCH_SIZE=1
DTYPE="auto"

BASE_PATH="/mnt/ML_Storage/models_global_slicing"

for MODEL_ENTRY in "${MODELS[@]}"; do
  # Split the string into two variables using the pipe character
  IFS="|" read -r MODEL_ID MODEL_NAME <<< "${MODEL_ENTRY}"

  IFS=' ' read -r -a LOCAL_PATHS_LIST \
    <<< "${LOCAL_PATHS[$MODEL_ID]-}"

  if [ "${#LOCAL_PATHS_LIST[@]}" -eq 0 ]; then
    echo "Error: missing local paths for ${MODEL_ID}" >&2
    exit 1
  fi

  for CAL_SET in "${CAL_SETS[@]}"; do
    for LOCAL_PATH in "${LOCAL_PATHS_LIST[@]}"; do
      echo "======================================================================"
      echo "Running Perplexity Eval for ${MODEL_NAME}: ${LOCAL_PATH}"
      echo "Calibration Dataset: ${CAL_SET}"
      echo "======================================================================"

      python experiments_trueslice/run_eval.py \
        --model "${MODEL_ID}" \
        --model-path "${BASE_PATH}/${LOCAL_PATH}" \
        --ppl-only \
        --cal-dataset "${CAL_SET}" \
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
  done


done