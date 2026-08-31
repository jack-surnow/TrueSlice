#!/bin/bash

set -e

RUN_DENSE_PPL=0
RUN_TRUESLICE=1
RUN_SLICEGPT=1

MODELS=(
  # "meta-llama/Llama-3.2-1B|Llama-3.2-1B"
  # "meta-llama/Llama-3.2-3B|Llama-3.2-3B"
  "meta-llama/Llama-3.1-8B|Llama-3.1-8B"
  # "Qwen/Qwen3-1.7B|Qwen3-1.7B"
  # "Qwen/Qwen3-4B|Qwen3-4B"
  # "Qwen/Qwen3-8B|Qwen3-8B"
  # "microsoft/Phi-3-mini-128k-instruct|Phi-3-mini"
)

SPARSITIES_TRUESLICE=(0.125000 0.250000 0.375000 0.500000)
SPARSITIES_SLICEGPT=(0.222168 0.333985 0.448731 0.566651)

CAL_SETS=(
  "wikitext2"
  # "alpaca"
)

CAL_BATCH_SIZE=2
EVAL_BATCH_SIZE=2
DTYPE="auto"

BASE_PATH="/mnt/ML_Storage/models_global_slicing"

for MODEL_ENTRY in "${MODELS[@]}"; do
  # Split the string into two variables using the pipe character
  IFS="|" read -r MODEL_ID MODEL_NAME <<< "${MODEL_ENTRY}"

  if [ "${RUN_DENSE_PPL}" -eq 1 ]; then
    for CAL_SET in "${CAL_SETS[@]}"; do
      echo "======================================================================"
      echo "Running Perplexity Eval for ${MODEL_NAME}"
      echo "Calibration Dataset: ${CAL_SET}"
      echo "======================================================================"

      python experiments/run_trueslice.py \
        --model "${MODEL_ID}" \
        --ppl-only \
        --cal-dataset "${CAL_SET}" \
        --cal-nsamples 128 \
        --cal-batch-size "${CAL_BATCH_SIZE}" \
        --cal-max-seqlen 2048 \
        --seed 42 \
        --final-orientation random \
        --ppl-eval-seqlen 2048 \
        --ppl-eval-batch-size "${EVAL_BATCH_SIZE}" \
        --ppl-eval-nsamples 128 \
        --dtype "${DTYPE}" \
        --distribute-model \
        --no-wandb
    done
  fi

  if [ "${RUN_TRUESLICE}" -eq 1 ]; then
    for CAL_SET in "${CAL_SETS[@]}"; do
      for SPARSITY in "${SPARSITIES_TRUESLICE[@]}"; do
        echo "======================================================================"
        echo "Running TrueSlice for ${MODEL_NAME}"
        echo "Sparsity: ${SPARSITY} | Calibration Dataset: ${CAL_SET}"
        echo "======================================================================"

        python experiments/run_trueslice.py \
          --model "${MODEL_ID}" \
          --save-dir "${BASE_PATH}/${MODEL_NAME}_SLICEGPT_${SPARSITY}_${CAL_SET}" \
          --cal-dataset "${CAL_SET}" \
          --cal-nsamples 128 \
          --cal-batch-size "${CAL_BATCH_SIZE}" \
          --cal-max-seqlen 2048 \
          --seed 42 \
          --sparsity "${SPARSITY}" \
          --round-interval 1 \
          --final-orientation random \
          --ppl-eval-seqlen 2048 \
          --ppl-eval-batch-size "${EVAL_BATCH_SIZE}" \
          --ppl-eval-nsamples 128 \
          --dtype "${DTYPE}" \
          --distribute-model \
          --no-wandb
      done
    done
  fi

  if [ "${RUN_SLICEGPT}" -eq 1 ]; then
    for CAL_SET in "${CAL_SETS[@]}"; do
      for SPARSITY in "${SPARSITIES_SLICEGPT[@]}"; do
        echo "======================================================================"
        echo "Running SliceGPT for ${MODEL_NAME}"
        echo "Sparsity: ${SPARSITY} | Calibration Dataset: ${CAL_SET}"
        echo "======================================================================"

        python experiments/run_slicegpt.py \
          --model "${MODEL_ID}" \
          --save-dir "${BASE_PATH}/${MODEL_NAME}_SLICEGPT_${SPARSITY}_${CAL_SET}" \
          --cal-dataset "${CAL_SET}" \
          --cal-nsamples 128 \
          --cal-batch-size "${CAL_BATCH_SIZE}" \
          --cal-max-seqlen 2048 \
          --seed 42 \
          --sparsity "${SPARSITY}" \
          --round-interval 1 \
          --final-orientation random \
          --ppl-eval-seqlen 2048 \
          --ppl-eval-batch-size "${EVAL_BATCH_SIZE}" \
          --ppl-eval-nsamples 128 \
          --dtype "${DTYPE}" \
          --distribute-model \
          --no-wandb
      done
    done
  fi
done