#!/bin/bash

set -e

RUN_DENSE_PPL=0
RUN_TRUESLICE=1
RUN_SLICEGPT=0

# MODEL_ID="microsoft/phi-2"
# MODEL_NAME="Phi-2-2.7B"

MODEL_ID="meta-llama/Llama-3.2-1B"
MODEL_NAME="Llama-3.2-1B"

# SPARSITIES=(0.125 0.25 0.375 0.5)
SPARSITIES=(0.125)

CAL_SETS=("wikitext2")

CAL_BATCH_SIZE=2
EVAL_BATCH_SIZE=2
DTYPE="fp32"

BASE_PATH="/mnt/ML_Storage/models_global_slicing"

if [ "${RUN_DENSE_PPL}" -eq 1 ]; then
  for CAL_SET in "${CAL_SETS[@]}"; do
    echo "======================================================================"
    echo "Running Perplexity Eval for ${MODEL_NAME}"
    echo "Calibration Dataset: ${CAL_SET}"
    echo "======================================================================"

    python experiments/run_slicegpt.py \
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
    for SPARSITY in "${SPARSITIES[@]}"; do
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
    for SPARSITY in "${SPARSITIES[@]}"; do
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