#!/bin/bash
#SBATCH --job-name=arr-epi-eval
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH --time=04:00:00
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --output=arr_runs/epistemic/slurm/%x-%A_%a.out
#SBATCH --error=arr_runs/epistemic/slurm/%x-%A_%a.err

set -euo pipefail
source cluster/jean_zay/common.sh
arr_preamble

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
MODELS=("mistralai/Mistral-7B-Instruct-v0.3" "meta-llama/Llama-3.1-8B-Instruct")
SLUGS=(mistral llama)
LOSSES=(mse listnet)

if [[ ! "$TASK_ID" =~ ^[0-7]$ ]]; then
  printf 'Epistemic evaluation array index must be 0..7\n' >&2
  exit 2
fi
CONDITION_INDEX=$((TASK_ID / 2))
DATASET_INDEX=$((TASK_ID % 2))
MODEL_INDEX=$((CONDITION_INDEX / 2))
LOSS_INDEX=$((CONDITION_INDEX % 2))
CHECKPOINT="$ARR_PROJECT_ROOT/arr_runs/epistemic/judges/${SLUGS[$MODEL_INDEX]}/${LOSSES[$LOSS_INDEX]}/seed-42/best"
if (( DATASET_INDEX == 0 )); then
  DATA="$ARR_PROJECT_ROOT/data/arr/esnli_test.jsonl"
  DATASET_SLUG=esnli
else
  DATA="$ARR_PROJECT_ROOT/data/arr/ds_critique_external_test.jsonl"
  DATASET_SLUG=ds-critique
fi
OUTPUT="$ARR_PROJECT_ROOT/arr_runs/epistemic/evaluations/${SLUGS[$MODEL_INDEX]}/${LOSSES[$LOSS_INDEX]}/seed-42/$DATASET_SLUG"

arr_run -m src.arr.compression_cli evaluate-epistemic \
  --config configs/arr/compression.yaml \
  --data "$DATA" \
  --checkpoint "$CHECKPOINT" \
  --output-dir "$OUTPUT" \
  --model-name "${MODELS[$MODEL_INDEX]}" \
  --loss "${LOSSES[$LOSS_INDEX]}" \
  --seed 42 \
  --set evaluation.local_files_only=true \
  --set statistics.bootstrap_samples=2000
