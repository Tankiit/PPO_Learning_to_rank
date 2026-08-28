#!/bin/bash
#SBATCH --job-name=arr-epi-train
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH --time=10:00:00
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --output=arr_runs/epistemic/slurm/%x-%A_%a.out
#SBATCH --error=arr_runs/epistemic/slurm/%x-%A_%a.err

set -euo pipefail
source cluster/jean_zay/common.sh
arr_preamble

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
MODELS=("mistralai/Mistral-7B-Instruct-v0.3" "meta-llama/Llama-3.1-8B-Instruct")
REVISIONS=("c170c708c41dac9275d15a8fff4eca08d52bab71" "0e9e39f249a16976918f6564b8830bc894c89659")
SLUGS=(mistral llama)
LOSSES=(mse listnet)

if [[ ! "$TASK_ID" =~ ^[0-3]$ ]]; then
  printf 'Epistemic scalar array index must be 0..3\n' >&2
  exit 2
fi
MODEL_INDEX=$((TASK_ID / 2))
LOSS_INDEX=$((TASK_ID % 2))
OUTPUT="$ARR_PROJECT_ROOT/arr_runs/epistemic/judges/${SLUGS[$MODEL_INDEX]}/${LOSSES[$LOSS_INDEX]}/seed-42"

arr_run -m src.arr.compression_cli train-epistemic-judge \
  --config configs/arr/compression.yaml \
  --train-data "$ARR_PROJECT_ROOT/data/arr/esnli_train.jsonl" \
  --validation-data "$ARR_PROJECT_ROOT/data/arr/esnli_val.jsonl" \
  --output-dir "$OUTPUT" \
  --model "${MODELS[$MODEL_INDEX]}" \
  --loss "${LOSSES[$LOSS_INDEX]}" \
  --seed 42 \
  --head-count 5 \
  --set "judge_training.revision=${REVISIONS[$MODEL_INDEX]}" \
  --set judge_training.local_files_only=true \
  --set judge_training.epochs=3
