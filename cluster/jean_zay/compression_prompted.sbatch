#!/bin/bash
#SBATCH --job-name=arr-comp-prompt
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH --time=04:00:00
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --output=arr_runs/compression/slurm/%x-%A_%a.out
#SBATCH --error=arr_runs/compression/slurm/%x-%A_%a.err

set -euo pipefail
source cluster/jean_zay/common.sh
arr_preamble

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
MODELS=("mistralai/Mistral-7B-Instruct-v0.3" "meta-llama/Llama-3.1-8B-Instruct")
REVISIONS=("c170c708c41dac9275d15a8fff4eca08d52bab71" "0e9e39f249a16976918f6564b8830bc894c89659")
SLUGS=(mistral llama)

if [[ ! "$TASK_ID" =~ ^[0-3]$ ]]; then
  printf 'Prompted compression array index must be 0..3\n' >&2
  exit 2
fi
MODEL_INDEX=$((TASK_ID / 2))
DATASET_INDEX=$((TASK_ID % 2))
if (( DATASET_INDEX == 0 )); then
  DATA="$ARR_PROJECT_ROOT/data/arr/esnli_test.jsonl"
  DATASET_SLUG=esnli
  EXTRA=(--group-count 2000)
else
  DATA="$ARR_PROJECT_ROOT/data/arr/ds_critique_external_test.jsonl"
  DATASET_SLUG=ds-critique
  EXTRA=()
fi
OUTPUT="$ARR_PROJECT_ROOT/arr_runs/compression/prompted/${SLUGS[$MODEL_INDEX]}/direct/$DATASET_SLUG"

arr_run -m src.arr.compression_cli score-prompted \
  --config configs/arr/compression.yaml \
  --data "$DATA" \
  --model "${MODELS[$MODEL_INDEX]}" \
  --prompt-mode direct \
  --output-dir "$OUTPUT" \
  --seed 42 \
  --set "prompted.revision=${REVISIONS[$MODEL_INDEX]}" \
  --set prompted.local_files_only=true \
  --set prompted.batch_size=8 \
  "${EXTRA[@]}"
