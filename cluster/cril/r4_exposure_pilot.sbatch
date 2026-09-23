#!/bin/bash
#SBATCH --job-name=arr-r4-pilot
#SBATCH --partition=h100
#SBATCH --account=cril
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:h100:1
#SBATCH --gpu-bind=closest
#SBATCH --time=00:40:00
#SBATCH --array=0-5%2
#SBATCH --export=NIL
#SBATCH --output=runs/r4_exposure/slurm/%x-%A_%a.out
#SBATCH --error=runs/r4_exposure/slurm/%x-%A_%a.err

set -euo pipefail
source "${SLURM_SUBMIT_DIR:?}/cluster/cril/common.sh"
arr_preamble

EXPOSURES=(000 050 100)
TASK_ID="${SLURM_ARRAY_TASK_ID:?}"
EXPOSURE="${EXPOSURES[$((TASK_ID / 2))]}"
if (( TASK_ID % 2 == 0 )); then
  CONSTRUCTION=shared
else
  CONSTRUCTION=independent
fi
TRAIN="data/arr/r4_exposure/train_qid100_exposure${EXPOSURE}.jsonl"
VALIDATION="data/arr/r4_exposure/gpt4_qid_holdout.jsonl"
OUTPUT="$ARR_PROJECT_ROOT/runs/r4_exposure/pilots/exposure${EXPOSURE}/${CONSTRUCTION}"

if [[ "$CONSTRUCTION" == shared ]]; then
  arr_run -m src.arr.tier2_cli train-shared \
    --config configs/arr/tier2_pythia.yaml \
    --train-data "$TRAIN" --validation-data "$VALIDATION" \
    --loss listnet --global-seed 42 --arm baseline --pilot --output "$OUTPUT"
else
  arr_run -m src.arr.tier2_cli train-member \
    --config configs/arr/tier2_pythia.yaml \
    --train-data "$TRAIN" --validation-data "$VALIDATION" \
    --loss listnet --global-seed 42 --arm independent \
    --member-id 0 --pilot --output "$OUTPUT"
fi
arr_run -m src.arr.tier2_cli verify-run --run "$OUTPUT"
arr_run -m src.arr.tier2_score score-checkpoint \
  --run "$OUTPUT" --data "$VALIDATION" --output "$OUTPUT/gpt4_score_check"
