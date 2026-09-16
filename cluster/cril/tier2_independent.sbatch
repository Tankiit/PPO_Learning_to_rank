#!/bin/bash
#SBATCH --job-name=arr-pythia-indep
#SBATCH --partition=h100
#SBATCH --account=cril
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:h100:1
#SBATCH --gpu-bind=closest
#SBATCH --time=02:00:00
#SBATCH --array=0-89%2
#SBATCH --export=NIL
#SBATCH --output=runs/tier2_pythia/slurm/%x-%A_%a.out
#SBATCH --error=runs/tier2_pythia/slurm/%x-%A_%a.err

set -euo pipefail
source "${SLURM_SUBMIT_DIR:?}/cluster/cril/common.sh"
arr_preamble

SEEDS=(42 123 777)
LOSSES=(listnet listmle mse)
ARMS=(independent independent_bootstrap)
TASK_ID="${SLURM_ARRAY_TASK_ID:?}"
MEMBER_ID=$((TASK_ID % 5))
CELL=$((TASK_ID / 5))
ARM="${ARMS[$((CELL % 2))]}"
LOSS="${LOSSES[$(((CELL / 2) % 3))]}"
GLOBAL_SEED="${SEEDS[$((CELL / 6))]}"
MEMBER_SEED=$((GLOBAL_SEED + MEMBER_ID))

case "$LOSS" in
  listnet) STEM=independent_backbones ;;
  listmle) STEM=independent_backbones_listmle ;;
  mse) STEM=independent_backbones_mse ;;
esac
ROOT="$ARR_PROJECT_ROOT/runs/tier2_pythia/${STEM}_qid_50ep_seed${GLOBAL_SEED}"
OUTPUT="$ROOT/runs/$ARM/member_${MEMBER_ID}_seed${MEMBER_SEED}"

arr_run -m src.arr.tier2_cli train-member \
  --config configs/arr/tier2_pythia.yaml \
  --loss "$LOSS" --global-seed "$GLOBAL_SEED" \
  --member-id "$MEMBER_ID" --arm "$ARM" --output "$OUTPUT"
arr_run -m src.arr.tier2_cli verify-run --run "$OUTPUT"
