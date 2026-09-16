#!/bin/bash
#SBATCH --job-name=arr-pythia-combine
#SBATCH --partition=h100
#SBATCH --account=cril
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:30:00
#SBATCH --array=0-17%2
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
ARM="${ARMS[$((TASK_ID % 2))]}"
LOSS="${LOSSES[$(((TASK_ID / 2) % 3))]}"
GLOBAL_SEED="${SEEDS[$((TASK_ID / 6))]}"
case "$LOSS" in
  listnet) STEM=independent_backbones ;;
  listmle) STEM=independent_backbones_listmle ;;
  mse) STEM=independent_backbones_mse ;;
esac
ROOT="$ARR_PROJECT_ROOT/runs/tier2_pythia/${STEM}_qid_50ep_seed${GLOBAL_SEED}/runs/$ARM"
RUNS=()
for MEMBER_ID in 0 1 2 3 4; do
  RUNS+=("$ROOT/member_${MEMBER_ID}_seed$((GLOBAL_SEED + MEMBER_ID))")
done

arr_run -m src.arr.tier2_cli combine-independent \
  --runs "${RUNS[@]}" --output "$ROOT/ensemble"
