#!/bin/bash
#SBATCH --job-name=arr-r4-loso-combine
#SBATCH --partition=h100
#SBATCH --account=cril
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:h100:1
#SBATCH --gpu-bind=closest
#SBATCH --time=02:00:00
#SBATCH --array=0-2%2
#SBATCH --export=NIL
#SBATCH --output=runs/r4_exposure/slurm/%x-%A_%a.out
#SBATCH --error=runs/r4_exposure/slurm/%x-%A_%a.err

set -euo pipefail
source "${SLURM_SUBMIT_DIR:?}/cluster/cril/common.sh"
arr_preamble

SEEDS=(42 123 777)
GLOBAL_SEED="${SEEDS[${SLURM_ARRAY_TASK_ID:?}]}"
ROOT="$ARR_PROJECT_ROOT/runs/r4_exposure/loso/seed${GLOBAL_SEED}/independent"
MEMBERS=()
SCORED=()
LOSO_SCORED=()
for MEMBER_ID in 0 1 2 3 4; do
  MEMBER_SEED=$((GLOBAL_SEED + MEMBER_ID))
  MEMBER="$ROOT/member_${MEMBER_ID}_seed${MEMBER_SEED}"
  MEMBERS+=("$MEMBER")
  SCORED+=("$MEMBER/gpt4_holdout")
  LOSO_SCORED+=("$MEMBER/gpt4_loso")
done

arr_run -m src.arr.tier2_cli combine-independent \
  --runs "${MEMBERS[@]}" --output "$ROOT/ensemble"
arr_run -m src.arr.tier2_score combine-independent \
  --runs "${SCORED[@]}" \
  --data data/arr/r4_exposure/gpt4_qid_holdout.jsonl \
  --output "$ROOT/gpt4_ensemble"
arr_run -m src.arr.tier2_score combine-independent \
  --runs "${LOSO_SCORED[@]}" \
  --data data/arr/r4_exposure/gpt4_loso_train_qids.jsonl \
  --output "$ROOT/gpt4_loso_ensemble"
