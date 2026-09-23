#!/bin/bash
#SBATCH --job-name=arr-r4-loso-indep
#SBATCH --partition=h100
#SBATCH --account=cril
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:h100:1
#SBATCH --gpu-bind=closest
#SBATCH --time=02:00:00
#SBATCH --array=0-14%2
#SBATCH --export=NIL
#SBATCH --output=runs/r4_exposure/slurm/%x-%A_%a.out
#SBATCH --error=runs/r4_exposure/slurm/%x-%A_%a.err

set -euo pipefail
source "${SLURM_SUBMIT_DIR:?}/cluster/cril/common.sh"
arr_preamble

SEEDS=(42 123 777)
TASK_ID="${SLURM_ARRAY_TASK_ID:?}"
MEMBER_ID=$((TASK_ID % 5))
GLOBAL_SEED="${SEEDS[$((TASK_ID / 5))]}"
MEMBER_SEED=$((GLOBAL_SEED + MEMBER_ID))
ROOT="$ARR_PROJECT_ROOT/runs/r4_exposure/loso/seed${GLOBAL_SEED}/independent"
OUTPUT="$ROOT/member_${MEMBER_ID}_seed${MEMBER_SEED}"
TRAIN="data/arr/r4_exposure/train_qid100_exposure000.jsonl"
VALIDATION="data/arr/ds_critique_qidsplit_dev.jsonl"
GPT4="data/arr/r4_exposure/gpt4_qid_holdout.jsonl"
GPT4_LOSO="data/arr/r4_exposure/gpt4_loso_train_qids.jsonl"

arr_run -m src.arr.tier2_cli train-member \
  --config configs/arr/tier2_pythia.yaml \
  --train-data "$TRAIN" --validation-data "$VALIDATION" \
  --loss listnet --global-seed "$GLOBAL_SEED" \
  --member-id "$MEMBER_ID" --arm independent --output "$OUTPUT"
arr_run -m src.arr.tier2_cli verify-run --run "$OUTPUT"
arr_run -m src.arr.tier2_score score-checkpoint \
  --run "$OUTPUT" --data "$GPT4" --output "$OUTPUT/gpt4_holdout"
arr_run -m src.arr.tier2_score score-checkpoint \
  --run "$OUTPUT" --data "$GPT4_LOSO" --output "$OUTPUT/gpt4_loso"
