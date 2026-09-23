#!/bin/bash
# Five genuinely independent Pythia backbones per R4 cell and global seed.
#SBATCH --job-name=arr-r4-dose-indep
#SBATCH --partition=h100
#SBATCH --account=cril
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:h100:1
#SBATCH --gpu-bind=closest
#SBATCH --time=02:00:00
#SBATCH --array=0-119%2
#SBATCH --export=NIL
#SBATCH --output=runs/r4_exposure/slurm/%x-%A_%a.out
#SBATCH --error=runs/r4_exposure/slurm/%x-%A_%a.err

set -euo pipefail
source "${SLURM_SUBMIT_DIR:?}/cluster/cril/common.sh"
arr_preamble

QIDS=(025 025 025 050 050 050 100 100)
EXPOSURES=(000 050 100 000 050 100 050 100)
SEEDS=(42 123 777)
TASK_ID="${SLURM_ARRAY_TASK_ID:?}"
MEMBER_ID=$((TASK_ID % 5))
SEED_CELL=$((TASK_ID / 5))
GLOBAL_SEED="${SEEDS[$((SEED_CELL % 3))]}"
CELL=$((SEED_CELL / 3))
QID="${QIDS[$CELL]}"
EXPOSURE="${EXPOSURES[$CELL]}"
MEMBER_SEED=$((GLOBAL_SEED + MEMBER_ID))
ROOT="$ARR_PROJECT_ROOT/runs/r4_exposure/dose/qid${QID}/exposure${EXPOSURE}/seed${GLOBAL_SEED}/independent"
OUTPUT="$ROOT/member_${MEMBER_ID}_seed${MEMBER_SEED}"
TRAIN="data/arr/r4_exposure/train_qid${QID}_exposure${EXPOSURE}.jsonl"
VALIDATION="data/arr/ds_critique_qidsplit_dev.jsonl"
GPT4="data/arr/r4_exposure/gpt4_qid_holdout.jsonl"

arr_run -m src.arr.tier2_cli train-member \
  --config configs/arr/tier2_pythia.yaml \
  --train-data "$TRAIN" --validation-data "$VALIDATION" \
  --loss listnet --global-seed "$GLOBAL_SEED" \
  --member-id "$MEMBER_ID" --arm independent --output "$OUTPUT"
arr_run -m src.arr.tier2_cli verify-run --run "$OUTPUT"
arr_run -m src.arr.tier2_score score-checkpoint \
  --run "$OUTPUT" --data "$GPT4" --output "$OUTPUT/gpt4_holdout"
