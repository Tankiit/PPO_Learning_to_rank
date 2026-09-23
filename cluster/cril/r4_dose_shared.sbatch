#!/bin/bash
# R4 assumption: GPT-4 is the held-out provenance. Exposure is varied across
# training QIDs while the disjoint 54-QID GPT-4 evaluation remains fixed.
#SBATCH --job-name=arr-r4-dose-shared
#SBATCH --partition=h100
#SBATCH --account=cril
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:h100:1
#SBATCH --gpu-bind=closest
#SBATCH --time=02:00:00
#SBATCH --array=0-23%2
#SBATCH --export=NIL
#SBATCH --output=runs/r4_exposure/slurm/%x-%A_%a.out
#SBATCH --error=runs/r4_exposure/slurm/%x-%A_%a.err

set -euo pipefail
source "${SLURM_SUBMIT_DIR:?}/cluster/cril/common.sh"
arr_preamble

# qid100/exposure000 is omitted because jobs 13138794--13138796 already
# produced the exact three-seed baseline with the same config and data.
QIDS=(025 025 025 050 050 050 100 100)
EXPOSURES=(000 050 100 000 050 100 050 100)
SEEDS=(42 123 777)
TASK_ID="${SLURM_ARRAY_TASK_ID:?}"
CELL=$((TASK_ID / 3))
GLOBAL_SEED="${SEEDS[$((TASK_ID % 3))]}"
QID="${QIDS[$CELL]}"
EXPOSURE="${EXPOSURES[$CELL]}"
ROOT="$ARR_PROJECT_ROOT/runs/r4_exposure/dose/qid${QID}/exposure${EXPOSURE}/seed${GLOBAL_SEED}/shared"
TRAIN="data/arr/r4_exposure/train_qid${QID}_exposure${EXPOSURE}.jsonl"
VALIDATION="data/arr/ds_critique_qidsplit_dev.jsonl"
GPT4="data/arr/r4_exposure/gpt4_qid_holdout.jsonl"

arr_run -m src.arr.tier2_cli train-shared \
  --config configs/arr/tier2_pythia.yaml \
  --train-data "$TRAIN" --validation-data "$VALIDATION" \
  --loss listnet --global-seed "$GLOBAL_SEED" --arm baseline --output "$ROOT"
arr_run -m src.arr.tier2_cli verify-run --run "$ROOT"
arr_run -m src.arr.tier2_score score-checkpoint \
  --run "$ROOT" --data "$GPT4" --output "$ROOT/gpt4_holdout"
