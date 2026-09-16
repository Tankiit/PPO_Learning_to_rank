#!/bin/bash
#SBATCH --job-name=arr-pythia-shared
#SBATCH --partition=h100
#SBATCH --account=cril
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:h100:1
#SBATCH --gpu-bind=closest
#SBATCH --time=02:00:00
#SBATCH --array=0-62%2
#SBATCH --export=NIL
#SBATCH --output=runs/tier2_pythia/slurm/%x-%A_%a.out
#SBATCH --error=runs/tier2_pythia/slurm/%x-%A_%a.err

set -euo pipefail
source "${SLURM_SUBMIT_DIR:?}/cluster/cril/common.sh"
arr_preamble

SEEDS=(42 123 777)
LOSSES=(listnet listmle mse)
ARMS=(baseline bootstrap features bootstrap_features lambda_0p01 lambda_0p1 lambda_1)
TASK_ID="${SLURM_ARRAY_TASK_ID:?}"
ARM="${ARMS[$((TASK_ID % 7))]}"
LOSS="${LOSSES[$(((TASK_ID / 7) % 3))]}"
GLOBAL_SEED="${SEEDS[$((TASK_ID / 21))]}"
ROOT="$ARR_PROJECT_ROOT/runs/tier2_pythia/shared_ablation_qid_50ep_seed${GLOBAL_SEED}"
OUTPUT="$ROOT/$LOSS/$ARM"

arr_run -m src.arr.tier2_cli train-shared \
  --config configs/arr/tier2_pythia.yaml \
  --loss "$LOSS" --global-seed "$GLOBAL_SEED" \
  --arm "$ARM" --output "$OUTPUT"
arr_run -m src.arr.tier2_cli verify-run --run "$OUTPUT"
