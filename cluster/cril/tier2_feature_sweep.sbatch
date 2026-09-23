#!/bin/bash
#SBATCH --job-name=arr-pythia-mask-k
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
ARMS=(features bootstrap_features)
KEEP_COUNTS=(20 100 180 260 340)
TASK_ID="${SLURM_ARRAY_TASK_ID:?}"
K="${KEEP_COUNTS[$((TASK_ID % 5))]}"
ARM="${ARMS[$(((TASK_ID / 5) % 2))]}"
LOSS="${LOSSES[$(((TASK_ID / 10) % 3))]}"
GLOBAL_SEED="${SEEDS[$((TASK_ID / 30))]}"

ROOT="$ARR_PROJECT_ROOT/runs/tier2_pythia/shared_feature_k_sweep_qid_50ep_seed${GLOBAL_SEED}"
OUTPUT="$ROOT/$LOSS/${ARM}_k${K}"

arr_run -m src.arr.tier2_cli train-shared \
  --config configs/arr/tier2_pythia.yaml \
  --loss "$LOSS" --global-seed "$GLOBAL_SEED" \
  --arm "$ARM" --feature-keep-count "$K" --output "$OUTPUT"
arr_run -m src.arr.tier2_cli verify-run --run "$OUTPUT"
