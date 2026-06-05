#!/bin/bash
#SBATCH --job-name=explrank_loss
#SBATCH --array=0-11
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/loss_comparison_%A_%a.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

LOSSES=(mse listnet ranknet approxndcg)
SEEDS=(42 123 456)
IDX=${SLURM_ARRAY_TASK_ID:-0}
LOSS_IDX=$((IDX / 3))
SEED_IDX=$((IDX % 3))
LOSS=${LOSSES[$LOSS_IDX]}
SEED=${SEEDS[$SEED_IDX]}

export EXPLRANK_DATA_DIR="${EXPLRANK_DATA_DIR:-./data}"
export EXPLRANK_OUTPUT_DIR="${EXPLRANK_OUTPUT_DIR:-./outputs}"

python scripts/train.py \
  experiment=loss_comparison \
  loss=${LOSS} \
  seed=${SEED} \
  data=ds_critique
