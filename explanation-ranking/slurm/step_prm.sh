#!/bin/bash
#SBATCH --job-name=explrank_step_prm
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --output=logs/step_prm_%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

export EXPLRANK_DATA_DIR="${EXPLRANK_DATA_DIR:-./data}"
export EXPLRANK_OUTPUT_DIR="${EXPLRANK_OUTPUT_DIR:-./outputs}"

# Step PRM training script to be wired when step trainer lands; for now use experiment config
python scripts/train.py \
  experiment=step_prm \
  data=ds_critique \
  seed=${SEED:-42}
