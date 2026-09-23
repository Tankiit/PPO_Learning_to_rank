#!/bin/bash
#SBATCH --job-name=arr-pythia-mask-k-pilot
#SBATCH --partition=h100
#SBATCH --account=cril
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:h100:1
#SBATCH --gpu-bind=closest
#SBATCH --time=00:30:00
#SBATCH --export=NIL
#SBATCH --output=runs/tier2_pythia/slurm/%x-%j.out
#SBATCH --error=runs/tier2_pythia/slurm/%x-%j.err

set -euo pipefail
source "${SLURM_SUBMIT_DIR:?}/cluster/cril/common.sh"
arr_preamble

OUTPUT="$ARR_PROJECT_ROOT/runs/tier2_pythia/pilots/shared_listnet_features_k20"
arr_run -m src.arr.tier2_cli train-shared \
  --config configs/arr/tier2_pythia.yaml \
  --loss listnet --global-seed 42 --arm features \
  --feature-keep-count 20 --pilot --output "$OUTPUT"
arr_run -m src.arr.tier2_cli verify-run --run "$OUTPUT"
