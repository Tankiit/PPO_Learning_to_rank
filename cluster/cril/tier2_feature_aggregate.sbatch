#!/bin/bash
#SBATCH --job-name=arr-pythia-mask-summary
#SBATCH --partition=h100
#SBATCH --account=cril
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:h100:1
#SBATCH --gpu-bind=closest
#SBATCH --time=01:00:00
#SBATCH --export=NIL
#SBATCH --output=runs/tier2_pythia/slurm/%x-%j.out
#SBATCH --error=runs/tier2_pythia/slurm/%x-%j.err

set -euo pipefail
source "${SLURM_SUBMIT_DIR:?}/cluster/cril/common.sh"
arr_preamble

arr_run -m scripts.analyze_tier2_feature_sweep \
  --root runs/tier2_pythia \
  --config configs/arr/tier2_pythia.yaml \
  --output runs/tier2_pythia/feature_sweep_analysis
