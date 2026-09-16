#!/bin/bash
#SBATCH --job-name=arr-pythia-check
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
#SBATCH --output=runs/pythia_infra/slurm/%x-%j.out
#SBATCH --error=runs/pythia_infra/slurm/%x-%j.err

set -euo pipefail
source "${SLURM_SUBMIT_DIR:?}/cluster/cril/common.sh"
arr_preamble

arr_run -m pytest -q \
  tests/arr/test_losses.py \
  tests/arr/test_gauge_invariance.py \
  tests/uncertainty_study/test_study.py
arr_run cluster/cril/verify_pythia_cluster.py
