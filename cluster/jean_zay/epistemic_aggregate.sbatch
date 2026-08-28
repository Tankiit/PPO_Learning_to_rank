#!/bin/bash
#SBATCH --job-name=arr-epi-agg
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH --time=00:20:00
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --output=arr_runs/epistemic/slurm/%x-%j.out
#SBATCH --error=arr_runs/epistemic/slurm/%x-%j.err

set -euo pipefail
source cluster/jean_zay/common.sh
arr_preamble
arr_run -m src.arr.epistemic_aggregate \
  --root "$ARR_PROJECT_ROOT/arr_runs/epistemic" \
  --output "$ARR_PROJECT_ROOT/arr_runs/epistemic/aggregate" \
  --expected-runs 8
