#!/bin/bash
#SBATCH --job-name=arr-comp-table
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH --time=01:00:00
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --output=arr_runs/compression/slurm/%x-%j.out
#SBATCH --error=arr_runs/compression/slurm/%x-%j.err

set -euo pipefail
source cluster/jean_zay/common.sh
arr_preamble
arr_run -m src.arr.compression_cli aggregate \
  --config configs/arr/compression.yaml \
  --roots "$ARR_PROJECT_ROOT/arr_runs/compression" \
  --output-dir "$ARR_PROJECT_ROOT/arr_runs/compression/aggregate" \
  --seed 42 \
  --set statistics.bootstrap_samples=2000
