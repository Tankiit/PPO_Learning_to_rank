#!/bin/bash

set -euo pipefail

DETECTED_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ARR_PROJECT_ROOT="${ARR_PROJECT_ROOT:-$DETECTED_PROJECT_ROOT}"
ARR_HF_HOME="${ARR_HF_HOME:-${HF_HOME:-$ARR_PROJECT_ROOT/.cache/huggingface}}"
ARR_VENV="${ARR_VENV:-$ARR_PROJECT_ROOT/.venv}"

if [[ -r /etc/profile.d/z_modules.sh ]]; then
  source /etc/profile.d/z_modules.sh
elif ! command -v module >/dev/null 2>&1 && [[ -r /etc/profile.d/modules.sh ]]; then
  source /etc/profile.d/modules.sh
fi

if [[ ! -f "$ARR_PROJECT_ROOT/configs/arr/compression.yaml" ]]; then
  printf 'ARR project not found at %s\n' "$ARR_PROJECT_ROOT" >&2
  exit 2
fi
if [[ ! -x "$ARR_VENV/bin/python" ]]; then
  printf 'Jean-Zay environment not found at %s\n' "$ARR_VENV" >&2
  exit 2
fi

module purge
module load arch/h100
module load pytorch-gpu/py3/2.4.0
source "$ARR_VENV/bin/activate"

export PYTHONNOUSERSITE=1
export PYTHONPATH="$ARR_PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$ARR_HF_HOME"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export WANDB_DISABLED=true
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-24}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy

if [[ -n "${JOBSCRATCH:-}" ]]; then
  export TMPDIR="$JOBSCRATCH"
else
  export TMPDIR="$ARR_PROJECT_ROOT/tmp/${SLURM_JOB_ID:-manual}"
  mkdir -p "$TMPDIR"
fi
mkdir -p "$ARR_PROJECT_ROOT/arr_runs/compression/slurm"

ARR_PYTHON="$ARR_VENV/bin/python"
cd "$ARR_PROJECT_ROOT"

arr_preamble() {
  printf 'date=%s\n' "$(date --iso-8601=seconds)"
  printf 'host=%s job=%s array=%s pwd=%s\n' \
    "$(hostname)" "${SLURM_JOB_ID:-none}" "${SLURM_ARRAY_TASK_ID:-none}" "$PWD"
  printf 'python=%s\n' "$($ARR_PYTHON --version 2>&1)"
  printf 'cuda_visible_devices=%s\n' "${CUDA_VISIBLE_DEVICES:-unset}"
  nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv,noheader
}

arr_run() {
  srun --unbuffered "$ARR_PYTHON" "$@"
}
