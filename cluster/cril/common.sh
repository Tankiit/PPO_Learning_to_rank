#!/bin/bash

set -euo pipefail

DETECTED_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ARR_PROJECT_ROOT="${ARR_PROJECT_ROOT:-$DETECTED_PROJECT_ROOT}"
ARR_BASE_VENV="${ARR_BASE_VENV:-$HOME/venv-pytorch2.5-py3.11}"
ARR_OVERLAY="${ARR_OVERLAY:-$ARR_PROJECT_ROOT/cluster/offline/site-packages}"
export ARR_PYTHIA_REVISION="${ARR_PYTHIA_REVISION:-a39f36b100fe8a5377810d56c3f4789b9c53ac42}"
export ARR_PYTHIA_MODEL_DIR="${ARR_PYTHIA_MODEL_DIR:-$ARR_PROJECT_ROOT/cluster/offline/models/pythia-70m-a39f36b}"

if [[ ! -f "$ARR_PROJECT_ROOT/configs/uncertainty_study/study.yaml" ]]; then
  printf 'Uncertainty-study project not found at %s\n' "$ARR_PROJECT_ROOT" >&2
  exit 2
fi
if [[ ! -x "$ARR_BASE_VENV/bin/python" ]]; then
  printf 'Cluster Python environment not found: %s\n' "$ARR_BASE_VENV" >&2
  exit 2
fi

if command -v module >/dev/null 2>&1; then
  module load cuda/12.6
fi
source "$ARR_BASE_VENV/bin/activate"

export SLURM_EXPORT_ENV=ALL
export PYTHONNOUSERSITE=1
if [[ -d "$ARR_OVERLAY" ]]; then
  export PYTHONPATH="$ARR_OVERLAY:$ARR_PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
else
  export PYTHONPATH="$ARR_PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
fi
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export WANDB_DISABLED=true
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export TMPDIR="${SLURM_TMPDIR:-/tmp/${USER}/arr-uncertainty-${SLURM_JOB_ID:-manual}}"
export MPLCONFIGDIR="$TMPDIR/matplotlib"
mkdir -p \
  "$TMPDIR" \
  "$MPLCONFIGDIR" \
  "$ARR_PROJECT_ROOT/runs/uncertainty_study/slurm" \
  "$ARR_PROJECT_ROOT/runs/tier2_pythia/slurm"

ARR_PYTHON="$ARR_BASE_VENV/bin/python"
cd "$ARR_PROJECT_ROOT"

arr_preamble() {
  printf 'date=%s\n' "$(date --iso-8601=seconds)"
  printf 'host=%s job=%s array=%s pwd=%s\n' \
    "$(hostname)" "${SLURM_JOB_ID:-none}" "${SLURM_ARRAY_TASK_ID:-none}" "$PWD"
  printf 'python=%s\n' "$($ARR_PYTHON --version 2>&1)"
  printf 'cuda_visible_devices=%s\n' "${CUDA_VISIBLE_DEVICES:-unset}"
  if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv,noheader
  fi
}

arr_run() {
  if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    srun --unbuffered "$ARR_PYTHON" "$@"
  else
    "$ARR_PYTHON" "$@"
  fi
}
