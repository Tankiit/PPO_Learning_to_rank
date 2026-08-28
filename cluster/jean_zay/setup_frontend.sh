#!/bin/bash

set -euo pipefail

# Non-interactive SSH commands do not source Jean-Zay's login profile.  The
# frontend proxy is required for PyPI and Hugging Face; compute jobs explicitly
# unset it again in common.sh because GPU nodes have no Internet route.
if [[ -r /etc/profile.d/proxy.sh ]]; then
  source /etc/profile.d/proxy.sh
fi
export PYTHONNOUSERSITE=1

DETECTED_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ARR_PROJECT_ROOT="${ARR_PROJECT_ROOT:-$DETECTED_PROJECT_ROOT}"
ARR_HF_HOME="${ARR_HF_HOME:-${HF_HOME:-$ARR_PROJECT_ROOT/.cache/huggingface}}"
BASE_PYTHON="${ARR_JZ_BASE_PYTHON:-/lustre/fshomisc/sup/pub/anaconda-py3/h100/2024.06/envs/pytorch-gpu-2.4.0+py3.11.9/bin/python}"
VENV="$ARR_PROJECT_ROOT/.venv"

if [[ ! -x "$BASE_PYTHON" ]]; then
  printf 'Jean-Zay module Python not found: %s\n' "$BASE_PYTHON" >&2
  exit 2
fi

mkdir -p "$ARR_PROJECT_ROOT" "$ARR_HF_HOME"
if [[ ! -x "$VENV/bin/python" ]]; then
  "$BASE_PYTHON" -m venv --system-site-packages "$VENV"
fi
"$VENV/bin/python" -m pip install --upgrade pip
"$VENV/bin/python" -m pip install --upgrade \
  -r "$ARR_PROJECT_ROOT/cluster/jean_zay/requirements.txt"

export HF_HOME="$ARR_HF_HOME"
"$VENV/bin/huggingface-cli" download mistralai/Mistral-7B-Instruct-v0.3 \
  --revision c170c708c41dac9275d15a8fff4eca08d52bab71 \
  --cache-dir "$ARR_HF_HOME/hub"
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH="$ARR_PROJECT_ROOT" \
  "$VENV/bin/python" -c \
  'from src.arr.utils import resolve_hf_source; print(resolve_hf_source("mistralai/Mistral-7B-Instruct-v0.3", "c170c708c41dac9275d15a8fff4eca08d52bab71", True)); print(resolve_hf_source("meta-llama/Llama-3.1-8B-Instruct", "0e9e39f249a16976918f6564b8830bc894c89659", True))'

printf 'Jean-Zay frontend setup complete at %s\n' "$ARR_PROJECT_ROOT"
