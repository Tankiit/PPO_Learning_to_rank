#!/bin/bash

set -euo pipefail
MODE="${1:---dry-run}"
if [[ "$MODE" != "--dry-run" && "$MODE" != "--submit" ]]; then
  printf 'Usage: bash cluster/cril/submit_pythia_preflight.sh [--dry-run|--submit]\n' >&2
  exit 2
fi
if [[ "$(hostname)" != "nodeC000" ]]; then
  printf 'Run from nodeC000 after: ssh clustercril; ssh nodec000\n' >&2
  exit 2
fi
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

[[ -f "$ARR_PROJECT_ROOT/runs/pythia_infra/model_manifest.json" ]] || {
  printf 'Run cluster/cril/prepare_pythia.sh first.\n' >&2
  exit 2
}
[[ -f "$ARR_PROJECT_ROOT/data/arr/ds_critique_qidsplit_manifest.json" ]] || {
  printf 'The audited QID split is missing.\n' >&2
  exit 2
}

if [[ "$MODE" == "--dry-run" ]]; then
  printf '/usr/local/bin/sbatch --parsable cluster/cril/pythia_preflight.sbatch\n'
  printf 'mode=dry-run scientific_training=false\n'
  exit 0
fi

JOB_ID="$(/usr/local/bin/sbatch --parsable cluster/cril/pythia_preflight.sbatch)"
printf 'preflight=%s\nscientific_training=false\n' "$JOB_ID"
