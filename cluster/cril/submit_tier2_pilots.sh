#!/bin/bash

set -euo pipefail
MODE="${1:---dry-run}"
if [[ "$MODE" != "--dry-run" && "$MODE" != "--submit" ]]; then
  printf 'Usage: bash cluster/cril/submit_tier2_pilots.sh [--dry-run|--submit]\n' >&2
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
mkdir -p "$ARR_PROJECT_ROOT/runs/tier2_pythia/slurm"

/usr/local/bin/sbatch --test-only cluster/cril/tier2_pilot.sbatch
if [[ "$MODE" == "--dry-run" ]]; then
  printf '/usr/local/bin/sbatch --parsable cluster/cril/tier2_pilot.sbatch\n'
  printf 'mode=dry-run scientific_training=pilot_only cells=6\n'
  exit 0
fi

JOB_ID="$(/usr/local/bin/sbatch --parsable cluster/cril/tier2_pilot.sbatch)"
printf 'tier2_pilots=%s\nscientific_training=pilot_only\n' "$JOB_ID"
