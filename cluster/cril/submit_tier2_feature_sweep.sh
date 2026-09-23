#!/bin/bash

set -euo pipefail
MODE="${1:---dry-run}"
if [[ "$MODE" != "--dry-run" && "$MODE" != "--pilot" && "$MODE" != "--submit" ]]; then
  printf 'Usage: bash cluster/cril/submit_tier2_feature_sweep.sh [--dry-run|--pilot|--submit]\n' >&2
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

/usr/local/bin/sbatch --test-only cluster/cril/tier2_feature_sweep_pilot.sbatch
/usr/local/bin/sbatch --test-only cluster/cril/tier2_feature_sweep.sbatch
if [[ "$MODE" == "--dry-run" ]]; then
  printf 'pilot: /usr/local/bin/sbatch --parsable cluster/cril/tier2_feature_sweep_pilot.sbatch\n'
  printf 'full:  /usr/local/bin/sbatch --parsable cluster/cril/tier2_feature_sweep.sbatch\n'
  printf 'mode=dry-run pilot_cells=1 full_cells=90\n'
  exit 0
fi
if [[ "$MODE" == "--pilot" ]]; then
  JOB_ID="$(/usr/local/bin/sbatch --parsable cluster/cril/tier2_feature_sweep_pilot.sbatch)"
  printf 'tier2_feature_sweep_pilot=%s\n' "$JOB_ID"
  exit 0
fi

PILOT="$ARR_PROJECT_ROOT/runs/tier2_pythia/pilots/shared_listnet_features_k20/_final.json"
[[ -f "$PILOT" ]] || {
  printf 'Pilot gate not satisfied: submit --pilot and inspect %s first.\n' "$PILOT" >&2
  exit 2
}
JOB_ID="$(/usr/local/bin/sbatch --parsable cluster/cril/tier2_feature_sweep.sbatch)"
printf 'tier2_feature_sweep=%s\nfull_cells=90\n' "$JOB_ID"
