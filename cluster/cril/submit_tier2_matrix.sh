#!/bin/bash

# The full matrix is intentionally separate from the pilot launcher. It must
# only be submitted after all six pilot cells have complete final manifests.
set -euo pipefail
MODE="${1:---dry-run}"
if [[ "$MODE" != "--dry-run" && "$MODE" != "--submit" ]]; then
  printf 'Usage: bash cluster/cril/submit_tier2_matrix.sh [--dry-run|--submit]\n' >&2
  exit 2
fi
if [[ "$(hostname)" != "nodeC000" ]]; then
  printf 'Run from nodeC000 after: ssh clustercril; ssh nodec000\n' >&2
  exit 2
fi
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

for construction in independent shared; do
  for loss in listnet listmle mse; do
    marker="$ARR_PROJECT_ROOT/runs/tier2_pythia/pilots/${construction}_${loss}/_final.json"
    [[ -f "$marker" ]] || {
      printf 'Pilot gate not satisfied: %s\n' "$marker" >&2
      exit 2
    }
  done
done

/usr/local/bin/sbatch --test-only cluster/cril/tier2_independent.sbatch
/usr/local/bin/sbatch --test-only cluster/cril/tier2_shared.sbatch
if [[ "$MODE" == "--dry-run" ]]; then
  printf 'independent: /usr/local/bin/sbatch --parsable cluster/cril/tier2_independent.sbatch\n'
  printf 'shared:      /usr/local/bin/sbatch --parsable cluster/cril/tier2_shared.sbatch\n'
  printf 'combine:     submit afterok on the independent array\n'
  printf 'mode=dry-run scientific_training=full cells=153\n'
  exit 0
fi

INDEPENDENT_ID="$(/usr/local/bin/sbatch --parsable cluster/cril/tier2_independent.sbatch)"
SHARED_ID="$(/usr/local/bin/sbatch --parsable cluster/cril/tier2_shared.sbatch)"
COMBINE_ID="$(/usr/local/bin/sbatch --parsable --dependency="afterok:$INDEPENDENT_ID" cluster/cril/tier2_combine.sbatch)"
printf 'independent=%s\nshared=%s\ncombine=%s\n' "$INDEPENDENT_ID" "$SHARED_ID" "$COMBINE_ID"
