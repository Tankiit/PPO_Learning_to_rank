#!/bin/bash

set -euo pipefail

if [[ -r /etc/profile.d/z_modules.sh ]]; then
  source /etc/profile.d/z_modules.sh
fi
if [[ -z "${MODULEPATH:-}" ]]; then
  printf 'Jean-Zay MODULEPATH is not initialized on the submission frontend\n' >&2
  exit 2
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export ARR_PROJECT_ROOT="${ARR_PROJECT_ROOT:-$PROJECT_ROOT}"
SBATCH_ACCOUNT_ARGS=()
if [[ -n "${ARR_SLURM_ACCOUNT:-}" ]]; then
  SBATCH_ACCOUNT_ARGS=(--account="$ARR_SLURM_ACCOUNT")
fi
cd "$PROJECT_ROOT"
mkdir -p arr_runs/epistemic/slurm

submit() {
  local label="$1"
  shift
  local job_id
  job_id="$(sbatch --parsable "${SBATCH_ACCOUNT_ARGS[@]}" "$@")"
  printf 'SUBMITTED %-12s %s\n' "$label" "$job_id" >&2
  printf '%s' "$job_id"
}

PREFLIGHT="$(submit preflight cluster/jean_zay/epistemic_preflight.sbatch)"
TRAIN="$(submit train --array=0-3%4 --dependency="afterok:$PREFLIGHT" cluster/jean_zay/epistemic_scalar_train.sbatch)"
EVALUATE="$(submit evaluate --array=0-7%4 --dependency="afterok:$TRAIN" cluster/jean_zay/epistemic_scalar_eval.sbatch)"
AGGREGATE="$(submit aggregate --dependency="afterok:$EVALUATE" cluster/jean_zay/epistemic_aggregate.sbatch)"

STAMP="$(date +%Y%m%dT%H%M%S)"
MANIFEST="arr_runs/epistemic/slurm/submission-$STAMP.txt"
{
  printf 'preflight=%s\n' "$PREFLIGHT"
  printf 'train=%s\n' "$TRAIN"
  printf 'evaluate=%s\n' "$EVALUATE"
  printf 'aggregate=%s\n' "$AGGREGATE"
} > "$MANIFEST"

printf 'Submission manifest: %s\n' "$MANIFEST"
squeue -u "$USER" -o '%.18i|%.12P|%.24j|%.8T|%.10M|%.6D|%R'
