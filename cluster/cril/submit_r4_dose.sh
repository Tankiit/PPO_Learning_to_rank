#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"
mkdir -p runs/r4_exposure/slurm

SCRIPTS=(r4_dose_shared.sbatch r4_dose_independent.sbatch r4_dose_combine.sbatch)
for SCRIPT in "${SCRIPTS[@]}"; do
  bash -n "cluster/cril/$SCRIPT"
done
for QID in 025 050 100; do
  for EXPOSURE in 000 050 100; do
    test -f "data/arr/r4_exposure/train_qid${QID}_exposure${EXPOSURE}.jsonl"
  done
done
test -f data/arr/r4_exposure/gpt4_qid_holdout.jsonl
test -f data/arr/ds_critique_qidsplit_dev.jsonl
test -f src/arr/tier2_score.py

MODE="${1:---dry-run}"
case "$MODE" in
  --dry-run)
    for SCRIPT in "${SCRIPTS[@]}"; do
      sbatch --test-only "cluster/cril/$SCRIPT"
    done
    ;;
  --submit)
    SHARED_JOB="$(sbatch --parsable cluster/cril/r4_dose_shared.sbatch)"
    INDEPENDENT_JOB="$(sbatch --parsable cluster/cril/r4_dose_independent.sbatch)"
    COMBINE_JOB="$(sbatch --parsable --dependency="afterok:$INDEPENDENT_JOB" cluster/cril/r4_dose_combine.sbatch)"
    printf 'shared=%s independent=%s combine_afterok=%s\n' \
      "$SHARED_JOB" "$INDEPENDENT_JOB" "$COMBINE_JOB"
    ;;
  *)
    printf 'Usage: %s [--dry-run|--submit]\n' "$0" >&2
    exit 2
    ;;
esac
