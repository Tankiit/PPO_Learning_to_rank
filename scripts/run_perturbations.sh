#!/usr/bin/env bash
# Controls 2 and 3: the perturbation suite, run as two separate axes.
#
# Only the two arms the controls need are trained - independent ListNet (five
# members) and the shared ListNet baseline - because the cross of both axes
# would be twelve cells and the axes answer independent questions. sigma=0 and
# fraction=1.0 are the Tier 2 qid runs and are not repaid here.
#
# Every stage is resumable: a completed output directory is skipped, and one
# whose recorded seed disagrees with the request is refused rather than reused.
set -u
LOG_DIR="${1:?usage: run_perturbations.sh <log-dir>}"
SEEDS="${2:-42,123,777}"
mkdir -p "$LOG_DIR"

stage () {
  local name="$1"; shift
  echo "=== [$(date -u +%H:%M:%S)] START $name"
  if "$@" > "$LOG_DIR/$name.log" 2>&1; then
    echo "=== [$(date -u +%H:%M:%S)] OK    $name"
  else
    echo "=== [$(date -u +%H:%M:%S)] FAIL  $name (see $LOG_DIR/$name.log)"
  fi
}

for CELL in qid_sigma0p05 qid_sigma0p1 qid_sigma0p2 qid_frac0p25 qid_frac0p5; do
  stage "indep_${CELL}" modal run modal_independent_backbones.py \
      --loss listnet --seeds "$SEEDS" --split "$CELL"
  stage "shared_${CELL}" modal run modal_shared_ablation.py \
      --seeds "$SEEDS" --split "$CELL" --losses listnet --arms baseline
done

echo "=== [$(date -u +%H:%M:%S)] perturbation suite finished"
