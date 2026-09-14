#!/usr/bin/env bash
# Tier 2: the replication the paper cannot ship without.
#
# Two axes, run as separate stages so a failure in one does not strand the rest.
# Every stage is resumable: each Modal function returns early on a completed
# output directory, and refuses one whose recorded seed disagrees with the
# request, so re-running this script after an interruption costs nothing.
#
#   published split x {123, 777}   the missing replication of the released runs
#   qid split      x {42,123,777}  the powered in-domain evaluation
#
set -u
LOG_DIR="${1:?usage: run_tier2.sh <log-dir>}"
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

# Published split: seeds 123 and 777. The shared arms for these seeds are
# launched separately and may already be running; this covers the independent
# ensembles, both losses.
stage indep_listnet_published modal run modal_independent_backbones.py \
    --loss listnet --seeds "123,777" --split published
stage indep_mse_published modal run modal_independent_backbones.py \
    --loss mse --seeds "123,777" --split published

# Question-disjoint split: all three seeds, full main-table matrix.
stage shared_qid modal run modal_shared_ablation.py \
    --seeds "42,123,777" --split qid
stage indep_listnet_qid modal run modal_independent_backbones.py \
    --loss listnet --seeds "42,123,777" --split qid
stage indep_mse_qid modal run modal_independent_backbones.py \
    --loss mse --seeds "42,123,777" --split qid

echo "=== [$(date -u +%H:%M:%S)] Tier 2 chain finished"
