#!/usr/bin/env bash
# The question-disjoint split matrix: full main-table arms, seeds 42/123/777.
#
# Held-out evaluation is 54 questions x 12 explanations = 648 candidates with no
# singletons, against a random-ranking NDCG@5 baseline of 0.6687 - versus the
# published split's 52 rankable groups against a baseline of 0.9260. This is the
# axis that can actually separate arms.
#
# Stages are sequential so one failure does not strand the rest, and every stage
# is resumable: a completed output directory is skipped, and one whose recorded
# seed disagrees with the request is refused rather than silently reused.
set -u
LOG_DIR="${1:?usage: run_qid_matrix.sh <log-dir>}"
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

stage shared_qid modal run modal_shared_ablation.py \
    --seeds "$SEEDS" --split qid
stage indep_listnet_qid modal run modal_independent_backbones.py \
    --loss listnet --seeds "$SEEDS" --split qid
stage indep_mse_qid modal run modal_independent_backbones.py \
    --loss mse --seeds "$SEEDS" --split qid

echo "=== [$(date -u +%H:%M:%S)] qid split matrix finished"
