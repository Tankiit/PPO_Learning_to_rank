#!/usr/bin/env bash
# Epistemic experiments for explanation-quality ranking.
# See docs/epistemic_experiments.md for what each stage is testing.
set -euo pipefail

CONFIG=${CONFIG:-configs/arr/epistemic_local.yaml}
RAW_TRAIN=${RAW_TRAIN:-data/arr/DSCB-train-crowd-anno.jsonl}
RAW_VALID=${RAW_VALID:-data/arr/DSCB-dev-crowd-anno.jsonl}
TRAIN=${TRAIN:-data/arr/ds_critique_external_test.jsonl}
VALID=${VALID:-data/arr/ds_critique_external_dev.jsonl}
RUNS=${RUNS:-runs/epistemic}
SEED=${SEED:-42}
TRL_API=${TRL_API:-modern}
EPOCHS=${EPOCHS:-5}
BATCH_SIZE=${BATCH_SIZE:-1}
DEVICE=${DEVICE:-auto}
CLI="python -m src.arr.compression_cli"

mkdir -p "$RUNS"

# Prepare the human-scored ranking groups when only raw DS-Critique sources
# are present. Non-annotated and expert files are never silently mixed into
# the crowd-supervised target.
if [[ ! -f "$TRAIN" ]]; then
  $CLI prepare-data --config "$CONFIG" --dataset ds-main \
    --ds-main-source "$RAW_TRAIN" --output-dir data/arr
fi
if [[ ! -f "$VALID" ]]; then
  $CLI prepare-data --config "$CONFIG" --dataset ds-dev \
    --ds-dev-source "$RAW_VALID" --output-dir data/arr
fi

# ---------------------------------------------------------------------------
# Stage 0 — equivalence check. lambda_div=0.0 must reproduce the existing
# objective exactly. If this loss curve does not match a previous run, stop:
# something is miswired and nothing downstream is interpretable.
# ---------------------------------------------------------------------------
if [[ ! -d "$RUNS/rm_lam0" ]]; then
  $CLI train-epistemic-judge --config "$CONFIG" \
    --train-data "$TRAIN" --validation-data "$VALID" \
    --lambda-div 0.0 --seed "$SEED" \
    --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" --device "$DEVICE" \
    --output-dir "$RUNS/rm_lam0"
fi

# ---------------------------------------------------------------------------
# Stage 1 — diagnostics gate. Exits non-zero when the epistemic signal is not
# supported.
#
# train_judge keeps two checkpoints, "best" and "last" -- there are no epoch_*
# directories to re-score. It does write validation_predictions_epoch_N.jsonl
# at the run root on every epoch, which is exactly the per-epoch series D3
# needs, so feed those in directly rather than re-scoring anything.
#
# D3 needs at least three epochs to call a decay monotonic. With fewer, it
# reports the widths and abstains.
# ---------------------------------------------------------------------------
$CLI evaluate-epistemic --config "$CONFIG" \
  --data "$VALID" --checkpoint "$RUNS/rm_lam0/best" \
  --output-dir "$RUNS/rm_lam0/best/eval" --seed "$SEED"

set +e
$CLI diagnose-epistemic --config "$CONFIG" \
  --data "$VALID" \
  --predictions "$RUNS/rm_lam0/best/eval/predictions.jsonl" \
  --epoch-predictions "$RUNS"/rm_lam0/validation_predictions_epoch_*.jsonl \
  --output-dir "$RUNS/diagnostics"
GATE=$?
set -e

if [[ $GATE -ne 0 ]]; then
  echo
  echo "Diagnostics say credal width is not carrying an epistemic signal."
  echo "Read $RUNS/diagnostics/epistemic_diagnostics.json, then run Stage 2."
  echo "Do not skip to Stage 3: the penalty would be penalising optimiser noise."
fi

# ---------------------------------------------------------------------------
# Stage 2 — decorrelation sweep. Only needed if Stage 1 failed.
# ---------------------------------------------------------------------------
if [[ $GATE -ne 0 ]]; then
  for LAM in 0.01 0.1 1.0; do
    $CLI train-epistemic-judge --config "$CONFIG" \
      --train-data "$TRAIN" --validation-data "$VALID" \
      --lambda-div "$LAM" --seed "$SEED" \
      --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" --device "$DEVICE" \
      --output-dir "$RUNS/rm_lamdiv_$LAM"
  done
  echo "Re-run Stage 1 against the best sweep point before continuing."
  exit "$GATE"
fi

# ---------------------------------------------------------------------------
# Stage 3 — PPO arms. Three seeds each; kl_coef fixed by the config.
# ---------------------------------------------------------------------------
RM=${RM:-$RUNS/rm_lam0/best}
for ARM in none var credal; do
  for S in 0 1 2; do
    $CLI train-ppo-epistemic --config "$CONFIG" \
      --data "$TRAIN" --reward-checkpoint "$RM" \
      --penalty "$ARM" --trl-api "$TRL_API" --seed "$S" \
      --output-dir "$RUNS/ppo_${ARM}_seed${S}"
  done
done

echo "Done. The headline figure is credal/width_mean over PPO steps, per arm."
