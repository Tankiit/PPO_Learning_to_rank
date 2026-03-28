#!/bin/bash
# =============================================================================
# test_deberta_fix.sh — Quick DeBERTa-v3 sanity check
#
# Runs 5 epochs with the NaN fixes applied. Should see:
#   - Loss: finite number (NOT nan)
#   - NDCG@5: < 1.0 (NOT degenerate 1.0)
#   - Spearman: > 0.0 (NOT zero)
#
# If you still see NaN after this, try:
#   1. Lower LR further: --learning_rate 5e-6
#   2. Smaller batch: --batch_size 8
#   3. Add warmup: already 500 steps by default
#
# Usage:
#   bash test_deberta_fix.sh
# =============================================================================

set -e

MODEL="microsoft/deberta-v3-base"
DATASET="ds_critique"  # change to your dataset
LOSS="listnet"
EPOCHS=5
OUTPUT="results/deberta_test"
LR="1e-5"  # lower than default 2e-5

echo "================================================"
echo "DeBERTa-v3 NaN Fix Test"
echo "Model:   $MODEL"
echo "Loss:    $LOSS"
echo "LR:      $LR"
echo "Epochs:  $EPOCHS"
echo "Dataset: $DATASET"
echo "================================================"
echo ""

python train_ranking_model.py \
    --base_model "$MODEL" \
    --dataset "$DATASET" \
    --loss_function "$LOSS" \
    --num_epochs $EPOCHS \
    --batch_size 16 \
    --learning_rate $LR \
    --output_dir "$OUTPUT" \
    --use_cuda \
    --val_frequency 1

echo ""
echo "================================================"
echo "Check results above:"
echo "  ✓ Loss should be a finite number (e.g., 0.5-2.0)"
echo "  ✓ NDCG@5 should be < 1.0 (e.g., 0.6-0.9)"
echo "  ✓ Spearman should be > 0.0"
echo "  ✗ If still NaN → try --learning_rate 5e-6"
echo "================================================"
