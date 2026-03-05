#!/bin/bash
# =============================================================================
# run_ds_critique_experiments.sh
#
# Non-NLI experiment: DS-Critique Bank (science reasoning)
# Demonstrates score compression is loss-function-level, not NLI-specific.
#
# What this proves for reviewers:
#   - MSE still compresses scores on science reasoning data
#   - ListNet still preserves separation on science reasoning data
#   - The ranking paradigm generalizes beyond NLI
#
# INSTRUCTIONS FOR CO-AUTHORS:
#   1. Make sure you have the repo cloned and dependencies installed
#   2. Run: bash scripts/run_ds_critique_experiments.sh
#   3. When done, run: python -m scripts.aggregate_results \
#          --input results/multi_seed/ds_critique
#   4. Send me the results/ directory
#
# Expected time: ~20h on V100 (5 losses × 5 seeds × ~45 min each)
# =============================================================================

set -e

SEEDS="42 123 456 789 1024"
OUTPUT_ROOT="results/multi_seed/ds_critique"
MODEL="roberta-base"
DATASET="ds_critique"
EPOCHS=50

echo "================================================================"
echo "DS-Critique Bank Experiments (Non-NLI validation)"
echo "Model: $MODEL"
echo "Losses: mse, binary, ranknet, approxndcg, listnet"
echo "Seeds: $SEEDS"
echo "================================================================"

# -------------------------------------------------------
# Step 0: Download dataset (only needs to happen once)
# -------------------------------------------------------
echo ""
echo "[Step 0] Downloading DS-Critique Bank..."
python -c "from src.data.ds_critique_loader import load_ds_critique_ranking; load_ds_critique_ranking()" 2>&1 || {
    echo "ERROR: Failed to download dataset. Check your internet connection."
    echo "You can also manually download from https://huggingface.co/datasets/allenai/DS_Critique_Bank"
    exit 1
}
echo "Dataset ready."

# -------------------------------------------------------
# Step 1: Loss function comparison (the key experiment)
# -------------------------------------------------------
echo ""
echo "================================================================"
echo "[Step 1] Loss function comparison on DS-Critique"
echo "================================================================"

for LOSS in mse binary ranknet approxndcg listnet; do
    for SEED in $SEEDS; do
        OUT="${OUTPUT_ROOT}/loss_comparison/${LOSS}/seed_${SEED}"
        if [ -f "${OUT}/results.json" ]; then
            echo "SKIP: ${LOSS}/seed_${SEED} (already done)"
            continue
        fi
        echo ""
        echo "RUN: loss=${LOSS} seed=${SEED}"
        python -m src.train \
            --model $MODEL \
            --loss $LOSS \
            --dataset $DATASET \
            --epochs $EPOCHS \
            --seed $SEED \
            --output_dir "$OUT" \
            --batch_size 16 \
            --lr 2e-5 \
            --pooling attention
    done
done

# -------------------------------------------------------
# Step 2: Zero-shot transfer (apply NLI-trained model to DS-Critique)
# -------------------------------------------------------
echo ""
echo "================================================================"
echo "[Step 2] Zero-shot transfer: NLI-trained → DS-Critique"
echo "================================================================"
echo ""
echo "This evaluates NLI-trained models (from results/multi_seed/loss_comparison)"
echo "on DS-Critique data WITHOUT any DS-Critique training."
echo ""

# Only need one seed for zero-shot (it's deterministic evaluation)
for LOSS in mse listnet; do
    NLI_MODEL="results/multi_seed/loss_comparison/${LOSS}/seed_42/best_model.pt"
    OUT="${OUTPUT_ROOT}/zero_shot_transfer/${LOSS}"
    
    if [ -f "${OUT}/results.json" ]; then
        echo "SKIP: zero-shot ${LOSS} (already done)"
        continue
    fi
    
    if [ ! -f "$NLI_MODEL" ]; then
        echo "SKIP: zero-shot ${LOSS} (NLI model not found at ${NLI_MODEL})"
        echo "  → Run NLI experiments first, then re-run this script"
        continue
    fi
    
    echo "RUN: zero-shot transfer loss=${LOSS}"
    python -m scripts.evaluate_zero_shot \
        --model_path "$NLI_MODEL" \
        --eval_dataset $DATASET \
        --output_dir "$OUT"
done

# -------------------------------------------------------
# Summary
# -------------------------------------------------------
echo ""
echo "================================================================"
echo "DONE! Next steps:"
echo "================================================================"
echo ""
echo "1. Aggregate results:"
echo "   python -m scripts.aggregate_results --input ${OUTPUT_ROOT}"
echo ""
echo "2. Check key claim — score compression on DS-Critique:"
echo "   Look for Sep. Ratio in the results."
echo "   Expected: MSE ~0.05-0.15, ListNet ~0.80-0.95"
echo ""
echo "3. Send results/ directory to Tanmoy"
echo ""
