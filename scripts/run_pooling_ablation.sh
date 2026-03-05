#!/bin/bash
# =============================================================================
# run_pooling_ablation.sh — Pooling strategy comparison
#
# Produces one row per strategy for an ablation table in the appendix.
# Uses RoBERTa-base + ListNet (best config) to isolate pooling effect.
#
# Expected results (based on literature):
#   attention > cls ≈ mean > max for discriminative ranking tasks
#
# Total time: ~12h on V100 (4 strategies × 5 seeds × ~35 min each)
# =============================================================================

set -e

SEEDS="42 123 456 789 1024"
OUTPUT_ROOT="results/multi_seed/pooling_ablation"
MODEL="roberta-base"
LOSS="listnet"
DATASET="multinli"
EPOCHS=50

echo "================================================"
echo "Pooling Ablation: $MODEL + $LOSS"
echo "Strategies: mean, cls, max, attention"
echo "Seeds: $SEEDS"
echo "================================================"

for POOLING in mean cls max attention; do
    for SEED in $SEEDS; do
        OUT="${OUTPUT_ROOT}/${POOLING}/seed_${SEED}"
        if [ -f "${OUT}/results.json" ]; then
            echo "SKIP: ${POOLING}/seed_${SEED} (already done)"
            continue
        fi
        echo "RUN: pooling=${POOLING} seed=${SEED}"
        python -m src.train \
            --model $MODEL \
            --loss $LOSS \
            --pooling $POOLING \
            --dataset $DATASET \
            --epochs $EPOCHS \
            --seed $SEED \
            --output_dir "$OUT" \
            --batch_size 16 \
            --lr 2e-5
    done
done

echo ""
echo "Done! Aggregate with:"
echo "  python -m scripts.aggregate_results --input $OUTPUT_ROOT"
