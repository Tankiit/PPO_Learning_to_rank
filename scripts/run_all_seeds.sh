#!/bin/bash
# =============================================================================
# run_all_seeds.sh — Run paper experiments across 5 seeds
#
# Co-authors: run this script on your GPU machine.
# Adjust DEVICE and OUTPUT_ROOT as needed.
#
# This reproduces:
#   Table 2 (loss comparison) — ~15h total on V100
#   Table 1 (model comparison) — ~40h total on V100
#   Table 3 (data creation ablation) — ~5h total on V100
# =============================================================================

set -e

SEEDS="42 123 456 789 1024"
OUTPUT_ROOT="results/multi_seed"
DATASET="multinli"
EPOCHS=50

echo "================================================"
echo "Multi-seed experiment runner"
echo "Seeds: $SEEDS"
echo "Output: $OUTPUT_ROOT"
echo "================================================"

# -----------------------------------------------------------------
# EXPERIMENT 1: Loss Function Comparison (Table 2)
# Uses RoBERTa-base across 5 loss functions × 5 seeds = 25 runs
# Each run: ~35 min on V100 → total ~15h
# -----------------------------------------------------------------
echo ""
echo "=== EXPERIMENT 1: Loss Function Comparison ==="
echo "Model: roberta-base | Dataset: $DATASET | 25 runs"
echo ""

for LOSS in mse binary ranknet approxndcg listnet; do
    for SEED in $SEEDS; do
        OUT="${OUTPUT_ROOT}/loss_comparison/${LOSS}/seed_${SEED}"
        if [ -f "${OUT}/results.json" ]; then
            echo "SKIP: ${LOSS}/seed_${SEED} (already done)"
            continue
        fi
        echo "RUN: ${LOSS}/seed_${SEED}"
        python -m src.train \
            --model roberta-base \
            --loss $LOSS \
            --dataset $DATASET \
            --epochs $EPOCHS \
            --seed $SEED \
            --output_dir "$OUT" \
            --batch_size 16 \
            --lr 2e-5
    done
done

# -----------------------------------------------------------------
# EXPERIMENT 2: Model Comparison (Table 1)
# Uses ListNet across 5+ models × 5 seeds
# Encoders: ~3h each → 45h total
# Decoders: ~10h each → if needed, run separately
# -----------------------------------------------------------------
echo ""
echo "=== EXPERIMENT 2: Model Comparison (Encoders) ==="
echo "Loss: listnet | Dataset: $DATASET"
echo ""

for MODEL in bert-base-uncased roberta-base microsoft/deberta-v3-base; do
    MODEL_SHORT=$(echo $MODEL | sed 's/.*\///')
    for SEED in $SEEDS; do
        OUT="${OUTPUT_ROOT}/model_comparison/${MODEL_SHORT}/seed_${SEED}"
        if [ -f "${OUT}/results.json" ]; then
            echo "SKIP: ${MODEL_SHORT}/seed_${SEED} (already done)"
            continue
        fi
        echo "RUN: ${MODEL_SHORT}/seed_${SEED}"
        python -m src.train \
            --model $MODEL \
            --loss listnet \
            --dataset $DATASET \
            --epochs $EPOCHS \
            --seed $SEED \
            --output_dir "$OUT" \
            --batch_size 16 \
            --lr 2e-5
    done
done

# Decoder models (optional — run if GPU time allows)
# Uncomment to include:
# echo ""
# echo "=== EXPERIMENT 2b: Model Comparison (Decoders, 4-bit) ==="
# for MODEL in microsoft/phi-2 mistralai/Mistral-7B-v0.1; do
#     MODEL_SHORT=$(echo $MODEL | sed 's/.*\///')
#     for SEED in $SEEDS; do
#         OUT="${OUTPUT_ROOT}/model_comparison/${MODEL_SHORT}/seed_${SEED}"
#         python -m src.train \
#             --model $MODEL \
#             --loss listnet \
#             --dataset $DATASET \
#             --epochs $EPOCHS \
#             --seed $SEED \
#             --output_dir "$OUT" \
#             --batch_size 4 \
#             --lr 1e-5 \
#             --quantize_4bit
#     done
# done

# -----------------------------------------------------------------
# EXPERIMENT 3: Data Creation Ablation (Table 3)
# Uses RoBERTa-base + ListNet across 3 methods × 5 seeds = 15 runs
# Each run: ~35 min → total ~9h
# -----------------------------------------------------------------
echo ""
echo "=== EXPERIMENT 3: Data Creation Ablation ==="
echo "Model: roberta-base | Loss: listnet"
echo ""

for METHOD in heuristic graded_delta overlap; do
    for SEED in $SEEDS; do
        OUT="${OUTPUT_ROOT}/data_ablation/${METHOD}/seed_${SEED}"
        if [ -f "${OUT}/results.json" ]; then
            echo "SKIP: ${METHOD}/seed_${SEED} (already done)"
            continue
        fi
        echo "RUN: ${METHOD}/seed_${SEED}"
        python -m src.train \
            --model roberta-base \
            --loss listnet \
            --dataset $DATASET \
            --data_method $METHOD \
            --epochs $EPOCHS \
            --seed $SEED \
            --output_dir "$OUT" \
            --batch_size 16 \
            --lr 2e-5
    done
done

echo ""
echo "================================================"
echo "All experiments complete!"
echo "Run: python -m scripts.aggregate_results --input $OUTPUT_ROOT"
echo "================================================"
