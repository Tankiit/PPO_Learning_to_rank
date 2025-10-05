#!/bin/bash

# Train models with all ranking loss functions and compare results
# This will take several hours but give you complete comparison data

MAIN_DIR="/Users/tanmoy/research/PPO_learning_to_rank/PPO_Learning_to_rank"
EPOCHS=30
BATCH_SIZE=16

echo "=========================================="
echo "Training Models with All Loss Functions"
echo "=========================================="
echo "This will train 4 models:"
echo "  1. ListNet (best for ranking)"
echo "  2. RankNet (pairwise)"
echo "  3. ApproxNDCG (directly optimizes NDCG)"
echo " 4. MSE (baseline - already trained)"
echo ""
echo "Total time: ~6-8 hours (2 hours per model)"
echo "=========================================="
echo ""

# Train ListNet
echo "1/3: Training with ListNet loss..."
python train_ranking_model.py \
    --dataset ds_critique \
    --loss_function listnet \
    --num_epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate 2e-5 \
    --output_dir "$MAIN_DIR/models/ranking_listnet" \
    --num_workers 4

echo "ListNet training complete!"
echo ""

# Train RankNet
echo "2/3: Training with RankNet loss..."
python train_ranking_model.py \
    --dataset ds_critique \
    --loss_function ranknet \
    --num_epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate 2e-5 \
    --output_dir "$MAIN_DIR/models/ranking_ranknet" \
    --num_workers 4

echo "RankNet training complete!"
echo ""

# Train ApproxNDCG
echo "3/3: Training with ApproxNDCG loss..."
python train_ranking_model.py \
    --dataset ds_critique \
    --loss_function approxndcg \
    --num_epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate 2e-5 \
    --output_dir "$MAIN_DIR/models/ranking_approxndcg" \
    --num_workers 4

echo "ApproxNDCG training complete!"
echo ""

echo "=========================================="
echo "All Training Complete!"
echo "=========================================="
echo ""
echo "Models saved to:"
echo "  - $MAIN_DIR/models/ranking_reward (MSE baseline)"
echo "  - $MAIN_DIR/models/ranking_listnet"
echo "  - $MAIN_DIR/models/ranking_ranknet"
echo "  - $MAIN_DIR/models/ranking_approxndcg"
echo ""
echo "Next: Run comparison script"
echo "  python compare_all_models.py"
