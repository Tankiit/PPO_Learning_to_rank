#!/bin/bash

# Train ranking models across all datasets
# Usage: ./train_all_datasets.sh [loss_function]

LOSS_FUNCTION=${1:-listnet}

echo "=========================================="
echo "Multi-Dataset Training"
echo "=========================================="
echo "Loss function: $LOSS_FUNCTION"
echo ""

# Train on all datasets
python train_all_datasets.py \
    --datasets all \
    --loss_function $LOSS_FUNCTION \
    --base_model bert-base-uncased \
    --num_epochs 100 \
    --batch_size 16 \
    --learning_rate 2e-5 \
    --early_stopping_patience 10 \
    --output_dir models/multi_dataset_ranking \
    --save_every 5 \
    --eval_every 1

echo ""
echo "=========================================="
echo "Training Complete!"
echo "=========================================="
echo "Results saved to: models/multi_dataset_ranking/"
echo ""
echo "To train on specific datasets:"
echo "  python train_all_datasets.py --datasets chaosnli esnli"
echo ""
echo "To try different loss functions:"
echo "  ./train_all_datasets.sh ranknet"
echo "  ./train_all_datasets.sh approxndcg"
