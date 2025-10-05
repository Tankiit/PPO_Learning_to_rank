#!/bin/bash

# Helper script to compare your model with mock PPO
# This uses the correct paths for your setup

MAIN_DIR="/Users/tanmoy/research/PPO_learning_to_rank/PPO_Learning_to_rank"
MODEL_PATH="$MAIN_DIR/models/ranking_reward/best_model.pt"

echo "=========================================="
echo "Comparing Supervised vs Mock PPO"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo ""

# Check if model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "ERROR: Model not found at $MODEL_PATH"
    exit 1
fi

echo "Running comparison with simulated PPO..."
echo "(No PPO training needed - using mock for demonstration)"
echo ""

python compare_supervised_vs_ppo.py \
    --supervised_model "$MODEL_PATH" \
    --use_mock_ppo \
    --output_dir "$MAIN_DIR/comparison_results" \
    --device cpu

echo ""
echo "=========================================="
echo "Comparison Complete!"
echo "=========================================="
echo "Results saved to: $MAIN_DIR/comparison_results/"
echo ""
echo "Check these files:"
echo "  - comparison_results/comparison_results.json"
echo "  - comparison_results/comparison_barplot.png"
echo "  - comparison_results/improvement_plot.png"
