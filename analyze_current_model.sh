#!/bin/bash

# Helper script to analyze your existing trained model
# This uses the correct paths for your setup

MAIN_DIR="/Users/tanmoy/research/PPO_learning_to_rank/PPO_Learning_to_rank"
MODEL_PATH="$MAIN_DIR/models/ranking_reward/best_model.pt"

echo "=========================================="
echo "Analyzing Your Trained Model"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo ""

# Check if model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "ERROR: Model not found at $MODEL_PATH"
    exit 1
fi

echo "1. Visualizing Score Distribution..."
python visualize_score_distribution.py \
    --model_path "$MODEL_PATH" \
    --output_dir "$MAIN_DIR/score_analysis" \
    --device cpu

echo ""
echo "=========================================="
echo "Analysis Complete!"
echo "=========================================="
echo "Results saved to: $MAIN_DIR/score_analysis/"
echo ""
echo "Next steps:"
echo "  1. Check score_analysis/score_distribution_analysis.png"
echo "  2. Check score_analysis/per_query_detailed.png"
echo ""
echo "To compare with mock PPO:"
echo "  ./compare_with_mock_ppo.sh"
