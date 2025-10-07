# Anonymous Submission: Ranking Model Training with PPO

This repository contains a minimal working implementation for training ranking models with PPO (Proximal Policy Optimization) experiments.

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Train a simple ranking model:**
   ```bash
   python train_ranking_model.py --dataset chaosnli --use_cuda --num_epochs 10
   ```

3. **Train with PPO:**
   ```bash
   python train_ppo_ranking.py --use_cuda --num_epochs 10
   ```

## Key Features

- **Multiple Training Approaches**: Supervised learning and PPO for ranking tasks
- **Flexible Data Support**: Easy data loading for ranking tasks
- **Model Compatibility**: Supports BERT, RoBERTa, and large language models with quantization
- **Comprehensive Evaluation**: NDCG, MAP, MRR, and other ranking metrics

## Files Included

- `ranking_models.py` - Core ranking model implementation
- `ranking_evaluator.py` - Evaluation metrics for ranking models
- `train_ranking_model.py` - Full training script with multiple datasets
- `train_ppo_ranking.py` - PPO-based training script
- `requirements.txt` - Required Python packages

## Supported Datasets

- ChaosNLI
- DS Critique Bank
- e-SNLI
- Stack Exchange
- Daily Dialog

## Technical Details

- **Framework**: PyTorch with Transformers
- **Models**: BERT, RoBERTa, DistilBERT, and large language models
- **Quantization**: 4-bit and 8-bit quantization support for large models
- **Loss Functions**: MSE, ListNet, RankNet, ListMLE, ApproxNDCG
- **Evaluation**: Comprehensive ranking metrics with batched inference

## Anonymous Submission

This code is submitted anonymously for review. All implementation details and technical contributions are provided for evaluation purposes.