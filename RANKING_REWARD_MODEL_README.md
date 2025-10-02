# Ranking Reward Model for Explanation Quality

Complete pipeline for training and evaluating ranking-based reward models on explanation quality assessment.

## Overview

This implementation provides a comprehensive framework for training reward models that can rank explanations by quality, using the Digital Socrates Critique Bank format and your comprehensive explanation dataset.

## Components

### 1. Data Loading (`ds_critique_loader.py`)

Loads and preprocesses DS Critique Bank data or synthetic data from your dataset.

**Features:**
- Automatic fallback to local synthetic data
- Converts between formats (candidate-level → query-level)
- Creates pairwise comparisons for preference learning
- Creates regression data for direct score prediction

**Usage:**
```python
from ds_critique_loader import DSCritiqueBankLoader

loader = DSCritiqueBankLoader()
dataset = loader.load_dataset()

# Get ranking format data
ranking_data = loader.convert_to_ranking_format('train')

# Get regression format
regression_data = loader.create_regression_data(ranking_data)

# Get pairwise comparisons
pairwise_data = loader.create_pairwise_comparisons(ranking_data)
```

**Output Format:**
```python
# Ranking format
{
    'query': "Question: Why does ice float? Explain why the answer is correct.",
    'explanations': [exp1, exp2, exp3, ...],
    'scores': [5, 4, 3, 2, 1],  # 1-5 scale
    'num_candidates': 5
}

# Regression format
{
    'query': "Question: ...",
    'explanation': "Ice is less dense...",
    'score': 4,  # Raw score 1-5
    'normalized_score': 0.75  # Normalized to [0, 1]
}

# Pairwise format
{
    'query': "Question: ...",
    'chosen': "Ice is less dense than water...",  # Higher score
    'rejected': "Ice is lighter.",  # Lower score
    'chosen_score': 4,
    'rejected_score': 2,
    'score_diff': 2
}
```

### 2. Ranking Models (`ranking_models.py`)

Reward models for ranking explanations by quality.

**Models:**
- `RankingRewardModel`: Base model with regression or ranking output
- `ListwiseRankingModel`: Extended model with listwise ranking loss
- `SimpleMockRankingModel`: Mock model for testing

**Architecture:**
```
Input: Query [SEP] Explanation
  ↓
BERT Encoder
  ↓
Pooler ([CLS] token)
  ↓
Projection Head (Hidden → Hidden/2 → 1)
  ↓
Sigmoid (for regression) or Identity (for ranking)
  ↓
Output: Quality Score [0, 1]
```

**Usage:**
```python
from ranking_models import RankingRewardModel

# Initialize model
model = RankingRewardModel(
    base_model="bert-base-uncased",
    output_mode="regression",  # or "ranking"
    dropout=0.1
)

# Rank explanations
query = "Why does ice float?"
explanations = [
    "Ice is lighter",
    "Ice is less dense than water",
    "Crystal structure makes it less dense"
]

scores = model.rank_explanations(query, explanations)
# Returns: [0.23, 0.76, 0.89]
```

### 3. Evaluation (`ranking_evaluator.py`)

Comprehensive evaluation metrics for ranking performance.

**Metrics:**
- **NDCG@k**: Normalized Discounted Cumulative Gain at ranks 1, 3, 5
- **MAP**: Mean Average Precision
- **MRR**: Mean Reciprocal Rank
- **Kendall's Tau**: Rank correlation coefficient
- **Spearman's ρ**: Rank correlation coefficient

**Usage:**
```python
from ranking_evaluator import RankingEvaluator

evaluator = RankingEvaluator(
    metrics=['ndcg', 'map', 'mrr', 'kendall_tau', 'spearman']
)

results = evaluator.evaluate(model, test_data)
evaluator.print_results(results)
```

**Output:**
```
============================================================
RANKING EVALUATION RESULTS
============================================================
ndcg@1              : 0.7234 ± 0.0123
ndcg@3              : 0.7891 ± 0.0156
ndcg@5              : 0.8234 ± 0.0142
map                 : 0.7654 ± 0.0178
mrr                 : 0.7845 ± 0.0134
kendall_tau         : 0.6789 ± 0.0234
spearman            : 0.7012 ± 0.0198
============================================================
```

### 4. Training (`train_ranking_model.py`)

Complete training pipeline for ranking reward models.

**Features:**
- MSE loss for regression
- Optional ranking loss for pairwise comparisons
- Gradient clipping and warmup
- Best model checkpointing
- Optional Weights & Biases logging

**Usage:**
```bash
# Basic training
python train_ranking_model.py \
    --base_model bert-base-uncased \
    --batch_size 16 \
    --learning_rate 2e-5 \
    --num_epochs 3 \
    --output_dir models/ranking_reward

# With GPU and wandb
python train_ranking_model.py \
    --base_model bert-base-uncased \
    --batch_size 32 \
    --learning_rate 2e-5 \
    --num_epochs 5 \
    --use_cuda \
    --use_wandb \
    --output_dir models/ranking_reward_v2
```

**Command-line Arguments:**
```
--base_model         Base transformer model (default: bert-base-uncased)
--batch_size         Batch size for training (default: 16)
--learning_rate      Learning rate (default: 2e-5)
--num_epochs         Number of training epochs (default: 3)
--dropout            Dropout rate (default: 0.1)
--output_dir         Directory to save models (default: models/ranking_reward)
--use_cuda           Use CUDA if available
--use_wandb          Use Weights & Biases for logging
```

## Quick Start

### 1. Install Dependencies
```bash
pip install torch transformers datasets scikit-learn scipy numpy tqdm
pip install wandb  # Optional, for experiment tracking
```

### 2. Prepare Data
Your dataset should already be created by `create_dataset.py`. The loader will automatically use it.

### 3. Train a Model
```bash
python train_ranking_model.py \
    --base_model bert-base-uncased \
    --batch_size 16 \
    --num_epochs 3 \
    --use_cuda \
    --output_dir models/my_ranking_model
```

### 4. Evaluate
```python
from ranking_models import RankingRewardModel
from ranking_evaluator import RankingEvaluator
from ds_critique_loader import DSCritiqueBankLoader
import torch

# Load model
model = RankingRewardModel(base_model="bert-base-uncased")
checkpoint = torch.load('models/my_ranking_model/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])

# Load test data
loader = DSCritiqueBankLoader()
test_data = loader.convert_to_ranking_format('test')

# Evaluate
evaluator = RankingEvaluator()
results = evaluator.evaluate(model, test_data)
evaluator.print_results(results)
```

## Dataset Statistics

Using the comprehensive ranking dataset:

**DS Critique Bank Subset:**
- Train: 2,331 queries → 11,655 candidates
- Validation: 666 queries → 3,330 candidates
- Test: 333 queries → 1,665 candidates

**Quality Distribution:**
Each query has 5 candidates with quality scores 0-4 (converted to 1-5):
- Score 5 (Excellent): 20%
- Score 4 (Good): 20%
- Score 3 (Fair): 20%
- Score 2 (Poor): 20%
- Score 1 (Nonsense): 20%

## Expected Performance

**Baseline (Random):**
- NDCG@5: ~0.65
- Spearman: ~0.0

**Length-based Heuristic:**
- NDCG@5: ~0.72
- Spearman: ~0.45

**Trained BERT Model:**
- NDCG@5: 0.80-0.85
- Spearman: 0.65-0.75
- MAP: 0.75-0.80

## Advanced Usage

### Custom Loss Functions

```python
from ranking_models import RankingRewardModel

model = RankingRewardModel(
    base_model="bert-base-uncased",
    output_mode="ranking"
)

# Use ranking loss instead of MSE
scores = model(input_ids, attention_mask)
labels = torch.tensor([5, 4, 3, 2, 1], dtype=torch.float)

ranking_loss = model.compute_ranking_loss(scores, labels, margin=1.0)
```

### Listwise Ranking

```python
from ranking_models import ListwiseRankingModel

model = ListwiseRankingModel(
    base_model="bert-base-uncased",
    output_mode="ranking"
)

# Listwise loss for joint ranking
scores = model(input_ids, attention_mask)
labels = torch.tensor([5, 4, 3, 2, 1], dtype=torch.float)

listwise_loss = model.compute_listwise_loss(scores, labels, temperature=1.0)
```

### Binary vs Ranking Comparison

```python
from ranking_evaluator import RankingEvaluator

# Compare two models
binary_model = load_binary_model()
ranking_model = load_ranking_model()

evaluator = RankingEvaluator()
comparison = evaluator.evaluate_binary_vs_ranking(
    binary_model,
    ranking_model,
    test_data
)

print("Improvements:", comparison['improvements'])
```

## Integration with PPO

Once trained, use the ranking reward model in PPO:

```python
from ranking_models import RankingRewardModel
import torch

# Load trained reward model
reward_model = RankingRewardModel(base_model="bert-base-uncased")
checkpoint = torch.load('models/ranking_reward/best_model.pt')
reward_model.load_state_dict(checkpoint['model_state_dict'])
reward_model.eval()

# Use in PPO reward function
def compute_reward(query, explanation):
    scores = reward_model.rank_explanations(query, [explanation])
    return scores[0]  # Return score for single explanation

# Or batch scoring
def compute_rewards_batch(queries, explanations):
    rewards = []
    for query, explanation in zip(queries, explanations):
        score = reward_model.rank_explanations(query, [explanation])[0]
        rewards.append(score)
    return rewards
```

## File Structure

```
.
├── ds_critique_loader.py      # Data loading and preprocessing
├── ranking_models.py           # Reward model architectures
├── ranking_evaluator.py        # Evaluation metrics
├── train_ranking_model.py      # Training script
├── models/                     # Saved models
│   └── ranking_reward/
│       ├── best_model.pt       # Best checkpoint
│       └── final_model.pt      # Final model
└── data/                       # Dataset (created by create_dataset.py)
    └── processed/
        └── comprehensive_ranking_dataset/
```

## Citation

If you use this ranking reward model framework, please cite:

```bibtex
@misc{ranking_reward_model,
  title={Ranking Reward Models for Explanation Quality Assessment},
  author={Your Name},
  year={2025},
  howpublished={\url{https://github.com/your-repo}}
}
```

## Troubleshooting

**Issue:** CUDA out of memory
**Solution:** Reduce batch size or use gradient accumulation

**Issue:** All scores are the same
**Solution:** Check that quality scores vary in your data; use data with diverse quality levels

**Issue:** Poor ranking performance
**Solution:** Try longer training, larger model, or add more training data

**Issue:** Slow evaluation
**Solution:** Use batch processing in rank_explanations() or reduce test set size

## Next Steps

1. ✅ Train baseline BERT model
2. ✅ Evaluate on DS Critique Bank subset
3. 🔄 Fine-tune on domain-specific data
4. 🔄 Integrate with PPO training
5. 🔄 Compare with Digital Socrates models
6. 🔄 Experiment with larger models (RoBERTa, DeBERTa)
7. 🔄 Add contrastive learning objectives
