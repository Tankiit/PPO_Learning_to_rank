# Comprehensive Evaluation Results

This directory contains comprehensive evaluation results for the explanation generation model.

## Files Generated

### JSON Results
- `evaluation_results.json` - Complete evaluation results
- `generation_metrics.json` - BLEU, METEOR, ROUGE scores
- `ranking_metrics.json` - NDCG, MAP, correlation metrics
- `quality_analysis.json` - Quality scores by source, label, and overall
- `dataset_stats.json` - Dataset statistics

### Visualizations
- `generation_metrics.png` - Bar chart of generation metrics
- `ranking_metrics.png` - Bar chart of ranking metrics
- `quality_by_source.png` - Quality scores by source dataset
- `quality_distribution.png` - Distribution of quality scores
- `dataset_quality_distribution.png` - Ground truth quality distribution

## Metrics Explained

### Generation Metrics
- **BLEU**: Measures n-gram overlap with reference explanations (0-1, higher is better)
- **METEOR**: Considers synonyms and stemming (0-1, higher is better)
- **ROUGE**: Measures recall-oriented overlap (0-1, higher is better)

### Ranking Metrics
- **NDCG@k**: Normalized Discounted Cumulative Gain at rank k (0-1, higher is better)
- **MAP**: Mean Average Precision (0-1, higher is better)
- **Kendall's Tau**: Rank correlation (-1 to 1, higher is better)
- **Spearman**: Rank correlation (-1 to 1, higher is better)

### Quality Metrics
- **Mean/Median Quality**: Average explanation quality (0-1, higher is better)
- **Length Stats**: Word count statistics for generated explanations

## Usage

To run evaluation on your own model:

```python
from evaluate_model import evaluate_model_comprehensive

# Your model must have:
# - generate(query, max_new_tokens, do_sample) method
# - rank_explanations(query, explanations) method (optional)

results = evaluate_model_comprehensive(
    model=your_model,
    test_dataset_path='data/processed/comprehensive_ranking_dataset',
    output_dir='experiments/your_evaluation',
    config={
        'generation_metrics': ['bleu', 'meteor', 'rouge'],
        'ranking_metrics': ['ndcg', 'map', 'kendall_tau', 'spearman'],
        'ndcg_k_values': [1, 3, 5],
        'sample_size': None  # Use all data, or set to int for sampling
    }
)
```

## Mock Model Results

The current results are from a mock model that generates simple explanations.
Replace with your actual trained model for real evaluation.
