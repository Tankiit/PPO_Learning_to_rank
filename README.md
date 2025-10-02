# PPO Learning to Rank

This repository contains a comprehensive PPO (Proximal Policy Optimization) implementation for learning to rank tasks, specifically focused on explanation quality ranking for natural language inference datasets.

## Overview

The project implements PPO-based reward learning for ranking explanations across multiple datasets in natural language inference tasks. It includes dataset creation, model training, and evaluation components.

## Key Components

### 1. Dataset Creation (`create_dataset.py`)
- **ComprehensiveDatasetBuilder**: Creates cross-source explanation ranking datasets in HuggingFace-friendly format
- Supports multiple NLI datasets: e-SNLI, Alpha-NLI, Delta-NLI, WinoWhy, Sens-Making, DS-Critique
- Generates quality-ranked explanation candidates from excellent (4) to nonsense (0)
- Saves normalized per-source datasets and merged comprehensive dataset

### 2. Model Evaluation (`evaluate_model.py`)
- Comprehensive evaluation framework for ranking models
- Evaluates explanation quality ranking across multiple metrics
- Supports visualizations and detailed reporting
- Integration with PPO training pipeline

### 3. Dataset Structure
- **Query-level format**: `{premise, hypothesis, label, gold_explanation, source, query_text, query_id}`
- **Ranking format**: `{query_id, source, premise, hypothesis, label, query_text, candidate, quality_score, generation_method}`

## Quality Levels

The dataset includes 5 quality levels for explanations:
- **4 (Excellent)**: Gold/high-quality explanations with detailed reasoning
- **3 (Good)**: Correct but brief explanations  
- **2 (Fair)**: Simple, basic explanations
- **1 (Poor)**: Incorrect explanations with wrong labels
- **0 (Nonsense)**: Random, irrelevant explanations

## Usage

### Creating Datasets
```python
from create_dataset import ComprehensiveDatasetBuilder

config = {
    "max_samples_per_source": 10000,
    "samples_per_query": 5,
    "output_long_table_name": "comprehensive_ranking_dataset"
}

builder = ComprehensiveDatasetBuilder(config)
dataset = builder.create_comprehensive_dataset(
    sources=["e-snli", "delta-nli", "winowhy", "ds-critique"],
    output_dir="data"
)
```

### Evaluating Models
```python
from evaluate_model import ModelEvaluator

evaluator = ModelEvaluator()
results = evaluator.evaluate_model(model_path, dataset_path)
```

## Features

- **Multi-source support**: Aggregates data from multiple NLI datasets
- **Quality ranking**: Implements explanation quality scoring from excellent to nonsense
- **Configurable generation**: Supports both template-based and model-generated explanations
- **Comprehensive evaluation**: Multiple ranking metrics and visualizations
- **HuggingFace integration**: Native compatibility with HF datasets and models

## Dataset Sources

- **e-SNLI**: Entailment-based explanation dataset
- **Alpha-NLI**: Commonsense reasoning with explanations
- **Delta-NLI**: Defeasible reasoning dataset
- **WinoWhy**: Winograd schema with explanations
- **Sens-Making**: Sentence making tasks with rationales
- **DS-Critique**: Synthetic data science critique bank style data

## Configuration

The system supports extensive configuration for:
- Dataset sampling parameters
- Quality generation strategies
- Model selection for generation
- Output formatting and organization

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- Datasets
- NumPy
- tqdm

## License

This project is licensed under the MIT License.
