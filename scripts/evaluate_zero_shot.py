"""
Zero-shot transfer evaluation.

Loads a model trained on NLI data and evaluates it on DS-Critique Bank
(or any other dataset) WITHOUT any fine-tuning.

This tests whether score compression is a property of the loss function
(generalizes across domains) or the data (NLI-specific).

Expected outcome:
    - MSE-trained model: low separation ratio on DS-Critique (~0.1)
    - ListNet-trained model: higher separation ratio (~0.5-0.7)
    - Gap confirms: compression is loss-level, not domain-specific
    - Absolute numbers will be lower than in-domain (expected)

Usage:
    python -m scripts.evaluate_zero_shot \
        --model_path results/multi_seed/loss_comparison/listnet/seed_42/best_model.pt \
        --eval_dataset ds_critique \
        --output_dir results/multi_seed/ds_critique/zero_shot_transfer/listnet
"""

import argparse
import json
import os
import sys

import torch
from transformers import AutoTokenizer

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.ranking_reward_model import RankingRewardModel
from src.evaluation.metrics import compute_ranking_metrics
from src.data.ds_critique_loader import load_ds_critique_ranking


def evaluate_zero_shot(model_path: str, eval_dataset: str, output_dir: str):
    """Run zero-shot evaluation."""
    
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    print(f"Loading model from {model_path}")
    
    # TODO: Adapt this to your actual model saving format
    # Option A: Full model checkpoint
    # model = torch.load(model_path, map_location=device)
    
    # Option B: State dict + config
    checkpoint = torch.load(model_path, map_location=device)
    model_config = checkpoint.get("config", {})
    model_name = model_config.get("model_name", "roberta-base")
    pooling = model_config.get("pooling", "attention")
    
    model = RankingRewardModel(
        model_name=model_name,
        pooling=pooling,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # ------------------------------------------------------------------
    # Load evaluation data
    # ------------------------------------------------------------------
    print(f"\nLoading evaluation dataset: {eval_dataset}")
    
    if eval_dataset == "ds_critique":
        _, val_examples = load_ds_critique_ranking()
    else:
        raise ValueError(f"Unknown eval dataset: {eval_dataset}")
    
    print(f"Evaluating on {len(val_examples)} ranking groups")
    
    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------
    all_true_scores = []
    all_pred_scores = []
    
    for example in val_examples:
        # Tokenize query + each candidate
        texts = [
            f"{example['query_text']} {cand}"
            for cand in example["candidates"]
        ]
        
        enc = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        ).to(device)
        
        with torch.no_grad():
            pred = model(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
            )
            pred_scores = pred.squeeze(-1).cpu().tolist()
        
        # Normalize true scores to [0, 1]
        true_scores = [s / 5.0 for s in example["scores"]]
        
        all_true_scores.append(true_scores)
        all_pred_scores.append(pred_scores)
    
    # ------------------------------------------------------------------
    # Compute metrics
    # ------------------------------------------------------------------
    metrics = compute_ranking_metrics(all_true_scores, all_pred_scores)
    
    # Add zero-shot specific info
    metrics["eval_type"] = "zero_shot_transfer"
    metrics["source_domain"] = "nli"
    metrics["target_domain"] = eval_dataset
    metrics["n_eval_groups"] = len(val_examples)
    
    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n{'='*50}")
    print(f"Zero-shot transfer results ({eval_dataset})")
    print(f"{'='*50}")
    print(f"  NDCG@3:           {metrics.get('ndcg@3', 'N/A'):.4f}")
    print(f"  NDCG@5:           {metrics.get('ndcg@5', 'N/A'):.4f}")
    print(f"  Spearman ρ:       {metrics.get('spearman', 'N/A'):.4f}")
    print(f"  Sep. Ratio:       {metrics.get('separation_ratio', 'N/A'):.4f}")
    print(f"  Score Range:      {metrics.get('score_range', 'N/A'):.4f}")
    print(f"\nResults saved to {results_path}")
    
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--eval_dataset", default="ds_critique")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    
    evaluate_zero_shot(args.model_path, args.eval_dataset, args.output_dir)
