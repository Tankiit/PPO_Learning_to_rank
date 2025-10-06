#!/usr/bin/env python3
"""
Comprehensive ranking evaluation with detailed metrics.
Use this after training to get a full picture of model performance.
"""

import torch
import numpy as np
from ranking_models import RankingRewardModel
from ranking_evaluator import RankingEvaluator
import argparse
from tqdm import tqdm

def pairwise_accuracy(gold_scores, pred_scores):
    """Calculate pairwise ranking accuracy"""
    correct = 0
    total = 0

    for i in range(len(gold_scores)):
        for j in range(i+1, len(gold_scores)):
            if gold_scores[i] != gold_scores[j]:  # Skip ties
                total += 1
                # Check if predicted order matches gold order
                gold_prefers_i = gold_scores[i] > gold_scores[j]
                pred_prefers_i = pred_scores[i] > pred_scores[j]
                if gold_prefers_i == pred_prefers_i:
                    correct += 1

    return correct / total if total > 0 else 0.0

def detailed_ranking_evaluation(model_path, val_data, device='mps'):
    """
    Comprehensive evaluation with multiple metrics

    Args:
        model_path: Path to model checkpoint
        val_data: Validation data in ranking format
        device: Device to use
    """

    print("="*80)
    print("COMPREHENSIVE RANKING EVALUATION")
    print("="*80)

    # Load model
    print("\nLoading model...")
    checkpoint = torch.load(model_path, map_location=device)

    if 'config' in checkpoint:
        config = checkpoint['config']
    else:
        config = {
            'base_model': 'bert-base-uncased',
            'output_mode': 'regression',
            'dropout': 0.1
        }

    model = RankingRewardModel(
        base_model=config['base_model'],
        output_mode=config['output_mode'],
        dropout=config['dropout']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"Model loaded from {model_path}")
    print(f"Config: {config['base_model']}, dropout={config['dropout']}")

    # Standard metrics using evaluator
    print("\n" + "="*80)
    print("STANDARD RANKING METRICS")
    print("="*80)

    evaluator = RankingEvaluator()
    results = evaluator.evaluate(model, val_data, batch_size=32)

    print(f"\nNDCG Metrics:")
    print(f"  NDCG@1: {results.get('ndcg@1', 0):.4f}")
    print(f"  NDCG@3: {results.get('ndcg@3', 0):.4f}")
    print(f"  NDCG@5: {results.get('ndcg@5', 0):.4f}")

    print(f"\nCorrelation Metrics:")
    print(f"  Spearman: {results.get('spearman', 0):.4f}")
    print(f"  Kendall Tau: {results.get('kendall_tau', 0):.4f}")

    print(f"\nOther Metrics:")
    print(f"  MAP: {results.get('map', 0):.4f}")
    print(f"  MRR: {results.get('mrr', 0):.4f}")

    # Additional detailed metrics
    print("\n" + "="*80)
    print("ADDITIONAL METRICS")
    print("="*80)

    top1_correct = 0
    top3_correct = 0
    top5_correct = 0
    pairwise_accs = []
    score_errors = []

    with torch.no_grad():
        for example in tqdm(val_data, desc="Computing additional metrics"):
            query = example['query']
            explanations = example['explanations']
            gold_scores = np.array(example['scores'])

            if len(explanations) < 2:
                continue

            # Get predictions
            texts = [f"{query} {model.tokenizer.sep_token} {exp}" for exp in explanations]
            encoded = model.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors='pt'
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}

            pred_scores = model(
                input_ids=encoded['input_ids'],
                attention_mask=encoded['attention_mask']
            )
            pred_scores = pred_scores.cpu().numpy()

            # Top-K accuracy
            best_gold_idx = np.argmax(gold_scores)
            pred_ranking = np.argsort(-pred_scores)  # High to low

            if pred_ranking[0] == best_gold_idx:
                top1_correct += 1
            if best_gold_idx in pred_ranking[:min(3, len(pred_ranking))]:
                top3_correct += 1
            if best_gold_idx in pred_ranking[:min(5, len(pred_ranking))]:
                top5_correct += 1

            # Pairwise accuracy
            pairwise_acc = pairwise_accuracy(gold_scores, pred_scores)
            pairwise_accs.append(pairwise_acc)

            # Score prediction error
            errors = np.abs(gold_scores - pred_scores)
            score_errors.extend(errors.tolist())

    num_queries = len(val_data)

    print(f"\nTop-K Accuracy (best explanation in top K):")
    print(f"  Top-1: {top1_correct}/{num_queries} = {100*top1_correct/num_queries:.1f}%")
    print(f"  Top-3: {top3_correct}/{num_queries} = {100*top3_correct/num_queries:.1f}%")
    print(f"  Top-5: {top5_correct}/{num_queries} = {100*top5_correct/num_queries:.1f}%")

    print(f"\nPairwise Ranking Accuracy:")
    print(f"  Mean: {np.mean(pairwise_accs):.4f}")
    print(f"  Std: {np.std(pairwise_accs):.4f}")
    print(f"  (% of pairs where better item ranked higher)")

    print(f"\nScore Prediction Error:")
    print(f"  MAE (Mean Absolute Error): {np.mean(score_errors):.4f}")
    print(f"  RMSE (Root Mean Squared Error): {np.sqrt(np.mean(np.square(score_errors))):.4f}")
    print(f"  (On 0-1 scale)")

    # Performance interpretation
    print("\n" + "="*80)
    print("PERFORMANCE INTERPRETATION")
    print("="*80)

    ndcg5 = results.get('ndcg@5', 0)
    spearman = results.get('spearman', 0)
    top1_acc = top1_correct / num_queries

    print(f"\nOverall Ranking Quality (NDCG@5 = {ndcg5:.3f}):")
    if ndcg5 > 0.85:
        print("  ⭐⭐⭐⭐⭐ EXCELLENT - Outstanding ranking performance")
    elif ndcg5 > 0.75:
        print("  ⭐⭐⭐⭐ GOOD - Strong ranking with minor errors")
    elif ndcg5 > 0.6:
        print("  ⭐⭐⭐ FAIR - Decent ranking but room for improvement")
    else:
        print("  ⭐⭐ NEEDS WORK - Significant ranking errors")

    print(f"\nRanking Order Understanding (Spearman = {spearman:.3f}):")
    if spearman > 0.8:
        print("  ⭐⭐⭐⭐⭐ EXCELLENT - Understands relative quality very well")
    elif spearman > 0.6:
        print("  ⭐⭐⭐⭐ GOOD - Good grasp of relative quality")
    elif spearman > 0.4:
        print("  ⭐⭐⭐ FAIR - Moderate understanding of quality ordering")
    else:
        print("  ⭐⭐ NEEDS WORK - Struggles with quality ordering")

    print(f"\nTop Result Quality (Top-1 Accuracy = {100*top1_acc:.1f}%):")
    if top1_acc > 0.8:
        print("  ⭐⭐⭐⭐⭐ EXCELLENT - Consistently finds best explanation")
    elif top1_acc > 0.65:
        print("  ⭐⭐⭐⭐ GOOD - Usually finds best explanation")
    elif top1_acc > 0.5:
        print("  ⭐⭐⭐ FAIR - Sometimes finds best explanation")
    else:
        print("  ⭐⭐ NEEDS WORK - Rarely finds best explanation")

    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)

    if ndcg5 > 0.75 and spearman > 0.6:
        print("\n✓ Model is performing well!")
        print("  - Ready for use in production/evaluation")
        print("  - Consider testing on harder examples")
    elif ndcg5 > 0.6 or spearman > 0.5:
        print("\n⚠ Model shows promise but needs improvement:")
        print("  - Consider more training epochs")
        print("  - Try different hyperparameters (learning rate, dropout)")
        print("  - Check if training loss is still decreasing")
    else:
        print("\n✗ Model needs significant improvement:")
        print("  - Check data quality and preprocessing")
        print("  - Verify model architecture is appropriate")
        print("  - Consider increasing model capacity")
        print("  - Ensure sufficient training time")

    print()

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Comprehensive ranking evaluation")
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--dataset', type=str, default='ds_critique',
                       help='Dataset to evaluate on')
    parser.add_argument('--device', type=str, default='mps',
                       help='Device to use (cuda, mps, or cpu)')
    args = parser.parse_args()

    # Load validation data (simplified - adapt to your dataset loader)
    print(f"Loading validation data for {args.dataset}...")

    if args.dataset == 'ds_critique':
        print("\nWARNING: ds_critique has only 5 queries - results may not be representative")
        print("Consider using demo_ranking_quality.py for better evaluation\n")

    # You'll need to add your dataset loading logic here
    print("Note: Please implement dataset loading in this script")
    print("For now, use demo_ranking_quality.py for qualitative evaluation")
