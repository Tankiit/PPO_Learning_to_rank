import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import ndcg_score
from scipy.stats import kendalltau, spearmanr
import json
from tqdm import tqdm
from collections import defaultdict

class RankingEvaluator:
    """
    Comprehensive evaluator for ranking-based explanation models
    Handles both binary and multi-level quality assessment
    """

    def __init__(self, metrics: List[str] = None):
        if metrics is None:
            metrics = ['ndcg', 'map', 'mrr', 'kendall_tau', 'spearman']

        self.metrics = metrics
        self.results = {}

    def evaluate(self,
                model,
                test_data: List[Dict],
                batch_size: int = 32) -> Dict[str, float]:
        """
        Evaluate model on ranking metrics

        Args:
            model: Model with rank_explanations method
            test_data: List of examples with 'query', 'explanations', 'scores'
            batch_size: Batch size for evaluation

        Returns:
            Dictionary of metric scores
        """
        all_results = defaultdict(list)

        for example in tqdm(test_data, desc="Evaluating"):
            query = example['query']
            explanations = example['explanations']
            gold_scores = example['scores']

            # Skip if too few candidates
            if len(explanations) < 2:
                continue

            # Get model predictions
            try:
                pred_scores = model.rank_explanations(query, explanations)
            except Exception as e:
                print(f"Error ranking: {e}")
                continue

            # Compute metrics
            metrics = self.compute_metrics(gold_scores, pred_scores)

            for metric, value in metrics.items():
                if not np.isnan(value):
                    all_results[metric].append(value)

        # Aggregate results
        final_results = {}
        for metric, values in all_results.items():
            if values:
                final_results[metric] = float(np.mean(values))
                final_results[f"{metric}_std"] = float(np.std(values))

        self.results = final_results
        return final_results

    def compute_metrics(self,
                       gold_scores: List[float],
                       pred_scores: List[float]) -> Dict[str, float]:
        """Compute all ranking metrics for a single query"""

        metrics = {}

        # Ensure numpy arrays
        gold_scores = np.array(gold_scores)
        pred_scores = np.array(pred_scores)

        # NDCG at various k
        for k in [1, 3, 5]:
            if len(gold_scores) >= k:
                try:
                    ndcg_k = ndcg_score([gold_scores], [pred_scores], k=k)
                    metrics[f'ndcg@{k}'] = ndcg_k
                except:
                    metrics[f'ndcg@{k}'] = np.nan

        # Mean Average Precision
        if 'map' in self.metrics:
            metrics['map'] = self._compute_map(gold_scores, pred_scores)

        # Mean Reciprocal Rank
        if 'mrr' in self.metrics:
            metrics['mrr'] = self._compute_mrr(gold_scores, pred_scores)

        # Correlation metrics
        if len(set(gold_scores)) > 1:  # Need variation
            if 'kendall_tau' in self.metrics:
                tau, _ = kendalltau(gold_scores, pred_scores)
                metrics['kendall_tau'] = tau

            if 'spearman' in self.metrics:
                rho, _ = spearmanr(gold_scores, pred_scores)
                metrics['spearman'] = rho

        return metrics

    def _compute_map(self, gold_scores: np.ndarray, pred_scores: np.ndarray) -> float:
        """Compute Mean Average Precision"""
        # Define relevance threshold (e.g., score >= 4 is relevant)
        threshold = 4.0
        relevance = (gold_scores >= threshold).astype(int)

        # Sort by predicted scores
        sorted_indices = np.argsort(-pred_scores)
        sorted_relevance = relevance[sorted_indices]

        # Compute AP
        ap = 0.0
        num_relevant = 0

        for i, rel in enumerate(sorted_relevance):
            if rel == 1:
                num_relevant += 1
                precision_at_i = num_relevant / (i + 1)
                ap += precision_at_i

        return ap / max(relevance.sum(), 1)

    def _compute_mrr(self, gold_scores: np.ndarray, pred_scores: np.ndarray) -> float:
        """Compute Mean Reciprocal Rank"""
        # Find best gold item
        best_gold_idx = np.argmax(gold_scores)

        # Get rank of best gold item in predictions
        pred_ranking = np.argsort(-pred_scores)

        for rank, idx in enumerate(pred_ranking):
            if idx == best_gold_idx:
                return 1.0 / (rank + 1)

        return 0.0

    def evaluate_binary_vs_ranking(self,
                                  binary_model,
                                  ranking_model,
                                  test_data: List[Dict]) -> Dict[str, Dict]:
        """
        Compare binary reward model vs ranking reward model
        """
        print("Evaluating binary model...")
        binary_results = self.evaluate(binary_model, test_data)

        print("Evaluating ranking model...")
        ranking_results = self.evaluate(ranking_model, test_data)

        # Compute improvements
        improvements = {}
        for metric in set(binary_results.keys()) & set(ranking_results.keys()):
            if not metric.endswith('_std'):
                binary_val = binary_results[metric]
                ranking_val = ranking_results[metric]
                improvements[metric] = {
                    'absolute': ranking_val - binary_val,
                    'relative': (ranking_val - binary_val) / max(abs(binary_val), 1e-6) * 100
                }

        return {
            'binary_results': binary_results,
            'ranking_results': ranking_results,
            'improvements': improvements
        }

    def print_results(self, results: Dict[str, float]):
        """Pretty print evaluation results"""
        print("\n" + "="*60)
        print("RANKING EVALUATION RESULTS")
        print("="*60)

        for metric, value in sorted(results.items()):
            if not metric.endswith('_std'):
                std_key = f"{metric}_std"
                if std_key in results:
                    print(f"{metric:20s}: {value:.4f} ± {results[std_key]:.4f}")
                else:
                    print(f"{metric:20s}: {value:.4f}")

        print("="*60)


if __name__ == "__main__":
    print("Testing Ranking Evaluator...")

    # Create mock test data
    test_data = [
        {
            'query': "Why does ice float on water?",
            'explanations': [
                "Ice is lighter",
                "Ice is less dense than water",
                "Ice has a crystalline structure with more space between molecules",
                "Water expands when frozen",
                "Random explanation"
            ],
            'scores': [2.0, 3.0, 5.0, 4.0, 1.0]
        },
        {
            'query': "What causes gravity?",
            'explanations': [
                "Mass attracts mass",
                "Einstein's general relativity explains it",
                "It's a force",
                "Space-time curvature"
            ],
            'scores': [3.0, 5.0, 2.0, 4.0]
        }
    ]

    # Create simple mock model
    class SimpleMockModel:
        def rank_explanations(self, query, explanations):
            # Rank by length
            return [len(exp) / 50.0 for exp in explanations]

    model = SimpleMockModel()

    # Evaluate
    evaluator = RankingEvaluator()
    results = evaluator.evaluate(model, test_data)

    # Print results
    evaluator.print_results(results)

    # Test individual metrics
    print("\n\nTesting individual metric computation...")
    gold = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    pred = np.array([4.5, 4.0, 3.5, 2.5, 1.5])

    metrics = evaluator.compute_metrics(gold.tolist(), pred.tolist())
    print("\nMetrics for perfect ranking:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")

    print("\n✅ Ranking evaluator test complete!")
