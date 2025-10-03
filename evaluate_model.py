# src/evaluation/comprehensive_evaluator.py

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
import evaluate
from sklearn.metrics import ndcg_score
from scipy.stats import kendalltau, spearmanr
import json
import os
from datasets import load_from_disk, Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import defaultdict

class ComprehensiveEvaluator:
    """
    Comprehensive evaluation framework for explanation generation models
    Includes generation metrics, ranking metrics, and quality analysis
    """

    def __init__(self, config: Dict = None):
        self.config = config or self.get_default_config()
        self.load_metrics()

    def get_default_config(self) -> Dict:
        return {
            'generation_metrics': ['bleu', 'meteor', 'rouge'],
            'ranking_metrics': ['ndcg', 'map', 'kendall_tau', 'spearman'],
            'ndcg_k_values': [1, 3, 5],
            'save_detailed_results': True,
            'create_visualizations': True,
            'sample_size': None  # None = use all data, or set to int for sampling
        }

    def load_metrics(self):
        """Load evaluation metrics"""
        self.metrics = {}

        # Generation metrics
        try:
            if 'bleu' in self.config['generation_metrics']:
                self.metrics['bleu'] = evaluate.load("bleu")
        except:
            print("Warning: Could not load BLEU metric")

        try:
            if 'meteor' in self.config['generation_metrics']:
                self.metrics['meteor'] = evaluate.load("meteor")
        except:
            print("Warning: Could not load METEOR metric")

        try:
            if 'rouge' in self.config['generation_metrics']:
                self.metrics['rouge'] = evaluate.load("rouge")
        except:
            print("Warning: Could not load ROUGE metric")

    def prepare_query_grouped_data(self, dataset: Dataset) -> Dict[str, List[Dict]]:
        """
        Convert long format dataset to query-grouped format
        Each query has multiple candidate explanations with quality scores
        """
        query_groups = defaultdict(lambda: {
            'premise': '',
            'hypothesis': '',
            'label': '',
            'query_text': '',
            'source': '',
            'candidates': []
        })

        for example in dataset:
            query_id = example['query_id']

            # Set query info (will be same for all candidates of this query)
            query_groups[query_id]['premise'] = example['premise']
            query_groups[query_id]['hypothesis'] = example['hypothesis']
            query_groups[query_id]['label'] = example['label']
            query_groups[query_id]['query_text'] = example['query_text']
            query_groups[query_id]['source'] = example['source']

            # Add candidate
            query_groups[query_id]['candidates'].append({
                'text': example['candidate'],
                'quality_score': example['quality_score'],
                'generation_method': example['generation_method']
            })

        return dict(query_groups)

    def evaluate_all(self,
                    model,
                    test_data: Dataset,
                    output_dir: str) -> Dict:
        """
        Run comprehensive evaluation on all metrics
        """
        print("Starting comprehensive evaluation...")

        os.makedirs(output_dir, exist_ok=True)

        # Sample data if requested
        if self.config['sample_size']:
            # Get unique query_ids
            query_ids = list(set(test_data['query_id']))
            if len(query_ids) > self.config['sample_size']:
                sampled_ids = np.random.choice(query_ids, self.config['sample_size'], replace=False)
                test_data = test_data.filter(lambda x: x['query_id'] in sampled_ids)
                print(f"Sampled {self.config['sample_size']} queries for evaluation")

        # Prepare query-grouped data
        print("Preparing query-grouped data...")
        query_data = self.prepare_query_grouped_data(test_data)
        print(f"Evaluating on {len(query_data)} unique queries")

        results = {
            'generation_metrics': {},
            'ranking_metrics': {},
            'quality_analysis': {},
            'dataset_stats': {}
        }

        # Generate predictions
        print("Generating predictions...")
        predictions = self.generate_predictions(model, query_data)

        # Evaluate generation quality
        print("Evaluating generation metrics...")
        results['generation_metrics'] = self.evaluate_generation_metrics(
            predictions, query_data
        )

        # Evaluate ranking performance
        print("Evaluating ranking metrics...")
        results['ranking_metrics'] = self.evaluate_ranking_metrics(
            model, query_data
        )

        # Quality analysis
        print("Analyzing explanation quality...")
        results['quality_analysis'] = self.analyze_explanation_quality(
            predictions, query_data
        )

        # Dataset statistics
        results['dataset_stats'] = self.compute_dataset_stats(query_data)

        # Save results
        self.save_results(results, output_dir)

        # Create visualizations
        if self.config['create_visualizations']:
            print("Creating visualizations...")
            self.create_visualizations(results, output_dir)

        print(f"Evaluation complete! Results saved to {output_dir}")
        return results

    def generate_predictions(self, model, query_data: Dict) -> List[Dict]:
        """Generate model predictions for test data"""
        predictions = []

        for query_id, data in tqdm(query_data.items(), desc="Generating predictions"):
            query = data['query_text']

            # Generate single best explanation
            try:
                best_explanation = model.generate(query, max_new_tokens=128, do_sample=False)
            except Exception as e:
                print(f"Warning: Generation failed for query {query_id}: {e}")
                best_explanation = ""

            # Get gold explanations (quality >= 3)
            gold_explanations = [
                c['text'] for c in data['candidates']
                if c['quality_score'] >= 3
            ]

            predictions.append({
                'query_id': query_id,
                'query': query,
                'premise': data['premise'],
                'hypothesis': data['hypothesis'],
                'label': data['label'],
                'source': data['source'],
                'best_explanation': best_explanation,
                'gold_explanations': gold_explanations
            })

        return predictions

    def evaluate_generation_metrics(self,
                                   predictions: List[Dict],
                                   query_data: Dict) -> Dict:
        """Evaluate standard generation metrics"""
        results = {}

        # Extract predictions and references
        pred_explanations = []
        ref_explanations = []

        for p in predictions:
            pred_explanations.append(p['best_explanation'])

            # Use highest quality gold explanation as reference
            if p['gold_explanations']:
                ref_explanations.append(p['gold_explanations'][0])
            else:
                # Fallback to any available candidate
                query_id = p['query_id']
                candidates = query_data[query_id]['candidates']
                if candidates:
                    ref_explanations.append(candidates[0]['text'])
                else:
                    ref_explanations.append("")

        # Filter out empty predictions/references
        valid_pairs = [(p, r) for p, r in zip(pred_explanations, ref_explanations)
                       if p and r]

        if not valid_pairs:
            return {'error': 'No valid prediction-reference pairs'}

        pred_explanations, ref_explanations = zip(*valid_pairs)
        pred_explanations = list(pred_explanations)
        ref_explanations = list(ref_explanations)

        # BLEU score
        if 'bleu' in self.metrics:
            try:
                bleu_result = self.metrics['bleu'].compute(
                    predictions=pred_explanations,
                    references=[[ref] for ref in ref_explanations]
                )
                results['bleu'] = bleu_result['bleu']
            except Exception as e:
                print(f"Warning: BLEU computation failed: {e}")

        # METEOR score
        if 'meteor' in self.metrics:
            try:
                meteor_result = self.metrics['meteor'].compute(
                    predictions=pred_explanations,
                    references=ref_explanations
                )
                results['meteor'] = meteor_result['meteor']
            except Exception as e:
                print(f"Warning: METEOR computation failed: {e}")

        # ROUGE scores
        if 'rouge' in self.metrics:
            try:
                rouge_result = self.metrics['rouge'].compute(
                    predictions=pred_explanations,
                    references=ref_explanations
                )
                results['rouge'] = {
                    'rouge1': rouge_result['rouge1'],
                    'rouge2': rouge_result['rouge2'],
                    'rougeL': rouge_result['rougeL']
                }
            except Exception as e:
                print(f"Warning: ROUGE computation failed: {e}")

        return results

    def evaluate_ranking_metrics(self, model, query_data: Dict) -> Dict:
        """Evaluate ranking performance metrics"""
        if not hasattr(model, 'rank_explanations'):
            return {'error': 'Model does not support ranking'}

        ndcg_scores = {f'ndcg@{k}': [] for k in self.config['ndcg_k_values']}
        map_scores = []
        kendall_tau_scores = []
        spearman_scores = []

        for query_id, data in tqdm(query_data.items(), desc="Evaluating ranking"):
            candidates = data['candidates']

            if len(candidates) < 2:
                continue

            query = data['query_text']

            # Get explanations and their gold quality scores
            explanations = [c['text'] for c in candidates]
            gold_scores = [c['quality_score'] for c in candidates]

            # Get model ranking scores
            try:
                model_scores = model.rank_explanations(query, explanations)
            except Exception as e:
                print(f"Warning: Ranking failed for query {query_id}: {e}")
                continue

            # NDCG@k
            for k in self.config['ndcg_k_values']:
                if len(explanations) >= k:
                    try:
                        ndcg_k = ndcg_score(
                            [gold_scores], [model_scores], k=k
                        )
                        ndcg_scores[f'ndcg@{k}'].append(ndcg_k)
                    except:
                        pass

            # MAP (Mean Average Precision)
            try:
                map_score = self.compute_map(gold_scores, model_scores)
                map_scores.append(map_score)
            except:
                pass

            # Kendall's Tau correlation
            if len(set(gold_scores)) > 1:  # Need variation in scores
                try:
                    tau, _ = kendalltau(gold_scores, model_scores)
                    if not np.isnan(tau):
                        kendall_tau_scores.append(tau)
                except:
                    pass

            # Spearman correlation
            if len(set(gold_scores)) > 1:
                try:
                    rho, _ = spearmanr(gold_scores, model_scores)
                    if not np.isnan(rho):
                        spearman_scores.append(rho)
                except:
                    pass

        # Aggregate results
        results = {}

        for k in self.config['ndcg_k_values']:
            if ndcg_scores[f'ndcg@{k}']:
                results[f'ndcg@{k}'] = float(np.mean(ndcg_scores[f'ndcg@{k}']))

        results['map'] = float(np.mean(map_scores)) if map_scores else 0.0
        results['kendall_tau'] = float(np.mean(kendall_tau_scores)) if kendall_tau_scores else 0.0
        results['spearman'] = float(np.mean(spearman_scores)) if spearman_scores else 0.0
        results['num_queries_ranked'] = len(map_scores)

        return results

    def compute_map(self, gold_scores: List[float], pred_scores: List[float]) -> float:
        """Compute Mean Average Precision"""
        # Convert to relevance (top 50% are relevant)
        threshold = np.median(gold_scores)
        relevance = [1 if score > threshold else 0 for score in gold_scores]

        # Sort by predicted scores (descending)
        sorted_indices = np.argsort(pred_scores)[::-1]
        sorted_relevance = [relevance[i] for i in sorted_indices]

        # Compute AP
        ap = 0.0
        num_relevant = 0

        for i, rel in enumerate(sorted_relevance):
            if rel == 1:
                num_relevant += 1
                precision_at_i = num_relevant / (i + 1)
                ap += precision_at_i

        return ap / max(sum(relevance), 1)

    def analyze_explanation_quality(self,
                                   predictions: List[Dict],
                                   query_data: Dict) -> Dict:
        """Analyze explanation quality across different dimensions"""
        quality_results = {}

        # Quality by source dataset
        source_quality = defaultdict(list)
        for pred in predictions:
            source = pred['source']
            quality_score = self.compute_explanation_quality(
                pred['query'], pred['best_explanation']
            )
            source_quality[source].append(quality_score)

        quality_results['by_source'] = {
            source: float(np.mean(scores)) for source, scores in source_quality.items()
        }

        # Quality by NLI relation
        label_quality = defaultdict(list)
        for pred in predictions:
            label = pred['label']
            quality_score = self.compute_explanation_quality(
                pred['query'], pred['best_explanation']
            )
            label_quality[label].append(quality_score)

        quality_results['by_label'] = {
            label: float(np.mean(scores)) for label, scores in label_quality.items()
        }

        # Overall quality distribution
        all_quality_scores = []
        for pred in predictions:
            quality_score = self.compute_explanation_quality(
                pred['query'], pred['best_explanation']
            )
            all_quality_scores.append(quality_score)

        quality_results['overall'] = {
            'mean': float(np.mean(all_quality_scores)),
            'std': float(np.std(all_quality_scores)),
            'median': float(np.median(all_quality_scores)),
            'distribution': np.histogram(all_quality_scores, bins=10)[0].tolist()
        }

        # Length statistics
        lengths = [len(pred['best_explanation'].split()) for pred in predictions]
        quality_results['length_stats'] = {
            'mean': float(np.mean(lengths)),
            'std': float(np.std(lengths)),
            'median': float(np.median(lengths)),
            'min': int(np.min(lengths)),
            'max': int(np.max(lengths))
        }

        return quality_results

    def compute_explanation_quality(self, query: str, explanation: str) -> float:
        """
        Compute explanation quality score using simple heuristics
        """
        if not explanation:
            return 0.0

        scores = {}

        # Length score (not too short, not too long)
        length = len(explanation.split())
        if 5 <= length <= 50:
            scores['length'] = 1.0
        elif length < 5:
            scores['length'] = length / 5.0
        else:
            scores['length'] = max(0.5, 1.0 - (length - 50) / 100)

        # Keyword relevance
        query_words = set(query.lower().split())
        exp_words = set(explanation.lower().split())
        overlap = len(query_words.intersection(exp_words))
        scores['relevance'] = min(1.0, overlap / max(len(query_words), 1))

        # Coherence (simplified - count of logical connectors)
        connectors = ['because', 'since', 'therefore', 'thus', 'so', 'hence', 'as', 'when', 'if']
        connector_count = sum(1 for word in explanation.lower().split() if word in connectors)
        scores['coherence'] = min(1.0, connector_count / 2.0)

        # Overall score (weighted average)
        weights = {'length': 0.3, 'relevance': 0.4, 'coherence': 0.3}
        overall_score = sum(weights[dim] * scores[dim] for dim in weights)

        return overall_score

    def compute_dataset_stats(self, query_data: Dict) -> Dict:
        """Compute dataset statistics"""
        stats = {
            'num_queries': len(query_data),
            'candidates_per_query': {},
            'quality_distribution': defaultdict(int),
            'source_distribution': defaultdict(int),
            'label_distribution': defaultdict(int)
        }

        candidate_counts = []
        for query_id, data in query_data.items():
            num_candidates = len(data['candidates'])
            candidate_counts.append(num_candidates)

            # Quality distribution
            for c in data['candidates']:
                stats['quality_distribution'][c['quality_score']] += 1

            # Source and label distribution
            stats['source_distribution'][data['source']] += 1
            stats['label_distribution'][data['label']] += 1

        stats['candidates_per_query'] = {
            'mean': float(np.mean(candidate_counts)),
            'min': int(np.min(candidate_counts)),
            'max': int(np.max(candidate_counts))
        }

        # Convert defaultdicts to regular dicts
        stats['quality_distribution'] = dict(stats['quality_distribution'])
        stats['source_distribution'] = dict(stats['source_distribution'])
        stats['label_distribution'] = dict(stats['label_distribution'])

        return stats

    def compute_ranking_rewards_with_loss(self, queries, responses, ranking_model):
        """
        Use ranking loss as negative reward for PPO
        """
        # Import ComprehensiveRankingLoss (assumes it's defined elsewhere or needs to be created)
        from ranking_models import RankingLosses

        # Create a combined loss function
        def combined_loss(scores, labels):
            losses = {}

            # Use multi-margin hinge loss instead of simple pairwise
            losses['hinge'] = RankingLosses.multi_margin_hinge_loss(scores, labels) if len(scores) > 1 else torch.tensor(0.0)

            losses['listnet'] = RankingLosses.listnet_loss(scores.unsqueeze(0), labels.unsqueeze(0))
            losses['ndcg'] = RankingLosses.approxndcg_loss(scores.unsqueeze(0), labels.unsqueeze(0))

            # Weighted combination
            total_loss = 0.4 * losses['hinge'] + 0.3 * losses['listnet'] + 0.3 * losses['ndcg']
            return total_loss, losses

        all_rewards = []

        for query, response_list in zip(queries, responses):
            # Score all responses
            if hasattr(ranking_model, 'rank_explanations'):
                scores = torch.tensor(ranking_model.rank_explanations(query, response_list))
            else:
                # Fallback to simple scoring
                scores = torch.tensor([len(r.split()) / 50.0 for r in response_list])

            # Get pseudo-labels based on heuristics or DS critique
            pseudo_labels = self.get_quality_estimates(query, response_list)

            # Compute loss
            loss, loss_dict = combined_loss(scores, pseudo_labels)

            # Convert to rewards (negative loss)
            rewards = -loss.detach()

            # Add per-response rewards based on ranking position
            sorted_indices = scores.argsort(descending=True)
            position_rewards = torch.zeros_like(scores)
            for rank, idx in enumerate(sorted_indices):
                position_rewards[idx] = 1.0 - (rank / len(scores))

            # Combine loss-based and position-based rewards
            combined_rewards = 0.6 * (-loss) + 0.4 * position_rewards

            all_rewards.extend(combined_rewards.tolist())

        return torch.tensor(all_rewards)

    def get_quality_estimates(self, query: str, responses: List[str]) -> torch.Tensor:
        """
        Estimate quality scores for responses using heuristics
        """
        quality_scores = []

        for response in responses:
            # Use the compute_explanation_quality method
            quality = self.compute_explanation_quality(query, response)
            quality_scores.append(quality)

        return torch.tensor(quality_scores)

    def save_results(self, results: Dict, output_dir: str):
        """Save evaluation results to files"""
        # Save main results
        with open(os.path.join(output_dir, 'evaluation_results.json'), 'w') as f:
            json.dump(results, f, indent=2, default=str)

        # Save detailed breakdown
        if self.config['save_detailed_results']:
            for key in ['generation_metrics', 'ranking_metrics', 'quality_analysis', 'dataset_stats']:
                if key in results:
                    with open(os.path.join(output_dir, f'{key}.json'), 'w') as f:
                        json.dump(results[key], f, indent=2, default=str)

    def create_visualizations(self, results: Dict, output_dir: str):
        """Create visualization plots for results"""
        viz_dir = os.path.join(output_dir, 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)

        # Set style
        try:
            plt.style.use('seaborn-v0_8')
        except:
            plt.style.use('default')

        # 1. Generation metrics comparison
        if results['generation_metrics']:
            self.plot_generation_metrics(results['generation_metrics'], viz_dir)

        # 2. Ranking metrics radar chart
        if results['ranking_metrics'] and 'error' not in results['ranking_metrics']:
            self.plot_ranking_metrics(results['ranking_metrics'], viz_dir)

        # 3. Quality analysis charts
        if results['quality_analysis']:
            self.plot_quality_analysis(results['quality_analysis'], viz_dir)

        # 4. Dataset statistics
        if results['dataset_stats']:
            self.plot_dataset_stats(results['dataset_stats'], viz_dir)

        print(f"Visualizations saved to {viz_dir}")

    def plot_generation_metrics(self, metrics: Dict, output_dir: str):
        """Plot generation metrics"""
        if not metrics or 'error' in metrics:
            return

        # Extract metrics for plotting
        metric_names = []
        metric_values = []

        for metric, value in metrics.items():
            if isinstance(value, dict):
                for sub_metric, sub_value in value.items():
                    metric_names.append(f"{metric}_{sub_metric}")
                    metric_values.append(sub_value)
            else:
                metric_names.append(metric)
                metric_values.append(value)

        if not metric_values:
            return

        # Create bar plot
        plt.figure(figsize=(12, 6))
        bars = plt.bar(metric_names, metric_values, color='skyblue', alpha=0.7)
        plt.title('Generation Metrics Performance', fontsize=16, fontweight='bold')
        plt.ylabel('Score', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, 1.0)

        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'generation_metrics.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def plot_ranking_metrics(self, metrics: Dict, output_dir: str):
        """Plot ranking metrics as radar chart"""
        if not metrics or 'error' in metrics:
            return

        # Filter out non-metric keys
        metric_data = {k: v for k, v in metrics.items() if k != 'num_queries_ranked'}

        if not metric_data:
            return

        metric_names = list(metric_data.keys())
        metric_values = list(metric_data.values())

        # Normalize values to 0-1 range for radar chart
        normalized_values = []
        for i, value in enumerate(metric_values):
            if 'ndcg' in metric_names[i] or 'map' in metric_names[i]:
                normalized_values.append(value)  # Already 0-1
            else:  # correlation metrics can be -1 to 1
                normalized_values.append((value + 1) / 2)

        # Create bar chart instead of radar for simplicity
        plt.figure(figsize=(10, 6))
        bars = plt.bar(metric_names, metric_values, color='coral', alpha=0.7)
        plt.title('Ranking Metrics Performance', fontsize=16, fontweight='bold')
        plt.ylabel('Score', fontsize=12)
        plt.xticks(rotation=45, ha='right')

        # Add value labels
        for bar, value in zip(bars, metric_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'ranking_metrics.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def plot_quality_analysis(self, quality_results: Dict, output_dir: str):
        """Plot quality analysis results"""
        if not quality_results:
            return

        # Quality by source
        if 'by_source' in quality_results:
            plt.figure(figsize=(10, 6))
            sources = list(quality_results['by_source'].keys())
            qualities = list(quality_results['by_source'].values())

            bars = plt.bar(sources, qualities, color='lightgreen', alpha=0.7)
            plt.title('Explanation Quality by Source Dataset', fontsize=16, fontweight='bold')
            plt.ylabel('Average Quality Score', fontsize=12)
            plt.ylim(0, 1.0)
            plt.xticks(rotation=45, ha='right')

            # Add value labels
            for bar, value in zip(bars, qualities):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'quality_by_source.png'), dpi=300, bbox_inches='tight')
            plt.close()

        # Quality distribution
        if 'overall' in quality_results and 'distribution' in quality_results['overall']:
            plt.figure(figsize=(10, 6))
            distribution = quality_results['overall']['distribution']
            bin_edges = np.linspace(0, 1, len(distribution) + 1)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            plt.bar(bin_centers, distribution, width=0.08, alpha=0.7, color='orange')
            plt.title('Distribution of Explanation Quality Scores', fontsize=16, fontweight='bold')
            plt.xlabel('Quality Score', fontsize=12)
            plt.ylabel('Number of Examples', fontsize=12)

            # Add statistics
            mean_qual = quality_results['overall']['mean']
            median_qual = quality_results['overall']['median']
            plt.axvline(mean_qual, color='red', linestyle='--', alpha=0.8, label=f'Mean: {mean_qual:.3f}')
            plt.axvline(median_qual, color='blue', linestyle='--', alpha=0.8, label=f'Median: {median_qual:.3f}')
            plt.legend()

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'quality_distribution.png'), dpi=300, bbox_inches='tight')
            plt.close()

    def plot_dataset_stats(self, stats: Dict, output_dir: str):
        """Plot dataset statistics"""
        if not stats:
            return

        # Quality distribution
        if 'quality_distribution' in stats:
            plt.figure(figsize=(10, 6))
            quality_scores = sorted(stats['quality_distribution'].keys())
            counts = [stats['quality_distribution'][q] for q in quality_scores]

            colors = ['red', 'orange', 'yellow', 'lightgreen', 'green']
            plt.bar(quality_scores, counts, color=colors[:len(quality_scores)], alpha=0.7)
            plt.title('Distribution of Quality Scores in Dataset', fontsize=16, fontweight='bold')
            plt.xlabel('Quality Score', fontsize=12)
            plt.ylabel('Number of Candidates', fontsize=12)

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'dataset_quality_distribution.png'), dpi=300, bbox_inches='tight')
            plt.close()


# Mock model for testing
class MockModel:
    """Simple mock model for testing the evaluator"""

    def generate(self, query: str, max_new_tokens: int = 128, do_sample: bool = False) -> str:
        """Generate a simple explanation based on query"""
        # Extract premise and hypothesis from query
        if "If: '" in query and "', why" in query:
            parts = query.split("', ")
            return f"This is because of the relationship between the statements."
        return "This is a simple explanation."

    def rank_explanations(self, query: str, explanations: List[str]) -> List[float]:
        """Rank explanations by length (simple heuristic)"""
        # Longer explanations get higher scores (simple heuristic)
        scores = [len(exp.split()) / 100.0 for exp in explanations]
        return scores


# Utility function for easy evaluation
def evaluate_model_comprehensive(model, test_dataset_path: str, output_dir: str, config: Dict = None):
    """
    Convenience function for comprehensive model evaluation

    Args:
        model: Trained model with generate() and optionally rank_explanations() methods
        test_dataset_path: Path to test dataset
        output_dir: Directory to save results
        config: Evaluation configuration
    """
    # Load test data
    dataset_dict = load_from_disk(test_dataset_path)
    test_data = dataset_dict['test']

    # Initialize evaluator
    evaluator = ComprehensiveEvaluator(config)

    # Run evaluation
    results = evaluator.evaluate_all(model, test_data, output_dir)

    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)

    if 'dataset_stats' in results:
        print(f"\nDataset Statistics:")
        print(f"  Number of queries: {results['dataset_stats']['num_queries']}")
        print(f"  Avg candidates per query: {results['dataset_stats']['candidates_per_query']['mean']:.1f}")

    if 'generation_metrics' in results and 'error' not in results['generation_metrics']:
        print("\nGeneration Metrics:")
        for metric, value in results['generation_metrics'].items():
            if isinstance(value, dict):
                for sub_metric, sub_value in value.items():
                    print(f"  {metric}_{sub_metric}: {sub_value:.4f}")
            else:
                print(f"  {metric}: {value:.4f}")

    if 'ranking_metrics' in results and 'error' not in results['ranking_metrics']:
        print("\nRanking Metrics:")
        for metric, value in results['ranking_metrics'].items():
            print(f"  {metric}: {value:.4f}")

    if 'quality_analysis' in results and 'overall' in results['quality_analysis']:
        overall_quality = results['quality_analysis']['overall']
        print(f"\nQuality Analysis:")
        print(f"  Mean Quality: {overall_quality['mean']:.4f}")
        print(f"  Median Quality: {overall_quality['median']:.4f}")
        print(f"  Std Quality: {overall_quality['std']:.4f}")

        if 'length_stats' in results['quality_analysis']:
            length_stats = results['quality_analysis']['length_stats']
            print(f"\nLength Statistics:")
            print(f"  Mean length: {length_stats['mean']:.1f} words")
            print(f"  Median length: {length_stats['median']:.1f} words")
            print(f"  Range: {length_stats['min']}-{length_stats['max']} words")

    print("="*60)
    return results


# Example usage
if __name__ == "__main__":
    # Example configuration
    eval_config = {
        'generation_metrics': ['bleu', 'meteor', 'rouge'],
        'ranking_metrics': ['ndcg', 'map', 'kendall_tau', 'spearman'],
        'ndcg_k_values': [1, 3, 5],
        'save_detailed_results': True,
        'create_visualizations': True,
        'sample_size': 100  # Use 100 queries for quick testing
    }

    # Create mock model for testing
    print("Creating mock model for testing...")
    model = MockModel()

    # Run evaluation
    print("\nRunning evaluation with mock model...")
    results = evaluate_model_comprehensive(
        model=model,
        test_dataset_path='data/processed/comprehensive_ranking_dataset',
        output_dir='experiments/evaluation_test',
        config=eval_config
    )

    print("\nEvaluation completa! Check experiments/evaluation_test/ for results.")
