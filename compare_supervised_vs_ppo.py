"""
Compare supervised ranking model vs PPO ranking model

This script evaluates both approaches side-by-side on the same test data
and provides detailed comparison metrics.
"""

import argparse
import torch
import numpy as np
from typing import Dict, List
import json
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

from ranking_models import RankingRewardModel
from ranking_evaluator import RankingEvaluator

# Conditionally import PPO (only if not using mock)
try:
    from train_ppo_ranking import PPORankingPolicy
except ImportError:
    PPORankingPolicy = None  # Will be handled if use_mock_ppo=True


class ModelComparator:
    """
    Compare supervised and PPO ranking models
    """

    def __init__(self, supervised_path: str, ppo_path: str = None, device: str = 'cpu', use_mock_ppo: bool = False):
        self.device = device
        self.supervised_path = supervised_path
        self.ppo_path = ppo_path
        self.use_mock_ppo = use_mock_ppo

        # Load supervised model
        print("Loading supervised model...")
        self.supervised_model = self.load_supervised_model(supervised_path)

        # Load PPO policy (or use mock)
        if use_mock_ppo:
            print("Using mock PPO (simulated with better discrimination) for demonstration...")
            self.ppo_policy = None  # Will use mock in rank_with_ppo
        elif ppo_path:
            print("Loading PPO policy...")
            self.ppo_policy = self.load_ppo_model(ppo_path)
        else:
            raise ValueError("Must provide ppo_path or set use_mock_ppo=True")

        # Evaluator
        self.evaluator = RankingEvaluator()

    def load_supervised_model(self, path: str) -> RankingRewardModel:
        """Load supervised ranking model"""
        checkpoint = torch.load(path, map_location=self.device)

        # Get config
        if 'config' in checkpoint:
            config = checkpoint['config']
            base_model = config.get('base_model', 'bert-base-uncased')
            dropout = config.get('dropout', 0.1)
        else:
            # Default config
            base_model = 'bert-base-uncased'
            dropout = 0.1

        # Initialize model
        model = RankingRewardModel(
            base_model=base_model,
            output_mode="regression",
            dropout=dropout
        )

        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()

        return model

    def load_ppo_model(self, path: str) -> PPORankingPolicy:
        """Load PPO policy network"""
        checkpoint = torch.load(path, map_location=self.device)

        # Get config
        if 'config' in checkpoint:
            config = checkpoint['config']
            base_model = config.get('base_model', 'bert-base-uncased')
            dropout = config.get('dropout', 0.1)
        else:
            base_model = 'bert-base-uncased'
            dropout = 0.1

        # Initialize policy
        policy = PPORankingPolicy(
            base_model=base_model,
            dropout=dropout
        )

        # Load weights
        policy.load_state_dict(checkpoint['policy_state_dict'])
        policy.to(self.device)
        policy.eval()

        return policy

    def rank_with_supervised(self, query: str, explanations: List[str]) -> np.ndarray:
        """Get ranking scores from supervised model"""
        scores = self.supervised_model.rank_explanations(query, explanations, return_scores=True)
        return np.array(scores)

    def rank_with_ppo(self, query: str, explanations: List[str]) -> np.ndarray:
        """Get ranking scores from PPO policy (or mock)"""
        if self.use_mock_ppo:
            # Mock PPO: Simulate better score discrimination
            # Use supervised scores but spread them out more
            sup_scores = self.rank_with_supervised(query, explanations)

            # Apply transformation to increase discrimination
            # This simulates what PPO might achieve with exploration
            mean_score = np.mean(sup_scores)
            std_score = np.std(sup_scores)

            # Spread out scores (simulate PPO's better discrimination)
            mock_scores = (sup_scores - mean_score) * 2.5 + mean_score  # Amplify differences
            mock_scores = np.clip(mock_scores, 0.0, 1.0)  # Keep in [0, 1]

            # Add small random perturbation (simulate exploration)
            noise = np.random.normal(0, 0.02, size=mock_scores.shape)
            mock_scores = np.clip(mock_scores + noise, 0.0, 1.0)

            return mock_scores
        else:
            # Real PPO policy
            # Tokenize all query-explanation pairs
            texts = [f"{query} {self.ppo_policy.tokenizer.sep_token} {exp}" for exp in explanations]

            encoded = self.ppo_policy.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors='pt'
            )

            # Move to device
            encoded = {k: v.to(self.device) for k, v in encoded.items()}

            # Get logits
            with torch.no_grad():
                logits = self.ppo_policy(
                    input_ids=encoded['input_ids'],
                    attention_mask=encoded['attention_mask']
                )

            # Convert to probabilities
            scores = torch.softmax(logits, dim=0).cpu().numpy()
            return scores

    def compare_on_examples(self, test_data: List[Dict]) -> Dict:
        """
        Compare both models on test examples

        Args:
            test_data: List of dicts with 'query', 'explanations', 'scores'

        Returns:
            Comparison results
        """
        print("\\nRunning comparison on test data...")

        supervised_results = []
        ppo_results = []

        for example in tqdm(test_data, desc="Evaluating"):
            query = example['query']
            explanations = example['explanations']
            gold_scores = np.array(example['scores'])

            if len(explanations) < 2:
                continue

            # Get predictions
            supervised_scores = self.rank_with_supervised(query, explanations)
            ppo_scores = self.rank_with_ppo(query, explanations)

            # Compute metrics
            supervised_metrics = self.evaluator.compute_metrics(gold_scores.tolist(), supervised_scores.tolist())
            ppo_metrics = self.evaluator.compute_metrics(gold_scores.tolist(), ppo_scores.tolist())

            supervised_results.append(supervised_metrics)
            ppo_results.append(ppo_metrics)

            # Store for detailed analysis
            example['supervised_scores'] = supervised_scores
            example['ppo_scores'] = ppo_scores
            example['gold_scores'] = gold_scores

        # Aggregate results
        comparison = self.aggregate_results(supervised_results, ppo_results)

        # Add detailed examples
        comparison['detailed_examples'] = test_data[:10]  # First 10 for inspection

        return comparison

    def aggregate_results(self, supervised_results: List[Dict], ppo_results: List[Dict]) -> Dict:
        """Aggregate metrics across all examples"""

        # Collect all metrics
        supervised_metrics = defaultdict(list)
        ppo_metrics = defaultdict(list)

        for sup_res in supervised_results:
            for metric, value in sup_res.items():
                if not np.isnan(value):
                    supervised_metrics[metric].append(value)

        for ppo_res in ppo_results:
            for metric, value in ppo_res.items():
                if not np.isnan(value):
                    ppo_metrics[metric].append(value)

        # Compute means and stds
        comparison = {
            'supervised': {},
            'ppo': {},
            'improvements': {}
        }

        for metric in supervised_metrics.keys():
            if metric in ppo_metrics:
                sup_values = supervised_metrics[metric]
                ppo_values = ppo_metrics[metric]

                sup_mean = np.mean(sup_values)
                sup_std = np.std(sup_values)
                ppo_mean = np.mean(ppo_values)
                ppo_std = np.std(ppo_values)

                comparison['supervised'][metric] = {
                    'mean': float(sup_mean),
                    'std': float(sup_std)
                }
                comparison['ppo'][metric] = {
                    'mean': float(ppo_mean),
                    'std': float(ppo_std)
                }

                # Compute improvement
                abs_improvement = ppo_mean - sup_mean
                rel_improvement = (abs_improvement / max(abs(sup_mean), 1e-6)) * 100

                comparison['improvements'][metric] = {
                    'absolute': float(abs_improvement),
                    'relative_percent': float(rel_improvement)
                }

        return comparison

    def print_comparison(self, comparison: Dict):
        """Pretty print comparison results"""
        print("\\n" + "="*80)
        print("SUPERVISED VS PPO RANKING MODEL COMPARISON")
        print("="*80)

        print(f"\\nSupervised Model: {self.supervised_path}")
        print(f"PPO Model: {self.ppo_path}")

        print("\\n" + "-"*80)
        print(f"{'Metric':<20} {'Supervised':<25} {'PPO':<25} {'Improvement':<15}")
        print("-"*80)

        for metric in sorted(comparison['supervised'].keys()):
            sup_mean = comparison['supervised'][metric]['mean']
            sup_std = comparison['supervised'][metric]['std']
            ppo_mean = comparison['ppo'][metric]['mean']
            ppo_std = comparison['ppo'][metric]['std']
            improvement = comparison['improvements'][metric]['relative_percent']

            # Format with color
            imp_str = f"{improvement:+.2f}%"
            if improvement > 5:
                imp_str = f"✓ {imp_str}"
            elif improvement < -5:
                imp_str = f"✗ {imp_str}"
            else:
                imp_str = f"≈ {imp_str}"

            print(f"{metric:<20} {sup_mean:.4f} ± {sup_std:.4f}      "
                  f"{ppo_mean:.4f} ± {ppo_std:.4f}      {imp_str:<15}")

        print("-"*80)

        # Summary
        ndcg5_imp = comparison['improvements'].get('ndcg@5', {}).get('relative_percent', 0)
        if ndcg5_imp > 5:
            print("\\n✅ PPO shows significant improvement over supervised learning")
        elif ndcg5_imp < -5:
            print("\\n⚠️  Supervised learning outperforms PPO")
        else:
            print("\\n➖ Both approaches perform similarly")

        print("="*80)

    def save_comparison(self, comparison: Dict, output_path: str):
        """Save comparison results to JSON"""
        # Remove detailed examples for cleaner JSON
        comparison_clean = {k: v for k, v in comparison.items() if k != 'detailed_examples'}

        with open(output_path, 'w') as f:
            json.dump(comparison_clean, f, indent=2)

        print(f"\\nComparison results saved to {output_path}")

    def plot_comparison(self, comparison: Dict, output_dir: str):
        """Create visualization comparing both models"""
        os.makedirs(output_dir, exist_ok=True)

        metrics = list(comparison['supervised'].keys())
        supervised_means = [comparison['supervised'][m]['mean'] for m in metrics]
        ppo_means = [comparison['ppo'][m]['mean'] for m in metrics]

        # Bar plot
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(metrics))
        width = 0.35

        ax.bar(x - width/2, supervised_means, width, label='Supervised', alpha=0.8)
        ax.bar(x + width/2, ppo_means, width, label='PPO', alpha=0.8)

        ax.set_xlabel('Metrics')
        ax.set_ylabel('Score')
        ax.set_title('Supervised vs PPO Ranking Model Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'comparison_barplot.png'), dpi=300)
        print(f"Bar plot saved to {output_dir}/comparison_barplot.png")

        # Improvement plot
        fig, ax = plt.subplots(figsize=(12, 6))
        improvements = [comparison['improvements'][m]['relative_percent'] for m in metrics]

        colors = ['green' if imp > 0 else 'red' for imp in improvements]
        ax.barh(metrics, improvements, color=colors, alpha=0.7)
        ax.axvline(0, color='black', linestyle='-', linewidth=0.8)
        ax.set_xlabel('Relative Improvement (%)')
        ax.set_title('PPO Improvement over Supervised Learning')
        ax.grid(axis='x', alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'improvement_plot.png'), dpi=300)
        print(f"Improvement plot saved to {output_dir}/improvement_plot.png")

        plt.close('all')


def load_test_data(dataset_path: str, max_samples: int = 100) -> List[Dict]:
    """
    Load test data for comparison

    Args:
        dataset_path: Path to test dataset or use demo examples
        max_samples: Maximum number of samples to use

    Returns:
        List of test examples
    """
    # For now, use demo examples
    # TODO: Load from actual dataset if path provided
    test_data = [
        {
            'query': "Why does ice float on water?",
            'explanations': [
                "Ice floats.",
                "Ice is lighter than water.",
                "Ice is less dense than water because water expands when it freezes.",
                "Water molecules form a hexagonal crystal structure when frozen, which is less dense than liquid water.",
                "The molecular structure of ice creates a lattice with more space between molecules.",
                "Because magic."
            ],
            'scores': [0.2, 0.5, 0.8, 1.0, 0.9, 0.0]
        },
        {
            'query': "What causes gravity?",
            'explanations': [
                "Mass attracts other mass through gravitational force.",
                "Heavy things pull on light things.",
                "Gravity is a fundamental force.",
                "Massive objects bend spacetime, and this curvature is what we experience as gravity according to Einstein's general relativity.",
                "Magic.",
                "According to Einstein's general relativity, gravity is caused by the curvature of spacetime by mass and energy."
            ],
            'scores': [0.6, 0.4, 0.4, 1.0, 0.0, 1.0]
        },
        {
            'query': "Why is the sky blue?",
            'explanations': [
                "Rayleigh scattering causes shorter wavelength blue light to scatter more in Earth's atmosphere.",
                "Blue light scatters more in the atmosphere because of its shorter wavelength.",
                "The atmosphere makes it blue.",
                "The sky scatters blue light more than red light.",
                "It reflects the ocean."
            ],
            'scores': [1.0, 0.8, 0.3, 0.6, 0.1]
        },
        {
            'query': "How do plants make food?",
            'explanations': [
                "Chloroplasts in plant cells convert sunlight, water, and carbon dioxide into glucose through photosynthesis, releasing oxygen as a byproduct.",
                "Through photosynthesis, plants use chlorophyll to capture light energy and convert CO2 and H2O into glucose.",
                "Photosynthesis converts sunlight to food.",
                "Plants use sunlight to make sugar from water and CO2.",
                "They grow food in their leaves.",
                "Plants eat sunlight."
            ],
            'scores': [1.0, 1.0, 0.6, 0.8, 0.3, 0.2]
        },
        {
            'query': "Why do seasons change?",
            'explanations': [
                "Earth's axial tilt of 23.5 degrees causes different hemispheres to receive varying amounts of sunlight throughout the year.",
                "The tilt of Earth's axis causes seasons as different parts receive more or less direct sunlight.",
                "The Earth moves closer and farther from the sun.",
                "Earth's tilt causes seasons.",
                "Because of temperature changes."
            ],
            'scores': [1.0, 0.8, 0.1, 0.5, 0.2]
        }
    ]

    return test_data[:max_samples]


def main():
    parser = argparse.ArgumentParser(description="Compare supervised vs PPO ranking models")
    parser.add_argument('--supervised_model', type=str, required=True,
                       help='Path to supervised model checkpoint')
    parser.add_argument('--ppo_model', type=str, default=None,
                       help='Path to PPO model checkpoint (optional if --use_mock_ppo)')
    parser.add_argument('--use_mock_ppo', action='store_true',
                       help='Use mock PPO for demonstration without training')
    parser.add_argument('--test_data', type=str, default=None,
                       help='Path to test dataset (optional, uses demo data by default)')
    parser.add_argument('--output_dir', type=str, default='comparison_results',
                       help='Directory to save comparison results')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to use (cpu, cuda, mps)')
    parser.add_argument('--max_samples', type=int, default=100,
                       help='Maximum number of test samples to use')

    args = parser.parse_args()

    # Validate arguments
    if not args.use_mock_ppo and args.ppo_model is None:
        parser.error("Must provide --ppo_model or use --use_mock_ppo")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize comparator
    print("Initializing model comparator...")
    comparator = ModelComparator(
        supervised_path=args.supervised_model,
        ppo_path=args.ppo_model,
        device=args.device,
        use_mock_ppo=args.use_mock_ppo
    )

    # Load test data
    print("Loading test data...")
    test_data = load_test_data(args.test_data, args.max_samples)
    print(f"Loaded {len(test_data)} test examples")

    # Run comparison
    comparison = comparator.compare_on_examples(test_data)

    # Print results
    comparator.print_comparison(comparison)

    # Save results
    output_json = os.path.join(args.output_dir, 'comparison_results.json')
    comparator.save_comparison(comparison, output_json)

    # Create plots
    comparator.plot_comparison(comparison, args.output_dir)

    print("\\n✅ Comparison complete!")
    print(f"Results saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
