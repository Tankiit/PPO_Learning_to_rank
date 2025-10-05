"""
Visualize Score Distribution and Discrimination

This script analyzes and visualizes:
1. Score distribution (histogram)
2. Score compression analysis
3. Ranking quality metrics
4. Per-query discrimination

Usage:
    python visualize_score_distribution.py --model_path models/ranking_reward/best_model.pt
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict
import os
from tqdm import tqdm
from collections import defaultdict

from ranking_models import RankingRewardModel


class ScoreDistributionAnalyzer:
    """Analyze and visualize score distribution from ranking model"""

    def __init__(self, model_path: str, device: str = 'cpu'):
        self.device = device
        self.model = self.load_model(model_path)

    def load_model(self, path: str) -> RankingRewardModel:
        """Load trained ranking model"""
        checkpoint = torch.load(path, map_location=self.device)

        if 'config' in checkpoint:
            config = checkpoint['config']
            base_model = config.get('base_model', 'bert-base-uncased')
            dropout = config.get('dropout', 0.1)
        else:
            base_model = 'bert-base-uncased'
            dropout = 0.1

        model = RankingRewardModel(
            base_model=base_model,
            output_mode="regression",
            dropout=dropout
        )

        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()

        return model

    def get_test_data(self) -> List[Dict]:
        """Generate test queries with varying quality explanations"""
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
                'true_scores': [0.2, 0.5, 0.8, 1.0, 0.9, 0.0]
            },
            {
                'query': "What causes gravity?",
                'explanations': [
                    "Mass attracts other mass through gravitational force.",
                    "Heavy things pull on light things.",
                    "Gravity is a fundamental force.",
                    "Massive objects bend spacetime, and this curvature is what we experience as gravity.",
                    "Magic.",
                    "According to Einstein's general relativity, gravity results from spacetime curvature."
                ],
                'true_scores': [0.6, 0.4, 0.4, 1.0, 0.0, 1.0]
            },
            {
                'query': "Why is the sky blue?",
                'explanations': [
                    "Rayleigh scattering causes shorter wavelength blue light to scatter more.",
                    "Blue light scatters more in the atmosphere.",
                    "The atmosphere makes it blue.",
                    "The sky scatters blue light more than red light.",
                    "It reflects the ocean."
                ],
                'true_scores': [1.0, 0.8, 0.3, 0.6, 0.1]
            },
            {
                'query': "How do plants make food?",
                'explanations': [
                    "Chloroplasts convert sunlight, water, and CO2 into glucose through photosynthesis.",
                    "Through photosynthesis, plants use chlorophyll to capture light energy.",
                    "Photosynthesis converts sunlight to food.",
                    "Plants use sunlight to make sugar from water and CO2.",
                    "They grow food in their leaves.",
                    "Plants eat sunlight."
                ],
                'true_scores': [1.0, 1.0, 0.6, 0.8, 0.3, 0.2]
            },
            {
                'query': "Why do seasons change?",
                'explanations': [
                    "Earth's axial tilt of 23.5 degrees causes different hemispheres to receive varying sunlight.",
                    "The tilt of Earth's axis causes seasons.",
                    "The Earth moves closer and farther from the sun.",
                    "Earth's tilt causes seasons.",
                    "Because of temperature changes."
                ],
                'true_scores': [1.0, 0.8, 0.1, 0.5, 0.2]
            }
        ]
        return test_data

    def analyze_scores(self, test_data: List[Dict]) -> Dict:
        """Analyze score distribution across test data"""
        all_pred_scores = []
        all_true_scores = []
        per_query_results = []

        print("\nAnalyzing score distribution...")

        for example in tqdm(test_data):
            query = example['query']
            explanations = example['explanations']
            true_scores = example['true_scores']

            # Get predicted scores
            pred_scores = self.model.rank_explanations(query, explanations, return_scores=True)

            all_pred_scores.extend(pred_scores)
            all_true_scores.extend(true_scores)

            # Per-query analysis
            per_query_results.append({
                'query': query,
                'pred_scores': pred_scores,
                'true_scores': true_scores,
                'pred_range': max(pred_scores) - min(pred_scores),
                'pred_std': np.std(pred_scores),
                'true_range': max(true_scores) - min(true_scores),
                'true_std': np.std(true_scores)
            })

        # Overall statistics
        results = {
            'all_pred_scores': np.array(all_pred_scores),
            'all_true_scores': np.array(all_true_scores),
            'per_query': per_query_results,
            'statistics': {
                'pred_mean': np.mean(all_pred_scores),
                'pred_std': np.std(all_pred_scores),
                'pred_min': np.min(all_pred_scores),
                'pred_max': np.max(all_pred_scores),
                'pred_range': np.max(all_pred_scores) - np.min(all_pred_scores),
                'true_mean': np.mean(all_true_scores),
                'true_std': np.std(all_true_scores),
                'true_range': np.max(all_true_scores) - np.min(all_true_scores),
                'compression_ratio': (np.max(all_pred_scores) - np.min(all_pred_scores)) /
                                    (np.max(all_true_scores) - np.min(all_true_scores))
            }
        }

        return results

    def print_analysis(self, results: Dict):
        """Print analysis results"""
        stats = results['statistics']

        print("\n" + "="*80)
        print("SCORE DISTRIBUTION ANALYSIS")
        print("="*80)

        print("\nOverall Statistics:")
        print("-"*80)
        print(f"{'Metric':<30} {'Predicted':<25} {'True':<25}")
        print("-"*80)
        print(f"{'Mean':<30} {stats['pred_mean']:.6f}                {stats['true_mean']:.6f}")
        print(f"{'Std Dev':<30} {stats['pred_std']:.6f}                {stats['true_std']:.6f}")
        print(f"{'Min':<30} {stats['pred_min']:.6f}                {0.0:.6f}")
        print(f"{'Max':<30} {stats['pred_max']:.6f}                {1.0:.6f}")
        print(f"{'Range':<30} {stats['pred_range']:.6f}                {stats['true_range']:.6f}")
        print("-"*80)

        # Compression analysis
        compression = stats['compression_ratio']
        print(f"\n📊 Compression Ratio: {compression:.4f}")
        if compression < 0.1:
            print("   ⚠️  SEVERE compression - scores are too similar!")
            print("   Recommendation: Use ranking losses (listnet, ranknet) or PPO")
        elif compression < 0.3:
            print("   ⚠️  Moderate compression - some discrimination but not ideal")
            print("   Recommendation: Consider ranking losses for better separation")
        else:
            print("   ✅ Good score discrimination")

        # Per-query analysis
        print("\n\nPer-Query Score Discrimination:")
        print("-"*80)
        print(f"{'Query':<50} {'Pred Range':<15} {'Pred Std':<15}")
        print("-"*80)

        for query_result in results['per_query']:
            query = query_result['query'][:47] + "..." if len(query_result['query']) > 47 else query_result['query']
            print(f"{query:<50} {query_result['pred_range']:.6f}        {query_result['pred_std']:.6f}")

        print("-"*80)
        avg_range = np.mean([q['pred_range'] for q in results['per_query']])
        avg_std = np.mean([q['pred_std'] for q in results['per_query']])
        print(f"{'Average':<50} {avg_range:.6f}        {avg_std:.6f}")
        print("="*80 + "\n")

    def visualize(self, results: Dict, output_dir: str):
        """Create visualizations"""
        os.makedirs(output_dir, exist_ok=True)

        pred_scores = results['all_pred_scores']
        true_scores = results['all_true_scores']

        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (15, 10)

        # Create 2x2 subplot
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 1. Score Distribution Histogram
        ax = axes[0, 0]
        ax.hist(pred_scores, bins=30, alpha=0.7, label='Predicted', color='blue', edgecolor='black')
        ax.hist(true_scores, bins=30, alpha=0.5, label='True', color='orange', edgecolor='black')
        ax.set_xlabel('Score')
        ax.set_ylabel('Frequency')
        ax.set_title('Score Distribution Comparison')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        # 2. Predicted vs True Scatter
        ax = axes[0, 1]
        ax.scatter(true_scores, pred_scores, alpha=0.6, s=50)
        ax.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect Prediction')
        ax.set_xlabel('True Score')
        ax.set_ylabel('Predicted Score')
        ax.set_title('Predicted vs True Scores')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. Box plot per query
        ax = axes[1, 0]
        per_query = results['per_query']
        pred_data = [q['pred_scores'] for q in per_query]
        true_data = [q['true_scores'] for q in per_query]

        positions = np.arange(len(per_query))
        bp1 = ax.boxplot(pred_data, positions=positions - 0.2, widths=0.3,
                         patch_artist=True, boxprops=dict(facecolor='blue', alpha=0.6),
                         medianprops=dict(color='darkblue', linewidth=2))
        bp2 = ax.boxplot(true_data, positions=positions + 0.2, widths=0.3,
                         patch_artist=True, boxprops=dict(facecolor='orange', alpha=0.6),
                         medianprops=dict(color='darkorange', linewidth=2))

        ax.set_xlabel('Query')
        ax.set_ylabel('Score Range')
        ax.set_title('Score Distribution Per Query')
        ax.set_xticks(positions)
        ax.set_xticklabels([f"Q{i+1}" for i in range(len(per_query))])
        ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ['Predicted', 'True'])
        ax.grid(axis='y', alpha=0.3)

        # 4. Compression visualization
        ax = axes[1, 1]
        queries = [f"Q{i+1}" for i in range(len(per_query))]
        pred_ranges = [q['pred_range'] for q in per_query]
        true_ranges = [q['true_range'] for q in per_query]

        x = np.arange(len(queries))
        width = 0.35

        ax.bar(x - width/2, pred_ranges, width, label='Predicted Range', alpha=0.8, color='blue')
        ax.bar(x + width/2, true_ranges, width, label='True Range', alpha=0.8, color='orange')

        ax.set_xlabel('Query')
        ax.set_ylabel('Score Range')
        ax.set_title('Score Range Comparison (Compression Analysis)')
        ax.set_xticks(x)
        ax.set_xticklabels(queries)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        output_path = os.path.join(output_dir, 'score_distribution_analysis.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Visualization saved to {output_path}")

        # Additional plot: Detailed per-query comparison
        fig, axes = plt.subplots(len(per_query), 1, figsize=(12, 3*len(per_query)))
        if len(per_query) == 1:
            axes = [axes]

        for idx, query_result in enumerate(per_query):
            ax = axes[idx]
            explanations = [f"E{i+1}" for i in range(len(query_result['pred_scores']))]

            x = np.arange(len(explanations))
            width = 0.35

            ax.bar(x - width/2, query_result['pred_scores'], width,
                  label='Predicted', alpha=0.8, color='blue')
            ax.bar(x + width/2, query_result['true_scores'], width,
                  label='True', alpha=0.8, color='orange')

            ax.set_ylabel('Score')
            ax.set_title(f"Query {idx+1}: {query_result['query'][:60]}...")
            ax.set_xticks(x)
            ax.set_xticklabels(explanations)
            ax.legend()
            ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        output_path = os.path.join(output_dir, 'per_query_detailed.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Detailed per-query plot saved to {output_path}")

        plt.close('all')


def main():
    parser = argparse.ArgumentParser(description="Visualize score distribution and compression")
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--output_dir', type=str, default='score_analysis',
                       help='Directory to save visualizations')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to use (cpu, cuda, mps)')

    args = parser.parse_args()

    print("="*80)
    print("SCORE DISTRIBUTION ANALYZER")
    print("="*80)
    print(f"Model: {args.model_path}")
    print(f"Device: {args.device}")
    print("="*80)

    # Initialize analyzer
    analyzer = ScoreDistributionAnalyzer(args.model_path, args.device)

    # Get test data
    test_data = analyzer.get_test_data()

    # Analyze scores
    results = analyzer.analyze_scores(test_data)

    # Print analysis
    analyzer.print_analysis(results)

    # Create visualizations
    analyzer.visualize(results, args.output_dir)

    print(f"\n✅ Analysis complete! Results saved to {args.output_dir}/")
    print("\nRecommendations:")

    compression = results['statistics']['compression_ratio']
    if compression < 0.1:
        print("1. ⚠️  SEVERE score compression detected")
        print("2. Try training with ranking losses:")
        print("   python train_ranking_model.py --loss_function listnet")
        print("3. Or train with PPO:")
        print("   python train_ppo_ranking.py")
        print("4. Or use hybrid approach:")
        print("   python train_hybrid_ranking.py")
    elif compression < 0.3:
        print("1. ⚠️  Moderate score compression")
        print("2. Consider ranking losses for better discrimination")
        print("3. Hybrid training may help:")
        print("   python train_hybrid_ranking.py --loss_function listnet")
    else:
        print("1. ✅ Good score discrimination!")
        print("2. Your model is working well")


if __name__ == "__main__":
    main()
