#!/usr/bin/env python3
"""
Demo script to show ranking quality on diverse examples.
This shows how the model ranks different quality explanations.
"""

import torch
from ranking_models import RankingRewardModel
import argparse

def demo_ranking_examples(model_path, device='mps'):
    """Show ranking quality on diverse test examples"""

    # Load model
    print("Loading model...")
    checkpoint = torch.load(model_path, map_location=device)

    # Handle different checkpoint formats
    if 'config' in checkpoint:
        # final_model.pt format
        config = checkpoint['config']
    else:
        # best_model.pt format - use defaults
        print("Note: Using default config (checkpoint doesn't include config)")
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
    print("="*80)

    # Define diverse test cases with explanations of varying quality
    test_cases = [
        {
            'query': 'Why does ice float on water?',
            'explanations': [
                ('Ice is less dense than water because water expands when it freezes, creating a crystalline structure with more space between molecules.', 5),
                ('Water molecules form a hexagonal crystal structure when frozen, which is less dense than liquid water.', 5),
                ('Ice floats because it is lighter than water.', 3),
                ('Ice has lower density than water.', 3),
                ('It just does.', 1),
                ('Because gravity.', 1),
            ]
        },
        {
            'query': 'What causes gravity?',
            'explanations': [
                ('According to Einstein\'s general relativity, gravity is caused by the curvature of spacetime due to mass and energy.', 5),
                ('Massive objects bend spacetime, and this curvature is what we experience as gravitational attraction.', 5),
                ('Mass attracts other mass through gravitational force.', 3),
                ('Gravity is a fundamental force.', 2),
                ('Heavy things pull on light things.', 2),
                ('Magic.', 1),
            ]
        },
        {
            'query': 'Why is the sky blue?',
            'explanations': [
                ('Rayleigh scattering causes shorter wavelength blue light to scatter more than longer wavelengths when sunlight passes through the atmosphere.', 5),
                ('Blue light scatters more in the atmosphere because of its shorter wavelength.', 4),
                ('The sky scatters blue light more than red light.', 3),
                ('The atmosphere makes it blue.', 2),
                ('It reflects the ocean.', 1),
            ]
        },
        {
            'query': 'How do plants make food?',
            'explanations': [
                ('Through photosynthesis, plants use chlorophyll to capture light energy and convert CO2 and water into glucose and oxygen.', 5),
                ('Chloroplasts in plant cells convert sunlight, water, and carbon dioxide into sugar through photosynthesis.', 5),
                ('Plants use sunlight to make sugar from water and CO2.', 4),
                ('Photosynthesis converts sunlight to food.', 3),
                ('Plants eat sunlight.', 2),
                ('They grow food in their leaves.', 2),
            ]
        },
        {
            'query': 'Why do seasons change?',
            'explanations': [
                ('Earth\'s axial tilt of 23.5 degrees causes different hemispheres to receive varying amounts of solar radiation as Earth orbits the sun.', 5),
                ('The tilt of Earth\'s axis causes seasons as different parts receive more direct sunlight during different times of year.', 4),
                ('Earth\'s tilt causes seasons.', 3),
                ('The Earth moves closer and farther from the sun.', 1),
                ('Because of temperature changes.', 1),
            ]
        }
    ]

    print("\n" + "="*80)
    print("RANKING QUALITY DEMONSTRATION")
    print("="*80)

    total_correct_top1 = 0
    total_correct_top3 = 0
    total_queries = len(test_cases)

    with torch.no_grad():
        for i, test_case in enumerate(test_cases, 1):
            query = test_case['query']
            explanations_with_gold = test_case['explanations']

            print(f"\n{'='*80}")
            print(f"Query {i}: {query}")
            print('='*80)

            # Get model predictions
            explanations = [exp for exp, _ in explanations_with_gold]
            gold_scores = [score for _, score in explanations_with_gold]

            # Tokenize
            texts = [f"{query} {model.tokenizer.sep_token} {exp}" for exp in explanations]
            encoded = model.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors='pt'
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}

            # Get predictions
            pred_scores = model(
                input_ids=encoded['input_ids'],
                attention_mask=encoded['attention_mask']
            )
            pred_scores = pred_scores.cpu().numpy()

            # Rank by predicted scores
            ranked_indices = pred_scores.argsort()[::-1]  # High to low

            print("\nModel Ranking (Best to Worst):")
            print("-" * 80)

            for rank, idx in enumerate(ranked_indices, 1):
                exp = explanations[idx]
                pred_score = pred_scores[idx]
                gold_score = gold_scores[idx]

                # Truncate long explanations
                exp_short = exp if len(exp) <= 70 else exp[:67] + "..."

                # Color coding for quality
                if gold_score >= 5:
                    quality = "⭐⭐⭐⭐⭐ EXCELLENT"
                elif gold_score >= 4:
                    quality = "⭐⭐⭐⭐ GOOD"
                elif gold_score >= 3:
                    quality = "⭐⭐⭐ OKAY"
                elif gold_score >= 2:
                    quality = "⭐⭐ POOR"
                else:
                    quality = "⭐ VERY POOR"

                print(f"{rank}. [Predicted: {pred_score:.3f}] {quality}")
                print(f"   {exp_short}")

            # Check if top quality explanation is in top-k
            best_gold_score = max(gold_scores)
            best_gold_indices = [i for i, s in enumerate(gold_scores) if s == best_gold_score]

            top1_correct = ranked_indices[0] in best_gold_indices
            top3_correct = any(idx in best_gold_indices for idx in ranked_indices[:3])

            if top1_correct:
                total_correct_top1 += 1
            if top3_correct:
                total_correct_top3 += 1

            print(f"\n✓ Top-1 Accuracy: {'CORRECT' if top1_correct else 'INCORRECT'}")
            print(f"✓ Top-3 Accuracy: {'CORRECT' if top3_correct else 'INCORRECT'}")

    # Summary
    print("\n" + "="*80)
    print("OVERALL RANKING PERFORMANCE")
    print("="*80)
    print(f"Top-1 Accuracy: {total_correct_top1}/{total_queries} = {100*total_correct_top1/total_queries:.1f}%")
    print(f"  (Best explanation ranked #1)")
    print(f"\nTop-3 Accuracy: {total_correct_top3}/{total_queries} = {100*total_correct_top3/total_queries:.1f}%")
    print(f"  (Best explanation in top 3)")
    print("="*80)

    # Additional analysis
    print("\nInterpretation:")
    if total_correct_top1 >= total_queries * 0.8:
        print("✓ EXCELLENT: Model consistently ranks best explanations at #1")
    elif total_correct_top1 >= total_queries * 0.6:
        print("✓ GOOD: Model usually ranks best explanations highly")
    elif total_correct_top1 >= total_queries * 0.4:
        print("⚠ FAIR: Model sometimes ranks best explanations highly")
    else:
        print("✗ POOR: Model struggles to identify best explanations")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo ranking quality on diverse examples")
    parser.add_argument('--model_path', type=str, default='models/ranking_reward/best_model.pt',
                       help='Path to trained model checkpoint')
    parser.add_argument('--device', type=str, default='mps',
                       help='Device to use (cuda, mps, or cpu)')
    args = parser.parse_args()

    print("\n" + "="*80)
    print("RANKING MODEL QUALITY DEMONSTRATION")
    print("="*80)
    print(f"Model: {args.model_path}")
    print(f"Device: {args.device}")

    demo_ranking_examples(args.model_path, args.device)
