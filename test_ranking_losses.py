#!/usr/bin/env python3
"""Test script for ranking losses and PPO reward computation"""

import torch
from ranking_models import RankingLosses, SimpleMockRankingModel
from evaluate_model import ComprehensiveEvaluator

def test_ranking_losses():
    """Test all ranking loss functions"""
    print("=" * 60)
    print("Testing Ranking Losses")
    print("=" * 60)

    # Sample data
    scores = torch.tensor([0.8, 0.6, 0.4, 0.2, 0.1])
    labels = torch.tensor([4.0, 3.0, 2.0, 1.0, 0.0])

    print(f"\nScores: {scores.tolist()}")
    print(f"Labels: {labels.tolist()}")

    # Test pairwise hinge loss
    print("\n1. Pairwise Hinge Loss")
    scores_pos = scores[:2]
    scores_neg = scores[3:5]
    loss = RankingLosses.pairwise_hinge_loss(scores_pos, scores_neg, margin=1.0)
    print(f"   Loss: {loss.item():.4f}")

    # Test weighted hinge loss
    print("\n2. Weighted Hinge Loss")
    weights = torch.tensor([1.0, 0.8])
    loss = RankingLosses.weighted_hinge_loss(scores_pos, scores_neg, weights, margin=1.0)
    print(f"   Loss: {loss.item():.4f}")

    # Test multi-margin hinge loss
    print("\n3. Multi-Margin Hinge Loss")
    loss = RankingLosses.multi_margin_hinge_loss(scores, labels)
    print(f"   Loss: {loss.item():.4f}")

    # Test RankNet loss
    print("\n4. RankNet Loss")
    scores_i = scores[:3]
    scores_j = scores[2:]
    labels_i = labels[:3]
    labels_j = labels[2:]
    loss = RankingLosses.ranknet_loss(scores_i, scores_j, labels_i, labels_j)
    print(f"   Loss: {loss.item():.4f}")

    # Test LambdaRank loss
    print("\n5. LambdaRank Loss")
    loss = RankingLosses.lambdarank_loss(scores, labels)
    print(f"   Loss: {loss.item():.4f}")

    # Test ListNet loss
    print("\n6. ListNet Loss")
    loss = RankingLosses.listnet_loss(scores, labels, temp=1.0)
    print(f"   Loss: {loss.item():.4f}")

    # Test ListMLE loss
    print("\n7. ListMLE Loss")
    loss = RankingLosses.listmle_loss(scores.unsqueeze(0), labels.unsqueeze(0))
    print(f"   Loss: {loss.item():.4f}")

    # Test ApproxNDCG loss
    print("\n8. ApproxNDCG Loss")
    loss = RankingLosses.approxndcg_loss(scores.unsqueeze(0), labels.unsqueeze(0))
    print(f"   Loss: {loss.item():.4f}")

    # Test SoftRank loss
    print("\n9. SoftRank Loss")
    loss = RankingLosses.softrank_loss(scores.unsqueeze(0), labels.unsqueeze(0))
    print(f"   Loss: {loss.item():.4f}")

    print("\n✅ All ranking losses computed successfully!")

def test_ppo_rewards():
    """Test PPO reward computation"""
    print("\n" + "=" * 60)
    print("Testing PPO Reward Computation")
    print("=" * 60)

    # Create mock model and evaluator
    model = SimpleMockRankingModel()
    evaluator = ComprehensiveEvaluator()

    # Sample queries and responses
    queries = [
        "If: 'The cat is on the mat' entails: 'There is a cat', why is that true?",
        "If: 'It is raining' contradicts: 'The sun is shining', why is that true?"
    ]

    responses = [
        [
            "The first statement explicitly mentions a cat, so a cat must exist.",
            "Because cats exist on mats.",
            "Cat on mat means cat.",
        ],
        [
            "Rain and sunshine are mutually exclusive weather conditions.",
            "Because it rains.",
            "Sun and rain don't mix.",
        ]
    ]

    print(f"\nQuery 1: {queries[0]}")
    print(f"Responses: {len(responses[0])}")
    for i, resp in enumerate(responses[0]):
        print(f"  {i+1}. {resp}")

    print(f"\nQuery 2: {queries[1]}")
    print(f"Responses: {len(responses[1])}")
    for i, resp in enumerate(responses[1]):
        print(f"  {i+1}. {resp}")

    # Compute rewards
    print("\nComputing rewards...")
    rewards = evaluator.compute_ranking_rewards_with_loss(queries, responses, model)

    print(f"\nRewards shape: {rewards.shape}")
    print(f"Rewards: {rewards.tolist()}")
    print(f"Mean reward: {rewards.mean().item():.4f}")
    print(f"Std reward: {rewards.std().item():.4f}")

    print("\n✅ PPO reward computation successful!")

def test_quality_estimates():
    """Test quality estimation"""
    print("\n" + "=" * 60)
    print("Testing Quality Estimation")
    print("=" * 60)

    evaluator = ComprehensiveEvaluator()

    query = "If: 'The dog is barking' entails: 'There is a dog', why is that true?"
    responses = [
        "The statement explicitly mentions a dog, so a dog must be present for it to bark.",
        "Because dogs bark.",
        "Dog barking means dog exists.",
        "Quantum physics explains this through entanglement."
    ]

    print(f"\nQuery: {query}")
    print("\nResponses and Quality Scores:")

    quality_scores = evaluator.get_quality_estimates(query, responses)

    for i, (resp, score) in enumerate(zip(responses, quality_scores)):
        print(f"\n{i+1}. Response: {resp}")
        print(f"   Quality Score: {score.item():.4f}")

    print(f"\nMean Quality: {quality_scores.mean().item():.4f}")
    print(f"Std Quality: {quality_scores.std().item():.4f}")

    print("\n✅ Quality estimation successful!")

if __name__ == "__main__":
    # Run all tests
    test_ranking_losses()
    test_ppo_rewards()
    test_quality_estimates()

    print("\n" + "=" * 60)
    print("All tests completed successfully! ✅")
    print("=" * 60)
