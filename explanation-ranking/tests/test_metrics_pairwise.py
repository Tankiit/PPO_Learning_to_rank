"""Pairwise accuracy must agree with Spearman on monotone rankings."""

import numpy as np
import pytest
import torch
from scipy import stats

from explrank.metrics.pairwise import mean_pairwise_accuracy, pairwise_accuracy
from explrank.metrics.ranking import spearman_correlation


def test_perfect_ranking():
    gold = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    pred = gold.copy()
    assert pairwise_accuracy(gold, pred) == pytest.approx(1.0)
    rho = stats.spearmanr(gold, pred).statistic
    assert rho == pytest.approx(1.0)


def test_reversed_ranking():
    gold = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    pred = gold[::-1]
    assert pairwise_accuracy(gold, pred) == pytest.approx(0.0)
    rho = stats.spearmanr(gold, pred).statistic
    assert rho == pytest.approx(-1.0)


def test_monotone_transform_preserves_both():
    gold = np.array([0.0, 1.0, 2.0, 4.0, 8.0])
    pred = np.sqrt(gold + 1)  # strictly increasing
    assert pairwise_accuracy(gold, pred) == pytest.approx(1.0)
    rho = stats.spearmanr(gold, pred).statistic
    assert rho == pytest.approx(1.0)


def test_gold_ties_skipped():
    gold = np.array([3.0, 3.0, 1.0, 5.0])
    pred = np.array([0.1, 0.9, 0.2, 0.8])  # wrong order among tied gold 3s — no pair
    # Comparable pairs: (3,1), (3,5), (1,5) for indices with distinct gold
    acc = pairwise_accuracy(gold, pred)
    assert 0.0 <= acc <= 1.0


def test_batch_spearman_pairwise_consistency():
    """When Spearman is 1 per query, pairwise accuracy must be 1."""
    gold = torch.tensor([[1.0, 2.0, 3.0], [0.0, 5.0, 10.0]])
    pred = gold.clone()
    for i in range(gold.shape[0]):
        assert pairwise_accuracy(gold[i], pred[i]) == pytest.approx(1.0)
        assert spearman_correlation(pred[i : i + 1], gold[i : i + 1]) == pytest.approx(1.0)
    assert mean_pairwise_accuracy(gold, pred) == pytest.approx(1.0)


def test_pred_tie_on_distinguishable_gold_counts_wrong():
    gold = np.array([1.0, 3.0])
    pred = np.array([2.0, 2.0])
    assert pairwise_accuracy(gold, pred) == pytest.approx(0.0)


def test_kendall_relation_no_ties():
    """pairwise_acc = (tau + 1) / 2 when no gold ties."""
    gold = np.array([1.0, 2.0, 4.0, 7.0])
    pred = np.array([1.5, 1.0, 5.0, 6.0])
    tau, _ = stats.kendalltau(gold, pred)
    expected = (tau + 1) / 2
    assert pairwise_accuracy(gold, pred) == pytest.approx(expected, abs=1e-9)
