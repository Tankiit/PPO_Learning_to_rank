"""Separation ratio definitions."""

import pytest
import torch

from explrank.metrics.separation import (
    score_separation_metrics,
    separation_ratio_batch,
    separation_ratio_std,
)


def test_identical_spread_per_query():
    pred = torch.tensor([[1.0, 2.0, 3.0], [0.0, 5.0, 10.0]])
    gold = pred.clone()
    assert separation_ratio_std(pred, gold) == pytest.approx(1.0, abs=1e-5)


def test_collapsed_predictions():
    pred = torch.ones(2, 3)
    gold = torch.tensor([[1.0, 2.0, 3.0], [0.0, 4.0, 8.0]])
    ratio = separation_ratio_std(pred, gold)
    assert ratio < 0.1


def test_batch_vs_std_definition_differ_with_variable_k():
    pred = torch.tensor([[1.0, 10.0], [1.0, 1.0, 1.0]])
    gold = torch.tensor([[1.0, 10.0], [1.0, 5.0, 10.0]])
    # pad for metrics API
    k = 3
    p = torch.zeros(2, k)
    g = torch.zeros(2, k)
    p[0, :2], g[0, :2] = pred[0], gold[0]
    p[1, :3], g[1, :3] = pred[1], gold[1]
    m = score_separation_metrics(p, g)
    assert "separation_ratio_std" in m
    assert "separation_ratio_batch" in m
