"""Ranking losses return finite scalars on toy batches."""

import pytest
import torch

from explrank.losses.ranking import get_loss_function


@pytest.fixture
def batch():
    pred = torch.tensor([[0.2, 0.5, 0.8, 0.3]], requires_grad=True)
    true = torch.tensor([[1.0, 2.0, 4.0, 3.0]])
    return pred, true



@pytest.mark.parametrize("name", ["mse", "listnet", "ranknet", "approxndcg", "lambdarank"])
def test_loss_forward(name, batch):
    pred, true = batch
    loss_fn = get_loss_function(name)
    loss = loss_fn(pred, true)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()
    assert pred.grad is not None
