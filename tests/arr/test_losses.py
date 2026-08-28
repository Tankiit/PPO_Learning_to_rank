import unittest

import torch

from src.arr.losses import lambdarank_loss, listnet_loss, masked_mse, ranknet_loss


class RankingLossTests(unittest.TestCase):
    def setUp(self) -> None:
        self.targets = torch.tensor([[1.0, 0.6, 0.1, 0.0], [0.2, 0.8, 0.0, 0.0]])
        self.mask = torch.tensor([[True, True, True, False], [True, True, False, False]])

    def test_all_losses_are_finite_and_backpropagate(self) -> None:
        for function in (masked_mse, listnet_loss, ranknet_loss, lambdarank_loss):
            with self.subTest(loss=function.__name__):
                scores = torch.tensor(
                    [[0.4, 0.3, 0.2, 99.0], [0.4, 0.5, -99.0, 99.0]],
                    requires_grad=True,
                )
                loss = function(scores, self.targets, self.mask)
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                self.assertIsNotNone(scores.grad)
                self.assertTrue(torch.isfinite(scores.grad).all())
                self.assertEqual(float(scores.grad[0, 3]), 0.0)
                self.assertEqual(float(scores.grad[1, 2]), 0.0)

    def test_correct_order_beats_reversed_order(self) -> None:
        target = torch.tensor([[1.0, 0.5, 0.0]])
        mask = torch.ones_like(target, dtype=torch.bool)
        correct = torch.tensor([[0.9, 0.5, 0.1]])
        reversed_scores = torch.tensor([[0.1, 0.5, 0.9]])
        for function in (listnet_loss, ranknet_loss, lambdarank_loss):
            with self.subTest(loss=function.__name__):
                self.assertLess(float(function(correct, target, mask)), float(function(reversed_scores, target, mask)))


if __name__ == "__main__":
    unittest.main()
