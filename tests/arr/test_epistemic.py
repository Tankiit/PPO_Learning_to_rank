import unittest

import torch

from src.arr.epistemic import (
    build_diverse_scalar_head,
    credence_dropout_rates,
    evaluate_epistemic_predictions,
    make_epistemic_record,
    summarize_head_scores,
)
from src.arr.losses import listnet_loss, masked_mse
from src.arr.schema import Candidate, RankingGroup
from src.arr.training import _multihead_ranking_loss


def _group() -> RankingGroup:
    return RankingGroup(
        group_id="q1",
        split="test",
        domain="nli",
        question="Why?",
        candidates=(
            Candidate("good", "Good explanation", 1.0, "human"),
            Candidate("bad", "Bad explanation", 0.0, "human"),
        ),
        data_fingerprint="fingerprint",
    )


class EpistemicScalarTests(unittest.TestCase):
    def test_credence_dropout_schedule_has_exact_endpoints(self) -> None:
        rates = credence_dropout_rates(5, 0.05, 0.30)
        self.assertEqual(len(rates), 5)
        self.assertAlmostEqual(rates[0], 0.30)
        self.assertAlmostEqual(rates[-1], 0.05)
        self.assertTrue(all(left > right for left, right in zip(rates, rates[1:])))

    def test_diverse_head_returns_one_logit_per_head(self) -> None:
        head, rates = build_diverse_scalar_head(8, 5, hidden_dim=4)
        logits = head(torch.randn(2, 3, 8))
        self.assertEqual(tuple(logits.shape), (2, 3, 5))
        self.assertEqual(len(rates), 5)

    def test_summary_uses_mean_min_max_and_population_variance(self) -> None:
        summary = summarize_head_scores([0.1, 0.3, 0.5])
        self.assertAlmostEqual(summary["score_mean"], 0.3)
        self.assertAlmostEqual(summary["credal_lower"], 0.1)
        self.assertAlmostEqual(summary["credal_upper"], 0.5)
        self.assertAlmostEqual(summary["credal_width"], 0.4)
        self.assertAlmostEqual(summary["epistemic_variance"], 0.02666666666666667)

    def test_each_head_receives_the_selected_ranking_loss_and_gradients(self) -> None:
        targets = torch.tensor([[1.0, 0.5, 0.0]])
        mask = torch.ones_like(targets, dtype=torch.bool)
        for loss_function in (masked_mse, listnet_loss):
            logits = torch.randn(1, 3, 5, requires_grad=True)
            scores = torch.sigmoid(logits)
            loss = _multihead_ranking_loss(loss_function, scores, targets, mask)
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertIsNotNone(logits.grad)
            self.assertTrue(torch.isfinite(logits.grad).all())
            self.assertTrue((logits.grad.abs().sum(dim=(0, 1)) > 0).all())

    def test_epistemic_metrics_include_pairwise_credal_order(self) -> None:
        group = _group()
        records = [
            make_epistemic_record(
                group=group,
                candidate_id="good",
                head_scores=[0.6, 0.7, 0.8],
                model_name="judge",
                model_revision="rev",
                seed=42,
                inference_ms=1.0,
            ),
            make_epistemic_record(
                group=group,
                candidate_id="bad",
                head_scores=[0.2, 0.3, 0.4],
                model_name="judge",
                model_revision="rev",
                seed=42,
                inference_ms=1.0,
            ),
        ]
        metrics = evaluate_epistemic_predictions([group], records)
        self.assertEqual(metrics["head_count"], 3)
        self.assertEqual(metrics["candidate_count"], 2)
        self.assertEqual(
            metrics["aggregate"]["pairwise_correct_nonoverlap_rate"], 1.0
        )
        self.assertEqual(len(metrics["individual_head_metrics"]), 3)

    def test_missing_candidate_is_never_imputed(self) -> None:
        group = _group()
        record = make_epistemic_record(
            group=group,
            candidate_id="good",
            head_scores=[0.6, 0.7, 0.8],
            model_name="judge",
            model_revision="rev",
            seed=42,
            inference_ms=1.0,
        )
        with self.assertRaisesRegex(ValueError, "missing epistemic prediction"):
            evaluate_epistemic_predictions([group], [record])


if __name__ == "__main__":
    unittest.main()
