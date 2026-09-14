import unittest

import torch
import numpy as np

from src.arr.epistemic import (
    combine_independent_member_records,
    center_listwise_member_logits,
    build_diverse_scalar_head,
    credence_dropout_rates,
    evaluate_epistemic_predictions,
    make_epistemic_record,
    summarize_head_scores,
)
from src.arr.losses import listnet_loss, masked_mse
from src.arr.schema import Candidate, RankingGroup, ScoreRecord
from src.arr.training import _multihead_ranking_loss, bootstrap_member_counts
from src.arr.epistemic_diagnostics import (
    _partial_spearman,
    effective_ensemble_size,
    mean_kl_to_consensus,
    head_redundancy,
)


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
    def test_partial_spearman_removes_a_shared_confidence_mediator(self) -> None:
        rng = np.random.default_rng(123)
        confidence = rng.normal(size=5000)
        width = confidence + 0.2 * rng.normal(size=5000)
        error = confidence + 0.2 * rng.normal(size=5000)
        marginal = __import__("scipy.stats", fromlist=["spearmanr"]).spearmanr(
            width, error
        ).correlation
        partial = _partial_spearman(width, error, confidence)
        self.assertGreater(marginal, 0.8)
        self.assertLess(abs(partial), 0.05)

    def test_independent_member_records_are_aligned_and_combined(self) -> None:
        def record(candidate_id: str, score: float, seed: int) -> ScoreRecord:
            return ScoreRecord(
                group_id="q1",
                candidate_id=candidate_id,
                model_name=f"member-{seed}",
                model_revision="main",
                prompt_hash="prompt",
                data_fingerprint="fingerprint",
                score=score,
                raw_output=str(score),
                parsing_status="ok",
                seed=seed,
                inference_ms=1.0,
            )

        first = [record("good", 0.8, 42), record("bad", 0.2, 42)]
        second = [record("bad", 0.4, 43), record("good", 0.6, 43)]
        combined = combine_independent_member_records([first, second])
        self.assertEqual(combined[0].metadata["head_scores"], [0.8, 0.6])
        self.assertAlmostEqual(combined[0].score, 0.7)
        self.assertEqual(combined[0].metadata["method"], "independent_backbone_ensemble")

    def test_listwise_gauge_alignment_preserves_order_and_removes_offset(self) -> None:
        def records(offset: float) -> list[ScoreRecord]:
            logits = [offset + 2.0, offset - 1.0]
            return [
                ScoreRecord(
                    group_id="q1",
                    candidate_id=candidate,
                    model_name="member",
                    model_revision="main",
                    prompt_hash="prompt",
                    data_fingerprint="fingerprint",
                    score=float(torch.sigmoid(torch.tensor(logit, dtype=torch.float64))),
                    raw_output="",
                    parsing_status="ok",
                    seed=42,
                    inference_ms=0.0,
                )
                for candidate, logit in zip(("good", "bad"), logits)
            ]

        low = center_listwise_member_logits(records(-8.0))
        high = center_listwise_member_logits(records(8.0))
        self.assertGreater(low[0].score, low[1].score)
        self.assertTrue(
            np.allclose([row.score for row in low], [row.score for row in high], atol=1e-5)
        )

    def test_effective_ensemble_size_detects_collapse_and_independence(self) -> None:
        rng = np.random.default_rng(42)
        shared = rng.normal(size=(400, 1))
        collapsed = np.repeat(shared, 5, axis=1)
        independent = rng.normal(size=(4000, 5))
        self.assertAlmostEqual(effective_ensemble_size(collapsed), 1.0, places=6)
        self.assertGreater(effective_ensemble_size(independent), 4.8)
        self.assertTrue(head_redundancy(collapsed)["redundant"])

    def test_effective_size_ignores_shared_irreducible_error(self) -> None:
        rng = np.random.default_rng(7)
        shared_error = 20.0 * rng.normal(size=(5000, 1))
        independent_disagreement = rng.normal(size=(5000, 5))
        residuals = shared_error + independent_disagreement
        self.assertGreater(effective_ensemble_size(residuals), 4.8)

    def test_kl_diversity_detects_magnitude_shrink(self) -> None:
        rng = np.random.default_rng(3)
        logits = rng.normal(size=(40, 5, 5))
        probs = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs /= probs.sum(axis=1, keepdims=True)
        consensus = probs.mean(axis=2, keepdims=True)
        shrunk = consensus + 0.02 * (probs - consensus)
        full = mean_kl_to_consensus(probs)
        small = mean_kl_to_consensus(shrunk)
        self.assertLess(small, full * 0.01)

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

    def test_fixed_feature_bags_are_reproducible_and_member_specific(self) -> None:
        left, _ = build_diverse_scalar_head(
            32, 5, hidden_dim=4, feature_keep_fraction=0.5, feature_seed=7
        )
        right, _ = build_diverse_scalar_head(
            32, 5, hidden_dim=4, feature_keep_fraction=0.5, feature_seed=7
        )
        self.assertTrue(torch.equal(left.feature_masks, right.feature_masks))
        self.assertFalse(torch.equal(left.feature_masks[0], left.feature_masks[1]))
        self.assertTrue((left.feature_masks.sum(dim=1) > 0).all())

    def test_group_bootstrap_is_exact_and_reproducible(self) -> None:
        groups = [_group()]
        for index in range(1, 20):
            group = _group()
            groups.append(
                RankingGroup(
                    group_id=f"q{index + 1}",
                    split=group.split,
                    domain=group.domain,
                    question=group.question,
                    candidates=group.candidates,
                    data_fingerprint=group.data_fingerprint,
                )
            )
        first = bootstrap_member_counts(groups, 5, seed=42, epoch=0)
        again = bootstrap_member_counts(groups, 5, seed=42, epoch=0)
        next_epoch = bootstrap_member_counts(groups, 5, seed=42, epoch=1)
        self.assertEqual(first, again)
        self.assertNotEqual(first, next_epoch)
        for head in range(5):
            self.assertEqual(sum(first[g.group_id][head] for g in groups), len(groups))

    def test_single_backbone_bootstrap_weights_its_group_losses(self) -> None:
        scores = torch.tensor([[2.0, 0.0], [0.0, 2.0]], requires_grad=True)
        targets = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        mask = torch.ones_like(targets, dtype=torch.bool)
        first_only = _multihead_ranking_loss(
            listnet_loss,
            scores,
            targets,
            mask,
            member_weights=torch.tensor([[2.0], [0.0]]),
        )
        expected = listnet_loss(scores[:1], targets[:1], mask[:1])
        self.assertTrue(torch.allclose(first_only, expected))

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

    def test_listnet_uses_raw_logit_scale_without_sigmoid_saturation(self) -> None:
        targets = torch.tensor([[1.0, 0.5, 0.0]])
        mask = torch.ones_like(targets, dtype=torch.bool)
        logits = torch.tensor(
            [[[-20.0, 20.0], [0.0, 0.0], [20.0, -20.0]]],
            requires_grad=True,
        )
        loss = _multihead_ranking_loss(listnet_loss, logits, targets, mask)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(logits.grad.abs().sum()), 0.1)

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
