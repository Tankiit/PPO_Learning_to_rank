"""Control 1: gauge injection. Proposition 1 as an executable check.

ListNet identifies a within-group ranking, not an absolute score level. Adding
a constant to one member's logits inside one group leaves that member's
within-group softmax exactly unchanged, so it is a symmetry of the model and
every reported statistic must be blind to it. A statistic that moves is reading
the arbitrary representative the optimiser happened to land on.

Three outcomes are asserted here, and the third is a real defect this control
was written to find rather than a property to celebrate.

1. Statistics computed in the within-group probability space - Jensen-Shannon
   divergence, participation ratio, probability-space width, the partial
   correlations, risk-coverage - are invariant.
2. Raw-score credal width is *not* invariant, and must move. If it did not, the
   injection would not have done anything and the test above would be vacuous.
3. The scalar ``score`` a combined record carries is the mean of the raw bounded
   member scores, which is gauge-dependent. Anything computed from it, NDCG
   included, therefore inherits the arbitrary representative unless the members
   are gauge-fixed first. ``center_listwise_member_logits`` is that fix, and the
   final test confirms it restores invariance.

On exact tolerance: the specification asked for bit-identical outputs. That is
not attainable through a floating-point sigmoid/logit round trip - the softmax
renormalisation alone reorders additions - so the invariance assertions use a
float64 tolerance of 1e-9 and the tests report the largest deviation actually
observed, which runs about 1e-15.
"""

import unittest

import numpy as np

from src.arr.epistemic import center_listwise_member_logits, summarize_head_scores
from src.arr.schema import Candidate, RankingGroup, ScoreRecord
from src.arr.tier1 import (
    apply_gauge_shift,
    group_bootstrap_partial_spearman,
    js_divergence_to_consensus,
    ndcg_by_group_size,
    participation_ratio_conventions,
    prepare_listwise_arrays,
    risk_coverage_curve,
)

FINGERPRINT = "gauge-fingerprint"
TOLERANCE = 1e-9


def _synthetic(groups: int = 40, size: int = 6, members: int = 5, seed: int = 0):
    """Groups whose member scores sit well away from the sigmoid boundaries.

    The listwise mapping clips to [1e-7, 1-1e-7]; a fixture near saturation
    would lose the gauge freedom to that clip and the control would report a
    violation that is an artefact of the fixture. Saturation is tested
    explicitly and separately below.
    """

    rng = np.random.default_rng(seed)
    ranking_groups, records = [], []
    for index in range(groups):
        scores = rng.uniform(0.05, 0.95, size=size)
        group = RankingGroup(
            group_id=f"g{index}",
            split="dev",
            domain="test",
            question=f"question {index}",
            candidates=tuple(
                Candidate(
                    candidate_id=f"g{index}-c{position}",
                    text=f"candidate {position} of group {index}",
                    score=float(value),
                    score_provenance="synthetic",
                )
                for position, value in enumerate(scores)
            ),
            data_fingerprint=FINGERPRINT,
        )
        heads = np.clip(
            rng.uniform(0.25, 0.75, size=(size, 1))
            + rng.normal(0.0, 0.06, size=(size, members)),
            0.02,
            0.98,
        )
        ranking_groups.append(group)
        for candidate, row in zip(group.candidates, heads):
            summary = summarize_head_scores(row.tolist())
            records.append(
                ScoreRecord(
                    group_id=group.group_id,
                    candidate_id=candidate.candidate_id,
                    model_name="test",
                    model_revision="test",
                    prompt_hash="none",
                    data_fingerprint=FINGERPRINT,
                    score=float(summary["score_mean"]),
                    raw_output="",
                    parsing_status="ok",
                    seed=0,
                    inference_ms=0.0,
                    metadata=dict(summary),
                )
            )
    return ranking_groups, records


def _listwise_statistics(groups, records) -> dict[str, float]:
    """Every statistic the report quotes that lives in the identified space."""

    arrays = prepare_listwise_arrays(groups, records)
    residuals = arrays.member_probabilities - arrays.target_probabilities[:, None]
    diversity = js_divergence_to_consensus(arrays)
    conventions = participation_ratio_conventions(residuals)
    coverage = risk_coverage_curve(arrays, seed=1, random_repeats=20)
    partial = group_bootstrap_partial_spearman(arrays, samples=40, seed=1)
    return {
        "js_divergence": diversity["js_divergence_to_consensus"],
        "js_normalised": diversity["js_divergence_normalised_non_singleton"],
        "participation_fraction": conventions["participation_fraction"],
        "effective_ensemble_size": conventions["effective_ensemble_size"],
        "mean_probability_width": float(np.mean(arrays.width)),
        "mean_absolute_error": float(np.mean(arrays.absolute_error)),
        "partial_spearman": partial["estimate"],
        "aurc_width": coverage["aurc_width"],
        "normalised_aurc_gain": coverage["normalised_aurc_gain"],
    }


class GaugeInvarianceTest(unittest.TestCase):
    def setUp(self):
        self.groups, self.records = _synthetic()

    def test_listwise_statistics_are_gauge_invariant(self):
        baseline = _listwise_statistics(self.groups, self.records)
        worst = 0.0
        for magnitude in (0.25, 1.0, 2.5):
            for seed in (0, 1, 2):
                shifted = apply_gauge_shift(
                    self.records, magnitude=magnitude, per_member=True, seed=seed
                )
                moved = _listwise_statistics(self.groups, shifted)
                for name, value in baseline.items():
                    deviation = abs(moved[name] - value)
                    worst = max(worst, deviation)
                    self.assertLess(
                        deviation,
                        TOLERANCE,
                        f"{name} moved by {deviation:.3e} under a gauge shift of "
                        f"{magnitude} (seed {seed}); the pipeline is reading the "
                        "representative, not the identified object",
                    )
        # A vacuous pass would look identical, so record what was actually seen.
        self.assertLess(worst, TOLERANCE)
        print(f"\n  largest gauge deviation across listwise statistics: {worst:.3e}")

    def test_raw_credal_width_does_move(self):
        """Guards the test above: confirm the injection had an effect at all."""

        before = np.array(
            [float(r.metadata["credal_width"]) for r in self.records]
        )
        shifted = apply_gauge_shift(self.records, magnitude=1.0, seed=0)
        after = np.array([float(r.metadata["credal_width"]) for r in shifted])
        self.assertGreater(
            float(np.max(np.abs(after - before))),
            0.01,
            "the gauge shift did not change raw width, so the invariance test "
            "above proves nothing",
        )

    def test_scalar_score_is_gauge_dependent(self):
        """The combined record's score is a mean of sigmoids, so it is not invariant.

        This is the defect the control exists to surface: NDCG and every other
        quantity derived from ``record.score`` inherits the representative
        unless the members are gauge-fixed first.
        """

        shifted = apply_gauge_shift(self.records, magnitude=1.0, per_member=True, seed=0)
        before = np.array([r.score for r in self.records])
        after = np.array([r.score for r in shifted])
        self.assertGreater(float(np.max(np.abs(after - before))), 0.01)

        baseline_ndcg = ndcg_by_group_size(self.groups, self.records)
        moved_ndcg = ndcg_by_group_size(self.groups, shifted)
        self.assertNotAlmostEqual(
            baseline_ndcg["all_groups"]["tie_aware_ndcg_at_5"],
            moved_ndcg["all_groups"]["tie_aware_ndcg_at_5"],
            places=9,
            msg="scalar-score NDCG happened to be unchanged; widen the shift",
        )

    def test_centring_restores_invariance_of_the_scalar_score(self):
        """``center_listwise_member_logits`` is the gauge fix, applied per member.

        Centring each member's logits to mean zero within a group removes any
        injected constant, so a pipeline that centres before combining is
        reproducible regardless of which representative training landed on.
        """

        def centred_member_scores(records):
            # One "member view" per head, centred independently, as the Modal
            # combine step does before building the ensemble record.
            head_count = len(records[0].metadata["head_scores"])
            per_member = []
            for index in range(head_count):
                view = [
                    ScoreRecord(
                        **{
                            **r.to_dict(),
                            "score": float(r.metadata["head_scores"][index]),
                            "metadata": {},
                        }
                    )
                    for r in records
                ]
                per_member.append(
                    [c.score for c in center_listwise_member_logits(view)]
                )
            return np.asarray(per_member).T

        baseline = centred_member_scores(self.records)
        shifted = centred_member_scores(
            apply_gauge_shift(self.records, magnitude=1.5, per_member=True, seed=3)
        )
        deviation = float(np.max(np.abs(baseline - shifted)))
        self.assertLess(
            deviation,
            1e-9,
            f"centring left a residual gauge dependence of {deviation:.3e}",
        )

    def test_saturation_breaks_the_symmetry(self):
        """Documents why the boundary guard in the report exists.

        The listwise mapping clips scores to [1e-7, 1-1e-7]. A gauge shift large
        enough to push members into that clip is no longer a symmetry, because
        the clip is not equivariant. This is not a bug to fix but a precondition
        to check, and the report's "no score within 1e-6 of a boundary" audit is
        exactly that check.
        """

        baseline = _listwise_statistics(self.groups, self.records)
        extreme = apply_gauge_shift(self.records, magnitude=40.0, per_member=True, seed=0)
        moved = _listwise_statistics(self.groups, extreme)
        self.assertGreater(
            abs(moved["js_divergence"] - baseline["js_divergence"]),
            TOLERANCE,
            "saturation was expected to break gauge invariance but did not; "
            "the clip may have been removed, in which case this test is stale",
        )


if __name__ == "__main__":
    unittest.main()
