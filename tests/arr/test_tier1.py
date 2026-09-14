"""Tests for the Tier 1 recomputation statistics.

These guard the properties the report now leans on: that the fast partial
Spearman matches the reference implementation, that the permutation nulls
actually destroy the association they claim to, that singleton groups
contribute nothing, and that the risk-coverage bounds order correctly.
"""

import unittest

import numpy as np

from src.arr.epistemic_diagnostics import _partial_spearman, effective_ensemble_size
from src.arr.schema import Candidate, RankingGroup, ScoreRecord
from src.arr.tier1 import (
    consensus_entropy,
    uncentred_participation_ratio,
    _average_ranks,
    _flat_index_for_groups,
    _group_starts,
    _partial_from_ranks,
    group_bootstrap_partial_spearman,
    js_divergence_to_consensus,
    ndcg_by_group_size,
    participation_ratio_conventions,
    permutation_null_partial_spearman,
    prepare_listwise_arrays,
    risk_coverage_curve,
)

FINGERPRINT = "fingerprint"


def _group(group_id: str, scores: list[float]) -> RankingGroup:
    return RankingGroup(
        group_id=group_id,
        split="dev",
        domain="test",
        question=f"question for {group_id}",
        candidates=tuple(
            Candidate(
                candidate_id=f"{group_id}-c{index}",
                text=f"candidate {index} of {group_id}",
                score=score,
                score_provenance="synthetic",
            )
            for index, score in enumerate(scores)
        ),
        data_fingerprint=FINGERPRINT,
    )


def _records(group: RankingGroup, head_scores: list[list[float]]) -> list[ScoreRecord]:
    return [
        ScoreRecord(
            group_id=group.group_id,
            candidate_id=candidate.candidate_id,
            model_name="test",
            model_revision="test",
            prompt_hash="none",
            data_fingerprint=FINGERPRINT,
            score=float(np.mean(heads)),
            raw_output="",
            parsing_status="ok",
            seed=0,
            inference_ms=0.0,
            metadata={"head_scores": [float(value) for value in heads]},
        )
        for candidate, heads in zip(group.candidates, head_scores)
    ]


def _synthetic(group_count: int = 60, size: int = 4, members: int = 5, seed: int = 0):
    rng = np.random.default_rng(seed)
    groups, records = [], []
    for index in range(group_count):
        scores = rng.uniform(0.05, 0.95, size=size).tolist()
        group = _group(f"g{index}", scores)
        heads = np.clip(
            rng.uniform(0.2, 0.8, size=(size, 1))
            + rng.normal(0, 0.05, size=(size, members)),
            0.01,
            0.99,
        )
        groups.append(group)
        records.extend(_records(group, heads.tolist()))
    return groups, records


class PartialSpearmanTest(unittest.TestCase):
    def test_rank_path_matches_reference(self):
        rng = np.random.default_rng(7)
        for _ in range(10):
            control = rng.normal(size=250)
            width = 0.5 * control + rng.normal(size=250)
            error = 0.3 * control + 0.4 * width + rng.normal(size=250)
            self.assertAlmostEqual(
                _partial_spearman(width, error, control),
                _partial_from_ranks(
                    _average_ranks(width), _average_ranks(error), _average_ranks(control)
                ),
                places=12,
            )

    def test_rank_path_handles_ties(self):
        rng = np.random.default_rng(11)
        width = np.round(rng.normal(size=200), 1)
        error = np.round(rng.normal(size=200), 1)
        control = np.round(rng.normal(size=200), 1)
        self.assertAlmostEqual(
            _partial_spearman(width, error, control),
            _partial_from_ranks(
                _average_ranks(width), _average_ranks(error), _average_ranks(control)
            ),
            places=12,
        )


class RaggedIndexTest(unittest.TestCase):
    def test_matches_explicit_concatenation(self):
        sizes = np.array([1, 3, 2, 4, 1, 2])
        starts = _group_starts(sizes)
        members = [np.arange(starts[i], starts[i] + sizes[i]) for i in range(sizes.size)]
        rng = np.random.default_rng(3)
        for _ in range(100):
            chosen = rng.integers(0, sizes.size, size=sizes.size)
            np.testing.assert_array_equal(
                _flat_index_for_groups(chosen, starts, sizes),
                np.concatenate([members[g] for g in chosen]),
            )


class ParticipationRatioTest(unittest.TestCase):
    def test_conventions_agree_with_existing_effective_size(self):
        rng = np.random.default_rng(5)
        residuals = rng.normal(size=(400, 5))
        conventions = participation_ratio_conventions(residuals)
        self.assertAlmostEqual(
            conventions["effective_ensemble_size"],
            effective_ensemble_size(residuals),
            places=12,
        )
        self.assertEqual(conventions["disagreement_participation_ratio_ceiling"], 4)
        self.assertEqual(conventions["effective_ensemble_size_ceiling"], 5)
        self.assertAlmostEqual(
            conventions["effective_ensemble_size"],
            1.0 + conventions["disagreement_participation_ratio"],
            places=12,
        )

    def test_collapsed_members_report_one(self):
        rng = np.random.default_rng(5)
        collapsed = np.repeat(rng.normal(size=(300, 1)), 5, axis=1)
        conventions = participation_ratio_conventions(collapsed)
        self.assertAlmostEqual(conventions["effective_ensemble_size"], 1.0, places=9)
        self.assertAlmostEqual(conventions["participation_fraction"], 0.0, places=9)


class PropositionTwoTest(unittest.TestCase):
    """The two participation ratios must answer different questions.

    Subtracting the target from every member is a per-example constant, so it
    vanishes under row-centring. The centred statistic is therefore identical on
    residuals and on raw member scores, and cannot register a growing shared
    error term. Only the uncentred form can, which is why both are reported.
    """

    def test_centred_ratio_ignores_shared_error(self):
        rng = np.random.default_rng(0)
        members = rng.normal(0, 0.1, size=(600, 5))
        shared = rng.normal(0, 1.0, size=600)
        baseline = participation_ratio_conventions(members)["effective_ensemble_size"]
        for scale in (0.5, 2.0, 8.0):
            moved = participation_ratio_conventions(
                members + scale * shared[:, None]
            )["effective_ensemble_size"]
            self.assertAlmostEqual(baseline, moved, places=9)

    def test_uncentred_ratio_collapses_with_shared_error(self):
        rng = np.random.default_rng(1)
        members = rng.normal(0, 0.1, size=(600, 5))
        shared = rng.normal(0, 1.0, size=600)
        values = [
            uncentred_participation_ratio(members + scale * shared[:, None])[
                "uncentred_participation_ratio"
            ]
            for scale in (0.0, 0.25, 1.0, 4.0)
        ]
        self.assertTrue(all(a > b for a, b in zip(values, values[1:])), values)
        self.assertGreater(values[0], 3.0)
        self.assertLess(values[-1], 1.1)


class ConsensusEntropyTest(unittest.TestCase):
    def test_flatter_consensus_has_higher_entropy(self):
        """The aleatoric proxy must rise as the consensus flattens."""

        groups, records = _synthetic(group_count=30, size=4, seed=3)
        arrays = prepare_listwise_arrays(groups, records)
        sharp = consensus_entropy(arrays)["consensus_entropy_normalised"]

        # Flatten every member toward the uniform distribution over the group.
        flat = arrays.member_probabilities * 0.0 + 1.0 / 4.0
        from src.arr.tier1 import ListwiseArrays

        flattened = ListwiseArrays(
            member_probabilities=flat,
            target_probabilities=arrays.target_probabilities,
            width=np.zeros(arrays.candidate_count),
            absolute_error=arrays.absolute_error,
            confidence=arrays.confidence,
            entropy=arrays.entropy,
            group_index=arrays.group_index,
            group_sizes=arrays.group_sizes,
            group_ids=arrays.group_ids,
        )
        self.assertGreater(
            consensus_entropy(flattened)["consensus_entropy_normalised"], sharp
        )
        self.assertAlmostEqual(
            consensus_entropy(flattened)["consensus_entropy_normalised"], 1.0, places=9
        )


class SingletonTest(unittest.TestCase):
    """A singleton group must contribute nothing to width, error, or diversity."""

    def test_singleton_contributes_zero_width_and_error(self):
        group = _group("solo", [0.7])
        records = _records(group, [[0.3, 0.5, 0.9]])
        arrays = prepare_listwise_arrays([group], records)
        # One candidate: the within-group softmax is 1.0 for every member.
        np.testing.assert_allclose(arrays.member_probabilities, 1.0)
        self.assertAlmostEqual(float(arrays.width[0]), 0.0, places=12)
        self.assertAlmostEqual(float(arrays.absolute_error[0]), 0.0, places=12)

    def test_js_divergence_reports_non_singleton_separately(self):
        groups, records = _synthetic(group_count=20, size=3)
        solo = _group("solo", [0.7])
        groups = groups + [solo]
        records = records + _records(solo, [[0.3, 0.5, 0.9, 0.4, 0.6]])
        diversity = js_divergence_to_consensus(
            prepare_listwise_arrays(groups, records)
        )
        self.assertEqual(diversity["n_groups"], 21)
        self.assertEqual(diversity["n_non_singleton_groups"], 20)
        # The singleton's exact zero drags the all-group mean below the
        # conditioned one, which is precisely why both are reported.
        self.assertLess(
            diversity["js_divergence_to_consensus"],
            diversity["js_divergence_non_singleton"],
        )

    def test_ndcg_reports_evaluated_count_not_supplied_count(self):
        groups, records = _synthetic(group_count=10, size=3)
        solo = _group("solo", [0.7])
        groups = groups + [solo]
        records = records + _records(solo, [[0.3, 0.5, 0.9, 0.4, 0.6]])
        report = ndcg_by_group_size(groups, records)
        self.assertEqual(report["all_groups"]["supplied_query_count"], 11)
        # The singleton is not rankable, so it never enters the mean.
        self.assertEqual(report["all_groups"]["evaluated_query_count"], 10)
        self.assertEqual(
            report["all_groups"]["tie_aware_ndcg_at_5"],
            report["non_singleton"]["tie_aware_ndcg_at_5"],
        )


class PermutationNullTest(unittest.TestCase):
    def test_null_is_centred_when_there_is_no_association(self):
        groups, records = _synthetic(group_count=80, size=4, seed=2)
        arrays = prepare_listwise_arrays(groups, records)
        for scheme in ("between_groups", "within_groups"):
            null = permutation_null_partial_spearman(
                arrays, permutations=200, scheme=scheme, seed=1
            )
            # The null mean need not be exactly zero: group structure and the
            # confidence control impose a small systematic offset, which is
            # exactly why the p-value is read against this empirical null
            # rather than against a nominal zero.
            self.assertLess(abs(null["null_mean"]), 0.15)
            self.assertLess(null["null_std"], 0.5)
            self.assertGreater(null["two_sided_p_value"], 0.01)

    def test_p_value_never_reaches_zero(self):
        groups, records = _synthetic(group_count=40, size=4, seed=4)
        arrays = prepare_listwise_arrays(groups, records)
        null = permutation_null_partial_spearman(arrays, permutations=100, seed=1)
        self.assertGreaterEqual(null["two_sided_p_value"], 1.0 / 101.0)

    def test_schemes_randomise_the_expected_units(self):
        groups, records = _synthetic(group_count=30, size=4, seed=6)
        arrays = prepare_listwise_arrays(groups, records)
        between = permutation_null_partial_spearman(
            arrays, permutations=20, scheme="between_groups"
        )
        within = permutation_null_partial_spearman(
            arrays, permutations=20, scheme="within_groups"
        )
        self.assertEqual(between["randomised_group_count"], 30)
        self.assertEqual(within["randomised_candidate_count"], 120)

    def test_rejects_unknown_scheme(self):
        groups, records = _synthetic(group_count=5, size=3)
        arrays = prepare_listwise_arrays(groups, records)
        with self.assertRaises(ValueError):
            permutation_null_partial_spearman(arrays, scheme="nonsense")


class BootstrapTest(unittest.TestCase):
    def test_interval_brackets_the_point_estimate(self):
        groups, records = _synthetic(group_count=60, size=4, seed=8)
        arrays = prepare_listwise_arrays(groups, records)
        result = group_bootstrap_partial_spearman(arrays, samples=300, seed=1)
        self.assertLessEqual(result["low"], result["estimate"])
        self.assertGreaterEqual(result["high"], result["estimate"])
        self.assertEqual(result["resampling_unit"], "ranking_group")
        self.assertEqual(result["n_groups"], 60)


class RiskCoverageTest(unittest.TestCase):
    def test_oracle_bounds_random_and_gain_is_normalised(self):
        groups, records = _synthetic(group_count=50, size=4, seed=9)
        arrays = prepare_listwise_arrays(groups, records)
        curve = risk_coverage_curve(arrays, seed=1, random_repeats=50)
        # The oracle retains the smallest errors first, so it can never be worse
        # than a random ordering at any coverage.
        self.assertLess(curve["aurc_oracle"], curve["aurc_random"])
        self.assertTrue(np.isfinite(curve["normalised_aurc_gain"]))

    def _arrays(self, width, error):
        """A minimal ListwiseArrays with the fields risk_coverage_curve reads."""

        from src.arr.tier1 import ListwiseArrays

        count = width.size
        return ListwiseArrays(
            member_probabilities=np.tile(np.linspace(0.1, 0.9, 5), (count, 1)),
            target_probabilities=np.zeros(count),
            width=width,
            absolute_error=error,
            confidence=np.zeros(count),
            entropy=np.zeros(count),
            group_index=np.arange(count) // 2,
            group_sizes=np.full(count // 2, 2),
            group_ids=tuple(f"g{i}" for i in range(count // 2)),
        )

    def test_width_matching_error_reaches_the_oracle(self):
        rng = np.random.default_rng(12)
        error = rng.uniform(0, 1, size=400)
        curve = risk_coverage_curve(
            self._arrays(error.copy(), error), seed=1, random_repeats=50
        )
        self.assertAlmostEqual(curve["aurc_width"], curve["aurc_oracle"], places=12)
        self.assertAlmostEqual(curve["normalised_aurc_gain"], 1.0, places=9)

    def test_width_reversed_from_error_is_worse_than_random(self):
        rng = np.random.default_rng(13)
        error = rng.uniform(0, 1, size=400)
        curve = risk_coverage_curve(
            self._arrays(-error, error), seed=1, random_repeats=50
        )
        self.assertGreater(curve["aurc_width"], curve["aurc_random"])
        self.assertLess(curve["normalised_aurc_gain"], 0.0)


if __name__ == "__main__":
    unittest.main()
