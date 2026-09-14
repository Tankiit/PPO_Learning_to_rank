"""Tier 1 recomputation: statistics derived from saved predictions only.

Nothing here trains or scores a model. Every function consumes prediction
JSONL files that already exist on the Modal volume, so the whole tier runs on
CPU and is cheap to re-run when a claim changes.

Four corrections motivate this module.

``ndcg_by_group_size``
    The in-domain validation split is 141 singleton groups out of 197. A
    singleton has a degenerate within-group softmax and no ranking to get
    right, so any ranking number quoted over "the ID split" needs its
    evaluated group count stated alongside it.

``permutation_null_partial_spearman``
    A group-clustered bootstrap interval says how precisely the association is
    measured. It does not say whether an association of that size arises by
    chance under the same group structure. The permutation null does, by
    reassigning whole uncertainty vectors between size-matched groups.

``participation_ratio_conventions``
    ``effective_ensemble_size`` reports ``1 + PR`` on a disagreement subspace
    of rank at most ``M - 1``. Reporting the raw ``PR`` and the ``1 + PR``
    convention side by side removes the ambiguity about which ceiling (4 or 5)
    a value like 4.91 is approaching.

``risk_coverage_curve``
    Width is only useful if abstaining on the widest candidates lowers error
    on what remains. AURC against oracle and random orderings measures that
    directly, which correlation alone does not.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .epistemic import _record_head_scores, summarize_head_scores
from .epistemic_diagnostics import _partial_spearman, _records_in_listwise_space
from .metrics import evaluate_predictions
from .schema import RankingGroup, ScoreRecord


# ---------------------------------------------------------------------------
# Shared array preparation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ListwiseArrays:
    """Candidate-level arrays in ListNet's identifiable probability space.

    ``member_probabilities`` is ``[candidates, members]`` after a within-group
    softmax of logits, which is the only space in which ensembles built from
    different constructions are comparable.
    """

    member_probabilities: np.ndarray
    target_probabilities: np.ndarray
    width: np.ndarray
    absolute_error: np.ndarray
    confidence: np.ndarray
    entropy: np.ndarray
    group_index: np.ndarray
    group_sizes: np.ndarray
    group_ids: tuple[str, ...]

    @property
    def member_count(self) -> int:
        return int(self.member_probabilities.shape[1])

    @property
    def candidate_count(self) -> int:
        return int(self.member_probabilities.shape[0])

    @property
    def group_count(self) -> int:
        return int(self.group_sizes.size)


def prepare_listwise_arrays(
    groups: Sequence[RankingGroup], records: Sequence[ScoreRecord]
) -> ListwiseArrays:
    """Build every array the Tier 1 statistics need, once."""

    member_probabilities, target_probabilities = _records_in_listwise_space(
        groups, records
    )
    width = np.ptp(member_probabilities, axis=1)
    consensus = member_probabilities.mean(axis=1)
    absolute_error = np.abs(consensus - target_probabilities)

    sizes = np.asarray([len(group.candidates) for group in groups], dtype=int)
    group_index = np.repeat(np.arange(sizes.size), sizes)
    confidence = np.empty(member_probabilities.shape[0], dtype=float)
    entropy = np.empty(member_probabilities.shape[0], dtype=float)
    start = 0
    for position, size in enumerate(sizes):
        stop = start + int(size)
        central = consensus[start:stop]
        confidence[start:stop] = float(np.max(central))
        entropy[start:stop] = float(
            -np.sum(central * np.log(np.clip(central, 1e-12, None)))
        )
        start = stop
    return ListwiseArrays(
        member_probabilities=member_probabilities,
        target_probabilities=target_probabilities,
        width=width,
        absolute_error=absolute_error,
        confidence=confidence,
        entropy=entropy,
        group_index=group_index,
        group_sizes=sizes,
        group_ids=tuple(group.group_id for group in groups),
    )


def width_readouts(
    groups: Sequence[RankingGroup], records: Sequence[ScoreRecord]
) -> dict[str, Any]:
    """Three width read-outs of the same checkpoint, only one of them identified.

    A listwise model fixes scores only up to an additive constant per group, per
    member. Width can be read in three places, and they are not interchangeable:

    ``raw_logit_width``  spread of the members' pre-sigmoid scores. Carries the
                         arbitrary offset directly, so it is not a function of
                         anything the model identifies.
    ``sigmoid_width``    spread of the bounded scores, which is what
                         ``credal_width`` stores today. Squashing does not remove
                         the offset, it only compresses it non-linearly - so this
                         is still gauge-dependent, and additionally its scale
                         depends on where on the sigmoid the group happens to sit.
    ``quotient_width``   spread of the within-group softmax probabilities, i.e.
                         width measured in the quotient by the gauge group. This
                         is the only one of the three that is a function of the
                         identified object.

    Reporting all three across training is the point: the first two can drift or
    decay for reasons that have nothing to do with disagreement, and separating
    that from real changes in the credal set is what the quotient read-out buys.
    """

    member_probabilities, _ = _records_in_listwise_space(groups, records)
    raw = np.asarray(
        [_record_head_scores(r) for r in _ordered(groups, records)], dtype=float
    )
    bounded = np.clip(raw, 1e-7, 1.0 - 1e-7)
    logits = np.log(bounded) - np.log1p(-bounded)
    return {
        "raw_logit_width": float(np.mean(np.ptp(logits, axis=1))),
        "sigmoid_width": float(np.mean(np.ptp(bounded, axis=1))),
        "quotient_width": float(np.mean(np.ptp(member_probabilities, axis=1))),
        "n_observations": int(raw.shape[0]),
        "member_count": int(raw.shape[1]),
    }


def _ordered(
    groups: Sequence[RankingGroup], records: Sequence[ScoreRecord]
) -> list[ScoreRecord]:
    """Records in group/candidate order, matching the listwise arrays."""

    by_key = {(r.group_id, r.candidate_id): r for r in records}
    out = []
    for group in groups:
        for candidate in group.candidates:
            key = (group.group_id, candidate.candidate_id)
            if key not in by_key:
                raise ValueError(f"missing prediction for {key}")
            out.append(by_key[key])
    return out


# ---------------------------------------------------------------------------
# Gauge injection control (Proposition 1 as an executable check)
# ---------------------------------------------------------------------------


def apply_gauge_shift(
    records: Sequence[ScoreRecord],
    *,
    magnitude: float = 1.0,
    per_member: bool = True,
    seed: int = 0,
) -> list[ScoreRecord]:
    """Add an arbitrary constant to each member's logits within each group.

    ListNet identifies a ranking, not an absolute score level: adding a constant
    to every candidate of one group, for one member, leaves that member's
    within-group softmax exactly unchanged. Any statistic the paper reports must
    therefore be blind to this transform, and any statistic that moves is
    reading the arbitrary representative rather than the identified object.

    ``per_member`` shifts each (group, member) pair independently, which is the
    full gauge freedom of a member-wise listwise model. Setting it False applies
    one constant per group, shared across members - the weaker transform that
    even a gauge-dependent consensus survives.

    The scalar ``score`` is recomputed as the mean of the shifted bounded member
    scores, exactly as ``combine_independent_member_records`` computes it, so
    the returned records are what the pipeline would have produced had the model
    landed on a different representative.
    """

    generator = np.random.default_rng(seed)
    grouped: dict[str, list[ScoreRecord]] = {}
    for record in records:
        grouped.setdefault(record.group_id, []).append(record)

    shifted: dict[tuple[str, str], ScoreRecord] = {}
    epsilon = 1e-12
    for group_id, members in grouped.items():
        head_count = len(_record_head_scores(members[0]))
        offsets = generator.uniform(-magnitude, magnitude, size=head_count)
        if not per_member:
            offsets = np.full(head_count, float(offsets[0]))
        for record in members:
            values = np.clip(
                np.asarray(_record_head_scores(record), dtype=float), epsilon, 1.0 - epsilon
            )
            logits = np.log(values) - np.log1p(-values)
            moved = 1.0 / (1.0 + np.exp(-(logits + offsets)))
            summary = summarize_head_scores(moved.tolist())
            shifted[(record.group_id, record.candidate_id)] = ScoreRecord(
                **{
                    **record.to_dict(),
                    "score": float(summary["score_mean"]),
                    "raw_output": f"{summary['score_mean']:.10f}",
                    "metadata": {
                        **record.metadata,
                        **summary,
                        "gauge_shift_applied": offsets.tolist(),
                        "gauge_shift_per_member": bool(per_member),
                    },
                }
            )
    return [shifted[(r.group_id, r.candidate_id)] for r in records]


# ---------------------------------------------------------------------------
# 1. Ranking quality conditioned on group size
# ---------------------------------------------------------------------------


def _lift_interval(
    groups: Sequence[RankingGroup],
    records: Sequence[ScoreRecord],
    *,
    samples: int = 3000,
    seed: int = 42,
) -> dict[str, Any]:
    """Bootstrap the NDCG lift over random, resampling whole ranking groups.

    NDCG@5 on two-to-four candidate groups is close to saturated: a random
    ranking already scores about 0.926 in domain. The lift is the fraction of
    the remaining headroom the model captures, and on 52 groups it needs an
    interval before it can be compared between arms.
    """

    per_query = evaluate_predictions(groups, records)["per_query"]
    achieved = np.asarray(
        [row["tie_aware_ndcg_at_5"] for row in per_query], dtype=float
    )
    baseline = np.asarray([row["random_ndcg_at_5"] for row in per_query], dtype=float)
    mask = np.isfinite(achieved) & np.isfinite(baseline)
    achieved, baseline = achieved[mask], baseline[mask]
    if achieved.size == 0:
        return {"estimate": float("nan"), "low": float("nan"), "high": float("nan")}

    def _lift(a: np.ndarray, b: np.ndarray) -> float:
        headroom = 1.0 - float(np.mean(b))
        if headroom <= 1e-12:
            return float("nan")
        return float((np.mean(a) - np.mean(b)) / headroom)

    generator = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=float)
    for index in range(samples):
        pick = generator.integers(0, achieved.size, size=achieved.size)
        draws[index] = _lift(achieved[pick], baseline[pick])
    finite = draws[np.isfinite(draws)]
    return {
        "estimate": _lift(achieved, baseline),
        "low": float(np.quantile(finite, 0.025)) if finite.size else float("nan"),
        "high": float(np.quantile(finite, 0.975)) if finite.size else float("nan"),
        "n_groups": int(achieved.size),
        "resampling_unit": "ranking_group",
    }


def ndcg_by_group_size(
    groups: Sequence[RankingGroup],
    records: Sequence[ScoreRecord],
    *,
    minimum_candidates: int = 2,
) -> dict[str, Any]:
    """NDCG over all groups, over ``C >= minimum_candidates``, and per size.

    ``evaluate_predictions`` already drops groups it cannot rank, so the
    headline figure was never an average over singletons. What was missing is
    the count: this reports ``evaluated_query_count`` next to every mean so a
    number computed on 56 groups is never read as one computed on 197.
    """

    def _summary(subset: Sequence[RankingGroup]) -> dict[str, Any]:
        if not subset:
            return {"tie_aware_ndcg_at_5": float("nan"), "evaluated_query_count": 0}
        aggregate = evaluate_predictions(subset, records)["aggregate"]
        return {
            "tie_aware_ndcg_at_5": aggregate["tie_aware_ndcg_at_5"],
            "ndcg_at_5": aggregate["ndcg_at_5"],
            "random_ndcg_at_5": aggregate["random_ndcg_at_5"],
            "ndcg_lift_over_random": aggregate["ndcg_lift_over_random"],
            "spearman": aggregate["spearman"],
            "evaluated_query_count": int(aggregate["evaluated_query_count"]),
            "rankable_query_count": int(aggregate["rankable_query_count"]),
            "supplied_query_count": int(aggregate["query_count"]),
            "candidate_count": int(aggregate["candidate_count"]),
        }

    sizes = sorted({len(group.candidates) for group in groups})
    return {
        "all_groups": _summary(list(groups)),
        "lift_over_random_ci": _lift_interval(groups, records),
        "non_singleton": _summary(
            [g for g in groups if len(g.candidates) >= minimum_candidates]
        ),
        "by_candidate_count": {
            str(size): _summary([g for g in groups if len(g.candidates) == size])
            for size in sizes
        },
        "group_size_histogram": {
            str(size): int(sum(1 for g in groups if len(g.candidates) == size))
            for size in sizes
        },
        "minimum_candidates": int(minimum_candidates),
    }


# ---------------------------------------------------------------------------
# 2. Group-clustered bootstrap and the permutation null
# ---------------------------------------------------------------------------


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Tie-averaged ranks, the only part of Spearman that costs anything."""

    from scipy.stats import rankdata

    return rankdata(values, method="average")


def _pearson(left: np.ndarray, right: np.ndarray) -> float:
    left = left - left.mean()
    right = right - right.mean()
    denominator = float(np.sqrt(float(left @ left) * float(right @ right)))
    if denominator <= 1e-12:
        return float("nan")
    return float((left @ right) / denominator)


def _partial_from_ranks(
    width_ranks: np.ndarray, error_ranks: np.ndarray, control_ranks: np.ndarray
) -> float:
    """First-order partial Spearman from pre-computed ranks.

    Identical in value to ``_partial_spearman`` but takes ranks as input, so a
    resampling loop ranks only the vectors that actually changed. The
    permutation null holds error and confidence fixed, which makes this the
    difference between minutes and hours across nine arms.
    """

    lr = _pearson(width_ranks, error_ranks)
    lc = _pearson(width_ranks, control_ranks)
    rc = _pearson(error_ranks, control_ranks)
    denominator = np.sqrt(max((1.0 - lc * lc) * (1.0 - rc * rc), 0.0))
    if not np.isfinite(denominator) or denominator <= 1e-12:
        return float("nan")
    return float((lr - lc * rc) / denominator)


def _group_starts(group_sizes: np.ndarray) -> np.ndarray:
    return np.concatenate([[0], np.cumsum(group_sizes)[:-1]]).astype(int)


def _flat_index_for_groups(
    chosen: np.ndarray, starts: np.ndarray, sizes: np.ndarray
) -> np.ndarray:
    """Flat candidate indices for a ragged multiset of groups, without a loop."""

    selected_sizes = sizes[chosen]
    total = int(selected_sizes.sum())
    output_starts = np.concatenate([[0], np.cumsum(selected_sizes)[:-1]]).astype(int)
    offsets = np.repeat(starts[chosen] - output_starts, selected_sizes)
    return offsets + np.arange(total)


def _control_vector(arrays: ListwiseArrays, control: str) -> np.ndarray:
    if control == "confidence":
        return arrays.confidence
    if control == "entropy":
        return arrays.entropy
    raise ValueError(f"unsupported control: {control!r}")


def _partial_statistic(
    arrays: ListwiseArrays,
    width: np.ndarray,
    control: str = "confidence",
) -> float:
    return _partial_spearman(
        width, arrays.absolute_error, _control_vector(arrays, control)
    )


def group_bootstrap_partial_spearman(
    arrays: ListwiseArrays,
    *,
    control: str = "confidence",
    samples: int = 3000,
    seed: int = 42,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    """Resample whole ranking groups, because candidates within one are dependent."""

    generator = np.random.default_rng(seed)
    starts = _group_starts(arrays.group_sizes)
    control_values = _control_vector(arrays, control)
    draws = np.empty(samples, dtype=float)
    for draw in range(samples):
        chosen = generator.integers(0, arrays.group_count, size=arrays.group_count)
        index = _flat_index_for_groups(chosen, starts, arrays.group_sizes)
        draws[draw] = _partial_from_ranks(
            _average_ranks(arrays.width[index]),
            _average_ranks(arrays.absolute_error[index]),
            _average_ranks(control_values[index]),
        )
    finite = draws[np.isfinite(draws)]
    tail = (1.0 - confidence_level) / 2.0
    return {
        "estimate": _partial_statistic(arrays, arrays.width, control),
        "low": float(np.quantile(finite, tail)) if finite.size else float("nan"),
        "high": float(np.quantile(finite, 1.0 - tail)) if finite.size else float("nan"),
        "samples": int(samples),
        "finite_samples": int(finite.size),
        "control": control,
        "resampling_unit": "ranking_group",
        "n_groups": arrays.group_count,
        "n_observations": arrays.candidate_count,
    }


def permutation_null_partial_spearman(
    arrays: ListwiseArrays,
    *,
    control: str = "confidence",
    permutations: int = 2000,
    seed: int = 42,
    scheme: str = "between_groups",
) -> dict[str, Any]:
    """Null distribution for the width/error association, holding structure fixed.

    Two schemes answer two different questions, and reporting one alone is
    misleading.

    ``between_groups`` exchanges whole group width-vectors between groups of
    equal candidate count. The within-group shape of the width vector travels
    with it, so anything a fixed positional pattern can produce survives into
    the null. This is the conservative test: it asks whether the association
    identifies *which group* is uncertain beyond what a constant within-group
    pattern already yields.

    ``within_groups`` shuffles widths among the candidates of each group. Group
    mean width is preserved and the within-group ordering is destroyed, so it
    asks the complementary question: does width identify *which candidate*
    inside a group is the erroneous one?

    Under both, error and confidence are untouched, so the group clustering and
    the confidence control are preserved exactly.
    """

    if scheme not in {"between_groups", "within_groups"}:
        raise ValueError(f"unsupported scheme: {scheme!r}")
    generator = np.random.default_rng(seed)
    starts = _group_starts(arrays.group_sizes)
    strata: dict[int, list[int]] = {}
    for position, size in enumerate(arrays.group_sizes):
        strata.setdefault(int(size), []).append(position)
    # A stratum that cannot be randomised contributes nothing to the null: one
    # group has no partner to swap with under ``between_groups``, and a
    # singleton group has no second candidate under ``within_groups``.
    blocks = []
    for size, positions in sorted(strata.items()):
        if scheme == "between_groups" and len(positions) < 2:
            continue
        if scheme == "within_groups" and int(size) < 2:
            continue
        indices = np.asarray(positions, dtype=int)
        blocks.append(starts[indices][:, None] + np.arange(int(size))[None, :])

    error_ranks = _average_ranks(arrays.absolute_error)
    control_ranks = _average_ranks(_control_vector(arrays, control))
    observed = _partial_from_ranks(
        _average_ranks(arrays.width), error_ranks, control_ranks
    )

    draws = np.empty(permutations, dtype=float)
    for draw in range(permutations):
        permuted = arrays.width.copy()
        for block in blocks:
            if scheme == "between_groups":
                order = generator.permutation(block.shape[0])
                permuted[block.reshape(-1)] = arrays.width[block[order].reshape(-1)]
            else:
                shuffled = np.argsort(
                    generator.random(block.shape), axis=1, kind="stable"
                )
                source = np.take_along_axis(block, shuffled, axis=1)
                permuted[block.reshape(-1)] = arrays.width[source.reshape(-1)]
        draws[draw] = _partial_from_ranks(
            _average_ranks(permuted), error_ranks, control_ranks
        )
    finite = draws[np.isfinite(draws)]
    if not finite.size or not np.isfinite(observed):
        p_value = float("nan")
    else:
        # Add-one correction: a permutation p-value is never exactly zero.
        extreme = int(np.sum(np.abs(finite) >= abs(observed)))
        p_value = float((extreme + 1) / (finite.size + 1))
    return {
        "observed": observed,
        "null_mean": float(np.mean(finite)) if finite.size else float("nan"),
        "null_std": float(np.std(finite)) if finite.size else float("nan"),
        "null_p02_5": float(np.quantile(finite, 0.025)) if finite.size else float("nan"),
        "null_p97_5": float(np.quantile(finite, 0.975)) if finite.size else float("nan"),
        "two_sided_p_value": p_value,
        "permutations": int(permutations),
        "finite_permutations": int(finite.size),
        "randomised_group_count": int(sum(block.shape[0] for block in blocks)),
        "randomised_candidate_count": int(sum(block.size for block in blocks)),
        "control": control,
        "scheme": scheme,
        "scheme_detail": (
            "exchange_whole_group_width_vectors_within_candidate_count_strata"
            if scheme == "between_groups"
            else "shuffle_widths_among_candidates_inside_each_group"
        ),
    }


# ---------------------------------------------------------------------------
# 3. Participation-ratio conventions and Jensen--Shannon naming
# ---------------------------------------------------------------------------


def participation_ratio_conventions(residuals: np.ndarray) -> dict[str, Any]:
    """Report the disagreement participation ratio under both conventions.

    The centred disagreement subspace has rank at most ``M - 1``, so the raw
    participation ratio has ceiling ``M - 1`` while the reported ``1 + PR`` has
    ceiling ``M``. Both are given, with the ceiling each is approaching, so a
    reader never has to infer which one a bare number uses.
    """

    values = np.asarray(residuals, dtype=float)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("residuals must have shape [examples, members>=2]")
    member_count = int(values.shape[1])
    disagreement = values - values.mean(axis=1, keepdims=True)
    covariance = np.cov(disagreement, rowvar=False, ddof=0)
    eigenvalues = np.clip(np.linalg.eigvalsh(covariance), 0.0, None)
    denominator = float(np.square(eigenvalues).sum())
    raw = (
        0.0
        if denominator <= 1e-24
        else float(min(member_count - 1, np.square(eigenvalues.sum()) / denominator))
    )
    return {
        "member_count": member_count,
        "disagreement_participation_ratio": raw,
        "disagreement_participation_ratio_ceiling": member_count - 1,
        "effective_ensemble_size": 1.0 + raw,
        "effective_ensemble_size_ceiling": member_count,
        "participation_fraction": raw / max(member_count - 1, 1),
        "convention": "effective_ensemble_size = 1 + disagreement_participation_ratio",
        "eigenvalues": eigenvalues.tolist(),
    }


def uncentred_participation_ratio(residuals: np.ndarray) -> dict[str, Any]:
    """Participation ratio of the RAW residual covariance, without centring.

    This is the pre-repair statistic, and it is the one Proposition 2 makes a
    prediction about. A covariance of raw residuals carries the irreducible
    error shared by every member as a rank-one component; as that shared error
    grows, it dominates the spectrum and drives the participation ratio toward
    one even for genuinely independent models.

    Note that the centred statistic in ``participation_ratio_conventions``
    cannot be used to test this. Subtracting the target from every member is a
    per-example constant, so it vanishes under row-centring: the centred
    participation ratio of residuals is identically the centred participation
    ratio of raw member scores. Measuring the aleatoric prediction therefore
    requires this uncentred form, and reporting both is what separates
    "disagreement collapsed" from "shared error grew".
    """

    values = np.asarray(residuals, dtype=float)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("residuals must have shape [examples, members>=2]")
    member_count = int(values.shape[1])
    covariance = np.cov(values, rowvar=False, ddof=0)
    eigenvalues = np.clip(np.linalg.eigvalsh(covariance), 0.0, None)
    denominator = float(np.square(eigenvalues).sum())
    if denominator <= 1e-24:
        return {
            "uncentred_participation_ratio": 1.0,
            "member_count": member_count,
            "participation_fraction": 0.0,
        }
    ratio = float(min(member_count, np.square(eigenvalues.sum()) / denominator))
    return {
        "uncentred_participation_ratio": ratio,
        "member_count": member_count,
        "participation_fraction": (ratio - 1.0) / max(member_count - 1, 1),
        "convention": "PR of raw residual covariance, ceiling M, floor 1",
    }


def consensus_entropy(arrays: ListwiseArrays) -> dict[str, Any]:
    """Mean entropy of the ensemble consensus distribution, per group.

    This is the aleatoric proxy: it measures how flat the consensus itself is,
    which is what should rise when the labels the model was trained on become
    noisier. It is deliberately independent of member disagreement - a single
    model has a consensus entropy, but no Jensen-Shannon divergence.
    """

    probabilities = np.clip(arrays.member_probabilities, 1e-12, None)
    consensus = probabilities.mean(axis=1)
    values, normalised = [], []
    start = 0
    for size in arrays.group_sizes:
        stop = start + int(size)
        block = consensus[start:stop]
        block = block / max(block.sum(), 1e-30)
        entropy = float(-np.sum(block * np.log(np.clip(block, 1e-12, None))))
        values.append(entropy)
        ceiling = float(np.log(int(size))) if int(size) > 1 else float("nan")
        normalised.append(entropy / ceiling if ceiling > 0 else float("nan"))
        start = stop
    finite = np.asarray([v for v in normalised if np.isfinite(v)], dtype=float)
    return {
        "consensus_entropy": float(np.mean(values)),
        "consensus_entropy_normalised": float(np.mean(finite)) if finite.size else float("nan"),
        "n_groups": arrays.group_count,
    }


def js_divergence_to_consensus(
    arrays: ListwiseArrays, *, eps: float = 1e-12
) -> dict[str, Any]:
    """Jensen--Shannon diversity, named for what it computes.

    ``H(mean_m p_m) - mean_m H(p_m)`` is the Jensen--Shannon divergence of the
    member distributions, not a KL divergence to a consensus. Earlier reports
    called it ``mean_kl_to_consensus``; that key is retained as an alias so
    existing summaries stay readable, but the JS name is the correct one.

    ``normalised`` divides by ``log(C)``, the ceiling a C-candidate group can
    reach, which is what makes groups of different sizes comparable.
    """

    probabilities = np.clip(arrays.member_probabilities, eps, None)
    per_group: list[float] = []
    per_group_normalised: list[float] = []
    start = 0
    for size in arrays.group_sizes:
        stop = start + int(size)
        block = probabilities[start:stop]
        block = block / block.sum(axis=0, keepdims=True).clip(min=eps)
        consensus = block.mean(axis=1, keepdims=True)
        entropy_consensus = float(-np.sum(consensus * np.log(consensus)))
        entropy_members = float(
            np.mean(-np.sum(block * np.log(block), axis=0))
        )
        value = entropy_consensus - entropy_members
        per_group.append(value)
        ceiling = float(np.log(int(size))) if int(size) > 1 else float("nan")
        per_group_normalised.append(value / ceiling if ceiling > 0 else float("nan"))
        start = stop

    values = np.asarray(per_group, dtype=float)
    normalised = np.asarray(per_group_normalised, dtype=float)
    non_singleton = arrays.group_sizes >= 2
    finite = normalised[non_singleton & np.isfinite(normalised)]
    return {
        "js_divergence_to_consensus": float(np.mean(values)),
        "mean_kl_to_consensus": float(np.mean(values)),  # deprecated alias
        "js_divergence_non_singleton": (
            float(np.mean(values[non_singleton])) if non_singleton.any() else float("nan")
        ),
        "js_divergence_normalised_non_singleton": (
            float(np.mean(finite)) if finite.size else float("nan")
        ),
        "normalisation": "divided_by_log_candidate_count",
        "definition": "H(mean_m p_m) - mean_m H(p_m)",
        "statistic_name": "jensen_shannon_divergence",
        "n_groups": arrays.group_count,
        "n_non_singleton_groups": int(non_singleton.sum()),
    }


def shrink_control(
    arrays: ListwiseArrays, factors: Sequence[float] = (1.0, 0.5, 0.1, 0.02)
) -> dict[str, Any]:
    """Shrink member deviations toward consensus and watch the two metrics part.

    Members are pulled toward their within-group consensus in logit space by
    ``factor``, then renormalised. The participation ratio depends only on the
    *shape* of the disagreement covariance spectrum, so uniform shrinkage leaves
    it exactly unchanged; Jensen--Shannon divergence depends on the magnitude of
    the deviations and falls roughly as the square of the factor.

    This replaces the illustrative constants previously used to make that point:
    the numbers below are measured on the arm's own predictions.
    """

    probabilities = np.clip(arrays.member_probabilities, 1e-12, 1.0 - 1e-12)
    logits = np.log(probabilities) - np.log1p(-probabilities)
    consensus = logits.mean(axis=1, keepdims=True)
    rows = []
    for factor in factors:
        shrunk = consensus + float(factor) * (logits - consensus)
        bounded = 1.0 / (1.0 + np.exp(-shrunk))
        renormalised = np.empty_like(bounded)
        start = 0
        for size in arrays.group_sizes:
            stop = start + int(size)
            block = bounded[start:stop]
            renormalised[start:stop] = block / block.sum(axis=0, keepdims=True).clip(
                min=1e-30
            )
            start = stop
        scaled = ListwiseArrays(
            member_probabilities=renormalised,
            target_probabilities=arrays.target_probabilities,
            width=np.ptp(renormalised, axis=1),
            absolute_error=arrays.absolute_error,
            confidence=arrays.confidence,
            entropy=arrays.entropy,
            group_index=arrays.group_index,
            group_sizes=arrays.group_sizes,
            group_ids=arrays.group_ids,
        )
        residuals = renormalised - arrays.target_probabilities[:, None]
        conventions = participation_ratio_conventions(residuals)
        rows.append(
            {
                "factor": float(factor),
                "participation_fraction": conventions["participation_fraction"],
                "effective_ensemble_size": conventions["effective_ensemble_size"],
                "js_divergence_to_consensus": js_divergence_to_consensus(scaled)[
                    "js_divergence_to_consensus"
                ],
                "mean_width": float(np.mean(scaled.width)),
            }
        )
    base = rows[0]
    for row in rows:
        row["js_ratio_to_unshrunk"] = (
            row["js_divergence_to_consensus"] / base["js_divergence_to_consensus"]
            if base["js_divergence_to_consensus"] > 0
            else float("nan")
        )
    return {"rows": rows, "shrinkage_space": "within_group_logit"}


# ---------------------------------------------------------------------------
# 4. Abstention risk--coverage (T5)
# ---------------------------------------------------------------------------


def _risk_curve(errors: np.ndarray, order: np.ndarray) -> np.ndarray:
    """Selective risk at coverage ``k/N`` for ``k = 1..N``, retaining ``order`` first."""

    ranked = errors[order]
    return np.cumsum(ranked) / np.arange(1, ranked.size + 1)


def risk_coverage_curve(
    arrays: ListwiseArrays,
    *,
    seed: int = 42,
    random_repeats: int = 200,
    coverage_grid: Sequence[float] = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1),
) -> dict[str, Any]:
    """Does abstaining on the widest candidates actually lower error?

    Candidates are retained in ascending width order, so the most uncertain are
    dropped first. AURC is the mean selective risk across coverage levels.
    Lower is better. The oracle orders by true error and the random baseline
    averages uniform orderings, which together bound what any ordering can
    achieve on this error vector.
    """

    errors = arrays.absolute_error
    count = errors.size
    width_order = np.argsort(arrays.width, kind="stable")
    oracle_order = np.argsort(errors, kind="stable")

    width_curve = _risk_curve(errors, width_order)
    oracle_curve = _risk_curve(errors, oracle_order)

    generator = np.random.default_rng(seed)
    random_curve = np.zeros(count, dtype=float)
    for _ in range(random_repeats):
        random_curve += _risk_curve(errors, generator.permutation(count))
    random_curve /= random_repeats

    def _at(curve: np.ndarray, coverage: float) -> float:
        position = max(1, int(round(coverage * count)))
        return float(curve[min(position, count) - 1])

    aurc_width = float(np.mean(width_curve))
    aurc_oracle = float(np.mean(oracle_curve))
    aurc_random = float(np.mean(random_curve))
    span = aurc_random - aurc_oracle
    return {
        "n_observations": int(count),
        "n_groups": arrays.group_count,
        "full_coverage_risk": float(np.mean(errors)),
        "aurc_width": aurc_width,
        "aurc_oracle": aurc_oracle,
        "aurc_random": aurc_random,
        # 1.0 means the width ordering matches the oracle, 0.0 means it is no
        # better than random, and a negative value means it is actively harmful.
        "normalised_aurc_gain": (
            float((aurc_random - aurc_width) / span) if span > 1e-12 else float("nan")
        ),
        "coverage_grid": [float(c) for c in coverage_grid],
        "risk_by_width": [_at(width_curve, c) for c in coverage_grid],
        "risk_by_oracle": [_at(oracle_curve, c) for c in coverage_grid],
        "risk_by_random": [_at(random_curve, c) for c in coverage_grid],
        "selection_rule": "retain_ascending_width",
        "risk_definition": "mean_absolute_error_in_within_group_probability_space",
    }


def group_risk_coverage_curve(
    groups: Sequence[RankingGroup],
    records: Sequence[ScoreRecord],
    arrays: ListwiseArrays,
    *,
    seed: int = 42,
    random_repeats: int = 200,
    coverage_grid: Sequence[float] = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1),
) -> dict[str, Any]:
    """Risk--coverage at the level a ranking system actually abstains on.

    A deployed ranker abstains on a whole query, not on one candidate, so risk
    here is ``1 - NDCG@5`` per group and groups are retained in ascending mean
    width. Only rankable groups take part; a singleton has no ranking to score.
    """

    per_query = evaluate_predictions(groups, records)["per_query"]
    scored = {row["group_id"]: row["tie_aware_ndcg_at_5"] for row in per_query}
    positions = [
        index
        for index, group_id in enumerate(arrays.group_ids)
        if group_id in scored and np.isfinite(scored[group_id])
    ]
    if not positions:
        return {"n_groups": 0, "aurc_width": float("nan")}

    mean_width = np.asarray(
        [float(np.mean(arrays.width[arrays.group_index == index])) for index in positions],
        dtype=float,
    )
    risk = np.asarray(
        [1.0 - float(scored[arrays.group_ids[index]]) for index in positions], dtype=float
    )
    count = risk.size
    width_curve = _risk_curve(risk, np.argsort(mean_width, kind="stable"))
    oracle_curve = _risk_curve(risk, np.argsort(risk, kind="stable"))
    generator = np.random.default_rng(seed)
    random_curve = np.zeros(count, dtype=float)
    for _ in range(random_repeats):
        random_curve += _risk_curve(risk, generator.permutation(count))
    random_curve /= random_repeats

    def _at(curve: np.ndarray, coverage: float) -> float:
        position = max(1, int(round(coverage * count)))
        return float(curve[min(position, count) - 1])

    aurc_width = float(np.mean(width_curve))
    aurc_oracle = float(np.mean(oracle_curve))
    aurc_random = float(np.mean(random_curve))
    span = aurc_random - aurc_oracle
    return {
        "n_groups": int(count),
        "full_coverage_risk": float(np.mean(risk)),
        "aurc_width": aurc_width,
        "aurc_oracle": aurc_oracle,
        "aurc_random": aurc_random,
        "normalised_aurc_gain": (
            float((aurc_random - aurc_width) / span) if span > 1e-12 else float("nan")
        ),
        "coverage_grid": [float(c) for c in coverage_grid],
        "risk_by_width": [_at(width_curve, c) for c in coverage_grid],
        "risk_by_oracle": [_at(oracle_curve, c) for c in coverage_grid],
        "risk_by_random": [_at(random_curve, c) for c in coverage_grid],
        "selection_rule": "retain_ascending_group_mean_width",
        "risk_definition": "one_minus_tie_aware_ndcg_at_5",
    }
