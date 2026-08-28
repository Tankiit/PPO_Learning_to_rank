from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
from scipy.stats import kendalltau, spearmanr

from .schema import RankingGroup, ScoreRecord


def _dcg(relevance: np.ndarray, k: int) -> float:
    relevance = np.asarray(relevance, dtype=float)[:k]
    if relevance.size == 0:
        return float("nan")
    discounts = np.log2(np.arange(2, relevance.size + 2))
    return float(np.sum((np.power(2.0, relevance) - 1.0) / discounts))


def ndcg_at_k(reference: Sequence[float], prediction: Sequence[float], k: int = 5) -> float:
    truth = np.asarray(reference, dtype=float)
    predicted = np.asarray(prediction, dtype=float)
    if truth.shape != predicted.shape or truth.ndim != 1:
        raise ValueError("reference and prediction must be same-length vectors")
    order = np.argsort(-predicted, kind="stable")
    ideal = np.argsort(-truth, kind="stable")
    denominator = _dcg(truth[ideal], k)
    if not math.isfinite(denominator) or denominator <= 0:
        return float("nan")
    return _dcg(truth[order], k) / denominator


def tie_aware_ndcg_at_k(
    reference: Sequence[float], prediction: Sequence[float], k: int = 5
) -> float:
    """Expected NDCG under uniform permutations inside predicted tie blocks.

    A stable sort silently rewards the serialised candidate order when predictions
    are tied.  This implementation assigns every member of a tie block the average
    gain of that block at the positions occupied by the block.
    """

    truth = np.asarray(reference, dtype=float)
    predicted = np.asarray(prediction, dtype=float)
    if truth.shape != predicted.shape or truth.ndim != 1:
        raise ValueError("reference and prediction must be same-length vectors")
    if truth.size == 0:
        return float("nan")
    ideal = np.argsort(-truth, kind="stable")
    denominator = _dcg(truth[ideal], k)
    if not math.isfinite(denominator) or denominator <= 0:
        return float("nan")

    order = np.argsort(-predicted, kind="stable")
    sorted_predictions = predicted[order]
    sorted_gains = np.power(2.0, truth[order]) - 1.0
    cutoff = min(k, truth.size)
    discounts = 1.0 / np.log2(np.arange(2, cutoff + 2, dtype=float))
    expected_dcg = 0.0
    start = 0
    while start < truth.size:
        end = start + 1
        while end < truth.size and math.isclose(
            float(sorted_predictions[end]),
            float(sorted_predictions[start]),
            rel_tol=0.0,
            abs_tol=1e-8,
        ):
            end += 1
        visible_end = min(end, cutoff)
        if start < visible_end:
            expected_dcg += float(np.mean(sorted_gains[start:end])) * float(
                np.sum(discounts[start:visible_end])
            )
        start = end
    return expected_dcg / denominator


def random_ndcg_at_k(reference: Sequence[float], k: int = 5) -> float:
    """Expected NDCG of a uniformly random ranking for this relevance vector."""

    truth = np.asarray(reference, dtype=float)
    if truth.ndim != 1:
        raise ValueError("reference must be a vector")
    if truth.size == 0:
        return float("nan")
    ideal = np.argsort(-truth, kind="stable")
    denominator = _dcg(truth[ideal], k)
    if not math.isfinite(denominator) or denominator <= 0:
        return float("nan")
    cutoff = min(k, truth.size)
    discounts = 1.0 / np.log2(np.arange(2, cutoff + 2, dtype=float))
    mean_gain = float(np.mean(np.power(2.0, truth) - 1.0))
    return mean_gain * float(np.sum(discounts)) / denominator


def _safe_correlation(function: Any, truth: np.ndarray, prediction: np.ndarray) -> float:
    if truth.size < 2 or np.std(truth) <= 1e-12 or np.std(prediction) <= 1e-12:
        return float("nan")
    result = function(truth, prediction)
    value = result.statistic if hasattr(result, "statistic") else result[0]
    return float(value)


def _query_metrics(truth: np.ndarray, prediction: np.ndarray, k: int) -> dict[str, float]:
    best_truth = np.flatnonzero(np.isclose(truth, np.max(truth), atol=1e-12))
    best_prediction = np.flatnonzero(np.isclose(prediction, np.max(prediction), atol=1e-12))
    pairs = max(1, prediction.size * (prediction.size - 1) // 2)
    ties = sum(
        int(math.isclose(float(prediction[i]), float(prediction[j]), abs_tol=1e-8))
        for i in range(prediction.size)
        for j in range(i + 1, prediction.size)
    )
    truth_std = float(np.std(truth))
    prediction_std = float(np.std(prediction))
    tie_aware_ndcg = tie_aware_ndcg_at_k(truth, prediction, k)
    random_ndcg = random_ndcg_at_k(truth, k)
    lift_denominator = 1.0 - random_ndcg
    return {
        "ndcg_at_5": ndcg_at_k(truth, prediction, k),
        "tie_aware_ndcg_at_5": tie_aware_ndcg,
        "random_ndcg_at_5": random_ndcg,
        "ndcg_lift_over_random": (
            (tie_aware_ndcg - random_ndcg) / lift_denominator
            if lift_denominator > 1e-12
            else float("nan")
        ),
        "spearman": _safe_correlation(spearmanr, truth, prediction),
        "kendall": _safe_correlation(kendalltau, truth, prediction),
        "top1": float(bool(set(best_truth.tolist()) & set(best_prediction.tolist()))),
        "fractional_top1": float(
            len(set(best_truth.tolist()) & set(best_prediction.tolist())) / len(best_prediction)
        ),
        "separation_ratio": prediction_std / truth_std if truth_std > 1e-12 else float("nan"),
        "score_mean": float(np.mean(prediction)),
        "score_std": prediction_std,
        "score_range": float(np.max(prediction) - np.min(prediction)),
        "tie_rate": float(ties / pairs) if prediction.size > 1 else 0.0,
        "high_saturation": float(np.mean(prediction >= 0.95)),
        "low_saturation": float(np.mean(prediction <= 0.05)),
    }


def _nanmean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    if array.size == 0 or np.isnan(array).all():
        return float("nan")
    return float(np.nanmean(array))


def evaluate_predictions(
    groups: Sequence[RankingGroup],
    records: Sequence[ScoreRecord],
    k: int = 5,
) -> dict[str, Any]:
    """Compute query-macro metrics without imputing failed predictions."""

    record_map: dict[tuple[str, str], ScoreRecord] = {}
    duplicates: set[tuple[str, str]] = set()
    for record in records:
        key = (record.group_id, record.candidate_id)
        if key in record_map:
            duplicates.add(key)
        record_map[key] = record
    if duplicates:
        raise ValueError(f"duplicate predictions for {len(duplicates)} candidate(s)")

    per_query: list[dict[str, Any]] = []
    candidate_total = sum(len(group.candidates) for group in groups)
    parsed_total = 0
    complete_total = 0
    rankable_total = 0
    for group in groups:
        truth: list[float] = []
        prediction: list[float] = []
        complete = True
        for candidate in group.candidates:
            record = record_map.get((group.group_id, candidate.candidate_id))
            if record is None or record.score is None or record.parsing_status not in {"ok", "cached"}:
                complete = False
                continue
            if record.data_fingerprint != group.data_fingerprint:
                raise ValueError(f"fingerprint mismatch for group {group.group_id}")
            parsed_total += 1
            truth.append(candidate.score)
            prediction.append(record.score)
        rankable = len(group.candidates) >= 2 and float(np.std([candidate.score for candidate in group.candidates])) > 1e-12
        rankable_total += int(rankable)
        if not complete:
            continue
        complete_total += 1
        if not rankable:
            continue
        values = _query_metrics(np.asarray(truth), np.asarray(prediction), k)
        per_query.append(
            {
                "group_id": group.group_id,
                "split": group.split,
                "domain": group.domain,
                "candidate_count": len(group.candidates),
                **values,
            }
        )

    metric_names = [
        "ndcg_at_5",
        "tie_aware_ndcg_at_5",
        "random_ndcg_at_5",
        "ndcg_lift_over_random",
        "spearman",
        "kendall",
        "top1",
        "fractional_top1",
        "separation_ratio",
        "score_mean",
        "score_std",
        "score_range",
        "tie_rate",
        "high_saturation",
        "low_saturation",
    ]
    aggregate = {name: _nanmean(row[name] for row in per_query) for name in metric_names}
    aggregate.update(
        {
            "parsing_coverage": parsed_total / candidate_total if candidate_total else float("nan"),
            "complete_query_coverage": complete_total / len(groups) if groups else float("nan"),
            "rankable_query_coverage": len(per_query) / rankable_total if rankable_total else float("nan"),
            "candidate_count": candidate_total,
            "complete_query_count": complete_total,
            "evaluated_query_count": len(per_query),
            "rankable_query_count": rankable_total,
            "unrankable_query_count": len(groups) - rankable_total,
            "query_count": len(groups),
        }
    )
    by_domain: dict[str, dict[str, float]] = {}
    domains: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in per_query:
        domains[row["domain"]].append(row)
    for domain, rows in domains.items():
        by_domain[domain] = {name: _nanmean(row[name] for row in rows) for name in metric_names}
        by_domain[domain]["query_count"] = float(len(rows))
    return {"aggregate": aggregate, "by_domain": by_domain, "per_query": per_query}


def bootstrap_query_ci(
    per_query: Sequence[Mapping[str, Any]],
    metric: str,
    samples: int = 10_000,
    seed: int = 42,
    confidence: float = 0.95,
) -> dict[str, float]:
    values = np.asarray([row[metric] for row in per_query], dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"estimate": float("nan"), "low": float("nan"), "high": float("nan")}
    generator = np.random.default_rng(seed)
    draws = generator.choice(values, size=(samples, values.size), replace=True).mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    return {
        "estimate": float(values.mean()),
        "low": float(np.quantile(draws, tail)),
        "high": float(np.quantile(draws, 1.0 - tail)),
    }


def paired_bootstrap_test(
    left: Mapping[str, float],
    right: Mapping[str, float],
    samples: int = 10_000,
    seed: int = 42,
) -> dict[str, float]:
    common = sorted(set(left) & set(right))
    if not common:
        raise ValueError("paired test needs at least one common group_id")
    differences = np.asarray([left[key] - right[key] for key in common], dtype=float)
    differences = differences[np.isfinite(differences)]
    if differences.size == 0:
        raise ValueError("paired test has no finite differences")
    generator = np.random.default_rng(seed)
    draws = generator.choice(differences, size=(samples, differences.size), replace=True).mean(axis=1)
    probability_nonpositive = float(np.mean(draws <= 0.0))
    probability_nonnegative = float(np.mean(draws >= 0.0))
    return {
        "difference": float(differences.mean()),
        "p_value": min(1.0, 2.0 * min(probability_nonpositive, probability_nonnegative)),
        "query_count": float(differences.size),
    }


def holm_correction(p_values: Mapping[str, float], alpha: float = 0.05) -> dict[str, dict[str, Any]]:
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    count = len(ordered)
    adjusted_running = 0.0
    result: dict[str, dict[str, Any]] = {}
    rejected_so_far = True
    for rank, (name, value) in enumerate(ordered):
        adjusted_running = max(adjusted_running, min(1.0, (count - rank) * float(value)))
        threshold = alpha / (count - rank)
        rejected = rejected_so_far and value <= threshold
        rejected_so_far = rejected
        result[name] = {
            "p_value": float(value),
            "adjusted_p_value": adjusted_running,
            "threshold": threshold,
            "reject": rejected,
        }
    return result
