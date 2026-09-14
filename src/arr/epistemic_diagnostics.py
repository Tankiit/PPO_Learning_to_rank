"""Diagnostics for the multi-head epistemic signal.

Run these before extending the epistemic work. They answer one question: is
``credal_width`` measuring epistemic uncertainty, or is it measuring how far
the heads are from convergence?

The concern is structural. In ``training.py`` every head minimises the same
ranking loss against the same targets from the same hidden state, so they
share one minimiser and converge together. Dropout rate is a regulariser: it
perturbs the optimisation path, not the optimum. If that is what is happening,
width has the wrong monotonicity -- train longer, look more certain -- and no
downstream use of it is sound.

D1  head redundancy       pairwise Spearman between heads
D2  dynamic range         coefficient of variation of width
D3  width versus epoch    THE DECISIVE ONE

D3 is decisive because epistemic uncertainty is a property of data coverage
and should be roughly stable once the model fits, whereas an optimisation
artefact decays monotonically with training.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from .data import load_groups
from .epistemic import _record_head_scores, evaluate_epistemic_predictions
from .schema import RankingGroup, ScoreRecord
from .utils import read_jsonl, write_json


def segment_ids_from_sizes(
    group_sizes: Sequence[int], device: Any = None
) -> Tensor:
    """Return the ranking-group index of every flattened candidate."""

    return torch.repeat_interleave(
        torch.arange(len(group_sizes), device=device),
        torch.tensor(list(group_sizes), device=device),
    )


def segment_softmax(scores: Tensor, segment_ids: Tensor, group_count: int) -> Tensor:
    """Softmax candidates within each ranking group, independently per member."""

    if scores.ndim != 2:
        raise ValueError(
            f"expected [candidates, members] scores, found {tuple(scores.shape)}"
        )
    member_count = scores.shape[1]
    index = segment_ids.long().unsqueeze(-1).expand(-1, member_count)
    maxima = torch.full(
        (group_count, member_count),
        float("-inf"),
        dtype=scores.dtype,
        device=scores.device,
    ).scatter_reduce_(0, index, scores, reduce="amax", include_self=False)
    shifted = (scores - maxima.gather(0, index)).exp()
    totals = torch.zeros(
        (group_count, member_count), dtype=scores.dtype, device=scores.device
    ).index_add_(0, segment_ids.long(), shifted)
    return shifted / totals.gather(0, index).clamp(min=1e-30)


def _make_vmap_member(input_dim: int, hidden_dim: int, dropout: float) -> nn.Module:
    member = nn.Sequential(
        nn.Dropout(float(dropout)),
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 1),
    )
    for layer in member:
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain("relu"))
            nn.init.zeros_(layer.bias)
    return member


class VmapEnsembleHead(nn.Module):
    """Independently parameterised scalar members evaluated with ``vmap``."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        head_count: int,
        dropout: float | Sequence[float] = 0.1,
        use_vmap: bool = True,
    ) -> None:
        super().__init__()
        if head_count < 2:
            raise ValueError("an ensemble needs at least two members")
        self.head_count = int(head_count)
        self._use_vmap = bool(use_vmap)
        self.stochastic_inference = False
        rates = (
            [float(dropout)] * head_count
            if isinstance(dropout, (int, float))
            else [float(rate) for rate in dropout]
        )
        if len(rates) != head_count:
            raise ValueError("one dropout rate per member is required")
        if self._use_vmap and len(set(rates)) > 1:
            raise ValueError(
                "the vmap path requires one shared dropout rate; "
                "pass use_vmap=False for heterogeneous members"
            )
        members = [_make_vmap_member(input_dim, hidden_dim, rate) for rate in rates]
        if not self._use_vmap:
            self.members = nn.ModuleList(members)
            return
        from torch.func import stack_module_state

        stacked, buffers = stack_module_state(members)
        if buffers:
            raise ValueError("vmap members must be buffer-free")
        self._names = list(stacked)
        self.stacked = nn.ParameterList(
            nn.Parameter(stacked[name].detach().clone()) for name in self._names
        )
        object.__setattr__(
            self,
            "_template",
            _make_vmap_member(input_dim, hidden_dim, rates[0]).to("meta"),
        )

    @property
    def use_vmap(self) -> bool:
        return self._use_vmap

    @use_vmap.setter
    def use_vmap(self, value: bool) -> None:
        if bool(value) != self._use_vmap:
            raise AttributeError("use_vmap is fixed at construction")

    def member_parameters(self) -> dict[str, Tensor]:
        return {name: parameter for name, parameter in zip(self._names, self.stacked)}

    def forward(self, hidden_states: Tensor, member_batch: Tensor | None = None) -> Tensor:
        if not self.use_vmap:
            return torch.cat(
                [
                    member(
                        hidden_states if member_batch is None else member_batch[index]
                    )
                    for index, member in enumerate(self.members)
                ],
                dim=-1,
            )
        from torch.func import functional_call, vmap

        template = self._template
        template.train(self.training or self.stochastic_inference)

        def forward_member(
            parameters: dict[str, Tensor], buffers: dict[str, Tensor], inputs: Tensor
        ) -> Tensor:
            return functional_call(template, (parameters, buffers), (inputs,))

        in_dims = (0, 0, None) if member_batch is None else (0, 0, 0)
        inputs = hidden_states if member_batch is None else member_batch
        stacked = vmap(
            forward_member, in_dims=in_dims, randomness="different"
        )(self.member_parameters(), {}, inputs)
        return stacked.squeeze(-1).movedim(0, -1)


def bootstrap_member_views(
    hidden_states: Tensor,
    head_count: int,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Return one group-preserving caller-defined batch view per member."""

    batch = hidden_states.shape[0]
    indices = torch.randint(
        batch,
        (head_count, batch),
        device=hidden_states.device,
        generator=generator,
    )
    return hidden_states[indices]


def _load_records(path: Path) -> list[ScoreRecord]:
    return [ScoreRecord.from_dict(row) for row in read_jsonl(path)]


def checkpoint_label(path: Path) -> str:
    """Name the checkpoint a predictions file came from.

    Two layouts occur. ``train_judge`` writes
    ``<run>/validation_predictions_epoch_N.jsonl`` directly, and re-scored
    checkpoints land at ``<run>/epoch_N/eval/predictions.jsonl``. Keying on
    ``path.parent.name`` collapses every epoch onto one entry in both cases,
    which silently disables D3 -- the decisive check -- because
    ``monotonic_decay`` needs at least three checkpoints to fire. Read the
    epoch out of the filename first, then fall back to an ``epoch_*`` ancestor.
    """

    match = re.search(r"epoch[_-]?(\d+)", path.stem)
    if match:
        return f"epoch_{int(match.group(1))}"
    for parent in path.parents:
        if parent.name.startswith("epoch"):
            return parent.name
    return path.parent.name or path.stem


def _head_matrix(records: Sequence[ScoreRecord], head_count: int | None = None) -> np.ndarray:
    rows = [_record_head_scores(record, head_count) for record in records]
    return np.asarray(rows, dtype=float)


def effective_ensemble_size(residuals: np.ndarray) -> float:
    """Effective members from the within-example disagreement spectrum.

    A covariance of raw residuals contains the irreducible error shared by all
    members as a rank-one component. That drives the participation ratio to
    one even for independently trained models. Subtracting each example's
    ensemble-mean residual removes this common direction. The disagreement
    subspace has rank at most H-1, so ``1 + PR`` maps collapsed members to 1
    and independent members to approximately H.
    """

    values = np.asarray(residuals, dtype=float)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("residuals must have shape [examples, members>=2]")
    disagreement = values - values.mean(axis=1, keepdims=True)
    covariance = np.cov(disagreement, rowvar=False, ddof=0)
    eigenvalues = np.clip(np.linalg.eigvalsh(covariance), 0.0, None)
    denominator = float(np.square(eigenvalues).sum())
    if denominator <= 1e-24:
        return 1.0
    return float(min(values.shape[1], 1.0 + np.square(eigenvalues.sum()) / denominator))


def mean_kl_to_consensus(probabilities: np.ndarray, *, eps: float = 1e-12) -> float:
    """Mean KL divergence from each member to the ensemble consensus.

    ``probabilities`` has shape ``[groups, candidates, members]``. The returned Jensen--Shannon diversity is
    ``H(p_bar) - mean_m H(p_m)``.  Unlike participation ratio this statistic is
    not scale-invariant: shrinking member deviations toward their consensus
    reduces it (approximately quadratically for small deviations).
    """
    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 3 or values.shape[2] < 2:
        raise ValueError("probabilities must have shape [groups, candidates, members>=2]")
    values = np.clip(values, eps, None)
    values = values / values.sum(axis=1, keepdims=True).clip(min=eps)
    consensus = values.mean(axis=2, keepdims=True)
    entropy_consensus = -np.sum(consensus * np.log(consensus), axis=1)
    entropy_members = -np.sum(values * np.log(values), axis=1).mean(axis=1, keepdims=True)
    return float(np.mean(entropy_consensus - entropy_members))


def head_redundancy(
    head_scores: np.ndarray, residuals: np.ndarray | None = None
) -> dict[str, Any]:
    """D1. Mean off-diagonal Spearman above 0.98 means the heads are one head
    wearing five hats and the credal set is effectively a point."""

    from scipy.stats import spearmanr

    head_count = head_scores.shape[1]
    matrix = np.eye(head_count)
    for i in range(head_count):
        for j in range(i + 1, head_count):
            left, right = head_scores[:, i], head_scores[:, j]
            if np.std(left) == 0.0 or np.std(right) == 0.0:
                value = float("nan")
            else:
                value = float(spearmanr(left, right).correlation)
            matrix[i, j] = matrix[j, i] = value
    off = matrix[~np.eye(head_count, dtype=bool)]
    finite = off[np.isfinite(off)]
    effective_size = effective_ensemble_size(
        head_scores if residuals is None else residuals
    )
    return {
        "pairwise_spearman": matrix.tolist(),
        "mean_offdiagonal": float(np.mean(finite)) if finite.size else float("nan"),
        "max_offdiagonal": float(np.max(finite)) if finite.size else float("nan"),
        "min_offdiagonal": float(np.min(finite)) if finite.size else float("nan"),
        "effective_ensemble_size": effective_size,
        "effective_size_definition": "1_plus_disagreement_covariance_participation_ratio",
        "redundant": bool(
            (finite.size and np.mean(finite) > 0.98) or effective_size < 1.5
        ),
    }


def width_dynamic_range(widths: np.ndarray) -> dict[str, Any]:
    """D2. A coefficient of variation below 0.15 means width is an offset
    rather than a signal. Sigmoid-bounded converged heads typically land
    around 1e-3."""

    mean = float(np.mean(widths))
    p05 = float(np.percentile(widths, 5))
    p95 = float(np.percentile(widths, 95))
    return {
        "mean": mean,
        "std": float(np.std(widths)),
        "coefficient_of_variation": float(np.std(widths) / max(mean, 1e-12)),
        "p05": p05,
        "p95": p95,
        "p95_over_p05": float(p95 / max(p05, 1e-12)),
        "constant": bool(np.std(widths) / max(mean, 1e-12) < 0.15),
    }


def width_by_epoch(per_epoch_widths: dict[str, np.ndarray]) -> dict[str, Any]:
    """D3. Scores the same evaluation set with successive checkpoints.

    Monotonic decay in mean width means the quantity tracks optimiser state,
    not data coverage, and the epistemic reading is unsupportable as written.
    """

    epochs = sorted(per_epoch_widths, key=lambda name: (len(name), name))
    means = [float(np.mean(per_epoch_widths[name])) for name in epochs]
    deltas = [b - a for a, b in zip(means, means[1:])]
    if len(means) < 3:
        trend = float("nan")
    else:
        from scipy.stats import spearmanr

        trend = float(spearmanr(np.arange(len(means)), np.asarray(means)).correlation)
    return {
        "checkpoints": epochs,
        "mean_width": means,
        "deltas": deltas,
        "trend_spearman": trend,
        "monotonic_decay": bool(np.isfinite(trend) and trend < -0.9),
        "relative_drop": float((means[0] - means[-1]) / max(means[0], 1e-12)) if means else 0.0,
    }


def _records_in_listwise_space(
    groups: Sequence[RankingGroup], records: Sequence[ScoreRecord]
) -> tuple[np.ndarray, np.ndarray]:
    """Map bounded member outputs into ListNet's identifiable score space."""

    by_key = {(record.group_id, record.candidate_id): record for record in records}
    probabilities = []
    targets = []
    sizes = []
    for group in groups:
        sizes.append(len(group.candidates))
        for candidate in group.candidates:
            key = (group.group_id, candidate.candidate_id)
            if key not in by_key:
                raise ValueError(f"missing prediction for {key}")
            probabilities.append(_record_head_scores(by_key[key]))
            targets.append(float(candidate.score))
    bounded = np.clip(np.asarray(probabilities, dtype=float), 1e-7, 1.0 - 1e-7)
    logits = np.log(bounded) - np.log1p(-bounded)
    segments = segment_ids_from_sizes(sizes)
    member_distribution = segment_softmax(
        torch.from_numpy(logits), segments, len(sizes)
    ).numpy()
    target_distribution = segment_softmax(
        torch.tensor(targets, dtype=torch.float64).unsqueeze(-1),
        segments,
        len(sizes),
    ).squeeze(-1).numpy()
    return member_distribution, target_distribution


def _partial_spearman(
    left: np.ndarray, right: np.ndarray, control: np.ndarray
) -> float:
    """First-order partial Spearman correlation."""

    from scipy.stats import spearmanr

    lr = float(spearmanr(left, right).correlation)
    lc = float(spearmanr(left, control).correlation)
    rc = float(spearmanr(right, control).correlation)
    denominator = math.sqrt(max((1.0 - lc * lc) * (1.0 - rc * rc), 0.0))
    if denominator <= 1e-12:
        return float("nan")
    return float((lr - lc * rc) / denominator)


def run_diagnostics(
    groups: Sequence[RankingGroup],
    records: Sequence[ScoreRecord],
    per_epoch_records: dict[str, Sequence[ScoreRecord]] | None = None,
    *,
    listwise: bool = False,
) -> dict[str, Any]:
    if listwise:
        head_scores, truth = _records_in_listwise_space(groups, records)
        widths = np.ptp(head_scores, axis=1)
        score_space = "within_group_softmax_of_logits"
        grouped = []
        start = 0
        for group in groups:
            stop = start + len(group.candidates)
            grouped.append(head_scores[start:stop])
            start = stop
        # Ragged groups are handled one at a time, then averaged.
        js_values = []
        for member_probs in grouped:
            member_probs = np.clip(member_probs, 1e-12, None)
            member_probs /= member_probs.sum(axis=0, keepdims=True).clip(min=1e-12)
            consensus = member_probs.mean(axis=1, keepdims=True)
            h_bar = -np.sum(consensus * np.log(consensus))
            h_members = -np.sum(member_probs * np.log(member_probs), axis=0).mean()
            js_values.append(float(h_bar - h_members))
    else:
        head_scores = _head_matrix(records)
        truth_by_key = {
            (group.group_id, candidate.candidate_id): float(candidate.score)
            for group in groups
            for candidate in group.candidates
        }
        truth = np.asarray(
            [truth_by_key[(record.group_id, record.candidate_id)] for record in records],
            dtype=float,
        )
        widths = np.asarray(
            [float(record.metadata["credal_width"]) for record in records], dtype=float
        )
        score_space = "bounded_record_scores"
    existing_metrics = evaluate_epistemic_predictions(groups, records)["aggregate"]
    if listwise:
        from scipy.stats import spearmanr

        absolute_error = np.abs(head_scores.mean(axis=1) - truth)
        confidence = np.empty_like(absolute_error)
        entropy = np.empty_like(absolute_error)
        start = 0
        for group in groups:
            stop = start + len(group.candidates)
            central = head_scores[start:stop].mean(axis=1)
            group_confidence = float(np.max(central))
            group_entropy = float(
                -np.sum(central * np.log(np.clip(central, 1e-12, None)))
            )
            confidence[start:stop] = group_confidence
            entropy[start:stop] = group_entropy
            start = stop
        original_scores = _head_matrix(records)
        near_lower = original_scores <= 1e-6
        near_upper = original_scores >= 1.0 - 1e-6
        existing_metrics = {
            **existing_metrics,
            "n_observations": int(head_scores.shape[0]),
            "n_groups": int(len(groups)),
            "spearman_width_vs_absolute_error": float(
                spearmanr(widths, absolute_error).correlation
            ),
            "mean_absolute_error": float(absolute_error.mean()),
            "epistemic_metric_score_space": score_space,
            "spearman_width_vs_confidence": float(
                spearmanr(widths, confidence).correlation
            ),
            "spearman_error_vs_confidence": float(
                spearmanr(absolute_error, confidence).correlation
            ),
            "partial_spearman_width_vs_error_given_confidence": _partial_spearman(
                widths, absolute_error, confidence
            ),
            "partial_spearman_width_vs_error_given_entropy": _partial_spearman(
                widths, absolute_error, entropy
            ),
            "stored_score_min": float(original_scores.min()),
            "stored_score_max": float(original_scores.max()),
            "stored_scores_le_1e-6": int(near_lower.sum()),
            "stored_scores_ge_1_minus_1e-6": int(near_upper.sum()),
            "stored_score_count": int(original_scores.size),
            "mean_kl_to_consensus": float(np.mean(js_values)),
        }
    report: dict[str, Any] = {
        "diagnostic_score_space": score_space,
        "D1_head_redundancy": head_redundancy(
            head_scores, head_scores - truth[:, None]
        ),
        "D2_width_dynamic_range": width_dynamic_range(widths),
        "D4_existing_metrics": existing_metrics,
    }
    report["sample_size"] = {"n_observations": int(head_scores.shape[0]), "n_groups": int(len(groups))}
    if per_epoch_records:
        report["D3_width_by_epoch"] = width_by_epoch(
            {
                name: (
                    np.ptp(_records_in_listwise_space(groups, recs)[0], axis=1)
                    if listwise
                    else np.asarray(
                        [float(r.metadata["credal_width"]) for r in recs], dtype=float
                    )
                )
                for name, recs in per_epoch_records.items()
            }
        )
    report["verdict"] = _verdict(report)
    return report


def _verdict(report: dict[str, Any]) -> dict[str, Any]:
    d1 = report["D1_head_redundancy"]["redundant"]
    d2 = report["D2_width_dynamic_range"]["constant"]
    d3 = report.get("D3_width_by_epoch", {}).get("monotonic_decay")
    failures = []
    if d1:
        failures.append("heads are mutually redundant (D1)")
    if d2:
        failures.append("width has no dynamic range (D2)")
    if d3:
        failures.append("width decays monotonically with training (D3)")
    return {
        "epistemic_signal_supported": not failures,
        "failures": failures,
        "recommended_action": (
            "proceed" if not failures
            else "enable epistemic_lambda_div > 0 and re-run; see docs/epistemic_experiments.md"
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="epistemic signal diagnostics")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument(
        "--epoch-predictions",
        type=Path,
        nargs="*",
        default=(),
        help="one predictions file per training epoch, in order, for D3",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--listwise", action="store_true")
    args = parser.parse_args(argv)

    groups = load_groups(args.data)
    records = _load_records(args.predictions)
    per_epoch = {checkpoint_label(path): _load_records(path)
                 for path in args.epoch_predictions}

    report = run_diagnostics(
        groups, records, per_epoch or None, listwise=bool(args.listwise)
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "epistemic_diagnostics.json", report)
    print(json.dumps(report["verdict"], indent=2))
    return 0 if report["verdict"]["epistemic_signal_supported"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
