from __future__ import annotations

import json
import math
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import spearmanr
from torch import nn

from .judges import Judge, SCALAR_INPUT_TEMPLATE
from .metrics import evaluate_predictions
from .schema import RankingGroup, ScoreRecord
from .utils import hf_dtype_kwargs, resolve_hf_source, stable_hash


def credence_dropout_rates(
    head_count: int,
    minimum: float = 0.05,
    maximum: float = 0.30,
) -> list[float]:
    """Return the geometric dropout schedule used by CREDENCE.

    The schedule is geometric in keep probability and intentionally runs from
    the largest dropout to the smallest one, matching ``head_config.py`` in the
    official CREDENCE implementation.
    """

    if head_count < 1:
        raise ValueError("head_count must be positive")
    if not 0.0 <= minimum <= maximum < 1.0:
        raise ValueError("dropout bounds must satisfy 0 <= minimum <= maximum < 1")
    if head_count == 1:
        return [(minimum + maximum) / 2.0]
    log_keep_max = math.log(1.0 - maximum)
    log_keep_min = math.log(1.0 - minimum)
    return [
        1.0
        - math.exp(
            log_keep_max
            + (head_index / (head_count - 1)) * (log_keep_min - log_keep_max)
        )
        for head_index in range(head_count)
    ]


class _ScalarMLPHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dropout(hidden_states)
        hidden_states = torch.relu(self.fc1(hidden_states))
        return self.fc2(hidden_states)


class DiverseScalarHead(nn.Module):
    """CREDENCE-style ensemble of scalar heads over shared hidden states."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout_rates: Sequence[float],
    ) -> None:
        super().__init__()
        if not dropout_rates:
            raise ValueError("at least one dropout rate is required")
        self.heads = nn.ModuleList(
            _ScalarMLPHead(input_dim, hidden_dim, float(dropout))
            for dropout in dropout_rates
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.cat([head(hidden_states) for head in self.heads], dim=-1)


def build_diverse_scalar_head(
    input_dim: int,
    head_count: int,
    hidden_dim: int = 256,
    dropout_min: float = 0.05,
    dropout_max: float = 0.30,
) -> tuple[DiverseScalarHead, list[float]]:
    rates = credence_dropout_rates(head_count, dropout_min, dropout_max)
    return DiverseScalarHead(input_dim, hidden_dim, rates), rates


def summarize_head_scores(head_scores: Sequence[float]) -> dict[str, Any]:
    values = np.asarray(head_scores, dtype=float)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("epistemic scoring requires at least two scalar heads")
    if not np.isfinite(values).all() or np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("all scalar-head scores must be finite and lie in [0, 1]")
    lower = float(np.min(values))
    upper = float(np.max(values))
    return {
        "head_scores": values.tolist(),
        "score_mean": float(np.mean(values)),
        "credal_lower": lower,
        "credal_upper": upper,
        "credal_width": upper - lower,
        # Population variance implements Var_h[p_h] over the finite ensemble.
        "epistemic_variance": float(np.var(values, ddof=0)),
        "variance_estimator": "population_ddof_0",
    }


def make_epistemic_record(
    *,
    group: RankingGroup,
    candidate_id: str,
    head_scores: Sequence[float],
    model_name: str,
    model_revision: str,
    seed: int,
    inference_ms: float,
    checkpoint: str | Path | None = None,
) -> ScoreRecord:
    summary = summarize_head_scores(head_scores)
    metadata = {
        **summary,
        "method": "credence_head_disagreement",
        "head_count": len(summary["head_scores"]),
        "bounded_by": "sigmoid",
    }
    if checkpoint is not None:
        metadata["checkpoint"] = str(checkpoint)
    return ScoreRecord(
        group_id=group.group_id,
        candidate_id=candidate_id,
        model_name=model_name,
        model_revision=model_revision,
        prompt_hash=stable_hash("arr-epistemic-scalar-input-v1", SCALAR_INPUT_TEMPLATE),
        data_fingerprint=group.data_fingerprint,
        score=float(summary["score_mean"]),
        raw_output=json.dumps(summary, sort_keys=True, separators=(",", ":")),
        parsing_status="ok",
        seed=seed,
        inference_ms=inference_ms,
        metadata=metadata,
    )


def _record_head_scores(record: ScoreRecord, expected_count: int | None = None) -> list[float]:
    raw = record.metadata.get("head_scores")
    if not isinstance(raw, list):
        raise ValueError(
            f"missing head_scores for {record.group_id}/{record.candidate_id}"
        )
    values = [float(value) for value in raw]
    summarize_head_scores(values)
    if expected_count is not None and len(values) != expected_count:
        raise ValueError(
            f"inconsistent head count for {record.group_id}/{record.candidate_id}: "
            f"expected {expected_count}, found {len(values)}"
        )
    return values


def records_for_head(records: Sequence[ScoreRecord], head_index: int) -> list[ScoreRecord]:
    output: list[ScoreRecord] = []
    for record in records:
        scores = _record_head_scores(record)
        if not 0 <= head_index < len(scores):
            raise IndexError(f"head index {head_index} outside ensemble of size {len(scores)}")
        output.append(
            ScoreRecord(
                **{
                    **record.to_dict(),
                    "model_name": f"{record.model_name}#head-{head_index}",
                    "score": scores[head_index],
                    "raw_output": f"{scores[head_index]:.10f}",
                    "metadata": {
                        **record.metadata,
                        "ensemble_role": "individual_head",
                        "head_index": head_index,
                    },
                }
            )
        )
    return output


def _safe_spearman(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 2 or np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
        return float("nan")
    result = spearmanr(left, right)
    return float(result.statistic if hasattr(result, "statistic") else result[0])


def _selective_mae_auc(errors: np.ndarray, uncertainty: np.ndarray) -> float:
    if errors.size == 0:
        return float("nan")
    order = np.argsort(uncertainty, kind="stable")
    cumulative_risk = np.cumsum(errors[order]) / np.arange(1, errors.size + 1)
    # Discrete area over coverage levels 1/N, ..., 1.
    return float(np.mean(cumulative_risk))


def evaluate_epistemic_predictions(
    groups: Sequence[RankingGroup],
    records: Sequence[ScoreRecord],
) -> dict[str, Any]:
    """Evaluate CREDENCE-style uncertainty and its relation to scalar error."""

    by_key: dict[tuple[str, str], ScoreRecord] = {}
    head_count: int | None = None
    for record in records:
        key = (record.group_id, record.candidate_id)
        if key in by_key:
            raise ValueError(f"duplicate epistemic prediction for {key}")
        scores = _record_head_scores(record, head_count)
        head_count = head_count or len(scores)
        by_key[key] = record
    if head_count is None:
        raise ValueError("cannot evaluate an empty prediction set")

    truth_values: list[float] = []
    predicted_values: list[float] = []
    variances: list[float] = []
    widths: list[float] = []
    interval_hits: list[float] = []
    pair_correct = 0
    pair_incorrect = 0
    pair_overlap = 0

    for group in groups:
        group_rows: list[tuple[float, float, float]] = []
        for candidate in group.candidates:
            key = (group.group_id, candidate.candidate_id)
            if key not in by_key:
                raise ValueError(f"missing epistemic prediction for {key}")
            record = by_key[key]
            if record.data_fingerprint != group.data_fingerprint:
                raise ValueError(f"fingerprint mismatch for group {group.group_id}")
            scores = _record_head_scores(record, head_count)
            summary = summarize_head_scores(scores)
            if record.score is None or not math.isclose(
                record.score, summary["score_mean"], rel_tol=0.0, abs_tol=1e-7
            ):
                raise ValueError(f"central score is not the head mean for {key}")
            truth = float(candidate.score)
            prediction = float(summary["score_mean"])
            lower = float(summary["credal_lower"])
            upper = float(summary["credal_upper"])
            truth_values.append(truth)
            predicted_values.append(prediction)
            variances.append(float(summary["epistemic_variance"]))
            widths.append(float(summary["credal_width"]))
            interval_hits.append(float(lower <= truth <= upper))
            group_rows.append((truth, lower, upper))

        for left in range(len(group_rows)):
            for right in range(left + 1, len(group_rows)):
                if math.isclose(group_rows[left][0], group_rows[right][0], abs_tol=1e-12):
                    continue
                high, low = (
                    (group_rows[left], group_rows[right])
                    if group_rows[left][0] > group_rows[right][0]
                    else (group_rows[right], group_rows[left])
                )
                if high[1] > low[2]:
                    pair_correct += 1
                elif high[2] < low[1]:
                    pair_incorrect += 1
                else:
                    pair_overlap += 1

    truth_array = np.asarray(truth_values, dtype=float)
    prediction_array = np.asarray(predicted_values, dtype=float)
    variance_array = np.asarray(variances, dtype=float)
    width_array = np.asarray(widths, dtype=float)
    absolute_error = np.abs(prediction_array - truth_array)
    pair_total = pair_correct + pair_incorrect + pair_overlap

    score_bands: dict[str, dict[str, float | int]] = {}
    for threshold in (0.80, 0.90, 0.95):
        selected = prediction_array >= threshold
        score_bands[f"score_ge_{threshold:.2f}"] = {
            "candidate_count": int(np.sum(selected)),
            "candidate_fraction": float(np.mean(selected)),
            "mean_epistemic_variance": (
                float(np.mean(variance_array[selected])) if selected.any() else float("nan")
            ),
            "mean_credal_width": (
                float(np.mean(width_array[selected])) if selected.any() else float("nan")
            ),
            "mean_absolute_error": (
                float(np.mean(absolute_error[selected])) if selected.any() else float("nan")
            ),
        }

    head_metrics = [
        evaluate_predictions(groups, records_for_head(records, head_index))["aggregate"]
        for head_index in range(head_count)
    ]
    return {
        "method": "CREDENCE-style ensemble disagreement adapted to scalar judges",
        "head_count": head_count,
        "candidate_count": len(truth_values),
        "aggregate": {
            "mean_epistemic_variance": float(np.mean(variance_array)),
            "median_epistemic_variance": float(np.median(variance_array)),
            "mean_credal_width": float(np.mean(width_array)),
            "median_credal_width": float(np.median(width_array)),
            "reference_interval_coverage": float(np.mean(interval_hits)),
            "spearman_epistemic_vs_absolute_error": _safe_spearman(
                variance_array, absolute_error
            ),
            "spearman_width_vs_absolute_error": _safe_spearman(width_array, absolute_error),
            "selective_mae_auc_by_epistemic": _selective_mae_auc(
                absolute_error, variance_array
            ),
            "mean_absolute_error": float(np.mean(absolute_error)),
            "pairwise_correct_nonoverlap_rate": (
                pair_correct / pair_total if pair_total else float("nan")
            ),
            "pairwise_incorrect_nonoverlap_rate": (
                pair_incorrect / pair_total if pair_total else float("nan")
            ),
            "pairwise_interval_overlap_rate": (
                pair_overlap / pair_total if pair_total else float("nan")
            ),
            "pairwise_comparison_count": pair_total,
        },
        "score_bands": score_bands,
        "individual_head_metrics": head_metrics,
    }


class EpistemicScalarJudge(Judge):
    """Load a shared-backbone, multi-head scalar PEFT checkpoint."""

    def __init__(
        self,
        checkpoint: str | Path,
        seed: int = 42,
        batch_size: int = 16,
        max_length: int = 512,
        device_map: str | dict[str, Any] = "auto",
        dtype: str | None = None,
        local_files_only: bool = False,
    ) -> None:
        self.checkpoint = Path(checkpoint)
        self.seed = seed
        self.batch_size = batch_size
        self.max_length = max_length
        self.device_map = device_map
        self.dtype = dtype
        self.local_files_only = local_files_only
        self._model: Any = None
        self._tokenizer: Any = None
        self.model_name = str(self.checkpoint)
        self.model_revision = "local"
        self.head_count = 0

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from peft import PeftModel
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("transformers and peft are required for epistemic scoring") from exc

        manifest_path = self.checkpoint / "arr_model_manifest.json"
        if not manifest_path.exists():
            raise ValueError(f"missing epistemic checkpoint manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.head_count = int(manifest.get("epistemic_head_count", 0))
        if self.head_count < 2:
            raise ValueError("checkpoint is not a multi-head epistemic scalar judge")
        hidden_dim = int(manifest.get("epistemic_hidden_dim", 256))
        dropout_min = float(manifest.get("epistemic_dropout_min", 0.05))
        dropout_max = float(manifest.get("epistemic_dropout_max", 0.30))
        base_model = str(manifest["base_model"])
        base_revision = str(manifest.get("model_revision", "main"))
        resolved_base = resolve_hf_source(base_model, base_revision, self.local_files_only)
        tokenizer_source = self.checkpoint if any(self.checkpoint.glob("tokenizer*")) else resolved_base
        self._tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_source), use_fast=True)
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        model_dtype = getattr(torch, self.dtype or str(manifest.get("dtype", "bfloat16")))
        base = AutoModelForSequenceClassification.from_pretrained(
            resolved_base,
            num_labels=self.head_count,
            device_map=self.device_map,
            **hf_dtype_kwargs(model_dtype),
        )
        diverse_head, expected_rates = build_diverse_scalar_head(
            int(base.config.hidden_size),
            self.head_count,
            hidden_dim,
            dropout_min,
            dropout_max,
        )
        saved_rates = [float(value) for value in manifest.get("epistemic_dropout_rates", [])]
        if saved_rates and not np.allclose(saved_rates, expected_rates, rtol=0.0, atol=1e-12):
            raise ValueError("checkpoint dropout schedule does not match its manifest")
        diverse_head.to(device=next(base.parameters()).device, dtype=model_dtype)
        base.score = diverse_head
        base.config.num_labels = self.head_count
        base.config.pad_token_id = self._tokenizer.pad_token_id
        self._model = PeftModel.from_pretrained(base, str(self.checkpoint))
        self._model.config.pad_token_id = self._tokenizer.pad_token_id
        if hasattr(self._model, "get_base_model"):
            self._model.get_base_model().config.pad_token_id = self._tokenizer.pad_token_id
        self._model.eval()
        self.model_name = base_model
        self.model_revision = base_revision

    def score(self, groups: Sequence[RankingGroup]) -> list[ScoreRecord]:
        self._load()
        flattened = [(group, candidate) for group in groups for candidate in group.candidates]
        texts = [
            SCALAR_INPUT_TEMPLATE.format(question=group.question, candidate=candidate.text)
            for group, candidate in flattened
        ]
        predictions: list[tuple[list[float], float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            encoded = self._tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
            device = next(self._model.parameters()).device
            encoded = {key: value.to(device) for key, value in encoded.items()}
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            with torch.inference_mode():
                logits = self._model(**encoded).logits
                if logits.ndim != 2 or logits.shape[1] != self.head_count:
                    raise RuntimeError(
                        f"expected [batch, {self.head_count}] logits, found {tuple(logits.shape)}"
                    )
                scores = torch.sigmoid(logits.float()).cpu().tolist()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            per_candidate_ms = (time.perf_counter() - started) * 1000.0 / len(batch)
            predictions.extend((list(map(float, row)), per_candidate_ms) for row in scores)

        return [
            make_epistemic_record(
                group=group,
                candidate_id=candidate.candidate_id,
                head_scores=head_scores,
                model_name=self.model_name,
                model_revision=self.model_revision,
                seed=self.seed,
                inference_ms=inference_ms,
                checkpoint=self.checkpoint,
            )
            for (group, candidate), (head_scores, inference_ms) in zip(flattened, predictions)
        ]
