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
    """Ensemble of scalar heads over shared hidden states.

    Note on naming: this is not CREDENCE's construction. CREDENCE varies LoRA
    rank across {4, 8, 16, 32, 64}, which changes each head's hypothesis space
    so the heads do not share an optimum. Here the hypothesis space is
    identical across heads and only the regularisation strength differs, so
    head diversity has to be induced by the objective (see
    ``decorrelation_penalty``) or by stochastic inference (``mc_dropout``).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout_rates: Sequence[float],
        mc_dropout: int = 0,
        feature_keep_fraction: float = 1.0,
        feature_seed: int = 0,
    ) -> None:
        super().__init__()
        if not dropout_rates:
            raise ValueError("at least one dropout rate is required")
        if mc_dropout < 0:
            raise ValueError("mc_dropout must be non-negative")
        if not 0.0 < feature_keep_fraction <= 1.0:
            raise ValueError("feature_keep_fraction must lie in (0, 1]")
        self.heads = nn.ModuleList(
            _ScalarMLPHead(input_dim, hidden_dim, float(dropout))
            for dropout in dropout_rates
        )
        # 0 preserves the existing deterministic behaviour exactly.
        self.mc_dropout = int(mc_dropout)
        self.feature_keep_fraction = float(feature_keep_fraction)
        generator = torch.Generator().manual_seed(int(feature_seed))
        if feature_keep_fraction == 1.0:
            masks = torch.ones((len(dropout_rates), input_dim))
        else:
            masks = (
                torch.rand((len(dropout_rates), input_dim), generator=generator)
                < feature_keep_fraction
            ).float()
            # Never create a member with an empty hypothesis space.
            for member in range(masks.shape[0]):
                if not masks[member].any():
                    masks[member, member % input_dim] = 1.0
            masks /= feature_keep_fraction
        self.register_buffer("feature_masks", masks, persistent=True)

    def _member_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [
                head(hidden_states * self.feature_masks[index].to(hidden_states.dtype))
                for index, head in enumerate(self.heads)
            ],
            dim=-1,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.mc_dropout > 0 and not self.training:
            # Stochastic passes over the heads only; the backbone stays in
            # eval(). Without this the sole diversity mechanism is switched
            # off at exactly the moment credal width is read.
            previous = [head.dropout.training for head in self.heads]
            for head in self.heads:
                head.dropout.train()
            try:
                samples = [
                    self._member_logits(hidden_states)
                    for _ in range(self.mc_dropout)
                ]
            finally:
                for head, was_training in zip(self.heads, previous):
                    head.dropout.train(was_training)
            # The credal set is over (head, sample) pairs, which is the honest
            # reading of what MC dropout provides.
            return torch.cat(samples, dim=-1)
        return self._member_logits(hidden_states)


def decorrelation_penalty(
    scores: "torch.Tensor", targets: "torch.Tensor", mask: "torch.Tensor"
) -> "torch.Tensor":
    """Push head residuals apart so the heads do not share a single optimum.

    ``scores`` is [batch, candidates, heads]; ``targets`` and ``mask`` are
    [batch, candidates]. Residual r_h = s_h - y. The penalty is the squared
    off-diagonal Gram of the centred residuals: heads that make the same
    errors are redundant, heads that make different errors span a wider credal
    set.

    Without this term every head minimises the same loss against the same
    targets from the same hidden state, so they converge together and the
    credal width decays with training rather than tracking data coverage.
    Dropout rate alone does not fix that -- it perturbs the optimisation path,
    not the optimum.

    This is negative correlation learning, and it is the ensemble form of the
    orthogonality condition in the SLVM separation result.
    """

    if scores.ndim != 3:
        raise ValueError(f"expected [batch, candidates, heads] scores, found {tuple(scores.shape)}")
    selected = mask.bool().unsqueeze(-1).expand_as(scores)
    residual = (scores - targets.unsqueeze(-1))[selected].view(-1, scores.shape[-1])
    if residual.shape[0] < 2:
        return scores.new_zeros(())
    residual = residual - residual.mean(dim=0, keepdim=True)
    gram = (residual.T @ residual) / residual.shape[0]
    off_diagonal = gram - torch.diag_embed(torch.diagonal(gram))
    head_count = scores.shape[-1]
    return off_diagonal.pow(2).sum() / max(head_count * (head_count - 1), 1)


def credal_summary(scores: "torch.Tensor") -> dict[str, "torch.Tensor"]:
    """Credal read-out over the head dimension of a [..., heads] tensor."""

    lower = scores.min(dim=-1).values
    upper = scores.max(dim=-1).values
    return {
        "mean": scores.mean(dim=-1),
        "lower": lower,
        "upper": upper,
        "width": upper - lower,
        "variance": scores.var(dim=-1, unbiased=False),
    }


def build_diverse_scalar_head(
    input_dim: int,
    head_count: int,
    hidden_dim: int = 256,
    dropout_min: float = 0.05,
    dropout_max: float = 0.30,
    mc_dropout: int = 0,
    feature_keep_fraction: float = 1.0,
    feature_seed: int = 0,
) -> tuple[DiverseScalarHead, list[float]]:
    rates = credence_dropout_rates(head_count, dropout_min, dropout_max)
    return DiverseScalarHead(
        input_dim,
        hidden_dim,
        rates,
        mc_dropout=mc_dropout,
        feature_keep_fraction=feature_keep_fraction,
        feature_seed=feature_seed,
    ), rates


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


def combine_independent_member_records(
    member_records: Sequence[Sequence[ScoreRecord]],
    *,
    model_name: str = "independent-backbone-ensemble",
    model_revision: str = "independent-members",
) -> list[ScoreRecord]:
    """Combine aligned scalar predictions from independently trained models.

    Each input sequence is one complete ensemble member. Alignment is checked
    by identifiers rather than file order so independently produced artifacts
    cannot silently attach a member score to the wrong candidate.
    """

    if len(member_records) < 2:
        raise ValueError("an independent ensemble requires at least two members")
    indexed: list[dict[tuple[str, str], ScoreRecord]] = []
    for member_index, records in enumerate(member_records):
        by_key: dict[tuple[str, str], ScoreRecord] = {}
        for record in records:
            key = (record.group_id, record.candidate_id)
            if key in by_key:
                raise ValueError(f"duplicate member {member_index} prediction for {key}")
            if record.score is None or not math.isfinite(float(record.score)):
                raise ValueError(f"invalid member {member_index} score for {key}")
            by_key[key] = record
        indexed.append(by_key)
    reference_keys = set(indexed[0])
    for member_index, by_key in enumerate(indexed[1:], start=1):
        if set(by_key) != reference_keys:
            missing = sorted(reference_keys - set(by_key))[:3]
            extra = sorted(set(by_key) - reference_keys)[:3]
            raise ValueError(
                f"member {member_index} prediction keys differ; missing={missing}, extra={extra}"
            )

    output: list[ScoreRecord] = []
    for reference in member_records[0]:
        key = (reference.group_id, reference.candidate_id)
        aligned = [by_key[key] for by_key in indexed]
        if any(row.data_fingerprint != reference.data_fingerprint for row in aligned):
            raise ValueError(f"member fingerprint mismatch for {key}")
        scores = [float(row.score) for row in aligned]
        summary = summarize_head_scores(scores)
        output.append(
            ScoreRecord(
                group_id=reference.group_id,
                candidate_id=reference.candidate_id,
                model_name=model_name,
                model_revision=model_revision,
                prompt_hash=reference.prompt_hash,
                data_fingerprint=reference.data_fingerprint,
                score=float(summary["score_mean"]),
                raw_output=json.dumps(summary, sort_keys=True, separators=(",", ":")),
                parsing_status="ok",
                seed=reference.seed,
                inference_ms=sum(float(row.inference_ms or 0.0) for row in aligned),
                metadata={
                    **summary,
                    "method": "independent_backbone_ensemble",
                    "head_count": len(scores),
                    "bounded_by": "sigmoid",
                    "member_models": [row.model_name for row in aligned],
                    "member_seeds": [row.seed for row in aligned],
                },
            )
        )
    return output


def center_listwise_member_logits(records: Sequence[ScoreRecord]) -> list[ScoreRecord]:
    """Fix ListNet's additive-logit gauge before comparing ensemble members.

    ListNet identifies rankings but not the absolute offset of a group's
    logits. Independently trained models can therefore represent the same
    ranking near opposite sigmoid endpoints, creating an arbitrary width near
    one. This transform converts scores back to logits, centres each member
    within each ranking group, and returns bounded scores. It preserves the
    member's ordering exactly while making cross-member levels comparable.
    """

    grouped: dict[str, list[ScoreRecord]] = {}
    for record in records:
        if record.score is None or not math.isfinite(float(record.score)):
            raise ValueError(
                f"invalid scalar score for {record.group_id}/{record.candidate_id}"
            )
        grouped.setdefault(record.group_id, []).append(record)
    output_by_key: dict[tuple[str, str], ScoreRecord] = {}
    epsilon = 1e-7
    for group_id, group_records in grouped.items():
        probabilities = np.clip(
            np.asarray([float(record.score) for record in group_records]),
            epsilon,
            1.0 - epsilon,
        )
        logits = np.log(probabilities) - np.log1p(-probabilities)
        centred = logits - float(np.mean(logits))
        aligned = 1.0 / (1.0 + np.exp(-centred))
        for record, score in zip(group_records, aligned):
            value = float(score)
            output_by_key[(record.group_id, record.candidate_id)] = ScoreRecord(
                **{
                    **record.to_dict(),
                    "score": value,
                    "raw_output": f"{value:.10f}",
                    "metadata": {
                        **record.metadata,
                        "listwise_logit_gauge": "within_group_mean_zero",
                        "pre_alignment_score": float(record.score),
                    },
                }
            )
    return [output_by_key[(record.group_id, record.candidate_id)] for record in records]


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
        mc_dropout: int | None = None,
    ) -> None:
        self.checkpoint = Path(checkpoint)
        self.seed = seed
        self.batch_size = batch_size
        self.max_length = max_length
        self.device_map = device_map
        self.dtype = dtype
        self.local_files_only = local_files_only
        self.mc_dropout = mc_dropout
        self._model: Any = None
        self._tokenizer: Any = None
        self.model_name = str(self.checkpoint)
        self.model_revision = "local"
        self.head_count = 0

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("transformers is required for epistemic scoring") from exc

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
            int(
                self.mc_dropout
                if self.mc_dropout is not None
                else manifest.get("epistemic_mc_dropout", 0)
            ),
            float(manifest.get("epistemic_feature_keep_fraction", 1.0)),
            int(manifest.get("epistemic_feature_seed", 0)),
        )
        saved_rates = [float(value) for value in manifest.get("epistemic_dropout_rates", [])]
        if saved_rates and not np.allclose(saved_rates, expected_rates, rtol=0.0, atol=1e-12):
            raise ValueError("checkpoint dropout schedule does not match its manifest")
        diverse_head.to(device=next(base.parameters()).device, dtype=model_dtype)
        base.score = diverse_head
        base.config.num_labels = self.head_count
        base.config.pad_token_id = self._tokenizer.pad_token_id
        if (self.checkpoint / "adapter_config.json").exists():
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise RuntimeError("peft is required to load this adapter checkpoint") from exc
            self._model = PeftModel.from_pretrained(base, str(self.checkpoint))
        else:
            from safetensors.torch import load_file

            shard = self.checkpoint / "model.safetensors"
            if not shard.exists():
                raise FileNotFoundError(f"no model weights found in {self.checkpoint}")
            base.load_state_dict(load_file(str(shard)), strict=False)
            self._model = base
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
                expected_count = self.head_count * max(int(self.mc_dropout or 0), 1)
                if logits.ndim != 2 or logits.shape[1] != expected_count:
                    raise RuntimeError(
                        f"expected [batch, {expected_count}] logits, found {tuple(logits.shape)}"
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
