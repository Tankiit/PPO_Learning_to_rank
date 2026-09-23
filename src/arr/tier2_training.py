"""Pythia training for the independent and shared Tier-2 rankers.

The experiment models are deliberately built from
``src.models.ranking_reward_model``.  This module adds experiment mechanics
only: grouped ranking losses, fixed bootstraps, shared-head ablations,
checkpointing, and auditable predictions in all three score spaces.
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from torch import nn
from torch.optim import AdamW
from transformers import get_scheduler

from src.models.ranking_reward_model import ProjectionHead, RankingRewardModel

from .data import load_groups
from .epistemic import (
    center_listwise_member_logits,
    combine_independent_member_records,
    summarize_head_scores,
)
from .judges import SCALAR_INPUT_TEMPLATE
from .losses import get_loss
from .metrics import evaluate_predictions
from .schema import RankingGroup, ScoreRecord, validate_disjoint_splits
from .training import RankingBatchCollator, set_reproducible_seed
from .utils import canonical_json, stable_hash, stable_seed, write_json, write_jsonl


PYTHIA_REPO_ID = "EleutherAI/pythia-70m"
PYTHIA_REVISION = "a39f36b100fe8a5377810d56c3f4789b9c53ac42"
INDEPENDENT_ARMS = {"independent", "independent_bootstrap"}
SHARED_ARMS = {
    "baseline",
    "bootstrap",
    "features",
    "bootstrap_features",
    "lambda_0p01",
    "lambda_0p1",
    "lambda_1",
}
LISTWISE_LOSSES = {"listnet", "listmle"}


class SharedProjectionEnsemble(nn.Module):
    """Five independent reward heads applied to one shared representation."""

    def __init__(
        self,
        hidden_size: int,
        head_count: int = 5,
        dropout: float = 0.1,
        feature_keep_fraction: float | None = 1.0,
        feature_keep_count: int | None = None,
        feature_seed: int = 42,
    ) -> None:
        super().__init__()
        if head_count < 2:
            raise ValueError("a shared ensemble needs at least two heads")
        if feature_keep_count is None:
            if feature_keep_fraction is None or not 0.0 < feature_keep_fraction <= 1.0:
                raise ValueError("feature_keep_fraction must lie in (0, 1]")
            keep = max(1, int(round(feature_keep_fraction * hidden_size)))
            scale = 1.0 / feature_keep_fraction
        else:
            if (
                isinstance(feature_keep_count, bool)
                or not isinstance(feature_keep_count, int)
                or feature_keep_count < 1
            ):
                raise ValueError("feature_keep_count must be a positive integer")
            keep = min(hidden_size, feature_keep_count)
            scale = hidden_size / keep
        self.heads = nn.ModuleList(
            [ProjectionHead(hidden_size, dropout=dropout) for _ in range(head_count)]
        )
        generator = torch.Generator().manual_seed(int(feature_seed))
        masks = torch.zeros((head_count, hidden_size), dtype=torch.float32)
        for head_index in range(head_count):
            selected = torch.randperm(hidden_size, generator=generator)[:keep]
            masks[head_index, selected] = scale
        self.register_buffer("feature_masks", masks, persistent=True)

    @property
    def head_count(self) -> int:
        return len(self.heads)

    def forward(self, representation: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [
                head(representation * self.feature_masks[index].to(representation.dtype))
                for index, head in enumerate(self.heads)
            ],
            dim=-1,
        )


class SharedRankingRewardModel(nn.Module):
    """A :class:`RankingRewardModel` backbone with multiple scalar heads."""

    def __init__(
        self,
        model_name: str,
        *,
        head_count: int = 5,
        dropout: float = 0.1,
        pooling: str = "last",
        feature_keep_fraction: float | None = 1.0,
        feature_keep_count: int | None = None,
        feature_seed: int = 42,
        revision: str | None = None,
        local_files_only: bool = True,
        torch_dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        base = RankingRewardModel(
            model_name=model_name,
            dropout=dropout,
            pooling=pooling,
            revision=revision,
            local_files_only=local_files_only,
            torch_dtype=torch_dtype,
        )
        self.model_name = model_name
        self.backbone = base.backbone
        self.pooling = base.pooling
        self.tokenizer = base.tokenizer
        self.projection = SharedProjectionEnsemble(
            self.backbone.config.hidden_size,
            head_count=head_count,
            dropout=dropout,
            feature_keep_fraction=feature_keep_fraction,
            feature_keep_count=feature_keep_count,
            feature_seed=feature_seed,
        )

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs: Any
    ) -> torch.Tensor:
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        representation = self.pooling(outputs.last_hidden_state, attention_mask)
        return self.projection(representation)


def member_seed(global_seed: int, member_id: int) -> int:
    """Match the reference runs: global seed 42 gives member seeds 42..46."""

    if member_id < 0:
        raise ValueError("member_id must be non-negative")
    return int(global_seed) + int(member_id)


def fixed_bootstrap_indices(
    group_count: int, *, global_seed: int, member_id: int
) -> list[int]:
    """Draw one group bootstrap and reuse it for every training epoch."""

    if group_count < 1:
        raise ValueError("cannot bootstrap an empty dataset")
    seed = stable_seed("arr-tier2-fixed-bootstrap-v1", global_seed, member_id)
    generator = np.random.default_rng(seed)
    return generator.integers(0, group_count, size=group_count).tolist()


def fixed_shared_bootstrap_counts(
    group_count: int, *, global_seed: int, head_count: int
) -> torch.Tensor:
    """Return fixed ``[groups, heads]`` bootstrap multiplicities."""

    counts = torch.zeros((group_count, head_count), dtype=torch.float32)
    for head_index in range(head_count):
        sampled = fixed_bootstrap_indices(
            group_count, global_seed=global_seed, member_id=head_index
        )
        counts[:, head_index] = torch.bincount(
            torch.tensor(sampled), minlength=group_count
        ).float()
    return counts


def softmax_residual_decorrelation(
    scores: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Mean squared off-diagonal correlation of probability residuals."""

    if scores.ndim != 3 or scores.shape[:2] != targets.shape:
        raise ValueError("scores must be [batch, candidates, heads]")
    residuals = []
    for row in range(scores.shape[0]):
        valid = mask[row].bool()
        member_probability = torch.softmax(scores[row, valid], dim=0)
        target_probability = torch.softmax(targets[row, valid], dim=0).unsqueeze(-1)
        residuals.append(member_probability - target_probability)
    residual = torch.cat(residuals, dim=0)
    if residual.shape[0] < 2:
        return scores.sum() * 0.0
    residual = residual - residual.mean(dim=0, keepdim=True)
    covariance = residual.T @ residual / residual.shape[0]
    scale = torch.sqrt(torch.diagonal(covariance).clamp_min(1e-12))
    correlation = covariance / (scale[:, None] * scale[None, :])
    off_diagonal = correlation - torch.diag_embed(torch.diagonal(correlation))
    count = scores.shape[-1] * max(scores.shape[-1] - 1, 1)
    return off_diagonal.square().sum() / count


def _scatter_logits(
    model: nn.Module, batch: dict[str, Any], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    encoded = {key: value.to(device) for key, value in batch["encoded"].items()}
    targets = batch["targets"].to(device)
    mask = batch["mask"].to(device)
    flat = model(**encoded).float()
    if flat.ndim == 1:
        flat = flat.unsqueeze(-1)
    if flat.ndim != 2:
        raise RuntimeError(f"expected [candidates, heads], found {tuple(flat.shape)}")
    output = flat.new_zeros((*targets.shape, flat.shape[-1]))
    output = output.index_put(
        (
            batch["group_indices"].to(device),
            batch["candidate_indices"].to(device),
        ),
        flat,
    )
    return output, targets, mask


def _per_group_member_losses(
    scores: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    loss_name: str,
) -> torch.Tensor:
    loss_function = get_loss(loss_name)
    losses = []
    for row in range(scores.shape[0]):
        losses.append(
            torch.stack(
                [
                    loss_function(
                        scores[row : row + 1, :, head],
                        targets[row : row + 1],
                        mask[row : row + 1],
                    )
                    for head in range(scores.shape[-1])
                ]
            )
        )
    return torch.stack(losses)


def grouped_loss(
    scores: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    *,
    loss_name: str,
    member_weights: torch.Tensor | None = None,
    decorrelation_lambda: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute the same query loss for every independent scalar head."""

    per_group = _per_group_member_losses(scores, targets, mask, loss_name)
    if member_weights is None:
        ranking = per_group.mean()
    else:
        weights = member_weights.to(device=scores.device, dtype=scores.dtype)
        if weights.shape != per_group.shape:
            raise ValueError(
                f"member weights {tuple(weights.shape)} != losses {tuple(per_group.shape)}"
            )
        ranking = (per_group * weights).mean()
    penalty = (
        softmax_residual_decorrelation(scores, targets, mask)
        if decorrelation_lambda > 0.0 and scores.shape[-1] > 1
        else scores.sum() * 0.0
    )
    total = ranking + float(decorrelation_lambda) * penalty
    return total, {
        "ranking_loss": float(ranking.detach().cpu()),
        "decorrelation_penalty": float(penalty.detach().cpu()),
    }


def _chunks(values: Sequence[int], size: int) -> Iterable[list[int]]:
    for start in range(0, len(values), size):
        yield list(values[start : start + size])


def _prediction_records(
    model: nn.Module,
    tokenizer: Any,
    groups: Sequence[RankingGroup],
    *,
    config: dict[str, Any],
    training_seed: int,
) -> list[ScoreRecord]:
    collator = RankingBatchCollator(tokenizer, max_length=int(config["max_length"]))
    device = next(model.parameters()).device
    model.eval()
    output: list[ScoreRecord] = []
    with torch.inference_mode():
        for group in groups:
            started = time.perf_counter()
            raw, _, _ = _scatter_logits(model, collator([group]), device)
            raw = raw[0, : len(group.candidates)]
            bounded = torch.sigmoid(raw)
            group_probability = torch.softmax(raw.double(), dim=0).float()
            elapsed = (time.perf_counter() - started) * 1000.0 / len(group.candidates)
            for candidate_index, candidate in enumerate(group.candidates):
                raw_scores = raw[candidate_index].cpu().tolist()
                bounded_scores = bounded[candidate_index].cpu().tolist()
                listwise_scores = group_probability[candidate_index].cpu().tolist()
                metadata = {
                    "construction": config["construction"],
                    "arm": config["arm"],
                    "loss": config["loss"],
                    "raw_head_scores": raw_scores,
                    "group_softmax_scores": listwise_scores,
                    "head_count": len(raw_scores),
                    "bounded_by": "sigmoid",
                    "training_seed": training_seed,
                }
                if len(bounded_scores) > 1:
                    summary = summarize_head_scores(bounded_scores)
                    metadata.update(summary)
                    score = float(summary["score_mean"])
                else:
                    score = float(bounded_scores[0])
                    metadata["raw_score"] = float(raw_scores[0])
                    metadata["group_softmax_score"] = float(listwise_scores[0])
                output.append(
                    ScoreRecord(
                        group_id=group.group_id,
                        candidate_id=candidate.candidate_id,
                        model_name=PYTHIA_REPO_ID,
                        model_revision=str(config["revision"]),
                        prompt_hash=stable_hash(
                            "arr-tier2-ranking-reward-v1", SCALAR_INPUT_TEMPLATE
                        ),
                        data_fingerprint=group.data_fingerprint,
                        score=score,
                        raw_output=canonical_json(
                            {
                                "raw": raw_scores,
                                "sigmoid": bounded_scores,
                                "group_softmax": listwise_scores,
                            }
                        ),
                        parsing_status="ok",
                        seed=training_seed,
                        inference_ms=elapsed,
                        metadata=metadata,
                    )
                )
    return output


def _save_model(model: nn.Module, tokenizer: Any, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    state = {
        name: value.detach().cpu().contiguous()
        for name, value in model.state_dict().items()
    }
    save_file(state, directory / "model.safetensors")
    tokenizer.save_pretrained(directory)


def _load_model_state(model: nn.Module, path: Path) -> None:
    missing, unexpected = model.load_state_dict(load_file(path), strict=False)
    if missing or unexpected:
        raise RuntimeError(f"checkpoint mismatch: missing={missing}, unexpected={unexpected}")


def _arm_settings(config: dict[str, Any]) -> dict[str, Any]:
    construction = str(config["construction"])
    arm = str(config["arm"])
    feature_keep_count = config.get("feature_keep_count")
    if feature_keep_count is not None:
        if construction != "shared" or arm not in {"features", "bootstrap_features"}:
            raise ValueError("feature_keep_count is only valid for shared feature-mask arms")
        if (
            isinstance(feature_keep_count, bool)
            or not isinstance(feature_keep_count, int)
            or feature_keep_count < 1
        ):
            raise ValueError("feature_keep_count must be a positive integer")
    if construction == "independent":
        if arm not in INDEPENDENT_ARMS:
            raise ValueError(f"unknown independent arm: {arm}")
        return {
            "bootstrap": arm.endswith("bootstrap"),
            "feature_keep_fraction": 1.0,
            "decorrelation_lambda": 0.0,
        }
    if construction != "shared" or arm not in SHARED_ARMS:
        raise ValueError(f"unknown shared arm: {arm}")
    lambdas = {"lambda_0p01": 0.01, "lambda_0p1": 0.1, "lambda_1": 1.0}
    if feature_keep_count is not None:
        feature_keep_fraction = None
    elif "features" in arm:
        feature_keep_fraction = 0.8
    else:
        feature_keep_fraction = 1.0
    return {
        "bootstrap": arm in {"bootstrap", "bootstrap_features"},
        "feature_keep_fraction": feature_keep_fraction,
        "decorrelation_lambda": lambdas.get(arm, 0.0),
    }


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    resolved = dict(config)
    required = {
        "construction",
        "arm",
        "loss",
        "model_path",
        "train_data",
        "validation_data",
        "global_seed",
        "epochs",
        "max_length",
        "group_batch_size",
        "gradient_accumulation_steps",
        "learning_rate",
    }
    missing = sorted(required - resolved.keys())
    if missing:
        raise ValueError(f"missing Tier-2 settings: {missing}")
    if resolved["loss"] not in {"mse", "listnet", "listmle"}:
        raise ValueError("loss must be mse, listnet, or listmle")
    if int(resolved["epochs"]) < 1:
        raise ValueError("epochs must be positive")
    if int(resolved["group_batch_size"]) < 1:
        raise ValueError("group_batch_size must be positive")
    if int(resolved["gradient_accumulation_steps"]) < 1:
        raise ValueError("gradient_accumulation_steps must be positive")
    if str(resolved.get("revision", PYTHIA_REVISION)) != PYTHIA_REVISION:
        raise ValueError("the Pythia revision must remain pinned")
    resolved.setdefault("revision", PYTHIA_REVISION)
    resolved.setdefault("model_name", PYTHIA_REPO_ID)
    resolved.setdefault("head_count", 5)
    resolved.setdefault("dropout", 0.1)
    resolved.setdefault("pooling", "last")
    resolved.setdefault("weight_decay", 0.01)
    resolved.setdefault("warmup_ratio", 0.03)
    resolved.setdefault("max_grad_norm", 1.0)
    resolved.setdefault("checkpoint_epochs", [1, 10, 25, 50])
    resolved.setdefault("eval_batch_size", 1)
    resolved.setdefault("member_id", None)
    resolved.setdefault("max_train_groups", None)
    resolved.setdefault("max_validation_groups", None)
    settings = _arm_settings(resolved)
    resolved.update(settings)
    if resolved["construction"] == "independent":
        if resolved["member_id"] is None:
            raise ValueError("independent training requires member_id")
        if not 0 <= int(resolved["member_id"]) < int(resolved["head_count"]):
            raise ValueError("member_id lies outside the ensemble")
    elif resolved["member_id"] is not None:
        raise ValueError("shared training must not set member_id")
    return resolved


def _build_model(config: dict[str, Any]) -> tuple[nn.Module, Any, int]:
    construction = config["construction"]
    global_seed = int(config["global_seed"])
    training_seed = (
        member_seed(global_seed, int(config["member_id"]))
        if construction == "independent"
        else global_seed
    )
    set_reproducible_seed(training_seed)
    if torch.cuda.is_available():
        # Pythia otherwise selects memory-efficient SDPA on H100, whose
        # backward pass is non-deterministic in the cluster's PyTorch build.
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    common = {
        "model_name": str(config["model_path"]),
        "dropout": float(config["dropout"]),
        "pooling": str(config["pooling"]),
        "revision": None,
        "local_files_only": True,
        "torch_dtype": torch.float32,
    }
    if construction == "independent":
        model: nn.Module = RankingRewardModel(**common)
    else:
        model = SharedRankingRewardModel(
            **common,
            head_count=int(config["head_count"]),
            feature_keep_fraction=(
                None if config["feature_keep_fraction"] is None
                else float(config["feature_keep_fraction"])
            ),
            feature_keep_count=config.get("feature_keep_count"),
            feature_seed=int(
                stable_seed("arr-tier2-features-v1", global_seed) % (2**63 - 1)
            ),
        )
    return model, model.tokenizer, training_seed


def train_tier2(config: dict[str, Any], output_dir: str | Path) -> dict[str, Any]:
    """Train one independent member or one complete shared-head model."""

    config = validate_config(config)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config_fingerprint = stable_hash("arr-tier2-config-v1", config)
    final_path = output / "_final.json"
    if final_path.exists():
        previous = json.loads(final_path.read_text(encoding="utf-8"))
        if previous.get("config_fingerprint") != config_fingerprint:
            raise ValueError("completed output directory belongs to another configuration")
        if previous.get("status") != "complete":
            raise ValueError("invalid final marker")
        return previous
    resolved_path = output / "resolved_config.json"
    if resolved_path.exists():
        previous = json.loads(resolved_path.read_text(encoding="utf-8"))
        if stable_hash("arr-tier2-config-v1", previous) != config_fingerprint:
            raise ValueError("output directory contains a different configuration")
    else:
        write_json(resolved_path, config)

    train_groups = load_groups(config["train_data"])
    validation_groups = load_groups(config["validation_data"])
    if config["max_train_groups"] is not None:
        train_groups = train_groups[: int(config["max_train_groups"])]
    if config["max_validation_groups"] is not None:
        validation_groups = validation_groups[: int(config["max_validation_groups"])]
    validate_disjoint_splits({"train": train_groups, "validation": validation_groups})
    if not train_groups or not validation_groups:
        raise ValueError("training and validation data must be non-empty")

    model, tokenizer, training_seed = _build_model(config)
    if not torch.cuda.is_available():
        raise RuntimeError("Tier-2 Pythia training requires a CUDA allocation")
    device = torch.device("cuda")
    model.to(device)
    collator = RankingBatchCollator(tokenizer, max_length=int(config["max_length"]))
    optimizer = AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )

    base_indices = list(range(len(train_groups)))
    if config["construction"] == "independent" and config["bootstrap"]:
        base_indices = fixed_bootstrap_indices(
            len(train_groups),
            global_seed=int(config["global_seed"]),
            member_id=int(config["member_id"]),
        )
    group_batch_size = int(config["group_batch_size"])
    accumulation = int(config["gradient_accumulation_steps"])
    micro_batches = math.ceil(len(base_indices) / group_batch_size)
    updates_per_epoch = math.ceil(micro_batches / accumulation)
    total_updates = int(config["epochs"]) * updates_per_epoch
    # A one-update smoke run must actually change the weights. A one-step
    # warmup initializes that sole optimizer step at LR=0.
    warmup_steps = (
        0
        if total_updates == 1
        else max(1, round(total_updates * float(config["warmup_ratio"])))
    )
    scheduler = get_scheduler(
        "linear",
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_updates,
    )
    shared_counts = (
        fixed_shared_bootstrap_counts(
            len(train_groups),
            global_seed=int(config["global_seed"]),
            head_count=int(config["head_count"]),
        )
        if config["construction"] == "shared" and config["bootstrap"]
        else None
    )

    history: list[dict[str, Any]] = []
    start_epoch = 0
    optimizer_updates = 0
    last_dir = output / "last"
    state_path = last_dir / "training_state.pt"
    if state_path.exists():
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        if state.get("config_fingerprint") != config_fingerprint:
            raise ValueError("resume checkpoint belongs to another configuration")
        _load_model_state(model, last_dir / "model.safetensors")
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch = int(state["epoch"])
        optimizer_updates = int(state["optimizer_updates"])
        history = json.loads((output / "history.json").read_text(encoding="utf-8"))

    manifest = {
        "status": "running",
        "config_fingerprint": config_fingerprint,
        "model": PYTHIA_REPO_ID,
        "revision": config["revision"],
        "construction": config["construction"],
        "arm": config["arm"],
        "loss": config["loss"],
        "global_seed": int(config["global_seed"]),
        "training_seed": training_seed,
        "member_id": config["member_id"],
        "head_count": 1 if config["construction"] == "independent" else int(config["head_count"]),
        "train_groups": len(train_groups),
        "validation_groups": len(validation_groups),
        "train_fingerprint": train_groups[0].data_fingerprint,
        "validation_fingerprint": validation_groups[0].data_fingerprint,
        "epochs_requested": int(config["epochs"]),
        "updates_per_epoch": updates_per_epoch,
        "total_optimizer_updates": total_updates,
        "warmup_steps": warmup_steps,
        "target_space": "human_crowd_mean_divided_by_5",
        "loss_score_space": "raw_scalar_output",
        "prediction_spaces": ["raw", "sigmoid", "within_group_softmax"],
        "fixed_bootstrap": bool(config["bootstrap"]),
        "feature_keep_fraction": config["feature_keep_fraction"],
        "decorrelation_lambda": float(config["decorrelation_lambda"]),
        "started_at_unix": time.time(),
    }
    if config.get("feature_keep_count") is not None:
        mask = model.projection.feature_masks[0]
        actual_count = int((mask > 0).sum().item())
        manifest.update(
            {
                "feature_keep_count_requested": int(config["feature_keep_count"]),
                "feature_keep_count_effective": actual_count,
                "feature_keep_fraction_effective": actual_count / mask.numel(),
                "feature_mask_scale": float(mask.max().item()),
            }
        )
    write_json(output / "run_manifest.json", manifest)

    try:
        for epoch_index in range(start_epoch, int(config["epochs"])):
            epoch_seed = stable_seed(
                "arr-tier2-epoch-rng-v1", training_seed, epoch_index
            ) % (2**63 - 1)
            torch.manual_seed(epoch_seed)
            torch.cuda.manual_seed_all(epoch_seed)
            model.train()
            epoch_indices = list(base_indices)
            random.Random(
                stable_seed(
                    "arr-tier2-epoch-order-v1", training_seed, epoch_index
                )
            ).shuffle(epoch_indices)
            optimizer.zero_grad(set_to_none=True)
            epoch_loss = []
            epoch_ranking = []
            epoch_penalty = []
            update_learning_rates = []
            epoch_started = time.perf_counter()
            batches = list(_chunks(epoch_indices, group_batch_size))
            for micro_index, indices in enumerate(batches):
                groups = [train_groups[index] for index in indices]
                logits, targets, mask = _scatter_logits(model, collator(groups), device)
                weights = shared_counts[indices] if shared_counts is not None else None
                loss, parts = grouped_loss(
                    logits,
                    targets,
                    mask,
                    loss_name=str(config["loss"]),
                    member_weights=weights,
                    decorrelation_lambda=float(config["decorrelation_lambda"]),
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"non-finite loss at epoch {epoch_index + 1}, batch {micro_index}"
                    )
                (loss / accumulation).backward()
                epoch_loss.append(float(loss.detach().cpu()))
                epoch_ranking.append(parts["ranking_loss"])
                epoch_penalty.append(parts["decorrelation_penalty"])
                update = (micro_index + 1) % accumulation == 0 or micro_index + 1 == len(batches)
                if update:
                    norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), float(config["max_grad_norm"])
                    )
                    if not torch.isfinite(norm):
                        raise FloatingPointError("non-finite gradient norm")
                    update_learning_rates.append(float(optimizer.param_groups[0]["lr"]))
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    optimizer_updates += 1

            records = _prediction_records(
                model,
                tokenizer,
                validation_groups,
                config=config,
                training_seed=training_seed,
            )
            prediction_path = output / f"validation_predictions_epoch_{epoch_index}.jsonl"
            write_jsonl(prediction_path, records)
            evaluation = evaluate_predictions(validation_groups, records)
            row = {
                "epoch": epoch_index + 1,
                "optimizer_updates": optimizer_updates,
                "training_loss": float(np.mean(epoch_loss)),
                "ranking_loss": float(np.mean(epoch_ranking)),
                "decorrelation_penalty": float(np.mean(epoch_penalty)),
                "gradient_learning_rate": float(scheduler.get_last_lr()[0]),
                "optimizer_learning_rate_min": float(min(update_learning_rates)),
                "optimizer_learning_rate_max": float(max(update_learning_rates)),
                "seconds": time.perf_counter() - epoch_started,
                "validation": evaluation["aggregate"],
                "prediction_file": prediction_path.name,
            }
            history.append(row)
            write_json(output / "history.json", history)

            _save_model(model, tokenizer, last_dir)
            torch.save(
                {
                    "config_fingerprint": config_fingerprint,
                    "epoch": epoch_index + 1,
                    "optimizer_updates": optimizer_updates,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                },
                state_path,
            )
            if epoch_index + 1 in {int(value) for value in config["checkpoint_epochs"]}:
                snapshot = output / "checkpoints" / f"epoch_{epoch_index + 1:04d}"
                _save_model(model, tokenizer, snapshot)
                write_json(snapshot / "metrics.json", row)

        if optimizer_updates != total_updates:
            raise RuntimeError(
                f"completed {optimizer_updates} updates, expected {total_updates}"
            )
        # Optimizer state is needed only while a run is resumable. Once the
        # final marker is about to be written, remove that large transient and
        # hard-link the duplicate final/last model when the filesystem allows
        # it. Scientific checkpoint weights and all predictions are retained.
        state_path.unlink(missing_ok=True)
        final_snapshot_model = (
            output
            / "checkpoints"
            / f"epoch_{int(config['epochs']):04d}"
            / "model.safetensors"
        )
        last_model = last_dir / "model.safetensors"
        final_model_hardlinked = False
        if final_snapshot_model.exists() and last_model.exists():
            try:
                last_model.unlink()
                os.link(final_snapshot_model, last_model)
                final_model_hardlinked = True
            except OSError:
                if not last_model.exists():
                    save_file(
                        {
                            name: value.detach().cpu().contiguous()
                            for name, value in model.state_dict().items()
                        },
                        last_model,
                    )
        final = {
            **manifest,
            "status": "complete",
            "completed_at_unix": time.time(),
            "epochs_completed": len(history),
            "optimizer_updates": optimizer_updates,
            "final_validation": history[-1]["validation"],
            "final_prediction_file": history[-1]["prediction_file"],
            "optimizer_state_retained": False,
            "final_model_hardlinked": final_model_hardlinked,
        }
        write_json(output / "run_manifest.json", final)
        write_json(final_path, final)
        return final
    except Exception as exc:
        manifest.update(
            {
                "status": "failed",
                "failed_at_unix": time.time(),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        write_json(output / "run_manifest.json", manifest)
        raise


def combine_independent_runs(
    run_dirs: Sequence[str | Path], output_dir: str | Path
) -> dict[str, Any]:
    """Combine five completed independent members at every common epoch."""

    if len(run_dirs) != 5:
        raise ValueError("an independent ensemble requires exactly five runs")
    roots = [Path(value) for value in run_dirs]
    manifests = [json.loads((root / "_final.json").read_text()) for root in roots]
    if any(item.get("status") != "complete" for item in manifests):
        raise ValueError("all five independent runs must be complete")
    comparable = {
        (item["global_seed"], item["loss"], item["arm"], item["validation_fingerprint"])
        for item in manifests
    }
    if len(comparable) != 1:
        raise ValueError("independent member manifests are incompatible")
    ids = sorted(int(item["member_id"]) for item in manifests)
    seeds = sorted(int(item["training_seed"]) for item in manifests)
    if ids != list(range(5)) or len(set(seeds)) != 5:
        raise ValueError(f"invalid independent ensemble ids={ids}, seeds={seeds}")

    epochs = None
    for root in roots:
        present = {
            int(path.stem.rsplit("_", 1)[-1])
            for path in root.glob("validation_predictions_epoch_*.jsonl")
        }
        epochs = present if epochs is None else epochs & present
    if not epochs:
        raise ValueError("members have no common prediction epoch")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    loss = str(manifests[0]["loss"])
    rows = []
    for epoch_index in sorted(epochs):
        members = []
        original_members = []
        for root in roots:
            records = [
                ScoreRecord.from_dict(value)
                for value in _read_jsonl(root / f"validation_predictions_epoch_{epoch_index}.jsonl")
            ]
            original_members.append(records)
            members.append(
                center_listwise_member_logits(records)
                if loss in LISTWISE_LOSSES
                else records
            )
        combined = combine_independent_member_records(
            members,
            model_name=f"{PYTHIA_REPO_ID}::{manifests[0]['arm']}",
            model_revision="five-independent-ranking-reward-models",
        )
        originals = [
            {(record.group_id, record.candidate_id): record for record in records}
            for records in original_members
        ]
        enriched = []
        for record in combined:
            key = (record.group_id, record.candidate_id)
            source = [member[key] for member in originals]
            raw_scores = [float(item.metadata["raw_score"]) for item in source]
            group_probabilities = [
                float(item.metadata["group_softmax_score"]) for item in source
            ]
            enriched.append(
                ScoreRecord(
                    **{
                        **record.to_dict(),
                        "raw_output": canonical_json(
                            {
                                "raw": raw_scores,
                                "sigmoid_aligned": record.metadata["head_scores"],
                                "group_softmax": group_probabilities,
                            }
                        ),
                        "metadata": {
                            **record.metadata,
                            "raw_head_scores": raw_scores,
                            "group_softmax_scores": group_probabilities,
                            "source_member_scores": [float(item.score) for item in source],
                        },
                    }
                )
            )
        combined = enriched
        path = output / f"validation_predictions_epoch_{epoch_index}.jsonl"
        write_jsonl(path, combined)
        rows.append({"epoch": epoch_index + 1, "prediction_file": path.name})
    validation_path = json.loads(
        (roots[0] / "resolved_config.json").read_text(encoding="utf-8")
    )["validation_data"]
    validation_groups = load_groups(validation_path)
    final_records = [
        ScoreRecord.from_dict(value)
        for value in _read_jsonl(output / rows[-1]["prediction_file"])
    ]
    final_evaluation = evaluate_predictions(validation_groups, final_records)["aggregate"]
    final = {
        "status": "complete",
        "construction": "independent_ensemble",
        "loss": loss,
        "arm": manifests[0]["arm"],
        "global_seed": manifests[0]["global_seed"],
        "member_ids": ids,
        "member_seeds": seeds,
        "epochs": rows,
        "final_prediction_file": rows[-1]["prediction_file"],
        "final_validation": final_evaluation,
        "validation_fingerprint": manifests[0]["validation_fingerprint"],
        "listwise_alignment": (
            "within_group_mean_zero_logit" if loss in LISTWISE_LOSSES else "none"
        ),
    }
    write_json(output / "_final.json", final)
    return final


def verify_tier2_run(run_dir: str | Path) -> dict[str, Any]:
    """Reload a finished run and prove that backbone and head weights changed."""

    root = Path(run_dir)
    config = validate_config(
        json.loads((root / "resolved_config.json").read_text(encoding="utf-8"))
    )
    final = json.loads((root / "_final.json").read_text(encoding="utf-8"))
    if final.get("status") != "complete":
        raise ValueError("run is not complete")
    fingerprint = stable_hash("arr-tier2-config-v1", config)
    if final.get("config_fingerprint") != fingerprint:
        raise ValueError("final manifest/config mismatch")
    model, tokenizer, training_seed = _build_model(config)
    initial = model.state_dict()
    checkpoint = load_file(root / "last" / "model.safetensors")
    if set(initial) != set(checkpoint):
        raise ValueError("checkpoint parameter keys do not match the model")

    changes: dict[str, dict[str, float | int]] = {}
    for component, prefix in (("backbone", "backbone."), ("heads", "projection.")):
        deltas = [
            (checkpoint[name].float() - value.detach().cpu().float()).abs()
            for name, value in initial.items()
            if name.startswith(prefix) and value.is_floating_point()
        ]
        changes[component] = {
            "changed_tensors": sum(int(torch.count_nonzero(delta).item() > 0) for delta in deltas),
            "changed_elements": sum(int(torch.count_nonzero(delta).item()) for delta in deltas),
            "max_abs_delta": max(float(delta.max().item()) for delta in deltas),
        }
        if changes[component]["changed_elements"] == 0:
            raise AssertionError(f"{component} weights did not change")

    _load_model_state(model, root / "last" / "model.safetensors")
    model.to(torch.device("cuda"))
    validation_groups = load_groups(config["validation_data"])
    if config["max_validation_groups"] is not None:
        validation_groups = validation_groups[: int(config["max_validation_groups"])]
    reproduced = _prediction_records(
        model,
        tokenizer,
        validation_groups,
        config=config,
        training_seed=training_seed,
    )
    saved = [
        ScoreRecord.from_dict(value)
        for value in _read_jsonl(root / str(final["final_prediction_file"]))
    ]
    expected = {(row.group_id, row.candidate_id): row for row in saved}
    actual = {(row.group_id, row.candidate_id): row for row in reproduced}
    if set(expected) != set(actual):
        raise AssertionError("reloaded prediction keys do not match")
    score_delta = max(
        abs(float(actual[key].score) - float(expected[key].score)) for key in expected
    )
    raw_delta = max(
        max(
            abs(left - right)
            for left, right in zip(
                actual[key].metadata["raw_head_scores"],
                expected[key].metadata["raw_head_scores"],
            )
        )
        for key in expected
    )
    if score_delta > 1e-7 or raw_delta > 1e-6:
        raise AssertionError(
            f"checkpoint reload changed predictions: score={score_delta}, raw={raw_delta}"
        )
    history = json.loads((root / "history.json").read_text(encoding="utf-8"))
    if not history or min(row["optimizer_learning_rate_max"] for row in history) <= 0.0:
        raise AssertionError("training used a zero learning rate")
    result = {
        "status": "ok",
        "config_fingerprint": fingerprint,
        "construction": config["construction"],
        "loss": config["loss"],
        "training_seed": training_seed,
        "changes": changes,
        "reloaded_candidates": len(actual),
        "maximum_score_delta": score_delta,
        "maximum_raw_delta": raw_delta,
        "optimizer_learning_rate_max": max(
            row["optimizer_learning_rate_max"] for row in history
        ),
    }
    write_json(root / "verification.json", result)
    return result


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def model_path_from_environment(configured: str | None = None) -> str:
    path = configured or os.environ.get("ARR_PYTHIA_MODEL_DIR")
    if not path:
        raise ValueError("set model_path or ARR_PYTHIA_MODEL_DIR")
    if not Path(path).is_dir():
        raise FileNotFoundError(path)
    return str(Path(path).resolve())
