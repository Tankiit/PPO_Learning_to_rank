from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .epistemic import build_diverse_scalar_head, make_epistemic_record
from .judges import SCALAR_INPUT_TEMPLATE
from .losses import get_loss
from .metrics import evaluate_predictions
from .schema import RankingGroup, ScoreRecord
from .utils import hf_dtype_kwargs, resolve_hf_source, stable_hash, write_json, write_jsonl


@dataclass
class RankingBatchCollator:
    tokenizer: Any
    max_length: int = 512

    def __call__(self, groups: Sequence[RankingGroup]) -> dict[str, Any]:
        import torch

        if not groups:
            raise ValueError("cannot collate an empty batch")
        maximum = max(len(group.candidates) for group in groups)
        targets = torch.zeros((len(groups), maximum), dtype=torch.float32)
        mask = torch.zeros((len(groups), maximum), dtype=torch.bool)
        group_indices: list[int] = []
        candidate_indices: list[int] = []
        texts: list[str] = []
        for group_index, group in enumerate(groups):
            for candidate_index, candidate in enumerate(group.candidates):
                targets[group_index, candidate_index] = candidate.score
                mask[group_index, candidate_index] = True
                group_indices.append(group_index)
                candidate_indices.append(candidate_index)
                texts.append(
                    SCALAR_INPUT_TEMPLATE.format(question=group.question, candidate=candidate.text)
                )
        encoded = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        return {
            "encoded": encoded,
            "targets": targets,
            "mask": mask,
            "group_indices": torch.tensor(group_indices, dtype=torch.long),
            "candidate_indices": torch.tensor(candidate_indices, dtype=torch.long),
            "groups": list(groups),
        }


def set_reproducible_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)


def _dtype(name: str) -> Any:
    import torch

    if not hasattr(torch, name):
        raise ValueError(f"unknown torch dtype: {name}")
    return getattr(torch, name)


def load_trainable_judge(config: dict[str, Any]) -> tuple[Any, Any, dict[str, Any]]:
    """Create a scalar HF model, optionally with a trainable NF4 QLoRA adapter."""

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig

    base_model = str(config["base_model"])
    revision = str(config.get("revision", "main"))
    local_files_only = bool(config.get("local_files_only", False))
    resolved_base = resolve_hf_source(base_model, revision, local_files_only)
    architecture = str(config.get("architecture", "decoder"))
    qlora = bool(config.get("qlora", architecture == "decoder"))
    resume_from = config.get("resume_from")
    tokenizer_source = str(resume_from or resolved_base)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        else:
            tokenizer.pad_token = tokenizer.eos_token

    epistemic_head_count = int(config.get("epistemic_heads", 1))
    if epistemic_head_count < 1:
        raise ValueError("epistemic_heads must be positive")
    common: dict[str, Any] = {
        "num_labels": epistemic_head_count,
        "problem_type": "regression",
        **hf_dtype_kwargs(_dtype(str(config.get("dtype", "bfloat16")))),
    }
    device_map = config.get("device_map")
    if qlora and device_map is None and torch.cuda.is_available():
        device_map = {"": int(os.environ.get("LOCAL_RANK", "0"))}
    if device_map is not None:
        common["device_map"] = device_map
    quantization_config = None
    if qlora:
        if not torch.cuda.is_available():
            raise RuntimeError("NF4 QLoRA requires a CUDA device")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=_dtype(str(config.get("dtype", "bfloat16"))),
            bnb_4bit_use_double_quant=True,
        )
        common["quantization_config"] = quantization_config

    model_source = resolved_base if qlora or not resume_from else str(resume_from)
    model = AutoModelForSequenceClassification.from_pretrained(model_source, **common)
    model.config.pad_token_id = tokenizer.pad_token_id
    if len(tokenizer) > model.get_input_embeddings().num_embeddings:
        model.resize_token_embeddings(len(tokenizer))

    epistemic_dropout_rates: list[float] | None = None
    if epistemic_head_count > 1:
        if architecture != "decoder":
            raise ValueError("the initial epistemic experiment supports decoder judges only")
        diverse_head, epistemic_dropout_rates = build_diverse_scalar_head(
            int(model.config.hidden_size),
            epistemic_head_count,
            int(config.get("epistemic_hidden_dim", 256)),
            float(config.get("epistemic_dropout_min", 0.05)),
            float(config.get("epistemic_dropout_max", 0.30)),
        )
        diverse_head.to(
            device=next(model.parameters()).device,
            dtype=_dtype(str(config.get("dtype", "bfloat16"))),
        )
        model.score = diverse_head
        model.config.num_labels = epistemic_head_count

    if qlora:
        from peft import LoraConfig, PeftModel, TaskType, get_peft_model, prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=bool(config.get("gradient_checkpointing", True))
        )
        if resume_from:
            model = PeftModel.from_pretrained(model, str(resume_from), is_trainable=True)
        else:
            lora = LoraConfig(
                task_type=TaskType.SEQ_CLS,
                inference_mode=False,
                r=int(config.get("lora_r", 16)),
                lora_alpha=int(config.get("lora_alpha", 32)),
                lora_dropout=float(config.get("lora_dropout", 0.05)),
                target_modules=list(
                    config.get(
                        "target_modules",
                        ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                    )
                ),
                modules_to_save=list(config.get("modules_to_save", ["score"])),
            )
            model = get_peft_model(model, lora)
    elif torch.cuda.is_available() and device_map is None:
        model = model.cuda()

    metadata = {
        "base_model": base_model,
        "model_revision": revision,
        "resolved_model_source": resolved_base,
        "dtype": str(config.get("dtype", "bfloat16")),
        "architecture": architecture,
        "qlora": qlora,
        "quantization": "NF4" if qlora else None,
        "epistemic_method": (
            "credence_head_disagreement" if epistemic_head_count > 1 else None
        ),
        "epistemic_head_count": epistemic_head_count,
        "epistemic_hidden_dim": int(config.get("epistemic_hidden_dim", 256)),
        "epistemic_dropout_min": float(config.get("epistemic_dropout_min", 0.05)),
        "epistemic_dropout_max": float(config.get("epistemic_dropout_max", 0.30)),
        "epistemic_dropout_rates": epistemic_dropout_rates,
        "epistemic_variance_estimator": (
            "population_ddof_0" if epistemic_head_count > 1 else None
        ),
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
    }
    return model, tokenizer, metadata


def _forward_ranking_batch(model: Any, batch: dict[str, Any], device: Any) -> tuple[Any, Any, Any]:
    import torch

    encoded = {key: value.to(device) for key, value in batch["encoded"].items()}
    targets = batch["targets"].to(device)
    mask = batch["mask"].to(device)
    group_indices = batch["group_indices"].to(device)
    candidate_indices = batch["candidate_indices"].to(device)
    logits = model(**encoded).logits
    if logits.ndim != 2:
        raise RuntimeError(f"expected rank-2 scalar-head logits, found {tuple(logits.shape)}")
    bounded = torch.sigmoid(logits.float())
    if bounded.shape[-1] == 1:
        bounded = bounded.squeeze(-1)
        score_shape = targets.shape
    else:
        score_shape = (*targets.shape, bounded.shape[-1])
    scores = bounded.new_zeros(score_shape)
    scores = scores.index_put((group_indices, candidate_indices), bounded)
    return scores, targets, mask


def _multihead_ranking_loss(
    loss_function: Any,
    scores: Any,
    targets: Any,
    mask: Any,
) -> Any:
    """Apply the selected ranking objective independently to every head."""

    import torch

    if scores.ndim == 2:
        return loss_function(scores, targets, mask)
    if scores.ndim != 3 or scores.shape[:2] != targets.shape:
        raise ValueError(
            f"expected [batch, candidates, heads] scores, found {tuple(scores.shape)}"
        )
    return torch.stack(
        [loss_function(scores[..., head], targets, mask) for head in range(scores.shape[-1])]
    ).mean()


def predict_groups(
    model: Any,
    tokenizer: Any,
    groups: Sequence[RankingGroup],
    base_model: str,
    model_revision: str,
    seed: int,
    batch_size: int = 8,
    max_length: int = 512,
) -> list[ScoreRecord]:
    import torch
    from torch.utils.data import DataLoader

    collator = RankingBatchCollator(tokenizer, max_length=max_length)
    loader = DataLoader(list(groups), batch_size=batch_size, shuffle=False, collate_fn=collator)
    device = next(model.parameters()).device
    model.eval()
    output: list[ScoreRecord] = []
    with torch.inference_mode():
        for batch in loader:
            started = time.perf_counter()
            scores, _, _ = _forward_ranking_batch(model, batch, device)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            groups_in_batch = batch["groups"]
            per_candidate_ms = elapsed_ms / sum(len(group.candidates) for group in groups_in_batch)
            for group_index, group in enumerate(groups_in_batch):
                for candidate_index, candidate in enumerate(group.candidates):
                    candidate_scores = scores[group_index, candidate_index]
                    if candidate_scores.ndim == 0:
                        score = float(candidate_scores.cpu())
                        output.append(
                            ScoreRecord(
                                group_id=group.group_id,
                                candidate_id=candidate.candidate_id,
                                model_name=base_model,
                                model_revision=model_revision,
                                prompt_hash=stable_hash(
                                    "arr-scalar-input-v1", SCALAR_INPUT_TEMPLATE
                                ),
                                data_fingerprint=group.data_fingerprint,
                                score=score,
                                raw_output=f"{score:.10f}",
                                parsing_status="ok",
                                seed=seed,
                                inference_ms=per_candidate_ms,
                                metadata={"bounded_by": "sigmoid"},
                            )
                        )
                    else:
                        output.append(
                            make_epistemic_record(
                                group=group,
                                candidate_id=candidate.candidate_id,
                                head_scores=candidate_scores.float().cpu().tolist(),
                                model_name=base_model,
                                model_revision=model_revision,
                                seed=seed,
                                inference_ms=per_candidate_ms,
                            )
                        )
    return output


def _save_checkpoint(
    model: Any,
    tokenizer: Any,
    directory: Path,
    manifest: dict[str, Any],
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(directory, safe_serialization=True)
    tokenizer.save_pretrained(directory)
    write_json(directory / "arr_model_manifest.json", manifest)


def train_judge(
    train_groups: Sequence[RankingGroup],
    validation_groups: Sequence[RankingGroup],
    config: dict[str, Any],
    run_dir: str | Path,
) -> dict[str, Any]:
    import torch
    from torch.optim import AdamW
    from torch.utils.data import DataLoader
    from transformers import get_linear_schedule_with_warmup

    output = Path(run_dir)
    output.mkdir(parents=True, exist_ok=True)
    seed = int(config.get("seed", 42))
    set_reproducible_seed(seed)
    loss_name = str(config.get("loss", "mse")).lower()
    loss_function = get_loss(loss_name)
    model, tokenizer, model_metadata = load_trainable_judge(config)
    architecture = model_metadata["architecture"]
    requested_epochs = int(config.get("epochs", 3 if architecture == "decoder" else 10))
    maximum_epochs = 3 if architecture == "decoder" else 10
    if requested_epochs > maximum_epochs:
        raise ValueError(f"{architecture} judge is capped at {maximum_epochs} epochs")
    batch_size = int(config.get("group_batch_size", 2 if architecture == "decoder" else 8))
    accumulation = int(config.get("gradient_accumulation_steps", 1))
    collator = RankingBatchCollator(tokenizer, int(config.get("max_length", 512)))
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        list(train_groups),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=collator,
        num_workers=0,
    )
    optimizer = AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(config.get("learning_rate", 2e-5)),
        weight_decay=float(config.get("weight_decay", 0.01)),
    )
    updates_per_epoch = math.ceil(len(loader) / accumulation)
    total_updates = max(1, updates_per_epoch * requested_epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(float(config.get("warmup_ratio", 0.03)) * total_updates),
        num_training_steps=total_updates,
    )

    start_epoch = 0
    best_ndcg = -float("inf")
    best_epoch = -1
    resume_from = config.get("resume_from")
    if resume_from:
        state_path = Path(resume_from) / "training_state.json"
        optimizer_path = Path(resume_from) / "optimizer.pt"
        if state_path.exists():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            start_epoch = int(state.get("epoch", -1)) + 1
            best_ndcg = float(state.get("best_ndcg", best_ndcg))
            best_epoch = int(state.get("best_epoch", best_epoch))
        if optimizer_path.exists():
            optimizer.load_state_dict(torch.load(optimizer_path, map_location="cpu", weights_only=False))

    device = next(model.parameters()).device
    curves: list[dict[str, Any]] = []
    patience = int(config.get("patience", 3)) if architecture == "encoder" else requested_epochs
    without_improvement = 0
    run_manifest = {
        "pipeline": "arr",
        "task": "train-judge",
        "status": "running",
        "started_at_unix": time.time(),
        "seed": seed,
        "loss": loss_name,
        "train_fingerprint": train_groups[0].data_fingerprint if train_groups else None,
        "validation_fingerprint": validation_groups[0].data_fingerprint if validation_groups else None,
        **model_metadata,
    }
    write_json(output / "run_manifest.json", run_manifest)
    try:
        for epoch in range(start_epoch, requested_epochs):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            epoch_losses: list[float] = []
            for step, batch in enumerate(loader):
                scores, targets, mask = _forward_ranking_batch(model, batch, device)
                loss = _multihead_ranking_loss(loss_function, scores, targets, mask) / accumulation
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite {loss_name} loss at epoch {epoch}, step {step}")
                loss.backward()
                epoch_losses.append(float(loss.detach().cpu()) * accumulation)
                final_micro_batch = step + 1 == len(loader)
                if (step + 1) % accumulation == 0 or final_micro_batch:
                    torch.nn.utils.clip_grad_norm_(
                        [parameter for parameter in model.parameters() if parameter.requires_grad],
                        float(config.get("max_grad_norm", 1.0)),
                    )
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

            validation_records = predict_groups(
                model,
                tokenizer,
                validation_groups,
                base_model=str(config["base_model"]),
                model_revision=str(config.get("revision", "main")),
                seed=seed,
                batch_size=int(config.get("eval_batch_size", batch_size)),
                max_length=int(config.get("max_length", 512)),
            )
            evaluation = evaluate_predictions(validation_groups, validation_records)
            ndcg = float(evaluation["aggregate"]["tie_aware_ndcg_at_5"])
            if not math.isfinite(ndcg):
                raise FloatingPointError("validation tie-aware NDCG@5 is not finite")
            epoch_row = {
                "epoch": epoch,
                "train_loss": float(np.mean(epoch_losses)),
                "validation": evaluation["aggregate"],
                "learning_rate": float(scheduler.get_last_lr()[0]),
            }
            curves.append(epoch_row)
            write_jsonl(output / f"validation_predictions_epoch_{epoch}.jsonl", validation_records)
            write_json(output / "curves.json", curves)
            improved = ndcg > best_ndcg + float(config.get("min_delta", 1e-6))
            if improved:
                best_ndcg = ndcg
                best_epoch = epoch
                without_improvement = 0
                checkpoint_manifest = {
                    **run_manifest,
                    "status": "checkpoint",
                    "selection_metric": "tie_aware_ndcg_at_5",
                    "selection_value": best_ndcg,
                    "selected_epoch": epoch,
                    "output_activation": "sigmoid",
                    "score_range": [0.0, 1.0],
                    "max_length": int(config.get("max_length", 512)),
                    "model_revision": str(config.get("revision", "main")),
                }
                _save_checkpoint(model, tokenizer, output / "best", checkpoint_manifest)
            else:
                without_improvement += 1

            last_manifest = {
                **run_manifest,
                "status": "checkpoint",
                "epoch": epoch,
                "best_epoch": best_epoch,
                "best_ndcg": best_ndcg,
                "output_activation": "sigmoid",
            }
            _save_checkpoint(model, tokenizer, output / "last", last_manifest)
            torch.save(optimizer.state_dict(), output / "last" / "optimizer.pt")
            write_json(
                output / "last" / "training_state.json",
                {"epoch": epoch, "best_epoch": best_epoch, "best_ndcg": best_ndcg},
            )
            if without_improvement >= patience:
                break
        run_manifest.update(
            {
                "status": "complete",
                "completed_at_unix": time.time(),
                "best_epoch": best_epoch,
                "best_tie_aware_ndcg_at_5": best_ndcg,
                "epochs_completed": len(curves),
            }
        )
        write_json(output / "run_manifest.json", run_manifest)
        return run_manifest
    except Exception as exc:
        run_manifest.update(
            {
                "status": "failed",
                "completed_at_unix": time.time(),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        write_json(output / "run_manifest.json", run_manifest)
        raise
