#!/usr/bin/env python3
"""Validate the offline Pythia stack and all three objectives on one H100."""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch

from src.arr.losses import get_loss
from src.arr.schema import Candidate, RankingGroup
from src.arr.training import (
    RankingBatchCollator,
    _forward_ranking_batch,
    _multihead_ranking_loss,
    load_trainable_judge,
)


EXPECTED_REVISION = "a39f36b100fe8a5377810d56c3f4789b9c53ac42"


def _gradient_summary(parameters) -> dict[str, float | int]:
    gradients = [
        parameter.grad.detach().float()
        for parameter in parameters
        if parameter.requires_grad and parameter.grad is not None
    ]
    if not gradients or not all(torch.isfinite(gradient).all() for gradient in gradients):
        raise RuntimeError("missing or non-finite gradients")
    nonzero = sum(int(torch.count_nonzero(gradient).item()) for gradient in gradients)
    if nonzero == 0:
        raise RuntimeError("all checked gradients are zero")
    return {
        "tensors": len(gradients),
        "nonzero_elements": nonzero,
        "max_abs": max(float(gradient.abs().max().item()) for gradient in gradients),
    }


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable inside the Slurm allocation")
    device_name = torch.cuda.get_device_name(0)
    if "H100" not in device_name.upper():
        raise RuntimeError(f"expected an H100, found {device_name}")

    model_dir = Path(os.environ["ARR_PYTHIA_MODEL_DIR"])
    revision = os.environ.get("ARR_PYTHIA_REVISION", EXPECTED_REVISION)
    if revision != EXPECTED_REVISION:
        raise RuntimeError(f"unexpected Pythia revision: {revision}")

    config = {
        "base_model": str(model_dir),
        "revision": revision,
        "local_files_only": True,
        "architecture": "decoder",
        "qlora": False,
        "dtype": "float32",
        "device": "cuda",
        "epistemic_heads": 5,
        "scalar_mlp_head": True,
        "epistemic_hidden_dim": 256,
        "epistemic_dropout_min": 0.05,
        "epistemic_dropout_max": 0.30,
    }
    model, tokenizer, metadata = load_trainable_judge(config)
    group = RankingGroup(
        group_id="cril-pythia-preflight",
        split="preflight",
        domain="synthetic",
        question="Which candidate is the clearest explanation?",
        candidates=tuple(
            Candidate(
                candidate_id=f"candidate-{index}",
                text=text,
                score=score,
                score_provenance="preflight",
            )
            for index, (text, score) in enumerate(
                (
                    ("The answer follows from both stated facts.", 1.0),
                    ("The answer is plausible but omits one fact.", 0.5),
                    ("The answer gives no supporting reason.", 0.0),
                )
            )
        ),
        data_fingerprint="cril-pythia-preflight-v1",
    )
    batch = RankingBatchCollator(tokenizer, max_length=64)([group])
    device = torch.device("cuda")
    losses: dict[str, dict] = {}
    for name in ("listnet", "listmle", "mse"):
        model.zero_grad(set_to_none=True)
        scores, targets, mask = _forward_ranking_batch(model, batch, device, False)
        if tuple(scores.shape) != (1, 3, 5):
            raise RuntimeError(f"unexpected five-head score shape: {tuple(scores.shape)}")
        loss = _multihead_ranking_loss(get_loss(name), scores, targets, mask)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite {name} loss")
        loss.backward()
        losses[name] = {
            "value": float(loss.detach().cpu()),
            "head_gradients": _gradient_summary(model.score.parameters()),
            "backbone_gradients": _gradient_summary(model.gpt_neox.parameters()),
        }

    result = {
        "status": "ok",
        "device": device_name,
        "cuda": torch.version.cuda,
        "model": "EleutherAI/pythia-70m",
        "revision": revision,
        "model_dir": str(model_dir),
        "dtype": "float32",
        "head_count": 5,
        "losses": losses,
        "trainable_parameters": metadata["trainable_parameters"],
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
    }
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
