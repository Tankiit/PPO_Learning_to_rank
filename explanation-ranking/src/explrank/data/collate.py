"""Batching helpers for variable-length candidate lists."""

from __future__ import annotations

from typing import Any, Dict, List

import torch


def collate_by_query(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate query-level items. Does not tokenize — trainer calls model.encode_pairs.

    Returns lists per field; loss uses per-item forward in trainer for variable K.
    """
    return {
        "queries": [b["query"] for b in batch],
        "explanations": [b["explanations"] for b in batch],
        "scores": [b["scores"] for b in batch],
        "query_ids": [b.get("query_id", "") for b in batch],
    }


def pad_1d(tensors: List[torch.Tensor], pad_value: float = 0.0) -> torch.Tensor:
    max_len = max(t.numel() for t in tensors)
    out = torch.full((len(tensors), max_len), pad_value, dtype=tensors[0].dtype)
    for i, t in enumerate(tensors):
        out[i, : t.numel()] = t
    return out
