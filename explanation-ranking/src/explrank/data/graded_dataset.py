"""GradedExplanationDataset — query-level ranking groups."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import torch
from torch.utils.data import Dataset


def normalize_scores(scores: List[float], scale: float = 5.0) -> List[float]:
    mx = max(scores) if scores else 1.0
    if mx <= 0:
        return [0.0] * len(scores)
    return [min(s / scale, 1.0) for s in scores]


class GradedExplanationDataset(Dataset):
    """
    Each item is one query with K graded explanations.

    Expected dict keys: query, explanations, scores
    Optional: query_id, source
    """

    def __init__(
        self,
        examples: List[Dict[str, Any]],
        normalize: bool = True,
        score_scale: float = 5.0,
        min_candidates: int = 2,
        transform: Optional[Callable[[Dict], Dict]] = None,
    ):
        self.examples = []
        for ex in examples:
            if len(ex.get("explanations", [])) < min_candidates:
                continue
            scores = list(ex["scores"])
            if normalize:
                scores = normalize_scores(scores, scale=score_scale)
            item = {
                "query_id": ex.get("query_id", ""),
                "query": ex["query"],
                "explanations": ex["explanations"],
                "scores": scores,
                "source": ex.get("source", ""),
            }
            if transform:
                item = transform(item)
            self.examples.append(item)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self.examples[idx]
        return {
            "query": ex["query"],
            "explanations": ex["explanations"],
            "scores": torch.tensor(ex["scores"], dtype=torch.float32),
            "query_id": ex.get("query_id", str(idx)),
        }
