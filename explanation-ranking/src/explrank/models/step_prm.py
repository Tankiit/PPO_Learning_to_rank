"""StepLevelRankingModel — Extension 1 step-level PRM."""

from __future__ import annotations

import re
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from explrank.models.reward_model import EncoderRewardModel

STEP_TOKEN = "[STEP]"
SENT_SPLIT_RE = re.compile(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s+")


def split_into_steps(text: str, max_steps: int = 5) -> List[str]:
    if not text or not text.strip():
        return [""]
    parts = [p.strip() for p in SENT_SPLIT_RE.split(text.strip()) if p.strip()]
    return (parts or [text.strip()])[:max_steps]


class StepLevelRankingModel(nn.Module):
    """
    Step-level scores via [STEP] token hidden states; explanation score = min(step).
    """

    def __init__(
        self,
        base_model: str = "microsoft/deberta-v3-base",
        dropout: float = 0.1,
        use_cls_auxiliary: bool = False,
    ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        if STEP_TOKEN not in self.tokenizer.get_vocab():
            self.tokenizer.add_tokens([STEP_TOKEN])
        self.encoder = AutoModel.from_pretrained(base_model)
        self.encoder.resize_token_embeddings(len(self.tokenizer))
        hidden = self.encoder.config.hidden_size
        self.step_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )
        self.use_cls_auxiliary = use_cls_auxiliary
        if use_cls_auxiliary:
            self.cls_head = nn.Linear(hidden, 1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        step_positions: List[List[int]],
    ) -> Dict[str, torch.Tensor]:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        batch_step_scores = []
        for b, positions in enumerate(step_positions):
            if not positions:
                batch_step_scores.append(hidden.new_zeros(1))
                continue
            step_h = hidden[b, positions[: len(positions)]]
            step_logits = self.step_head(step_h).squeeze(-1)
            batch_step_scores.append(step_logits)

        expl_scores = torch.stack([s.min() for s in batch_step_scores])
        out = {"step_scores": batch_step_scores, "explanation_scores": expl_scores}
        if self.use_cls_auxiliary and hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            out["cls_scores"] = torch.sigmoid(self.cls_head(outputs.pooler_output).squeeze(-1))
        return out

    def combined_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        gold_scores: torch.Tensor,
        step_labels: List[List[int]],
        rank_loss_fn: nn.Module,
        lambda_rank: float = 1.0,
        lambda_step: float = 0.5,
    ) -> torch.Tensor:
        l_rank = rank_loss_fn(
            outputs["explanation_scores"].unsqueeze(0),
            gold_scores.unsqueeze(0),
        )
        step_bce = torch.tensor(0.0, device=gold_scores.device)
        n = 0
        for step_s, labels in zip(outputs["step_scores"], step_labels):
            m = min(len(step_s), len(labels))
            if m == 0:
                continue
            targets = torch.tensor(labels[:m], dtype=torch.float32, device=step_s.device)
            step_bce = step_bce + F.binary_cross_entropy_with_logits(step_s[:m], targets)
            n += 1
        if n > 0:
            step_bce = step_bce / n
        return lambda_rank * l_rank + lambda_step * step_bce
