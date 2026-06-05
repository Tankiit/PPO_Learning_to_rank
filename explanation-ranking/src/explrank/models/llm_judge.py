"""LLMJudgeRewardModel (student) + EncoderRewardModel (teacher) for distillation.

RECONSTRUCTED from the status-doc spec — verify against your src/models.py.
Doc says: "EncoderRewardModel (teacher) + LLMJudgeRewardModel (student),
shared scoring head."

Pad-token note (your cluster section): decoder tokenizers have no pad token;
set tokenizer.pad_token = tokenizer.eos_token BEFORE tokenizing or the
last-token pooling grabs the wrong position.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer


class ScoringHead(nn.Module):
    """d_hidden -> d_hidden/2 -> 1. Shared design so distillation matches
    like-for-like (same head as the accepted paper)."""

    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, h):  # (B, hidden) -> (B,)
        return self.net(h).squeeze(-1)


def _last_token_hidden(hidden, attention_mask):
    """Pool last non-pad token per sequence (causal LM convention)."""
    idx = (attention_mask.sum(dim=1) - 1).clamp(min=0).long()
    return hidden[torch.arange(hidden.size(0)), idx]


class LLMJudgeRewardModel(nn.Module):
    """Decoder LLM + scalar scoring head. The judge under study.
    RECONSTRUCTED — your version may use a TRL value head or different pooling."""

    def __init__(self, model_name="meta-llama/Llama-3.1-8B", dropout=0.1, load_in_4bit=True):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        kwargs = {}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        self.backbone = AutoModelForCausalLM.from_pretrained(
            model_name, output_hidden_states=True, **kwargs
        )
        self.scoring_head = ScoringHead(self.backbone.config.hidden_size, dropout)

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = _last_token_hidden(out.hidden_states[-1], attention_mask)
        return self.scoring_head(pooled)


class EncoderRewardModel(nn.Module):
    """Bidirectional encoder + scoring head. Frozen teacher for distillation
    (Signal 2 from the accepted paper)."""

    def __init__(self, model_name="microsoft/deberta-v3-base", dropout=0.1):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)
        self.scoring_head = ScoringHead(self.backbone.config.hidden_size, dropout)

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        return self.scoring_head(out.last_hidden_state[:, 0, :])
