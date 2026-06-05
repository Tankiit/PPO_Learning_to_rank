"""EncoderRewardModel — DeBERTa/RoBERTa teacher for explanation scoring."""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class EncoderRewardModel(nn.Module):
    """Cross-encoder that scores (query, explanation) pairs in [0, 1]."""

    def __init__(
        self,
        base_model: str = "microsoft/deberta-v3-base",
        output_mode: str = "regression",
        dropout: float = 0.1,
        use_quantization: bool = False,
        trust_remote_code: bool = False,
    ):
        super().__init__()
        self.output_mode = output_mode
        self.base_model_name = base_model
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model, trust_remote_code=trust_remote_code
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model_type = self._detect_model_type(base_model)
        if use_quantization:
            try:
                from transformers import BitsAndBytesConfig

                qconfig = BitsAndBytesConfig(load_in_8bit=True)
                self.encoder = AutoModel.from_pretrained(
                    base_model,
                    quantization_config=qconfig,
                    device_map="auto",
                    trust_remote_code=trust_remote_code,
                )
            except ImportError:
                self.encoder = AutoModel.from_pretrained(
                    base_model, trust_remote_code=trust_remote_code
                )
        else:
            self.encoder = AutoModel.from_pretrained(
                base_model, trust_remote_code=trust_remote_code
            )

        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.output_activation = nn.Sigmoid() if output_mode == "regression" else nn.Identity()

    @staticmethod
    def _detect_model_type(base_model: str) -> str:
        name = base_model.lower()
        if any(x in name for x in ["gpt", "llama", "mistral", "phi"]):
            return "decoder"
        if any(x in name for x in ["t5", "bart"]):
            return "encoder-decoder"
        return "encoder"

    def encode_pairs(self, queries: List[str], explanations: List[str], max_length: int = 256):
        texts = [
            f"{q} {self.tokenizer.sep_token} {e}" for q, e in zip(queries, explanations)
        ]
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if self.model_type != "decoder" and token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids

        outputs = self.encoder(**kwargs)
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            pooled = outputs.pooler_output
        else:
            hidden = outputs.last_hidden_state
            mask = attention_mask.unsqueeze(-1).expand(hidden.size()).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

        logits = self.classifier(self.dropout(pooled))
        scores = self.output_activation(logits).squeeze(-1)
        return scores

    @torch.no_grad()
    def score_batch(
        self,
        queries: List[str],
        explanations: List[str],
        device: torch.device,
        max_length: int = 256,
    ) -> torch.Tensor:
        self.eval()
        enc = self.encode_pairs(queries, explanations, max_length=max_length)
        enc = {k: v.to(device) for k, v in enc.items()}
        return self(**enc)
