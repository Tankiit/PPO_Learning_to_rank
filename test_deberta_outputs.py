#!/usr/bin/env python3
"""
Quick test: verify DeBERTa-v3 produces finite outputs.
Run this to verify DeBERTa produces finite values before/after NaN fixes.
"""


def test_deberta_outputs():
    """Run this to verify DeBERTa produces finite values."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    model_name = "microsoft/deberta-v3-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    text = "This is a test explanation for NLI"
    enc = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

    with torch.no_grad():
        outputs = model(**enc)

    hs = outputs.last_hidden_state
    print(f"Hidden states: shape={hs.shape}")
    print(f"  min={hs.min():.4f}, max={hs.max():.4f}, mean={hs.mean():.4f}")
    print(f"  has NaN: {torch.isnan(hs).any()}")
    print(f"  has Inf: {torch.isinf(hs).any()}")

    # Check if pooler_output exists
    if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
        print(f"pooler_output: shape={outputs.pooler_output.shape}")
    else:
        print("pooler_output: NONE (will use mean pooling)")

    # Simulate projection head
    import torch.nn as nn
    head = nn.Sequential(
        nn.Linear(hs.shape[-1], hs.shape[-1] // 2),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(hs.shape[-1] // 2, 1),
    )

    # Default init
    pooled = hs[:, 0, :]  # CLS token
    out_default = head(pooled)
    print(f"\nDefault init output: {out_default.item():.4f}")

    # Xavier small init
    for m in head.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=0.01)
            nn.init.zeros_(m.bias)

    out_small = head(pooled)
    print(f"Small init output: {out_small.item():.4f}")
    print(f"\nSmall init is safer: {abs(out_small.item()) < abs(out_default.item())}")


if __name__ == "__main__":
    test_deberta_outputs()
