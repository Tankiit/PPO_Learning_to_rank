"""
Test pooling strategies.

Run: python -m tests.test_pooling
"""

import torch
import torch.nn.functional as F
from src.models.ranking_reward_model import (
    MeanPooling, CLSPooling, MaxPooling, AttentionPooling, LastTokenPooling,
    get_pooling
)


def test_shapes():
    """All pooling strategies produce [B, hidden] output."""
    B, seq_len, hidden = 4, 32, 768
    hidden_states = torch.randn(B, seq_len, hidden)
    attention_mask = torch.ones(B, seq_len)
    attention_mask[:, -5:] = 0  # padding

    for name in ["mean", "cls", "max", "last"]:
        pool = get_pooling(name)
        out = pool(hidden_states, attention_mask)
        assert out.shape == (B, hidden), f"{name}: expected {(B, hidden)}, got {out.shape}"
        print(f"  {name:12s}: {out.shape} ✓")

    pool = get_pooling("attention", hidden_size=hidden)
    out = pool(hidden_states, attention_mask)
    assert out.shape == (B, hidden), f"attention: expected {(B, hidden)}, got {out.shape}"
    print(f"  {'attention':12s}: {out.shape} ✓")


def test_attention_masking():
    """Attention pooling should zero out padding positions."""
    B, seq_len, hidden = 2, 10, 64
    hidden_states = torch.randn(B, seq_len, hidden)
    attention_mask = torch.ones(B, seq_len)
    attention_mask[:, -3:] = 0  # last 3 are padding

    pool = AttentionPooling(hidden)
    
    # Get weights
    projected = torch.tanh(pool.attention(hidden_states))
    scores = (projected * pool.query).sum(dim=-1)
    scores = scores.masked_fill(attention_mask == 0, -1e9)
    weights = F.softmax(scores, dim=-1)

    # Padding weights should be ~0
    assert weights[:, -1].max() < 1e-6, f"Padding got weight: {weights[:, -1].max()}"
    
    # Weights should sum to 1
    assert torch.allclose(weights.sum(dim=-1), torch.ones(B), atol=1e-5)
    
    print("  Attention masking: ✓")


def test_mean_vs_attention_different():
    """Attention pooling should produce different results than mean pooling."""
    B, seq_len, hidden = 4, 20, 128
    hidden_states = torch.randn(B, seq_len, hidden)
    attention_mask = torch.ones(B, seq_len)

    mean_pool = MeanPooling()
    attn_pool = AttentionPooling(hidden)

    mean_out = mean_pool(hidden_states, attention_mask)
    attn_out = attn_pool(hidden_states, attention_mask)

    # They should not be identical (attention has learned weights)
    diff = (mean_out - attn_out).abs().mean()
    assert diff > 0.01, f"Mean and attention too similar: diff={diff}"
    print(f"  Mean vs Attention diff: {diff:.4f} ✓ (sufficiently different)")


def test_max_pooling_correct():
    """Max pooling should take element-wise max of non-padding tokens."""
    B, seq_len, hidden = 1, 5, 3
    hidden_states = torch.tensor([[[1.0, 2.0, 0.5],
                                    [3.0, 0.1, 0.2],
                                    [0.1, 4.0, 0.3],
                                    [0.0, 0.0, 0.0],   # padding
                                    [0.0, 0.0, 0.0]]])  # padding
    attention_mask = torch.tensor([[1, 1, 1, 0, 0]])

    pool = MaxPooling()
    out = pool(hidden_states, attention_mask)
    
    expected = torch.tensor([[3.0, 4.0, 0.5]])
    assert torch.allclose(out, expected, atol=1e-4), f"Got {out}, expected {expected}"
    print("  Max pooling correctness: ✓")


def test_gradient_flow():
    """Attention pooling should allow gradients to flow to all non-padding tokens."""
    B, seq_len, hidden = 2, 8, 32
    hidden_states = torch.randn(B, seq_len, hidden, requires_grad=True)
    attention_mask = torch.ones(B, seq_len)
    attention_mask[:, -2:] = 0

    pool = AttentionPooling(hidden)
    out = pool(hidden_states, attention_mask)
    loss = out.sum()
    loss.backward()

    # Gradients should exist for non-padding tokens
    grads = hidden_states.grad
    assert grads is not None
    
    # Non-padding tokens should have non-zero gradients
    non_pad_grads = grads[:, :6, :].abs().mean()
    assert non_pad_grads > 0, "Non-padding tokens should have gradients"
    
    # Padding tokens should have ~zero gradients (masked out in softmax)
    pad_grads = grads[:, 6:, :].abs().mean()
    assert pad_grads < non_pad_grads * 0.01, f"Padding grads too large: {pad_grads}"
    
    print(f"  Gradient flow: non-pad={non_pad_grads:.4f}, pad={pad_grads:.6f} ✓")


if __name__ == "__main__":
    print("Testing pooling strategies:")
    print()
    
    print("1. Shape tests:")
    test_shapes()
    
    print("\n2. Attention masking:")
    test_attention_masking()
    
    print("\n3. Mean vs Attention differentiation:")
    test_mean_vs_attention_different()
    
    print("\n4. Max pooling correctness:")
    test_max_pooling_correct()
    
    print("\n5. Gradient flow:")
    test_gradient_flow()
    
    print("\n✅ All tests passed!")
