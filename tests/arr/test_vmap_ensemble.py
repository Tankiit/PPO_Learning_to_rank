import torch

from src.arr.epistemic_diagnostics import (
    VmapEnsembleHead,
    segment_ids_from_sizes,
    segment_softmax,
)


def test_vmap_gradients_dropout_state_and_round_trip() -> None:
    torch.manual_seed(0)
    head = VmapEnsembleHead(16, 8, 5, dropout=0.2)
    inputs = torch.randn(6, 16)
    head.train()
    output = head(inputs)
    output.sum().backward()
    assert output.shape == (6, 5)
    assert all(parameter.grad is not None for parameter in head.stacked)

    head.eval()
    with torch.no_grad():
        deterministic = head(inputs)
        assert torch.allclose(deterministic, head(inputs))
    head.stochastic_inference = True
    with torch.no_grad():
        assert not torch.allclose(head(inputs), head(inputs))
    head.stochastic_inference = False

    state = head.state_dict()
    restored = VmapEnsembleHead(16, 8, 5, dropout=0.2).eval()
    restored.load_state_dict(state)
    with torch.no_grad():
        assert torch.allclose(head(inputs), restored(inputs), atol=1e-6)


def test_segment_softmax_is_ragged_and_gauge_invariant() -> None:
    torch.manual_seed(1)
    sizes = [3, 5, 2, 4]
    segments = segment_ids_from_sizes(sizes)
    scores = torch.randn(sum(sizes), 5, dtype=torch.float64)
    probabilities = segment_softmax(scores, segments, len(sizes))
    totals = torch.zeros(len(sizes), 5, dtype=torch.float64).index_add_(
        0, segments, probabilities
    )
    assert torch.allclose(totals, torch.ones_like(totals), atol=1e-12)
    offsets = torch.randn(len(sizes), 5, dtype=torch.float64)
    shifted = segment_softmax(scores + offsets[segments], segments, len(sizes))
    assert torch.allclose(probabilities, shifted, atol=1e-12)
