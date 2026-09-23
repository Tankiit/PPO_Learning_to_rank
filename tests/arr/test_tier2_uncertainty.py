from __future__ import annotations

import numpy as np
import pytest

from scripts.analyze_tier2_uncertainty import (
    _softmax,
    holm,
    r1_gauge_control,
    synthetic_controls,
)


@pytest.fixture
def example() -> tuple[np.ndarray, np.ndarray]:
    raw = np.asarray(
        [
            [[0.2, 0.1, 0.5, -0.1, 0.3], [0.0, 0.4, 0.1, 0.2, -0.2],
             [0.7, 0.3, -0.1, 0.6, 0.0]],
            [[-0.2, 0.3, 0.0, 0.5, 0.2], [0.3, -0.1, 0.4, 0.0, 0.1],
             [0.1, 0.5, 0.2, -0.3, 0.6]],
        ],
        dtype=np.float64,
    )
    target = np.asarray([[0.0, 0.4, 1.0], [0.3, 0.2, 0.8]], dtype=np.float64)
    return raw, target


@pytest.mark.parametrize("loss", ["listnet", "listmle"])
def test_listwise_gauge_preserves_loss_and_softmax(example, loss: str) -> None:
    raw, target = example
    result = r1_gauge_control(raw, target, loss)
    assert result["ranking_preserved"]
    assert result["loss_absolute_change"] < 1e-12
    assert result["softmax_max_absolute_change"] < 1e-12
    assert result["raw_width_before"] != pytest.approx(result["raw_width_after"])
    assert result["sigmoid_width_before"] != pytest.approx(result["sigmoid_width_after"])
    assert result["centred_sigmoid_width_before"] == pytest.approx(
        result["centred_sigmoid_width_after"], abs=1e-12
    )


def test_mse_has_no_listwise_shift_gauge(example) -> None:
    raw, target = example
    result = r1_gauge_control(raw, target, "mse")
    assert result["ranking_preserved"]
    assert result["loss_absolute_change"] > 1e-3


def test_synthetic_diversity_shape_and_scale() -> None:
    rows = synthetic_controls()
    assert rows["diverse"]["participation_ratio"] > 1.0
    assert rows["collapsed"]["participation_ratio"] == pytest.approx(0.0)
    assert rows["collapsed"]["js_normalised"] == pytest.approx(0.0, abs=1e-12)
    assert rows["affine_0p1"]["js_normalised"] < rows["diverse"]["js_normalised"]
    assert rows["affine_0p1"]["participation_ratio"] == pytest.approx(
        rows["diverse"]["participation_ratio"], abs=1e-10
    )
    assert rows["affine_0p1"]["covariance_trace"] == pytest.approx(
        rows["diverse"]["covariance_trace"] * 0.01, rel=1e-10
    )


def test_softmax_invariance_and_holm() -> None:
    scores = np.asarray([[[0.1, 0.3], [0.7, 0.2], [-0.3, 0.1]]])
    shifted = scores + np.asarray([[[10.0, -7.0]]])
    np.testing.assert_allclose(_softmax(scores, axis=1), _softmax(shifted, axis=1))
    adjusted = holm({"a": 0.01, "b": 0.02, "c": 0.2})
    assert adjusted == pytest.approx({"a": 0.03, "b": 0.04, "c": 0.2})
