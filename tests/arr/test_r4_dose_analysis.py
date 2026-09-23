from __future__ import annotations

import numpy as np

from scripts.analyze_r4_dose import hierarchical_endpoint_interval, scored_run


def test_r4_scored_run_reuses_only_exact_loso_baseline(tmp_path) -> None:
    assert scored_run(tmp_path, "100", "000", 42, "shared") == (
        tmp_path / "loso/seed42/shared/gpt4_holdout"
    )
    assert scored_run(tmp_path, "100", "000", 42, "independent") == (
        tmp_path / "loso/seed42/independent/gpt4_ensemble"
    )
    assert scored_run(tmp_path, "050", "000", 123, "shared") == (
        tmp_path / "dose/qid050/exposure000/seed123/shared/gpt4_holdout"
    )
    assert scored_run(tmp_path, "100", "050", 777, "independent") == (
        tmp_path / "dose/qid100/exposure050/seed777/independent/gpt4_ensemble"
    )


def test_hierarchical_endpoint_interval_preserves_paired_change() -> None:
    initial = np.arange(3 * 8 * 2, dtype=float).reshape(3, 8, 2)
    result = hierarchical_endpoint_interval(
        initial, initial - 0.25, samples=500, seed=7
    )
    assert result["final_minus_initial"] == -0.25
    assert result["low"] == -0.25
    assert result["high"] == -0.25
    assert result["two_sided_p"] == 0.0
