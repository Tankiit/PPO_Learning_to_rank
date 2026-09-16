from __future__ import annotations

import torch

from src.arr.schema import Candidate, RankingGroup, ScoreRecord
from src.arr.tier2_training import (
    PYTHIA_REVISION,
    SharedProjectionEnsemble,
    fixed_bootstrap_indices,
    fixed_shared_bootstrap_counts,
    grouped_loss,
    member_seed,
    softmax_residual_decorrelation,
    validate_config,
)
from src.arr.utils import read_jsonl, write_json, write_jsonl


def _config(**overrides):
    value = {
        "construction": "shared",
        "arm": "baseline",
        "loss": "listnet",
        "model_path": "/offline/pythia",
        "revision": PYTHIA_REVISION,
        "train_data": "train.jsonl",
        "validation_data": "validation.jsonl",
        "global_seed": 42,
        "member_id": None,
        "epochs": 50,
        "max_length": 256,
        "group_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "learning_rate": 2e-5,
    }
    value.update(overrides)
    return value


def test_reference_member_seeds_are_distinct():
    assert [member_seed(42, index) for index in range(5)] == [42, 43, 44, 45, 46]


def test_fixed_bootstrap_is_reproducible_and_member_specific():
    first = fixed_bootstrap_indices(216, global_seed=42, member_id=0)
    assert first == fixed_bootstrap_indices(216, global_seed=42, member_id=0)
    assert first != fixed_bootstrap_indices(216, global_seed=42, member_id=1)
    assert len(first) == 216
    assert min(first) >= 0 and max(first) < 216


def test_shared_bootstrap_has_one_complete_draw_per_head():
    counts = fixed_shared_bootstrap_counts(37, global_seed=123, head_count=5)
    assert counts.shape == (37, 5)
    assert torch.equal(counts.sum(dim=0), torch.full((5,), 37.0))


def test_projection_members_and_feature_masks_are_distinct_and_trainable():
    ensemble = SharedProjectionEnsemble(
        hidden_size=20,
        head_count=5,
        dropout=0.0,
        feature_keep_fraction=0.8,
        feature_seed=7,
    )
    assert len({id(head) for head in ensemble.heads}) == 5
    assert torch.equal(
        (ensemble.feature_masks > 0).sum(dim=1), torch.full((5,), 16)
    )
    representation = torch.randn(4, 20, requires_grad=True)
    scores = ensemble(representation)
    assert scores.shape == (4, 5)
    scores.square().mean().backward()
    assert representation.grad is not None
    assert all(any(parameter.grad is not None for parameter in head.parameters()) for head in ensemble.heads)


def test_grouped_loss_updates_every_head():
    scores = torch.randn(2, 4, 5, requires_grad=True)
    targets = torch.tensor([[0.0, 0.2, 0.7, 1.0], [0.1, 0.4, 0.8, 0.9]])
    mask = torch.ones(2, 4, dtype=torch.bool)
    loss, components = grouped_loss(
        scores, targets, mask, loss_name="listnet", decorrelation_lambda=0.1
    )
    assert torch.isfinite(loss)
    assert components["ranking_loss"] > 0.0
    loss.backward()
    assert scores.grad is not None
    assert torch.count_nonzero(scores.grad).item() == scores.numel()


def test_probability_residual_decorrelation_is_finite_and_differentiable():
    scores = torch.randn(2, 3, 5, requires_grad=True)
    targets = torch.tensor([[0.0, 0.5, 1.0], [0.2, 0.4, 0.8]])
    mask = torch.ones(2, 3, dtype=torch.bool)
    penalty = softmax_residual_decorrelation(scores, targets, mask)
    assert torch.isfinite(penalty)
    assert 0.0 <= penalty.item() <= 1.0
    penalty.backward()
    assert scores.grad is not None


def test_config_resolves_all_shared_ablation_controls():
    features = validate_config(_config(arm="features"))
    assert features["feature_keep_fraction"] == 0.8
    assert not features["bootstrap"]
    decorated = validate_config(_config(arm="lambda_0p1"))
    assert decorated["decorrelation_lambda"] == 0.1


def test_independent_configuration_requires_a_member():
    config = validate_config(
        _config(construction="independent", arm="independent_bootstrap", member_id=3)
    )
    assert config["bootstrap"]
    assert config["member_id"] == 3


def test_independent_combine_preserves_all_three_score_spaces(tmp_path):
    from src.arr.tier2_training import combine_independent_runs

    group = RankingGroup(
        group_id="q1",
        split="validation",
        domain="test",
        question="Why?",
        candidates=(
            Candidate("c1", "weak", 0.2, "human"),
            Candidate("c2", "strong", 0.8, "human"),
        ),
        data_fingerprint="fingerprint",
    )
    validation = tmp_path / "validation.jsonl"
    write_jsonl(validation, [group])
    roots = []
    for member in range(5):
        root = tmp_path / f"member-{member}"
        root.mkdir()
        roots.append(root)
        write_json(
            root / "_final.json",
            {
                "status": "complete",
                "global_seed": 42,
                "loss": "listnet",
                "arm": "independent",
                "validation_fingerprint": "fingerprint",
                "member_id": member,
                "training_seed": 42 + member,
            },
        )
        write_json(root / "resolved_config.json", {"validation_data": str(validation)})
        scores = [0.2 + member * 0.02, 0.8 - member * 0.02]
        records = [
            ScoreRecord(
                group_id="q1",
                candidate_id=candidate,
                model_name="pythia",
                model_revision=PYTHIA_REVISION,
                prompt_hash="prompt",
                data_fingerprint="fingerprint",
                score=score,
                raw_output=str(score),
                parsing_status="ok",
                seed=42 + member,
                inference_ms=1.0,
                metadata={
                    "raw_score": float(torch.logit(torch.tensor(score))),
                    "group_softmax_score": probability,
                },
            )
            for candidate, score, probability in zip(
                ("c1", "c2"), scores, (0.3, 0.7)
            )
        ]
        write_jsonl(root / "validation_predictions_epoch_0.jsonl", records)

    result = combine_independent_runs(roots, tmp_path / "ensemble")
    assert result["status"] == "complete"
    assert result["member_ids"] == [0, 1, 2, 3, 4]
    rows = [
        ScoreRecord.from_dict(row)
        for row in read_jsonl(tmp_path / "ensemble" / "validation_predictions_epoch_0.jsonl")
    ]
    assert all(len(row.metadata["head_scores"]) == 5 for row in rows)
    assert all(len(row.metadata["raw_head_scores"]) == 5 for row in rows)
    assert all(len(row.metadata["group_softmax_scores"]) == 5 for row in rows)
