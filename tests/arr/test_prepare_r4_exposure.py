from __future__ import annotations

from copy import deepcopy

from src.arr.schema import Candidate, RankingGroup
from scripts.prepare_r4_exposure import (
    gpt4_heldout,
    materialize_cell,
    select_qids,
)
from scripts.prepare_r4_loso import prepare as prepare_loso
from src.arr.data import load_groups
from src.arr.utils import write_jsonl


def _groups() -> list[RankingGroup]:
    output = []
    for index in range(8):
        candidates = []
        for model in ("known-a", "known-b", "known-c", "gpt-4-0613"):
            for candidate_index in range(3):
                candidates.append(Candidate(
                    candidate_id=f"candidate-{index}-{model}-{candidate_index}",
                    text=f"explanation {index} {model} {candidate_index}",
                    score=candidate_index / 3,
                    score_provenance="DS_Critique_Bank.explanation_annotations.human_crowd_mean",
                    metadata={"student_model": model},
                ))
        output.append(RankingGroup(
            group_id=f"group-{index}", split="source", domain="A" if index < 4 else "B",
            question=f"question {index}", candidates=tuple(candidates),
            data_fingerprint="source", metadata={"qid": f"qid-{index}"},
        ))
    return output


def test_qid_fractions_are_nested_and_domain_stratified() -> None:
    groups = _groups()
    small = select_qids(groups, 0.25, salt="same")
    medium = select_qids(groups, 0.5, salt="same")
    full = select_qids(groups, 1.0, salt="same")
    assert len(small) == 2
    assert len(medium) == 4
    assert len(full) == 8
    assert small < medium < full


def test_exposure_keeps_nine_candidates_and_fixed_qids() -> None:
    groups = _groups()
    absent, absent_info = materialize_cell(groups, 1.0, 0.0)
    partial, partial_info = materialize_cell(groups, 1.0, 0.5)
    full, full_info = materialize_cell(groups, 1.0, 1.0)
    assert [info["exposed_qids"] for info in (absent_info, partial_info, full_info)] == [0, 4, 8]
    for cell in (absent, partial, full):
        assert len(cell) == 8
        assert all(len(group["candidates"]) == 9 for group in cell)
        assert all(group["candidate_mask"] == [True] * 9 for group in cell)
    assert all(sum(c["metadata"]["student_model"] == "gpt-4-0613"
                   for c in group["candidates"]) == 0 for group in absent)
    assert all(sum(c["metadata"]["student_model"] == "gpt-4-0613"
                   for c in group["candidates"]) == 3 for group in full)
    assert [group["metadata"]["qid"] for group in absent] == [
        group["metadata"]["qid"] for group in full
    ]


def test_gpt4_eval_keeps_three_candidates_per_qid() -> None:
    rows, fingerprint = gpt4_heldout(_groups())
    assert len(rows) == 8
    assert len(fingerprint) == 64
    assert all(len(group["candidates"]) == 3 for group in rows)
    assert all(group["data_fingerprint"] == fingerprint for group in rows)


def test_loso_eval_withholds_gpt4_on_training_qids(tmp_path) -> None:
    groups = _groups()
    expanded = []
    for index in range(216):
        group = deepcopy(groups[index % len(groups)].to_dict())
        group["group_id"] = f"expanded-{index}"
        group["metadata"]["qid"] = f"expanded-qid-{index}"
        expanded.append(group)
    train = tmp_path / "source.jsonl"
    output = tmp_path / "loso.jsonl"
    write_jsonl(train, expanded)
    info = prepare_loso(train, output)
    scored = load_groups(output)
    assert info["groups"] == 216
    assert info["candidates"] == 648
    assert {group.metadata["qid"] for group in scored} == {
        group["metadata"]["qid"] for group in expanded
    }
    assert all(group.split == "r4_gpt4_loso_train_qids" for group in scored)
    assert all(candidate.metadata["student_model"] == "gpt-4-0613"
               for group in scored for candidate in group.candidates)
