from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.arr.schema import Candidate, RankingGroup, ScoreRecord
from src.arr.tier2_score import combine_scored_members
from src.arr.utils import read_jsonl, write_json, write_jsonl


def test_combine_scored_members_on_second_evaluation_set(tmp_path: Path) -> None:
    group = RankingGroup(
        group_id="g", split="r4_gpt4_heldout", domain="test", question="why?",
        data_fingerprint="eval-fingerprint", metadata={"qid": "q"},
        candidates=tuple(Candidate(
            candidate_id=f"c{index}", text=f"explanation {index}", score=index / 2,
            score_provenance="human", metadata={"student_model": "gpt-4-0613"},
        ) for index in range(3)),
    )
    data = tmp_path / "eval.jsonl"
    write_jsonl(data, [group.to_dict()])
    roots = []
    for member in range(5):
        root = tmp_path / f"member-{member}"
        root.mkdir()
        raw = np.asarray([0.1, 0.5, 0.8], dtype=float) + member * 0.2
        probability = np.exp(raw - raw.max())
        probability /= probability.sum()
        records = [ScoreRecord(
            group_id="g", candidate_id=f"c{index}", model_name="pythia",
            model_revision="pinned", prompt_hash="prompt", data_fingerprint="eval-fingerprint",
            score=float(1.0 / (1.0 + np.exp(-value))), raw_output=str(value),
            parsing_status="ok", seed=42 + member, inference_ms=0.0,
            metadata={"raw_score": float(value),
                      "group_softmax_score": float(probability[index])},
        ) for index, value in enumerate(raw)]
        write_jsonl(root / "predictions.jsonl", records)
        write_json(root / "_final.json", {
            "status": "complete", "construction": "independent", "global_seed": 42,
            "loss": "listnet", "arm": "independent", "evaluation_fingerprint": "eval-fingerprint",
            "member_id": member, "training_seed": 42 + member,
            "prediction_file": "predictions.jsonl",
        })
        roots.append(root)
    output = tmp_path / "ensemble"
    result = combine_scored_members(roots, data, output)
    assert result["status"] == "complete"
    assert result["member_seeds"] == [42, 43, 44, 45, 46]
    assert result["evaluation_candidates"] == 3
    rows = list(read_jsonl(output / "predictions.jsonl"))
    assert len(rows) == 3
    assert all(len(row["metadata"]["raw_head_scores"]) == 5 for row in rows)
    assert all(len(row["metadata"]["group_softmax_scores"]) == 5 for row in rows)
    assert json.loads((output / "_final.json").read_text())["status"] == "complete"
