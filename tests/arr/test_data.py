import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.arr.data import build_ds_critique_groups, build_esnli_groups


def esnli_row(label: int, suffix: str) -> dict:
    return {
        "premise": f"premise {suffix}",
        "hypothesis": f"hypothesis {suffix}",
        "label": label,
        "explanation_1": f"human explanation {suffix}",
        "explanation_2": "",
        "explanation_3": "",
    }


class DataConstructionTests(unittest.TestCase):
    def test_esnli_is_deterministic_and_uses_human_gold(self) -> None:
        source = {
            "train": [esnli_row(label, f"train-{label}-{index}") for label in range(3) for index in range(2)],
            "validation": [esnli_row(0, "val")],
            "test": [esnli_row(1, "test")],
        }
        with patch("src.arr.data.load_esnli_source", return_value=source):
            first = build_esnli_groups(train_groups=3)
            second = build_esnli_groups(train_groups=3)
        self.assertEqual(
            [group.to_dict() for group in first["train"]],
            [group.to_dict() for group in second["train"]],
        )
        self.assertEqual(len(first["train"]), 3)
        self.assertTrue(all(len(group.candidates) == 5 for group in first["train"]))
        for group in first["train"]:
            gold = group.candidates[0]
            self.assertTrue(gold.metadata["human_text"])
            self.assertIn("human explanation", gold.text)

    def test_ds_uses_only_human_explanation_annotations(self) -> None:
        row = {
            "id": "candidate-1",
            "qid": "question-1",
            "dataset": "WinoGrande",
            "question": "Question?",
            "gold_answer": "A",
            "student_explanation": "Because the evidence supports A.",
            "student_model": "fixture",
            "student_answer": "A",
            "student_accuracy": 1,
            "critiques": [{"critique_elements": {"explanation_score": 1}}],
            "explanation_annotations": [
                {"explanation_score": 4, "worker": "one"},
                {"explanation_score": 2, "worker": "two"},
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture-crowd-anno.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            groups = build_ds_critique_groups(path, "external_test", 1, 1)
            self.assertAlmostEqual(groups[0].candidates[0].score, 0.6)
            self.assertEqual(
                groups[0].candidates[0].score_provenance,
                "DS_Critique_Bank.explanation_annotations.human_crowd_mean",
            )

    def test_ds_rejects_automatic_score_fallback(self) -> None:
        row = {
            "id": "candidate-1",
            "qid": "question-1",
            "dataset": "fixture",
            "question": "Question?",
            "student_explanation": "Explanation.",
            "critiques": [{"critique_elements": {"explanation_score": 5}}],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture-crowd-anno.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "human explanation_annotations"):
                build_ds_critique_groups(path, "external_test")

    def test_ds_accepts_official_zero_human_score(self) -> None:
        row = {
            "id": "candidate-zero",
            "qid": "question-zero",
            "dataset": "fixture",
            "question": "Question?",
            "student_explanation": "Incorrect explanation.",
            "explanation_annotations": [{"explanation_score": 0, "worker": "one"}],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture-crowd-anno.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            groups = build_ds_critique_groups(path, "external_test")
            self.assertEqual(groups[0].candidates[0].score, 0.0)


if __name__ == "__main__":
    unittest.main()
