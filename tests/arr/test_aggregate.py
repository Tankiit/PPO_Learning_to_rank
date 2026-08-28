import json
import tempfile
import unittest
from pathlib import Path

from src.arr.aggregate import _arr_evaluations, _ppo_summaries, _run_level_ci


class AggregateFilteringTests(unittest.TestCase):
    @staticmethod
    def _write(path: Path, value: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    def test_pilots_are_excluded_from_paper_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metrics = {"aggregate": {"ndcg_at_5": 0.5}, "per_query": []}
            evaluation_manifest = {"pipeline": "arr", "status": "complete"}
            for prefix in (root / "pilots" / "evaluation", root / "evaluations" / "full"):
                self._write(prefix / "metrics.json", metrics)
                self._write(prefix / "run_manifest.json", evaluation_manifest)

            ppo_manifest = {
                "pipeline": "arr",
                "task": "train-ppo-generator",
                "status": "complete",
                "seed": 7,
                "reward_checkpoint": str(root / "reward"),
            }
            for prefix in (root / "pilots" / "ppo", root / "ppo" / "full"):
                self._write(prefix / "run_manifest.json", ppo_manifest)
                self._write(prefix / "baseline_evaluation.json", {"bertscore_f1": 0.1})
                self._write(
                    prefix / "checkpoints" / "update_0020" / "evaluation.json",
                    {"bertscore_f1": 0.2},
                )

            evaluations = _arr_evaluations([root])
            ppo = _ppo_summaries([root])
            self.assertEqual(len(evaluations), 1)
            self.assertNotIn("pilots", Path(evaluations[0]["path"]).parts)
            self.assertEqual(len(ppo), 1)
            self.assertNotIn("pilots", Path(ppo[0]["run_dir"]).parts)

    def test_run_level_coverage_does_not_become_nan(self) -> None:
        runs = [
            {"metrics": {"aggregate": {"parsing_coverage": 1.0}}},
            {"metrics": {"aggregate": {"parsing_coverage": 0.8}}},
        ]
        interval = _run_level_ci(runs, "parsing_coverage", samples=100, seed=42)
        self.assertAlmostEqual(interval["estimate"], 0.9)
        self.assertTrue(0.8 <= interval["low"] <= interval["high"] <= 1.0)


if __name__ == "__main__":
    unittest.main()
