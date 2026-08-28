import math
import unittest

from src.arr.judges import parse_listwise_json, parse_score_json
from src.arr.metrics import (
    evaluate_predictions,
    holm_correction,
    ndcg_at_k,
    random_ndcg_at_k,
    tie_aware_ndcg_at_k,
)
from src.arr.schema import Candidate, RankingGroup, ScoreRecord


def fixture_group() -> RankingGroup:
    return RankingGroup(
        group_id="g",
        split="test",
        domain="nli",
        question="question",
        candidates=(
            Candidate("a", "a", 0.0, "fixture"),
            Candidate("b", "b", 0.5, "fixture"),
            Candidate("c", "c", 1.0, "fixture"),
        ),
        data_fingerprint="fp",
    )


class ParsingAndMetricsTests(unittest.TestCase):
    def test_strict_direct_and_final_cot_json(self) -> None:
        self.assertEqual(parse_score_json('{"score": 3}', "direct"), 0.75)
        self.assertEqual(parse_score_json('Reasoning.\n{"score": 2.5}', "cot"), 0.625)
        with self.assertRaises(ValueError):
            parse_score_json('Reasoning {"score": 3}', "direct")
        with self.assertRaises(ValueError):
            parse_score_json('{"score": 5}', "direct")
        with self.assertRaises(ValueError):
            parse_score_json('{"score": 2, "note": "x"}', "direct")
        with self.assertRaises(ValueError):
            parse_score_json('{"score": 2}', "cot")

    def test_listwise_parser_checks_count_and_bounds(self) -> None:
        self.assertEqual(parse_listwise_json('ok\n{"scores": [0, 2, 4]}', 3), [0.0, 0.5, 1.0])
        with self.assertRaises(ValueError):
            parse_listwise_json('{"scores": [0, 4]}', 3)

    def test_macro_separation_ratio_and_coverage(self) -> None:
        group = fixture_group()
        records = [
            ScoreRecord(
                group_id="g",
                candidate_id=candidate.candidate_id,
                model_name="model",
                model_revision="revision",
                prompt_hash="prompt",
                data_fingerprint="fp",
                score=score,
                raw_output=str(score),
                parsing_status="ok",
                seed=7,
                inference_ms=1.0,
            )
            for candidate, score in zip(group.candidates, (0.0, 0.25, 0.5))
        ]
        metrics = evaluate_predictions([group], records)
        self.assertAlmostEqual(metrics["aggregate"]["separation_ratio"], 0.5)
        self.assertEqual(metrics["aggregate"]["parsing_coverage"], 1.0)
        self.assertAlmostEqual(metrics["aggregate"]["ndcg_at_5"], 1.0)

    def test_failed_parse_is_not_imputed(self) -> None:
        group = fixture_group()
        records = [
            ScoreRecord(
                "g", candidate.candidate_id, "m", "r", "p", "fp",
                None if index == 0 else 0.5,
                "bad" if index == 0 else "0.5",
                "parse_error" if index == 0 else "ok",
                7, 0.0,
            )
            for index, candidate in enumerate(group.candidates)
        ]
        metrics = evaluate_predictions([group], records)
        self.assertAlmostEqual(metrics["aggregate"]["parsing_coverage"], 2 / 3)
        self.assertEqual(metrics["aggregate"]["complete_query_count"], 0)
        self.assertTrue(math.isnan(metrics["aggregate"]["ndcg_at_5"]))

    def test_ties_do_not_inherit_serialised_candidate_order(self) -> None:
        truth = [1.0, 0.5, 0.0]
        constant = [0.5, 0.5, 0.5]
        self.assertAlmostEqual(
            tie_aware_ndcg_at_k(truth, constant), random_ndcg_at_k(truth)
        )
        group = RankingGroup(
            "ordered", "test", "nli", "question",
            tuple(
                Candidate(str(index), str(index), score, "fixture")
                for index, score in enumerate(truth)
            ),
            "fp",
        )
        records = [
            ScoreRecord(
                "ordered", candidate.candidate_id, "m", "r", "p", "fp",
                0.5, "0.5", "ok", 7, 0.0,
            )
            for candidate in group.candidates
        ]
        metrics = evaluate_predictions([group], records)["aggregate"]
        self.assertEqual(metrics["top1"], 1.0)
        self.assertAlmostEqual(metrics["fractional_top1"], 1 / 3)
        self.assertAlmostEqual(metrics["ndcg_lift_over_random"], 0.0)

    def test_constant_reference_group_is_reported_not_scored(self) -> None:
        group = RankingGroup(
            "constant", "dev", "domain", "question",
            (Candidate("only", "text", 0.8, "fixture"),), "fp",
        )
        record = ScoreRecord("constant", "only", "m", "r", "p", "fp", 0.5, "0.5", "ok", 7, 0.0)
        metrics = evaluate_predictions([group], [record])
        self.assertEqual(metrics["aggregate"]["unrankable_query_count"], 1)
        self.assertEqual(metrics["aggregate"]["evaluated_query_count"], 0)
        self.assertTrue(math.isnan(metrics["aggregate"]["ndcg_at_5"]))

    def test_holm_is_monotonic(self) -> None:
        corrected = holm_correction({"a": 0.001, "b": 0.02, "c": 0.2})
        self.assertLessEqual(corrected["a"]["adjusted_p_value"], corrected["b"]["adjusted_p_value"])
        self.assertLessEqual(corrected["b"]["adjusted_p_value"], corrected["c"]["adjusted_p_value"])


if __name__ == "__main__":
    unittest.main()
