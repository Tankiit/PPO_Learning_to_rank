import unittest

from src.arr.schema import Candidate, RankingGroup
from src.arr.utils import deterministic_uniform, stable_hash


class SchemaAndHashTests(unittest.TestCase):
    def test_variable_candidates_have_only_real_mask_entries(self) -> None:
        group = RankingGroup(
            group_id="g1",
            split="train",
            domain="nli",
            question="q",
            candidates=(
                Candidate("c1", "one", 0.2, "fixture"),
                Candidate("c2", "two", 0.8, "fixture"),
            ),
            data_fingerprint="fingerprint",
        )
        self.assertEqual(group.candidate_mask, (True, True))
        self.assertEqual(RankingGroup.from_dict(group.to_dict()), group)

    def test_empty_padding_candidate_is_forbidden(self) -> None:
        with self.assertRaisesRegex(ValueError, "candidate text"):
            Candidate("padding", "", 0.0, "fixture")

    def test_hash_and_draw_are_process_independent(self) -> None:
        self.assertEqual(
            stable_hash("example", {"b": 2, "a": 1}),
            stable_hash("example", {"a": 1, "b": 2}),
        )
        first = deterministic_uniform(0.3, 0.7, "group", "tier")
        second = deterministic_uniform(0.3, 0.7, "group", "tier")
        self.assertEqual(first, second)
        self.assertGreaterEqual(first, 0.3)
        self.assertLessEqual(first, 0.7)


if __name__ == "__main__":
    unittest.main()
