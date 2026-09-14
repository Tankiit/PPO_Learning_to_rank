"""Design invariants for the aleatoric and epistemic perturbation controls.

These tests exist because both controls have a failure mode that produces a
*passing* result for the wrong reason, and neither is visible in the output.

Control 2 fails silently if each ensemble member draws its own target noise:
that injects member disagreement, which is synthetic epistemic uncertainty, and
the estimator would then look correct while measuring the injection. The design
closes this by baking one noise draw into one training file that every member
reads, so the tests here check that the file is deterministic given (sigma,
seed) - not that a comment says so.

Control 3 fails silently if the evaluation set moves with the training
fraction, because coverage and evaluation would change together and the
dose-response would be uninterpretable. The tests check that every cell points
at one identical held-out file, and that the training subsets nest.
"""

import json
import unittest
from pathlib import Path

import numpy as np

from scripts.build_perturbations import (
    add_target_noise,
    nested_subset,
    pair_flip_rate,
)

BASE = Path("data/arr/ds_critique_qidsplit_train.jsonl")
REGISTRY = Path("data/arr/splits_registry.json")
MANIFEST = Path("data/arr/perturbations/perturbation_manifest.json")


def _load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


@unittest.skipUnless(BASE.exists(), "qid split not built")
class AleatoricControlTest(unittest.TestCase):
    def setUp(self):
        self.base = _load(BASE)

    def test_noise_is_deterministic_so_all_members_see_one_draw(self):
        """The trap: a per-member draw would inject synthetic disagreement.

        Because the noise lives in a file, two members can only differ if the
        file differs. Determinism given (sigma, seed) is therefore the property
        that makes 'shared across members' structural rather than advisory.
        """

        first = add_target_noise(self.base, 0.1, 20260903)
        second = add_target_noise(self.base, 0.1, 20260903)
        a = [c["score"] for g in first for c in g["candidates"]]
        b = [c["score"] for g in second for c in g["candidates"]]
        self.assertEqual(a, b)

    def test_different_noise_seeds_differ(self):
        """Guards the test above from passing because nothing happened."""

        a = add_target_noise(self.base, 0.1, 1)
        b = add_target_noise(self.base, 0.1, 2)
        self.assertNotEqual(
            [c["score"] for g in a for c in g["candidates"]],
            [c["score"] for g in b for c in g["candidates"]],
        )

    def test_noise_is_approximately_unbiased(self):
        """The target concept must not move on average.

        Clipping at the unit interval induces a small downward shift because
        30% of targets sit at exactly 1.0. It is bounded here well inside one
        within-group standard deviation (about 0.27), and ListNet's loss uses
        the within-group softmax of targets, for which a uniform shift is inert.
        """

        before = np.mean([c["score"] for g in self.base for c in g["candidates"]])
        for sigma, bound in ((0.05, 0.01), (0.1, 0.02), (0.2, 0.04)):
            after = np.mean(
                [
                    c["score"]
                    for g in add_target_noise(self.base, sigma, 20260903)
                    for c in g["candidates"]
                ]
            )
            self.assertLess(abs(after - before), bound, f"sigma={sigma}")

    def test_flip_rate_rises_with_sigma_and_is_recorded(self):
        """Ordering flips are a concept change, so they are measured, not assumed."""

        rates = [
            pair_flip_rate(self.base, add_target_noise(self.base, s, 20260903))
            for s in (0.05, 0.1, 0.2)
        ]
        self.assertTrue(all(a < b for a, b in zip(rates, rates[1:])), rates)
        self.assertLess(rates[0], 0.05)
        # sigma=0.2 is expected to exceed the advisory bound; the suite reports
        # it rather than hiding it, and results there are read as partly a
        # change of target concept.
        self.assertGreater(rates[-1], 0.05)


@unittest.skipUnless(BASE.exists(), "qid split not built")
class EpistemicControlTest(unittest.TestCase):
    def setUp(self):
        self.base = _load(BASE)

    def test_fractions_are_nested(self):
        """A dose-response needs coverage to grow, not to be resampled."""

        salt = "arr-coverage-v1"
        quarter = {g["group_id"] for g in nested_subset(self.base, 0.25, salt)}
        half = {g["group_id"] for g in nested_subset(self.base, 0.5, salt)}
        whole = {g["group_id"] for g in self.base}
        self.assertLess(len(quarter), len(half))
        self.assertTrue(quarter <= half <= whole)

    def test_subset_is_seed_independent(self):
        """Every replicate must train on the same questions at a given fraction."""

        a = {g["group_id"] for g in nested_subset(self.base, 0.5, "arr-coverage-v1")}
        b = {g["group_id"] for g in nested_subset(self.base, 0.5, "arr-coverage-v1")}
        self.assertEqual(a, b)


@unittest.skipUnless(REGISTRY.exists(), "splits registry not built")
class RegistryTest(unittest.TestCase):
    def test_every_qid_cell_shares_one_evaluation_file(self):
        """Coverage varies; the evaluation must not move underneath it."""

        registry = json.loads(REGISTRY.read_text())
        qid_cells = {k: v for k, v in registry.items() if k.startswith("qid")}
        self.assertGreater(len(qid_cells), 3)
        held_out = {tuple(v)[1] for v in qid_cells.values()}
        self.assertEqual(
            len(held_out), 1, f"cells disagree on the held-out file: {held_out}"
        )

    def test_training_files_differ_between_cells(self):
        registry = json.loads(REGISTRY.read_text())
        trains = [tuple(v)[0] for k, v in registry.items() if k.startswith("qid")]
        self.assertEqual(len(trains), len(set(trains)))

    @unittest.skipUnless(MANIFEST.exists(), "perturbation manifest not built")
    def test_manifest_records_the_trap_being_closed(self):
        manifest = json.loads(MANIFEST.read_text())
        self.assertTrue(manifest["nested_verified"])
        aleatoric = [c for c in manifest["cells"] if c["control"] == "aleatoric"]
        self.assertTrue(aleatoric)
        for cell in aleatoric:
            self.assertTrue(cell["shared_across_members"])
            self.assertIn("pair_flip_rate", cell)


if __name__ == "__main__":
    unittest.main()
