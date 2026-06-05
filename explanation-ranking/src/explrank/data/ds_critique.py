"""DS-Critique Bank loader and step-level adapter."""

from __future__ import annotations

import json
import os
import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    hf_hub_download = None

REPO_ID = "allenai/DS_Critique_Bank"
ANNOTATED_FILES = {
    "train": "DSCB-train-crowd-anno.jsonl",
    "val": "DSCB-dev-crowd-anno.jsonl",
}


class DSCritiqueLoader:
    def __init__(
        self,
        cache_dir: str = "data/ds_critique_bank",
        min_candidates_per_query: int = 3,
        use_human_scores: bool = True,
        seed: int = 42,
    ):
        self.cache_dir = cache_dir
        self.min_candidates_per_query = min_candidates_per_query
        self.use_human_scores = use_human_scores
        random.seed(seed)
        os.makedirs(cache_dir, exist_ok=True)

    def download_file(self, filename: str) -> str:
        local = os.path.join(self.cache_dir, filename)
        if os.path.exists(local):
            return local
        if hf_hub_download is None:
            raise ImportError("pip install huggingface_hub")
        return hf_hub_download(
            repo_id=REPO_ID, filename=filename, repo_type="dataset", local_dir=self.cache_dir
        )

    def load_jsonl(self, path: str) -> List[Dict]:
        rows = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return rows

    @staticmethod
    def get_critique_score(instance: Dict, prefer_human: bool = True) -> Optional[float]:
        if prefer_human and "explanation_annotations" in instance:
            scores = []
            for ann in instance["explanation_annotations"] or []:
                if isinstance(ann, dict) and ann.get("explanation_score") is not None:
                    scores.append(float(ann["explanation_score"]))
            if scores:
                return sum(scores) / len(scores)
        if instance.get("critiques"):
            scores = []
            for c in instance["critiques"]:
                elems = c.get("critique_elements", {}) if isinstance(c, dict) else {}
                if elems.get("explanation_score") is not None:
                    scores.append(float(elems["explanation_score"]))
            if scores:
                return sum(scores) / len(scores)
        return None

    @staticmethod
    def get_query_text(instance: Dict) -> str:
        q = instance.get("question", "")
        a = instance.get("gold_answer", "")
        return f"Question: {q} Correct answer: {a}"

    def group_by_question(self, instances: List[Dict]) -> Dict[str, List[Dict]]:
        groups: Dict[str, List[Dict]] = defaultdict(list)
        for inst in instances:
            groups[inst.get("qid", inst.get("id", "unknown"))].append(inst)
        return dict(groups)

    def convert_group(self, qid: str, instances: List[Dict]) -> Optional[Dict]:
        candidates, scores = [], []
        for inst in instances:
            expl = inst.get("student_explanation", "").strip()
            sc = self.get_critique_score(inst, prefer_human=self.use_human_scores)
            if expl and sc is not None:
                candidates.append(expl)
                scores.append(sc)
        if len(candidates) < self.min_candidates_per_query or len(set(scores)) < 2:
            return None
        return {
            "query_id": f"dscb_{qid}",
            "query": self.get_query_text(instances[0]),
            "explanations": candidates,
            "scores": scores,
            "source": "ds_critique",
        }

    def load_splits(self) -> Tuple[List[Dict], List[Dict]]:
        train_inst, val_inst = [], []
        for split, fn in ANNOTATED_FILES.items():
            path = self.download_file(fn)
            data = self.load_jsonl(path)
            if "train" in split:
                train_inst.extend(data)
            else:
                val_inst.extend(data)
        train = [self.convert_group(q, g) for q, g in self.group_by_question(train_inst).items()]
        val = [self.convert_group(q, g) for q, g in self.group_by_question(val_inst).items()]
        return [x for x in train if x], [x for x in val if x]


def load_ds_critique_ranking(
    cache_dir: str,
    min_candidates: int = 3,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict]]:
    loader = DSCritiqueLoader(cache_dir=cache_dir, min_candidates_per_query=min_candidates, seed=seed)
    return loader.load_splits()


def to_step_examples(ranking_examples: List[Dict], silver_threshold: float = 3.0) -> List[Dict]:
    """Adapter: attach per-step silver labels (freeprm heuristic)."""
    from explrank.models.step_prm import split_into_steps

    out = []
    for ex in ranking_examples:
        mean_score = sum(ex["scores"]) / len(ex["scores"])
        label = 1 if mean_score >= silver_threshold else 0
        for expl in ex["explanations"]:
            steps = split_into_steps(expl)
            out.append(
                {
                    "query": ex["query"],
                    "explanation": expl,
                    "steps": steps,
                    "step_labels": [label] * len(steps),
                    "gold_score": mean_score,
                }
            )
    return out
