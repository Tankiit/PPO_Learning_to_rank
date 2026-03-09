"""
DS-Critique Bank Data Loader for Ranking Experiments.

Loads the AllenAI Digital Socrates Critique Bank dataset and converts it
into the ranking format expected by our training pipeline.

Key insight: DS-Critique Bank already has explanation_score (0-5) from
multiple critique models (GPT-4, DS-13B, DS-7B) AND human annotations.
We DON'T need to generate graded data — quality tiers already exist naturally
because different student models (GPT-4, GPT-3.5, Llama-2-70B, Llama-2-7B)
produce different quality explanations for the same question.

Ranking groups: Each question has multiple student explanations (from different
models) each scored 0-5. This gives us natural ranking candidates WITHOUT
any synthetic data construction.

Usage:
    loader = DSCritiqueBankLoader()
    train_data, val_data = loader.load_ranking_data()

Note on HuggingFace loading bug:
    load_dataset("allenai/DS_Critique_Bank") fails because some JSONL files
    lack 'explanation_annotations' column. We download individual files instead.
"""

import json
import os
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import random

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    hf_hub_download = None

REPO_ID = "allenai/DS_Critique_Bank"

ANNOTATED_FILES = {
    "train": "DSCB-train-crowd-anno.jsonl",
    "val": "DSCB-dev-crowd-anno.jsonl",
}

NON_ANNOTATED_FILES = {
    "train_expert": "DSCB-train-expert.jsonl",
    "train_non_anno": "DSCB-train-non-anno.jsonl",
    "val_non_anno": "DSCB-dev-non-anno.jsonl",
}


class DSCritiqueBankLoader:
    def __init__(
        self,
        cache_dir: str = "data/ds_critique_bank",
        use_human_scores: bool = True,
        min_candidates_per_query: int = 3,
        score_source: str = "best_available",
        seed: int = 42,
    ):
        self.cache_dir = cache_dir
        self.use_human_scores = use_human_scores
        self.min_candidates_per_query = min_candidates_per_query
        self.score_source = score_source
        self.seed = seed
        random.seed(seed)
        os.makedirs(cache_dir, exist_ok=True)

    def download_file(self, filename: str) -> str:
        local_path = os.path.join(self.cache_dir, filename)
        if os.path.exists(local_path):
            return local_path
        if hf_hub_download is None:
            raise ImportError("huggingface_hub required. Install with: pip install huggingface_hub")
        print(f"  Downloading {filename}...")
        return hf_hub_download(repo_id=REPO_ID, filename=filename, repo_type="dataset", local_dir=self.cache_dir)

    def load_jsonl(self, filepath: str) -> List[Dict]:
        data = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return data

    @staticmethod
    def get_critique_score(instance: Dict, prefer_human: bool = True) -> Optional[float]:
        if prefer_human and "explanation_annotations" in instance:
            annotations = instance["explanation_annotations"]
            if annotations and len(annotations) > 0:
                human_scores = []
                for ann in annotations:
                    if isinstance(ann, dict) and "explanation_score" in ann:
                        s = ann["explanation_score"]
                        if s is not None:
                            human_scores.append(float(s))
                if human_scores:
                    return sum(human_scores) / len(human_scores)

        if "critiques" in instance and instance["critiques"]:
            critique_scores = []
            for critique in instance["critiques"]:
                if isinstance(critique, dict):
                    elems = critique.get("critique_elements", {})
                    if isinstance(elems, dict) and "explanation_score" in elems:
                        s = elems["explanation_score"]
                        if s is not None:
                            critique_scores.append(float(s))
            if critique_scores:
                return sum(critique_scores) / len(critique_scores)
        return None

    @staticmethod
    def get_query_text(instance: Dict) -> str:
        question = instance.get("question", "")
        gold_answer = instance.get("gold_answer", "")
        return f"Question: {question} Correct answer: {gold_answer}"

    @staticmethod
    def get_explanation_text(instance: Dict) -> str:
        return instance.get("student_explanation", "").strip()

    def group_by_question(self, instances: List[Dict]) -> Dict[str, List[Dict]]:
        groups = defaultdict(list)
        for inst in instances:
            qid = inst.get("qid", inst.get("id", "unknown"))
            groups[qid].append(inst)
        return dict(groups)

    def convert_group_to_ranking(self, qid: str, instances: List[Dict]) -> Optional[Dict]:
        candidates = []
        scores = []
        student_models = []

        for inst in instances:
            explanation = self.get_explanation_text(inst)
            score = self.get_critique_score(inst, prefer_human=(self.score_source != "critique_model"))
            if explanation and score is not None:
                candidates.append(explanation)
                scores.append(score)
                student_models.append(inst.get("student_model", "unknown"))

        if len(candidates) < self.min_candidates_per_query:
            return None
        if len(set(scores)) < 2:
            return None

        return {
            "query_id": f"dscb_{qid}",
            "query_text": self.get_query_text(instances[0]),
            "candidates": candidates,
            "scores": scores,
            "source": "ds_critique",
            "num_candidates": len(candidates),
            "dataset_origin": instances[0].get("dataset", "unknown"),
            "student_models": student_models,
        }

    def load_ranking_data(self, use_annotated: bool = True, use_non_annotated: bool = True) -> Tuple[List[Dict], List[Dict]]:
        print("=" * 60)
        print("Loading DS-Critique Bank for ranking experiments")
        print("=" * 60)

        train_instances = []
        val_instances = []

        if use_annotated:
            for split, filename in ANNOTATED_FILES.items():
                try:
                    path = self.download_file(filename)
                    data = self.load_jsonl(path)
                    print(f"  {filename}: {len(data)} instances (with human annotations)")
                    if "train" in split:
                        train_instances.extend(data)
                    else:
                        val_instances.extend(data)
                except Exception as e:
                    print(f"  WARNING: Failed to load {filename}: {e}")

        if use_non_annotated:
            for split, filename in NON_ANNOTATED_FILES.items():
                try:
                    path = self.download_file(filename)
                    data = self.load_jsonl(path)
                    print(f"  {filename}: {len(data)} instances (critique model scores)")
                    if "train" in split:
                        train_instances.extend(data)
                    else:
                        val_instances.extend(data)
                except Exception as e:
                    print(f"  WARNING: Failed to load {filename}: {e}")

        print(f"\nTotal: {len(train_instances)} train, {len(val_instances)} val instances")
        print("\nGrouping by question...")
        train_groups = self.group_by_question(train_instances)
        val_groups = self.group_by_question(val_instances)
        print(f"  Train: {len(train_groups)} unique questions")
        print(f"  Val: {len(val_groups)} unique questions")

        print(f"\nConverting to ranking format (min {self.min_candidates_per_query} candidates)...")
        train_examples = [self.convert_group_to_ranking(qid, group) for qid, group in train_groups.items()]
        train_examples = [ex for ex in train_examples if ex is not None]

        val_examples = [self.convert_group_to_ranking(qid, group) for qid, group in val_groups.items()]
        val_examples = [ex for ex in val_examples if ex is not None]

        print(f"\nFinal ranking dataset:")
        print(f"  Train: {len(train_examples)} ranking groups")
        print(f"  Val: {len(val_examples)} ranking groups")

        self._print_stats(train_examples, "Train")
        self._print_stats(val_examples, "Val")

        return train_examples, val_examples

    def _print_stats(self, examples: List[Dict], split: str):
        if not examples:
            print(f"  {split}: empty")
            return
        n_candidates = [e["num_candidates"] for e in examples]
        all_scores = [s for e in examples for s in e["scores"]]
        score_ranges = [max(e["scores"]) - min(e["scores"]) for e in examples]
        unique_scores_per_query = [len(set(e["scores"])) for e in examples]

        print(f"\n  {split} stats:")
        print(f"    Candidates/query: {sum(n_candidates)/len(n_candidates):.1f} avg, {min(n_candidates)}-{max(n_candidates)} range")
        print(f"    Score range: {min(all_scores):.1f}-{max(all_scores):.1f}")
        print(f"    Mean score range per query: {sum(score_ranges)/len(score_ranges):.2f}")
        print(f"    Unique scores/query: {sum(unique_scores_per_query)/len(unique_scores_per_query):.2f} avg")

        origins = defaultdict(int)
        for e in examples:
            origins[e["dataset_origin"]] += 1
        print(f"    Source datasets: {dict(origins)}")


def load_ds_critique_ranking(cache_dir: str = "data/ds_critique_bank", min_candidates: int = 3, seed: int = 42):
    loader = DSCritiqueBankLoader(cache_dir=cache_dir, min_candidates_per_query=min_candidates, seed=seed)
    return loader.load_ranking_data()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Load DS-Critique Bank")
    parser.add_argument("--cache_dir", default="data/ds_critique_bank")
    parser.add_argument("--min_candidates", type=int, default=3)
    parser.add_argument("--show_examples", type=int, default=2)
    args = parser.parse_args()

    train, val = load_ds_critique_ranking(cache_dir=args.cache_dir, min_candidates=args.min_candidates)
    print(f"\n{'='*60}\nExample ranking groups:\n{'='*60}")
    for i, ex in enumerate(val[:args.show_examples]):
        print(f"\n--- Example {i+1} ---")
        print(f"Query: {ex['query_text'][:100]}...")
        print(f"Student models: {ex['student_models']}")
        print(f"Scores: {ex['scores']}")
