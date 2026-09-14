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
    
    # Each example:
    # {
    #     'query_id': 'dscb_ARC-Easy_123',
    #     'query_text': 'Question: What happens when... Answer: B) ...',
    #     'candidates': ['GPT-4 explanation...', 'Llama explanation...', ...],
    #     'scores': [4.5, 3.0, 1.5, ...],  # from critique models or humans
    #     'source': 'ds_critique'
    # }

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


# ============================================================================
# File mapping for the dataset
# ============================================================================

REPO_ID = "allenai/DS_Critique_Bank"

# Files that have human annotations (explanation_annotations field)
ANNOTATED_FILES = {
    "train": "DSCB-train-crowd-anno.jsonl",   # ~1,200 instances with human scores
    "val": "DSCB-dev-crowd-anno.jsonl",        # ~270 instances with human scores
}

# Files without human annotations (larger, critique model scores only)
NON_ANNOTATED_FILES = {
    "train_expert": "DSCB-train-expert.jsonl",     # expert-curated training
    "train_non_anno": "DSCB-train-non-anno.jsonl",  # ~26k instances
    "val_non_anno": "DSCB-dev-non-anno.jsonl",       # ~6.3k instances
}


# ============================================================================
# Core loader
# ============================================================================

class DSCritiqueBankLoader:
    """
    Loads DS-Critique Bank and converts to ranking format.
    
    Args:
        cache_dir: Where to cache downloaded files
        use_human_scores: If True, prefer human annotations when available
        min_candidates_per_query: Minimum number of different explanations per question
        score_source: 'human', 'critique_model', or 'best_available'
    """
    
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
        os.makedirs(cache_dir, exist_ok=True)
    
    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------
    
    def download_file(self, filename: str) -> str:
        """Download a single JSONL file from HuggingFace."""
        local_path = os.path.join(self.cache_dir, filename)
        
        if os.path.exists(local_path):
            return local_path
        
        if hf_hub_download is None:
            raise ImportError(
                "huggingface_hub required. Install with: pip install huggingface_hub"
            )
        
        print(f"  Downloading {filename}...")
        downloaded = hf_hub_download(
            repo_id=REPO_ID,
            filename=filename,
            repo_type="dataset",
            local_dir=self.cache_dir,
        )
        return downloaded
    
    def load_jsonl(self, filepath: str) -> List[Dict]:
        """Load JSONL file, handling encoding issues."""
        data = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue  # skip malformed lines
        return data
    
    # ------------------------------------------------------------------
    # Extract scores from the nested structure
    # ------------------------------------------------------------------
    
    @staticmethod
    def get_critique_score(instance: Dict, prefer_human: bool = True) -> Optional[float]:
        """
        Extract explanation quality score from an instance.
        
        Priority:
        1. Human annotation (explanation_annotations[].explanation_score) — average
        2. Critique model scores (critiques[].critique_elements.explanation_score) — average
        
        Returns float in [0, 5] or None if no score available.
        """
        # Try human annotations first
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
        
        # Fall back to critique model scores
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
        """Format instance as query text for the ranking model."""
        question = instance.get("question", "")
        gold_answer = instance.get("gold_answer", "")
        return f"Question: {question} Correct answer: {gold_answer}"
    
    @staticmethod
    def get_explanation_text(instance: Dict) -> str:
        """Extract student explanation."""
        return instance.get("student_explanation", "").strip()
    
    # ------------------------------------------------------------------
    # Group by question to create ranking candidates
    # ------------------------------------------------------------------
    
    def group_by_question(self, instances: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Group instances by question ID (qid).
        
        Each question may have multiple student model explanations,
        creating natural ranking candidates.
        """
        groups = defaultdict(list)
        for inst in instances:
            qid = inst.get("qid", inst.get("id", "unknown"))
            groups[qid].append(inst)
        return dict(groups)
    
    # ------------------------------------------------------------------
    # Convert to ranking format
    # ------------------------------------------------------------------
    
    def convert_group_to_ranking(
        self, qid: str, instances: List[Dict]
    ) -> Optional[Dict]:
        """
        Convert a group of instances (same question, different student models)
        into a single ranking example.
        
        Returns None if insufficient candidates or scores.
        """
        candidates = []
        scores = []
        
        for inst in instances:
            explanation = self.get_explanation_text(inst)
            score = self.get_critique_score(
                inst, prefer_human=(self.score_source != "critique_model")
            )
            
            if explanation and score is not None:
                candidates.append(explanation)
                scores.append(score)
        
        # Need at least min_candidates
        if len(candidates) < self.min_candidates_per_query:
            return None
        
        # Need score variance (otherwise ranking is trivial)
        if len(set(scores)) < 2:
            return None
        
        query_text = self.get_query_text(instances[0])
        
        return {
            "query_id": f"dscb_{qid}",
            "query_text": query_text,
            "candidates": candidates,
            "scores": scores,
            "source": "ds_critique",
            "num_candidates": len(candidates),
            "dataset_origin": instances[0].get("dataset", "unknown"),
            "student_models": [
                inst.get("student_model", "unknown") for inst in instances
                if self.get_explanation_text(inst) and self.get_critique_score(inst) is not None
            ],
        }
    
    # ------------------------------------------------------------------
    # Main loading functions
    # ------------------------------------------------------------------
    
    def load_ranking_data(
        self, use_annotated: bool = True, use_non_annotated: bool = True
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Load and convert DS-Critique Bank to ranking format.
        
        Returns:
            (train_examples, val_examples) — each a list of ranking dicts
        """
        print("=" * 60)
        print("Loading DS-Critique Bank for ranking experiments")
        print("=" * 60)
        
        train_instances = []
        val_instances = []
        
        # Load annotated files (smaller, have human scores)
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
        
        # Load non-annotated files (larger, critique model scores only)
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
        
        # Group by question and convert to ranking format
        print("\nGrouping by question...")
        train_groups = self.group_by_question(train_instances)
        val_groups = self.group_by_question(val_instances)
        
        print(f"  Train: {len(train_groups)} unique questions")
        print(f"  Val: {len(val_groups)} unique questions")
        
        # Convert to ranking examples
        print(f"\nConverting to ranking format (min {self.min_candidates_per_query} candidates)...")
        
        train_examples = []
        for qid, group in train_groups.items():
            example = self.convert_group_to_ranking(qid, group)
            if example is not None:
                train_examples.append(example)
        
        val_examples = []
        for qid, group in val_groups.items():
            example = self.convert_group_to_ranking(qid, group)
            if example is not None:
                val_examples.append(example)
        
        print(f"\nFinal ranking dataset:")
        print(f"  Train: {len(train_examples)} ranking groups")
        print(f"  Val: {len(val_examples)} ranking groups")
        
        # Print stats
        self._print_stats(train_examples, "Train")
        self._print_stats(val_examples, "Val")
        
        return train_examples, val_examples
    
    def _print_stats(self, examples: List[Dict], split: str):
        """Print dataset statistics."""
        if not examples:
            print(f"  {split}: empty")
            return
        
        n_candidates = [e["num_candidates"] for e in examples]
        all_scores = [s for e in examples for s in e["scores"]]
        score_ranges = [max(e["scores"]) - min(e["scores"]) for e in examples]
        
        print(f"\n  {split} stats:")
        print(f"    Candidates/query: {sum(n_candidates)/len(n_candidates):.1f} avg, "
              f"{min(n_candidates)}-{max(n_candidates)} range")
        print(f"    Score range: {min(all_scores):.1f}-{max(all_scores):.1f}")
        print(f"    Mean score range per query: {sum(score_ranges)/len(score_ranges):.2f}")
        
        # Source dataset breakdown
        origins = defaultdict(int)
        for e in examples:
            origins[e["dataset_origin"]] += 1
        print(f"    Source datasets: {dict(origins)}")
    
    # ------------------------------------------------------------------
    # Convert to the batch format expected by src/train.py
    # ------------------------------------------------------------------
    
    def to_training_format(
        self, examples: List[Dict], tokenizer, max_length: int = 256
    ) -> List[Dict]:
        """
        Convert ranking examples to tokenized training batches.
        
        Expected output format (matches src/train.py):
        {
            'input_ids': [k, seq_len],       # k candidates per query
            'attention_mask': [k, seq_len],
            'scores': [k],                    # quality scores
        }
        """
        training_data = []
        
        for example in examples:
            # Concatenate query with each candidate
            texts = [
                f"{example['query_text']} {cand}"
                for cand in example["candidates"]
            ]
            
            encoded = tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            
            # Normalize scores to [0, 1] range
            raw_scores = example["scores"]
            max_possible = 5.0
            normalized = [s / max_possible for s in raw_scores]
            
            training_data.append({
                "input_ids": encoded["input_ids"],          # [k, seq_len]
                "attention_mask": encoded["attention_mask"], # [k, seq_len]
                "scores": normalized,                        # [k]
                "query_id": example["query_id"],
            })
        
        return training_data


# ============================================================================
# Convenience function for scripts
# ============================================================================

def load_ds_critique_ranking(
    cache_dir: str = "data/ds_critique_bank",
    min_candidates: int = 3,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict]]:
    """Convenience function to load DS-Critique Bank in ranking format."""
    loader = DSCritiqueBankLoader(
        cache_dir=cache_dir,
        min_candidates_per_query=min_candidates,
        seed=seed,
    )
    return loader.load_ranking_data()


# ============================================================================
# CLI for inspection
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Load DS-Critique Bank")
    parser.add_argument("--cache_dir", default="data/ds_critique_bank")
    parser.add_argument("--min_candidates", type=int, default=3)
    parser.add_argument("--show_examples", type=int, default=3)
    args = parser.parse_args()
    
    train, val = load_ds_critique_ranking(
        cache_dir=args.cache_dir,
        min_candidates=args.min_candidates,
    )
    
    print(f"\n{'='*60}")
    print(f"Example ranking groups:")
    print(f"{'='*60}")
    
    for i, ex in enumerate(val[:args.show_examples]):
        print(f"\n--- Example {i+1} ---")
        print(f"Query: {ex['query_text'][:100]}...")
        print(f"Student models: {ex['student_models']}")
        print(f"Scores: {ex['scores']}")
        for j, (cand, score) in enumerate(zip(ex['candidates'], ex['scores'])):
            print(f"  [{score:.1f}] {cand[:80]}...")
