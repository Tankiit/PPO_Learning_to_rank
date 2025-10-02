# src/data_creation/comprehensive_builder.py

import os
import json
import random
import numpy as np
from typing import Dict, List, Optional
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm

class ComprehensiveDatasetBuilder:
    """
    Build a cross-source, explanation-ranking dataset in HF-friendly LONG format.
    Also saves per-source normalized datasets under data/raw/<source>/normalized.
    """

    def __init__(self, config: Dict = None):
        self.config = config or self.get_default_config()
        self.quality_levels = {4: "excellent", 3: "good", 2: "fair", 1: "poor", 0: "nonsense"}
        random.seed(self.config["random_seed"])
        np.random.seed(self.config["random_seed"])

        self.generator = None
        self.tokenizer = None
        if self.config.get("use_generation", False):
            self.tokenizer = AutoTokenizer.from_pretrained(self.config["generation_model"])
            self.generator = AutoModelForCausalLM.from_pretrained(
                self.config["generation_model"],
                device_map="auto",
                torch_dtype=torch.float16
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

        self.source_ids = self.config.get("source_ids", {
            "e-snli": ("presencesw/esnli", None),
            "alpha-nli": ("RAR-b/alphanli", None),
            "delta-nli": ("tasksource/defeasible-nli", "snli"),
            "winowhy": ("tasksource/winowhy", None),
            "sens-making": ("tasksource/sen-making", None),
            "ds-critique": ("synthetic", None),  # Will use synthetic DS-style data
        })

    def get_default_config(self) -> Dict:
        return {
            "generation_model": "meta-llama/Llama-3.1-8B-Instruct",
            "max_samples_per_source": 20000,
            "samples_per_query": 5,
            "use_generation": False,
            "random_seed": 42,
            "output_long_table_name": "comprehensive_ranking_dataset",
            # new
            "save_normalized_sources": True,
            "save_per_source_long": False,
        }

    # ----------------------------- PUBLIC API -----------------------------

    def create_comprehensive_dataset(self, sources: List[str], output_dir: str) -> DatasetDict:
        """
        Build and save:
          - per-source normalized datasets -> data/raw/<source>/normalized
          - (optional) per-source long datasets -> data/processed/per_source/<source>
          - merged long dataset -> data/processed/comprehensive_ranking_dataset
        """
        print("Building comprehensive explanation ranking dataset...")

        per_split_queries = {"train": [], "validation": [], "test": []}

        # ensure base folders
        raw_base = os.path.join(output_dir, "raw")
        processed_base = os.path.join(output_dir, "processed")
        os.makedirs(raw_base, exist_ok=True)
        os.makedirs(processed_base, exist_ok=True)

        # 1) process each source and save normalized (query-level)
        for source in sources:
            hf_info = self.source_ids.get(source, ('MISSING', None))
            hf_id = hf_info[0] if isinstance(hf_info, tuple) else hf_info
            print(f"\nLoading & normalizing: {source} -> {hf_id}")
            source_splits = self.process_source_dataset(source)

            # save normalized per source (DatasetDict of queries)
            if self.config.get("save_normalized_sources", True):
                self._save_normalized_source(source, source_splits, raw_base)

            # aggregate for merged build
            for split in per_split_queries:
                per_split_queries[split].extend(source_splits.get(split, []))

            # optional: also save per-source LONG format
            if self.config.get("save_per_source_long", False):
                print(f"Creating per-source LONG rows for {source}...")
                per_source_long = {}
                for split in ["train", "validation", "test"]:
                    rows = []
                    for q in source_splits.get(split, []):
                        rows.extend(self.create_ranking_rows(q))
                    if rows:
                        per_source_long[split] = Dataset.from_list(rows)
                if per_source_long:
                    per_source_long_dd = DatasetDict(per_source_long)
                    per_source_long_path = os.path.join(processed_base, "per_source", source)
                    os.makedirs(per_source_long_path, exist_ok=True)
                    per_source_long_dd.save_to_disk(per_source_long_path)
                    # small metadata
                    with open(os.path.join(per_source_long_path, "metadata.json"), "w") as f:
                        json.dump(self.get_dataset_stats(per_source_long_dd), f, indent=2)
                else:
                    print(f"Skipping per-source LONG save for {source} (no data)")

        # 2) expand to LONG format (merged)
        print("\nExpanding queries into ranking candidates (merged)...")
        long_ds = {}
        for split in ["train", "validation", "test"]:
            print(f"Processing {split} split ({len(per_split_queries[split])} queries)...")
            rows = []
            for q in tqdm(per_split_queries[split]):
                rows.extend(self.create_ranking_rows(q))
            if rows:
                long_ds[split] = Dataset.from_list(rows)

        if not long_ds:
            raise ValueError("No data found in any split. Cannot create dataset.")

        final = DatasetDict(long_ds)

        # 3) save merged long dataset
        merged_path = os.path.join(processed_base, self.config["output_long_table_name"])
        os.makedirs(merged_path, exist_ok=True)
        final.save_to_disk(merged_path)
        self.save_dataset_metadata(final, merged_path)

        print(f"\nSaved merged dataset to {merged_path}")
        print(f"Stats: {json.dumps(self.get_dataset_stats(final), indent=2)}")
        return final

    # ----------------------------- SOURCES -----------------------------

    def process_source_dataset(self, source: str) -> Dict[str, List[Dict]]:
        """
        Normalize each HF dataset to unified query-level dicts:
        { 'premise','hypothesis','label','gold_explanation','source','query_text','query_id' }
        """
        if source not in self.source_ids:
            raise ValueError(f"Unknown source: {source}. Add an HF dataset id in config['source_ids'].")

        hf_id, config_name = self.source_ids[source]

        # Handle synthetic DS Critique Bank data
        if hf_id == "synthetic" and source == "ds-critique":
            ds = self._create_synthetic_ds_critique_data()
        elif config_name:
            ds = load_dataset(hf_id, config_name, trust_remote_code=True)
        else:
            ds = load_dataset(hf_id, trust_remote_code=True)

        def first_available_split(d, names):
            for n in names:
                if n in d:
                    return n
            return None

        out = {"train": [], "validation": [], "test": []}
        split_map = {
            "train": first_available_split(ds, ["train", "training"]),
            "validation": first_available_split(ds, ["validation", "valid", "dev", "val"]),
            "test": first_available_split(ds, ["test", "testing", "eval"])
        }

        parser = {
            "e-snli": self._parse_esnli,
            "alpha-nli": self._parse_alphanli,
            "delta-nli": self._parse_defeasible_nli,
            "winowhy": self._parse_winowhy,
            "sens-making": self._parse_sensemaking,
            "ds-critique": self._parse_ds_critique
        }[source]

        max_n = self.config["max_samples_per_source"]

        for our_split, real_split in split_map.items():
            if real_split is None:
                continue
            normalized = []
            for ex in ds[real_split]:
                item = parser(ex)
                if item is None:
                    continue
                # add query_text + query_id here so raw/normalized is self-contained
                item["query_text"] = self.format_query(item["premise"], item["hypothesis"], item["label"])
                item["query_id"] = f"{source}_{abs(hash(item['premise'] + '||' + item['hypothesis'] + '||' + item['label'])) % 10**9}"
                normalized.append(item)
                if len(normalized) >= max_n:
                    break
            out[our_split] = normalized

        return out

    # ---- parsers ----

    def _parse_esnli(self, ex: Dict) -> Optional[Dict]:
        # New format has question with "Premise: ... Hypothesis: ..." and answer
        question = ex.get("question", "")
        answer = ex.get("answer", "")
        cot = ex.get("cot", "")

        if not question or not answer:
            return None

        # Extract premise and hypothesis from question
        try:
            parts = question.split('Premise: "')[1].split('" Hypothesis: "')
            premise = parts[0].strip()
            hypothesis = parts[1].split('"')[0].strip()
        except:
            return None

        # Map answer to label
        label_map = {"yes": "entails", "no": "contradicts", "it is not possible to tell": "is neutral to"}
        label = label_map.get(answer.lower(), "is neutral to")

        return {
            "premise": premise,
            "hypothesis": hypothesis,
            "label": label,
            "gold_explanation": cot,
            "source": "e-snli"
        }

    def _parse_alphanli(self, ex: Dict) -> Optional[Dict]:
        o1 = ex.get("obs1") or ex.get("observation1") or ""
        o2 = ex.get("obs2") or ex.get("observation2") or ""
        h1 = ex.get("hyp1") or ex.get("hypothesis1") or ""
        h2 = ex.get("hyp2") or ex.get("hypothesis2") or ""
        lbl = ex.get("label")
        if lbl is None or (h1 == "" and h2 == ""):
            return None
        correct_h = h1 if str(lbl).strip() in ["0", "1"] and int(lbl) == 0 else h2
        reason = ex.get("reason", "")
        return {
            "premise": f"{o1} {o2}".strip(),
            "hypothesis": correct_h,
            "label": "entails",
            "gold_explanation": reason,
            "source": "alpha-nli"
        }

    def _parse_defeasible_nli(self, ex: Dict) -> Optional[Dict]:
        # Use capitalized keys
        prem = ex.get("Premise") or ex.get("premise") or ""
        hyp = ex.get("Hypothesis") or ex.get("hypothesis") or ""
        update = ex.get("Update") or ex.get("update") or ""
        update_type = ex.get("UpdateType") or ex.get("label") or ""

        if not prem or not hyp or not update_type:
            return None

        label = "entails" if update_type.lower() == "strengthener" else "contradicts"
        return {
            "premise": prem,
            "hypothesis": hyp,
            "label": label,
            "gold_explanation": update,
            "source": "delta-nli"
        }

    def _parse_winowhy(self, ex: Dict) -> Optional[Dict]:
        # Format: sentence contains explanation, wnli_sent1/wnli_sent2 are parts
        sentence = ex.get("sentence", "")
        sent1 = ex.get("wnli_sent1", "")
        sent2 = ex.get("wnli_sent2", "")
        lbl = ex.get("label", 0)

        if not sentence or not sent1:
            return None

        # sent1 is premise, sent2 is hypothesis/explanation
        label = "entails" if lbl == 1 else "contradicts"
        return {
            "premise": sent1.strip(),
            "hypothesis": sent2.strip() if sent2 else sentence,
            "label": label,
            "gold_explanation": sent2.strip() if sent2 else "",
            "source": "winowhy"
        }

    def _parse_sensemaking(self, ex: Dict) -> Optional[Dict]:
        s1 = ex.get("s1") or ex.get("sentence_good") or ex.get("sent1") or ""
        s2 = ex.get("s2") or ex.get("sentence_bad") or ex.get("sent2") or ""
        lbl = ex.get("label")
        if lbl is None:
            return None
        lbl_int = int(lbl) if str(lbl).isdigit() else (1 if str(lbl).lower() in ["a", "s1"] else 0)
        good = s1 if lbl_int == 1 else s2
        rationale = ex.get("rationale") or ex.get("exp") or ""
        return {
            "premise": s1 if lbl_int == 1 else s2,
            "hypothesis": good,
            "label": "entails",
            "gold_explanation": rationale,
            "source": "sens-making"
        }

    def _parse_ds_critique(self, ex: Dict) -> Optional[Dict]:
        """Parse DS Critique Bank style data"""
        question = ex.get("question", "")
        explanation = ex.get("student_explanation", "")
        correct_answer = ex.get("correct_answer", "")

        if not question or not explanation:
            return None

        # Use the question as premise and explanation as hypothesis
        # The task is to explain why an answer is correct
        return {
            "premise": question,
            "hypothesis": f"The answer is {correct_answer}",
            "label": "entails",
            "gold_explanation": explanation,
            "source": "ds-critique"
        }

    def _create_synthetic_ds_critique_data(self) -> DatasetDict:
        """
        Create synthetic DS Critique Bank style data
        Based on science questions with explanations
        """
        # Sample science questions
        questions_data = [
            {
                "question": "What happens when salt is dissolved in water?",
                "correct_answer": "The salt breaks into ions",
                "explanation": "When salt (NaCl) dissolves in water, the polar water molecules surround and separate the sodium and chloride ions, breaking the ionic bonds.",
                "alt_explanations": [
                    "Salt molecules spread throughout the water evenly.",
                    "The water breaks the salt into smaller particles.",
                    "Ionic bonds are broken by water molecules."
                ]
            },
            {
                "question": "Why do objects fall towards Earth?",
                "correct_answer": "Gravity pulls them",
                "explanation": "Earth's gravitational force attracts all objects with mass towards its center. This force is proportional to the object's mass and inversely proportional to the square of the distance.",
                "alt_explanations": [
                    "Earth's gravity field attracts mass.",
                    "Objects are pulled by gravitational acceleration.",
                    "The Earth's mass creates a gravitational pull."
                ]
            },
            {
                "question": "What causes seasons on Earth?",
                "correct_answer": "Earth's tilt",
                "explanation": "Earth's axis is tilted at 23.5 degrees. As Earth orbits the sun, different hemispheres receive varying amounts of direct sunlight, causing seasonal temperature changes.",
                "alt_explanations": [
                    "The tilt of Earth's rotation axis affects sunlight.",
                    "Axial tilt causes varied solar radiation.",
                    "Different angles of sunlight due to tilt."
                ]
            },
            {
                "question": "Why does ice float on water?",
                "correct_answer": "Ice is less dense than water",
                "explanation": "When water freezes, its molecules form a crystalline structure with more space between them than in liquid water, making ice less dense and causing it to float.",
                "alt_explanations": [
                    "Frozen water has lower density.",
                    "Ice molecules are more spread out.",
                    "The solid form is lighter than liquid."
                ]
            },
            {
                "question": "What makes plants green?",
                "correct_answer": "Chlorophyll",
                "explanation": "Chlorophyll is a pigment in plant cells that absorbs red and blue light for photosynthesis but reflects green light, making plants appear green to our eyes.",
                "alt_explanations": [
                    "Green pigment reflects green wavelengths.",
                    "Chlorophyll absorbs other colors except green.",
                    "The pigment for photosynthesis is green."
                ]
            }
        ]

        # Generate examples for each split
        train_examples = []
        val_examples = []
        test_examples = []

        max_samples = self.config.get("max_samples_per_source", 10000)
        samples_per_question = max(1, max_samples // (len(questions_data) * 3))

        for i, q_data in enumerate(questions_data):
            # Generate multiple examples per question
            for j in range(samples_per_question):
                # Use main explanation sometimes, alternatives other times
                if j % 4 == 0:
                    explanation = q_data["explanation"]
                else:
                    explanation = random.choice(q_data["alt_explanations"])

                example = {
                    "id": f"ds_synthetic_{i}_{j}",
                    "question": q_data["question"],
                    "correct_answer": q_data["correct_answer"],
                    "student_explanation": explanation,
                    "explanation_score": random.choice([3, 4, 5]),  # Quality scores
                }

                # Split into train/val/test
                split_num = (i * samples_per_question + j) % 10
                if split_num < 7:
                    train_examples.append(example)
                elif split_num < 9:
                    val_examples.append(example)
                else:
                    test_examples.append(example)

        # Create DatasetDict
        return DatasetDict({
            "train": Dataset.from_list(train_examples) if train_examples else Dataset.from_list([{"question": "", "correct_answer": "", "student_explanation": ""}]),
            "validation": Dataset.from_list(val_examples) if val_examples else Dataset.from_list([{"question": "", "correct_answer": "", "student_explanation": ""}]),
            "test": Dataset.from_list(test_examples) if test_examples else Dataset.from_list([{"question": "", "correct_answer": "", "student_explanation": ""}])
        })

    # ----------------------------- RANKING -----------------------------

    def create_ranking_rows(self, q: Dict) -> List[Dict]:
        premise = q["premise"]
        hypothesis = q["hypothesis"]
        label = q["label"]
        gold_expl = q.get("gold_explanation", "")
        source = q["source"]

        query_text = q.get("query_text") or self.format_query(premise, hypothesis, label)
        query_id = q.get("query_id") or f"{source}_{abs(hash(premise + '||' + hypothesis + '||' + label)) % 10**9}"

        candidates = []

        # Level 4: excellent
        if gold_expl:
            candidates.append(self._row(query_id, source, premise, hypothesis, label, query_text,
                                        gold_expl, 4, "gold"))
        else:
            candidates.append(self._row(query_id, source, premise, hypothesis, label, query_text,
                                        self.generate_quality_explanation(premise, hypothesis, label, "excellent"),
                                        4, "generated_excellent"))

        # Level 3: good
        candidates.append(self._row(query_id, source, premise, hypothesis, label, query_text,
                                    self.generate_quality_explanation(premise, hypothesis, label, "good"),
                                    3, "generated_good"))

        # Level 2: fair
        candidates.append(self._row(query_id, source, premise, hypothesis, label, query_text,
                                    self.generate_quality_explanation(premise, hypothesis, label, "fair"),
                                    2, "generated_fair"))

        # Level 1: poor (wrong label)
        wrong_labels = ["entails", "is neutral to", "contradicts"]
        if label in wrong_labels:
            wrong_labels.remove(label)
        wrong_label = random.choice(wrong_labels) if wrong_labels else "is neutral to"
        candidates.append(self._row(query_id, source, premise, hypothesis, label, query_text,
                                    self.generate_quality_explanation(premise, hypothesis, wrong_label, "poor"),
                                    1, "wrong_label"))

        # Level 0: nonsense
        candidates.append(self._row(query_id, source, premise, hypothesis, label, query_text,
                                    self.generate_nonsense_explanation(premise, hypothesis),
                                    0, "nonsense"))

        if self.config["samples_per_query"] and len(candidates) > self.config["samples_per_query"]:
            candidates = random.sample(candidates, self.config["samples_per_query"])

        return candidates

    def _row(self, qid, source, premise, hypothesis, label, qtext, candidate, qscore, method):
        return {
            "query_id": qid,
            "source": source,
            "premise": premise,
            "hypothesis": hypothesis,
            "label": label,
            "query_text": qtext,
            "candidate": candidate,
            "quality_score": int(qscore),
            "generation_method": method,
        }

    # ----------------------------- GENERATION -----------------------------

    def generate_quality_explanation(self, premise: str, hypothesis: str, label: str, quality: str) -> str:
        if not self.config.get("use_generation", False) or self.generator is None:
            return f"[{quality}] {premise} {label} {hypothesis}"
        quality_prompts = {
            "excellent": "Provide a detailed, logically sound explanation with clear reasoning:",
            "good": "Explain briefly but correctly:",
            "fair": "Give a short, simple explanation:",
            "poor": "Explain this quickly:"
        }
        system_msg = quality_prompts.get(quality, quality_prompts["good"])
        user_msg = f"If: '{premise}' {label}: '{hypothesis}', why is that true?"
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.generator.device)
        with torch.no_grad():
            outputs = self.generator.generate(
                **inputs,
                max_new_tokens=64 if quality in ["fair", "poor"] else 128,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id
            )
        response = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        return response

    def generate_nonsense_explanation(self, premise: str, hypothesis: str) -> str:
        nonsense_templates = [
            "Because cats are better than dogs when considering weather patterns.",
            "This is clearly related to the economic implications of pizza consumption.",
            "The answer involves quantum physics and butterfly migration patterns.",
            "Obviously, this connects to ancient civilizations and space exploration.",
            "It's all about musical theory and ocean currents interacting."
        ]
        return random.choice(nonsense_templates)

    def format_query(self, premise: str, hypothesis: str, label: str) -> str:
        return f"If: '{premise}' {label}: '{hypothesis}', why is that true?"

    # ----------------------------- SAVING & STATS -----------------------------

    def _save_normalized_source(self, source: str, source_splits: Dict[str, List[Dict]], raw_base: str):
        """
        Save per-source normalized (query-level) DatasetDict to data/raw/<source>/normalized
        plus a small metadata.json.
        """
        ddict = {}
        for split in ["train", "validation", "test"]:
            rows = source_splits.get(split, [])
            if rows:
                ddict[split] = Dataset.from_list(rows)

        if not ddict:
            return  # Skip saving if no data available

        ds = DatasetDict(ddict)
        out_dir = os.path.join(raw_base, source, "normalized")
        os.makedirs(out_dir, exist_ok=True)
        ds.save_to_disk(out_dir)

        meta = {
            "source": source,
            "rows": {k: len(ds[k]) for k in ds.keys()},
            "fields": list(ds[list(ds.keys())[0]].features.keys()) if len(ds.keys()) > 0 else [],
        }
        with open(os.path.join(out_dir, "metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"Saved normalized source to {out_dir}")

    def save_dataset_metadata(self, dataset: DatasetDict, output_dir: str):
        meta = {
            "config": self.config,
            "quality_levels": self.quality_levels,
            "stats": self.get_dataset_stats(dataset),
            "splits": {k: len(dataset[k]) for k in dataset.keys()}
        }
        with open(os.path.join(output_dir, "metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)

    def get_dataset_stats(self, dataset: DatasetDict) -> Dict:
        stats = {}
        for split in dataset.keys():
            ds = dataset[split]
            sources = {}
            quality_dist = {i: 0 for i in range(5)}
            for ex in ds:
                s = ex["source"]
                sources[s] = sources.get(s, 0) + 1
                quality_dist[int(ex["quality_score"])] += 1
            stats[split] = {
                "total_rows": len(ds),
                "source_row_counts": sources,
                "quality_distribution": quality_dist
            }
        return stats

# ----------------------------- CLI EXAMPLE -----------------------------

def main():
    config = {
        "generation_model": "meta-llama/Llama-3.1-8B-Instruct",
        "max_samples_per_source": 10000,
        "samples_per_query": 5,
        "use_generation": False,  # flip to True if you have the model access
        "random_seed": 42,
        "output_long_table_name": "comprehensive_ranking_dataset",
        "save_normalized_sources": True,
        "save_per_source_long": False,
        "source_ids": {
            "e-snli": ("presencesw/esnli", None),
            "alpha-nli": ("RAR-b/alphanli", None),
            "delta-nli": ("tasksource/defeasible-nli", "snli"),
            "winowhy": ("tasksource/winowhy", None),
            "sens-making": ("tasksource/sen-making", None),
            "ds-critique": ("synthetic", None),
        }
    }
    builder = ComprehensiveDatasetBuilder(config)

    out_dir = "data"  # <— top-level data folder
    dataset = builder.create_comprehensive_dataset(
        sources=["e-snli", "delta-nli", "winowhy", "ds-critique"],
        output_dir=out_dir
    )

    print("Done.")
    total = sum(len(dataset[s]) for s in dataset.keys())
    print(f"Total rows (long format): {total}")

if __name__ == "__main__":
    main()