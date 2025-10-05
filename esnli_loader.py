import json
from typing import Dict, List, Tuple, Optional
from datasets import load_dataset, Dataset, DatasetDict
import numpy as np
from tqdm import tqdm

import os

class ESNLILoader:
    """
    Loader for e-SNLI dataset
    Converts labels to scores for training ranking reward models
    """

    def __init__(self, cache_dir: Optional[str] = None, data_dir: Optional[str] = 'data/raw/e-snli/normalized', use_local: bool = True):
        # Dynamically determine the project root
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
        self.cache_dir = cache_dir
        self.dataset = None
        self.score_mapping = {
            "entailment": 3,
            "neutral": 2,
            "contradiction": 1
        }
        self.data_dir = os.path.join(project_root, data_dir)
        self.use_local = use_local

    def load_dataset(self) -> DatasetDict:
        """Load e-SNLI from local processed data"""
        print("Loading e-SNLI dataset...")
        if self.use_local:
            print(f"Loading from local directory: {self.data_dir}")
            from datasets import load_from_disk
            try:
                self.dataset = load_from_disk(self.data_dir)
            except Exception as e:
                print(f"Error loading local data: {e}")
                raise
        else:
            # Placeholder for loading from Hub if ever needed
            print("Loading from local, remote loading not implemented for e-SNLI.")
            self.dataset = self.load_dataset()


        return self.dataset

    def convert_to_ranking_format(self,
                                 split: str = 'train',
                                 include_critiques: bool = False) -> List[Dict]:
        """
        Convert e-SNLI format to ranking training format
        """
        if self.dataset is None:
            self.load_dataset()

        data = self.dataset[split]
        ranking_examples = []
        query_groups = {}

        for example in tqdm(data, desc=f"Processing {split} split"):
            query_id = example.get('query_id')
            premise = example.get('premise')
            explanation = example.get('gold_explanation')
            label = example.get('label')

            if not query_id:
                query_id = premise

            if query_id not in query_groups:
                query_groups[query_id] = {
                    'question': premise,
                    'explanations': [],
                    'scores': [],
                    'critiques': []
                }
            
            score = self.score_mapping.get(label, 0)

            query_groups[query_id]['explanations'].append(explanation)
            query_groups[query_id]['scores'].append(score)
            if include_critiques:
                query_groups[query_id]['critiques'].append(f"Label: {label}")

        for query_id, group_data in query_groups.items():
            explanations = group_data['explanations']
            scores = group_data['scores']

            # For e-SNLI, each example typically has 1 explanation
            # We keep them all for regression training
            if len(explanations) < 1:
                continue

            ranking_example = {
                'query': self._format_as_query(group_data['question']),
                'explanations': explanations,
                'scores': scores,
                'num_candidates': len(explanations)
            }
            if include_critiques:
                ranking_example['critiques'] = group_data['critiques']

            ranking_examples.append(ranking_example)

        return ranking_examples

    def create_regression_data(self, ranking_examples: List[Dict]) -> List[Dict]:
        """
        Create regression training data where model learns to predict scores
        """
        regression_examples = []

        for example in ranking_examples:
            query = example['query']

            for exp, score in zip(example['explanations'], example['scores']):
                regression_examples.append({
                    'query': query,
                    'explanation': exp,
                    'score': score,
                    'normalized_score': (score - 1) / 2.0  # Normalize to [0, 1]
                })

        return regression_examples

    def _format_as_query(self, premise: str) -> str:
        """Format premise as instruction-style query"""
        return f"Premise: {premise}"

    def get_statistics(self, split: str = 'train') -> Dict:
        """Get dataset statistics"""
        ranking_data = self.convert_to_ranking_format(split)

        all_scores = []
        num_explanations = []

        for example in ranking_data:
            all_scores.extend(example['scores'])
            num_explanations.append(example['num_candidates'])

        return {
            'num_questions': len(ranking_data),
            'total_explanations': sum(num_explanations),
            'avg_explanations_per_question': np.mean(num_explanations),
            'score_distribution': {
                score: all_scores.count(score) for score in self.score_mapping.values()
            },
            'avg_score': np.mean(all_scores),
            'std_score': np.std(all_scores)
        }
