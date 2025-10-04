import json
from typing import Dict, List, Optional
from datasets import Dataset, DatasetDict
import numpy as np
from tqdm import tqdm
import random

import os

class ChaosNLILoader:
    """
    Loader for ChaosNLI dataset.
    Converts multiple annotations into a single score for training reward models.
    """

    def __init__(self, data_path: str = 'data/raw/chaosnli/chaosNLI_v1.0/chaosNLI_snli.jsonl'):
        # Dynamically determine the project root by finding the directory that contains the 'data' directory
        current_dir = os.getcwd()
        project_root = current_dir
        while not os.path.exists(os.path.join(project_root, 'data')):
            parent_dir = os.path.dirname(project_root)
            if parent_dir == project_root:
                # Reached the root of the filesystem without finding the 'data' directory
                raise FileNotFoundError("Could not find the 'data' directory in any parent directory.")
            project_root = parent_dir

        self.data_path = os.path.join(project_root, data_path)
        self.dataset = None

    def load_dataset(self) -> DatasetDict:
        """Load ChaosNLI from the specified jsonl file and create splits."""
        print(f"Loading ChaosNLI from {self.data_path}...")
        
        with open(self.data_path, 'r') as f:
            lines = f.readlines()
        
        data = [json.loads(line) for line in lines]
        
        # Shuffle the data
        random.shuffle(data)
        
        # Create splits
        train_size = int(0.8 * len(data))
        val_size = int(0.1 * len(data))
        
        train_data = data[:train_size]
        val_data = data[train_size:train_size + val_size]
        test_data = data[train_size + val_size:]
        
        self.dataset = DatasetDict({
            'train': Dataset.from_list(train_data),
            'validation': Dataset.from_list(val_data),
            'test': Dataset.from_list(test_data)
        })
        
        return self.dataset

    def convert_to_ranking_format(self, split: str = 'train') -> List[Dict]:
        """Convert ChaosNLI format to ranking training format."""
        if self.dataset is None:
            self.load_dataset()

        data = self.dataset[split]
        ranking_examples = []

        for example in tqdm(data, desc=f"Processing {split} split"):
            premise = example['example']['premise']
            hypothesis = example['example']['hypothesis']
            label_counter = example['label_counter']
            
            # There is only one hypothesis per premise in ChaosNLI, so we can't create a ranking of choices.
            # Instead, we will create a single example with a score.
            
            score = self._calculate_score(label_counter)
            
            ranking_examples.append({
                'query': self._format_as_query(premise),
                'explanations': [hypothesis],
                'scores': [score],
                'num_candidates': 1
            })

        return ranking_examples

    def create_regression_data(self, ranking_examples: List[Dict]) -> List[Dict]:
        """Create regression training data."""
        regression_examples = []

        for example in ranking_examples:
            query = example['query']
            explanation = example['explanations'][0]
            score = example['scores'][0]
            
            regression_examples.append({
                'query': query,
                'explanation': explanation,
                'score': score,
                'normalized_score': score # The score is already normalized between 0 and 2
            })

        return regression_examples

    def _calculate_score(self, label_counter: Dict) -> float:
        """Calculate a score from the label counter."""
        entailment_count = label_counter.get('e', 0)
        neutral_count = label_counter.get('n', 0)
        contradiction_count = label_counter.get('c', 0)
        
        total_annotations = entailment_count + neutral_count + contradiction_count
        
        if total_annotations == 0:
            return 0.0
        
        # Score is the weighted average of the labels
        score = (entailment_count * 2 + neutral_count * 1 + contradiction_count * 0) / total_annotations
        
        return score

    def _format_as_query(self, premise: str) -> str:
        """Format the premise as a query."""
        return f"Premise: {premise}"
