from typing import Dict, List, Optional
from datasets import load_dataset, DatasetDict
from tqdm import tqdm

class StackExchangeLoader:
    """
    Loader for the Stack Exchange dataset.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir
        self.dataset = None

    def load_dataset(self) -> DatasetDict:
        """Load the Stack Exchange dataset from Hugging Face."""
        print("Loading Stack Exchange dataset...")
        self.dataset = load_dataset("lvwerra/stack-exchange-paired", cache_dir=self.cache_dir)
        return self.dataset

    def convert_to_ranking_format(self, split: str = 'train') -> List[Dict]:
        """Convert the Stack Exchange dataset to the ranking format."""
        if self.dataset is None:
            self.load_dataset()

        data = self.dataset[split]
        ranking_examples = []

        for example in tqdm(data, desc=f"Processing {split} split"):
            # The dataset is already in a pairwise preference format.
            # We will create a ranking with two choices.
            query = example['question']
            chosen_answer = example['response_j']
            rejected_answer = example['response_k']

            ranking_examples.append({
                'query': query,
                'explanations': [chosen_answer, rejected_answer],
                'scores': [1, 0],  # Higher score for the chosen answer
                'num_candidates': 2
            })

        return ranking_examples

    def create_regression_data(self, ranking_examples: List[Dict]) -> List[Dict]:
        """Create regression training data."""
        regression_examples = []

        for example in ranking_examples:
            query = example['query']
            chosen_answer = example['explanations'][0]
            rejected_answer = example['explanations'][1]

            regression_examples.append({
                'query': query,
                'explanation': chosen_answer,
                'score': 1,
                'normalized_score': 1.0
            })
            regression_examples.append({
                'query': query,
                'explanation': rejected_answer,
                'score': 0,
                'normalized_score': 0.0
            })

        return regression_examples
