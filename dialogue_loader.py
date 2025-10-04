from typing import Dict, List, Optional
from datasets import load_dataset, DatasetDict
from tqdm import tqdm
import numpy as np

class DialogueLoader:
    """
    Loader for the Daily Dialog dataset.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir
        self.dataset = None

    def load_dataset(self) -> DatasetDict:
        """Load the Daily Dialog dataset from Hugging Face."""
        print("Loading Daily Dialog dataset...")
        self.dataset = load_dataset("daily_dialog", cache_dir=self.cache_dir)
        return self.dataset

    def convert_to_ranking_format(self, split: str = 'train') -> List[Dict]:
        """Convert the Daily Dialog dataset to the ranking format."""
        if self.dataset is None:
            self.load_dataset()

        data = self.dataset[split]
        ranking_examples = []

        for example in tqdm(data, desc=f"Processing {split} split"):
            dialogue = example['dialog']
            topic = example['topic']
            emotion = example['emotion']
            act = example['act']

            # Create a synthetic quality score based on the annotations.
            # This is a simple example, and more sophisticated scoring functions can be designed.
            score = self._calculate_score(dialogue, topic, emotion, act)

            # For dialogue, the "query" can be the dialogue history,
            # and the "explanation" can be the next utterance.
            for i in range(1, len(dialogue)):
                query = "\n".join(dialogue[:i])
                explanation = dialogue[i]

                ranking_examples.append({
                    'query': query,
                    'explanations': [explanation],
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
                'normalized_score': score / 10.0  # Normalize to [0, 1]
            })

        return regression_examples

    def _calculate_score(self, dialogue: List[str], topic: int, emotion: List[int], act: List[int]) -> float:
        """Calculate a synthetic quality score for a dialogue."""
        # Simple scoring function:
        # - Add 1 point for each turn in the dialogue.
        # - Add 2 points if the dialogue has a topic.
        # - Add 1 point for each unique emotion.
        # - Add 1 point for each unique dialogue act.

        score = 0
        score += len(dialogue)
        if topic != 0:  # Topic 0 is "no topic"
            score += 2
        score += len(set(emotion))
        score += len(set(act))

        return float(score)
