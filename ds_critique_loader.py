import json
from typing import Dict, List, Tuple, Optional
from datasets import load_dataset, Dataset, DatasetDict
import numpy as np
from tqdm import tqdm

class DSCritiqueBankLoader:
    """
    Loader for Digital Socrates Critique Bank dataset
    Converts 5-point critique scores to training data for ranking reward models
    """

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir
        self.dataset = None
        self.score_mapping = {
            1: "nonsensical",
            2: "poor",
            3: "fair",
            4: "good",
            5: "excellent"
        }

    def load_dataset(self) -> DatasetDict:
        """Load DS_Critique_Bank from HuggingFace or use local synthetic data"""
        print("Loading Digital Socrates Critique Bank...")
        try:
            self.dataset = load_dataset(
                "allenai/DS_Critique_Bank",
                cache_dir=self.cache_dir
            )
        except Exception as e:
            print(f"Could not load DS_Critique_Bank from HF: {e}")
            print("Using local synthetic data instead...")
            self.dataset = self._load_local_synthetic_data()

        return self.dataset

    def _load_local_synthetic_data(self) -> DatasetDict:
        """Load synthetic DS-style data from our create_dataset.py output"""
        from datasets import load_from_disk

        try:
            # Load from our created dataset
            full_dataset = load_from_disk('data/processed/comprehensive_ranking_dataset')

            # Filter only DS critique examples
            train_ds = full_dataset['train'].filter(lambda x: x['source'] == 'ds-critique')
            val_ds = full_dataset['validation'].filter(lambda x: x['source'] == 'ds-critique')
            test_ds = full_dataset['test'].filter(lambda x: x['source'] == 'ds-critique') if 'test' in full_dataset else train_ds.select(range(min(100, len(train_ds))))

            # Convert to DS format - need to keep only essential columns
            def convert_format(example):
                return {
                    'question': example['premise'],
                    'explanation': example['candidate'],
                    'score': example['quality_score'] + 1,  # Convert 0-4 to 1-5
                    'critique': f"Quality level: {example['quality_score']}"
                }

            train_converted = train_ds.map(convert_format, remove_columns=train_ds.column_names)
            val_converted = val_ds.map(convert_format, remove_columns=val_ds.column_names)
            test_converted = test_ds.map(convert_format, remove_columns=test_ds.column_names)

            return DatasetDict({
                'train': train_converted,
                'validation': val_converted,
                'test': test_converted
            })
        except Exception as e:
            print(f"Error loading local data: {e}")
            raise

    def convert_to_ranking_format(self,
                                 split: str = 'train',
                                 include_critiques: bool = False) -> List[Dict]:
        """
        Convert DS format to ranking training format

        Returns list of examples with format:
        {
            'query': instruction-style query,
            'explanations': list of explanations,
            'scores': list of quality scores (1-5),
            'critiques': list of critiques (optional)
        }
        """
        if self.dataset is None:
            self.load_dataset()

        data = self.dataset[split]
        ranking_examples = []

        # Group by unique questions
        question_groups = {}

        for example in tqdm(data, desc=f"Processing {split} split"):
            # Handle both DS format and our format
            question = example.get('question', example.get('premise', ''))
            explanation = example.get('explanation', example.get('candidate', ''))
            score = example.get('score', example.get('quality_score', 0))

            # Convert score if needed (0-4 to 1-5)
            if score <= 4 and score >= 0:
                score = score + 1  # Convert 0-4 to 1-5

            if question not in question_groups:
                question_groups[question] = {
                    'question': question,
                    'explanations': [],
                    'scores': [],
                    'critiques': []
                }

            # Add explanation and score
            question_groups[question]['explanations'].append(explanation)
            question_groups[question]['scores'].append(score)

            if include_critiques:
                critique = example.get('critique', f"Quality: {score}")
                question_groups[question]['critiques'].append(critique)

        # Convert to ranking format
        for question, group_data in question_groups.items():
            if len(group_data['explanations']) >= 2:  # Need at least 2 for ranking
                ranking_example = {
                    'query': self._format_as_query(question),
                    'explanations': group_data['explanations'],
                    'scores': group_data['scores'],
                    'num_candidates': len(group_data['explanations'])
                }

                if include_critiques:
                    ranking_example['critiques'] = group_data['critiques']

                ranking_examples.append(ranking_example)

        return ranking_examples

    def create_pairwise_comparisons(self, ranking_examples: List[Dict]) -> List[Dict]:
        """
        Create pairwise comparisons from ranking data
        For training with preference-based methods
        """
        pairwise_examples = []

        for example in ranking_examples:
            explanations = example['explanations']
            scores = example['scores']
            query = example['query']

            # Create all valid pairs where score_i > score_j
            for i in range(len(explanations)):
                for j in range(len(explanations)):
                    if scores[i] > scores[j]:
                        pairwise_examples.append({
                            'query': query,
                            'chosen': explanations[i],
                            'rejected': explanations[j],
                            'chosen_score': scores[i],
                            'rejected_score': scores[j],
                            'score_diff': scores[i] - scores[j]
                        })

        return pairwise_examples

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
                    'normalized_score': (score - 1) / 4.0  # Normalize to [0, 1]
                })

        return regression_examples

    def _format_as_query(self, question: str) -> str:
        """Format DS question as instruction-style query"""
        # DS questions often already include context
        if "If:" in question and "why" in question.lower():
            return question
        elif "?" in question:
            return f"Question: {question}\nExplain why the answer is correct."
        else:
            return f"Explain: {question}"

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
                score: all_scores.count(score) for score in range(1, 6)
            },
            'avg_score': np.mean(all_scores),
            'std_score': np.std(all_scores)
        }


# Test the loader
if __name__ == "__main__":
    print("Testing DS Critique Bank Loader...")

    loader = DSCritiqueBankLoader()

    # Load dataset
    dataset = loader.load_dataset()
    print(f"\nDataset loaded with splits: {list(dataset.keys())}")

    # Get statistics
    for split in ['train', 'validation', 'test']:
        if split in dataset:
            stats = loader.get_statistics(split)
            print(f"\n{split.upper()} Statistics:")
            print(f"  Questions: {stats['num_questions']}")
            print(f"  Total explanations: {stats['total_explanations']}")
            print(f"  Avg per question: {stats['avg_explanations_per_question']:.2f}")
            print(f"  Score distribution: {stats['score_distribution']}")

    # Test ranking format conversion
    print("\n\nTesting ranking format conversion...")
    ranking_data = loader.convert_to_ranking_format('train')
    print(f"Created {len(ranking_data)} ranking examples")

    if ranking_data:
        print(f"\nSample ranking example:")
        sample = ranking_data[0]
        print(f"  Query: {sample['query'][:100]}...")
        print(f"  Num candidates: {sample['num_candidates']}")
        print(f"  Scores: {sample['scores']}")

    # Test pairwise comparisons
    print("\n\nTesting pairwise comparison creation...")
    pairwise = loader.create_pairwise_comparisons(ranking_data[:10])
    print(f"Created {len(pairwise)} pairwise comparisons from 10 ranking examples")

    if pairwise:
        print(f"\nSample pairwise comparison:")
        sample = pairwise[0]
        print(f"  Query: {sample['query'][:80]}...")
        print(f"  Chosen (score {sample['chosen_score']}): {sample['chosen'][:80]}...")
        print(f"  Rejected (score {sample['rejected_score']}): {sample['rejected'][:80]}...")

    print("\n✅ DS Critique Bank Loader test complete!")
