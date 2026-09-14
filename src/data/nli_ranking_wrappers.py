"""
NLI Ranking Dataset Wrappers

Converts existing NLI loaders (ESNLI, ChaosNLI) into the ranking format
expected by the training pipeline.

Expected output format:
{
    'input_ids': [k, seq_len],       # k candidates per query
    'attention_mask': [k, seq_len],
    'scores': [k],                    # quality scores
}
"""

import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional
import sys
import os

# Add parent directory to path to import loaders
sys.path.insert(0, os.path.dirname(__file__))

from esnli_loader import ESNLILoader
from chaosnli_loader import ChaosNLILoader


class ESNLIRankingDataset(Dataset):
    """
    Wrapper for E-SNLI dataset in ranking format.

    Uses the existing ESNLILoader and converts to ranking batches.
    """

    def __init__(
        self,
        tokenizer,
        split: str = 'train',
        max_length: int = 256,
        cache_dir: Optional[str] = None,
        data_dir: Optional[str] = 'data/raw/e-snli/normalized',
        use_local: bool = True,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split

        # Load data using existing loader
        print(f"Loading E-SNLI {split} split...")
        self.loader = ESNLILoader(
            cache_dir=cache_dir,
            data_dir=data_dir,
            use_local=use_local
        )
        self.loader.load_dataset()

        # Handle missing validation split - use train split with subset
        actual_split = split
        if split not in self.loader.dataset:
            print(f"  WARNING: Split '{split}' not found, using 'train' split instead")
            actual_split = 'train'

        self.ranking_examples = self.loader.convert_to_ranking_format(split=actual_split)

        print(f"Loaded {len(self.ranking_examples)} ranking examples from E-SNLI")

        # Tokenize all examples
        self.data = self._tokenize_examples()

    def _tokenize_examples(self) -> List[Dict]:
        """Convert ranking examples to tokenized format."""
        tokenized_data = []

        for example in self.ranking_examples:
            query = example['query']
            explanations = example['explanations']
            scores = example['scores']

            # Normalize scores to [0, 1] range
            # ESNLI scores are 1, 2, 3 (contradiction, neutral, entailment)
            normalized_scores = [(s - 1) / 2.0 for s in scores]

            # Concatenate query with each explanation
            texts = [f"{query} {exp}" for exp in explanations]

            encoded = self.tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )

            tokenized_data.append({
                "input_ids": encoded["input_ids"],          # [k, seq_len]
                "attention_mask": encoded["attention_mask"], # [k, seq_len]
                "scores": torch.tensor(normalized_scores, dtype=torch.float),  # [k]
            })

        return tokenized_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class ChaosNLIRankingDataset(Dataset):
    """
    Wrapper for ChaosNLI dataset in ranking format.

    Uses the existing ChaosNLILoader and converts to ranking batches.
    """

    def __init__(
        self,
        tokenizer,
        split: str = 'train',
        max_length: int = 256,
        data_path: str = 'data/raw/chaosnli/chaosNLI_v1.0/chaosNLI_snli.jsonl',
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split

        # Load data using existing loader
        print(f"Loading ChaosNLI {split} split...")
        self.loader = ChaosNLILoader(data_path=data_path)
        self.loader.load_dataset()
        self.ranking_examples = self.loader.convert_to_ranking_format(split=split)

        print(f"Loaded {len(self.ranking_examples)} ranking examples from ChaosNLI")

        # Tokenize all examples
        self.data = self._tokenize_examples()

    def _tokenize_examples(self) -> List[Dict]:
        """Convert ranking examples to tokenized format."""
        tokenized_data = []

        for example in self.ranking_examples:
            query = example['query']
            explanations = example['explanations']
            scores = example['scores']

            # Normalize scores to [0, 1] range
            # ChaosNLI scores are already 0-2 from _calculate_score
            normalized_scores = [s / 2.0 for s in scores]

            # Concatenate query with each explanation
            texts = [f"{query} {exp}" for exp in explanations]

            encoded = self.tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )

            tokenized_data.append({
                "input_ids": encoded["input_ids"],          # [k, seq_len]
                "attention_mask": encoded["attention_mask"], # [k, seq_len]
                "scores": torch.tensor(normalized_scores, dtype=torch.float),  # [k]
            })

        return tokenized_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_ranking(batch):
    """
    Collate ranking batches: flatten k candidates into batch dim.

    Input: list of dicts with keys [input_ids, attention_mask, scores]
    Output: single dict with flattened input_ids/attention_mask and stacked scores
    """
    input_ids = torch.cat([b["input_ids"] for b in batch], dim=0)       # [B*k, seq]
    attention_mask = torch.cat([b["attention_mask"] for b in batch], dim=0)
    scores = torch.stack([b["scores"] for b in batch], dim=0)           # [B, k]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "scores": scores,
    }


# ============================================================================
# Convenience functions
# ============================================================================

def create_esnli_dataloaders(
    tokenizer,
    batch_size: int = 8,
    max_length: int = 256,
    data_dir: str = 'data/raw/e-snli/normalized',
    num_workers: int = 0,
):
    """Create train and val dataloaders for E-SNLI."""
    train_ds = ESNLIRankingDataset(
        tokenizer,
        split='train',
        max_length=max_length,
        data_dir=data_dir,
    )
    val_ds = ESNLIRankingDataset(
        tokenizer,
        split='validation',
        max_length=max_length,
        data_dir=data_dir,
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_ranking,
        num_workers=num_workers,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_ranking,
        num_workers=num_workers,
    )

    return train_loader, val_loader


def create_chaosnli_dataloaders(
    tokenizer,
    batch_size: int = 8,
    max_length: int = 256,
    data_path: str = 'data/raw/chaosnli/chaosNLI_v1.0/chaosNLI_snli.jsonl',
    num_workers: int = 0,
):
    """Create train and val dataloaders for ChaosNLI."""
    train_ds = ChaosNLIRankingDataset(
        tokenizer,
        split='train',
        max_length=max_length,
        data_path=data_path,
    )
    val_ds = ChaosNLIRankingDataset(
        tokenizer,
        split='validation',
        max_length=max_length,
        data_path=data_path,
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_ranking,
        num_workers=num_workers,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_ranking,
        num_workers=num_workers,
    )

    return train_loader, val_loader
