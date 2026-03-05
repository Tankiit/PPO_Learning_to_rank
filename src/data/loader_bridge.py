"""
Loader Bridge - Unified data loading interface for ranking experiments.

This module provides a unified interface to load different datasets in the
ranking format expected by the training pipeline.

Supported datasets:
  - ds_critique: DS-Critique Bank (fully wired)
  - esnli: e-SNLI with label-based scores
  - chaosnli: ChaosNLI with annotation-based scores
  - multinli: MultiNLI (placeholder)
  - delta_nli: Delta-NLI (placeholder)
  - winowhy: WinoWhy (placeholder)

Usage:
    from src.data.loader_bridge import load_ranking_data

    train_loader, val_loader = load_ranking_data(
        dataset_name='ds_critique',
        tokenizer=tokenizer,
        batch_size=8,
    )
"""

import torch
from torch.utils.data import DataLoader
from typing import Optional, Tuple
import sys
import os

# Import dataset loaders
from .ds_critique_loader import DSCritiqueBankLoader
from .nli_ranking_wrappers import (
    ESNLIRankingDataset,
    ChaosNLIRankingDataset,
    collate_ranking,
)
from .placeholder import create_placeholder_data


def _normalize_dataset_name(dataset_name: str) -> str:
    """Normalize dataset aliases to canonical internal names."""
    alias_map = {
        "deltanli": "delta_nli",
        "delta-nli": "delta_nli",
    }
    return alias_map.get(dataset_name, dataset_name)


# ============================================================================
# Collate function
# ============================================================================

def collate_ranking_batch(batch):
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
# DS-Critique Bank (fully wired)
# ============================================================================

class DSCritiqueRankingDataset(torch.utils.data.Dataset):
    """
    DS-Critique Bank wrapper for ranking experiments.

    This is the template for how dataset wrappers should work.
    """

    def __init__(
        self,
        examples,
        tokenizer,
        max_length: int = 256,
        candidates_per_query: int = 5,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.candidates_per_query = candidates_per_query
        self.examples = examples

        # Tokenize all examples
        self.data = self._tokenize_examples()

    def _tokenize_examples(self):
        """Convert ranking examples to tokenized format."""
        data = []

        for example in self.examples:
            pairs = list(zip(example["candidates"], example["scores"]))
            pairs.sort(key=lambda item: item[1], reverse=True)
            pairs = pairs[:self.candidates_per_query]

            while len(pairs) < self.candidates_per_query:
                pairs.append(("", 0.0))

            candidates = [cand for cand, _ in pairs]
            raw_scores = [score for _, score in pairs]

            # Concatenate query with each candidate
            texts = [
                f"{example['query_text']} {cand}"
                for cand in candidates
            ]

            encoded = self.tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )

            # Normalize scores to [0, 1] range
            max_possible = 5.0
            normalized = [s / max_possible for s in raw_scores]

            data.append({
                "input_ids": encoded["input_ids"],          # [k, seq_len]
                "attention_mask": encoded["attention_mask"], # [k, seq_len]
                "scores": torch.tensor(normalized, dtype=torch.float),  # [k]
            })

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def _load_ds_critique(
    tokenizer,
    batch_size: int = 8,
    max_length: int = 256,
    candidates_per_query: int = 5,
    min_candidates: int = 3,
    cache_dir: str = "data/ds_critique_bank",
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """Load DS-Critique Bank dataset."""
    print("=" * 60)
    print("Loading DS-Critique Bank")
    print("=" * 60)

    # Load ranking data
    loader = DSCritiqueBankLoader(
        cache_dir=cache_dir,
        min_candidates_per_query=min_candidates,
    )
    train_examples, val_examples = loader.load_ranking_data()

    # Create datasets
    train_ds = DSCritiqueRankingDataset(
        train_examples,
        tokenizer,
        max_length=max_length,
        candidates_per_query=candidates_per_query,
    )
    val_ds = DSCritiqueRankingDataset(
        val_examples,
        tokenizer,
        max_length=max_length,
        candidates_per_query=candidates_per_query,
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_ranking_batch,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_ranking_batch,
        num_workers=num_workers,
    )

    return train_loader, val_loader


# ============================================================================
# NLI Datasets
# ============================================================================

def _load_nli(
    dataset_name: str,
    tokenizer,
    batch_size: int = 8,
    max_length: int = 256,
    num_workers: int = 0,
    **kwargs,
) -> Tuple[DataLoader, DataLoader]:
    """
    Load NLI datasets in ranking format.

    Args:
        dataset_name: One of ['esnli', 'chaosnli', 'multinli', 'delta_nli', 'winowhy']
        tokenizer: Tokenizer to use
        batch_size: Batch size
        max_length: Max sequence length
        num_workers: Number of workers for dataloader
        **kwargs: Additional dataset-specific arguments

    Returns:
        (train_loader, val_loader)
    """
    print("=" * 60)
    print(f"Loading NLI dataset: {dataset_name}")
    print("=" * 60)

    if dataset_name == 'esnli':
        return _load_esnli(
            tokenizer=tokenizer,
            batch_size=batch_size,
            max_length=max_length,
            num_workers=num_workers,
            **kwargs,
        )
    elif dataset_name == 'chaosnli':
        return _load_chaosnli(
            tokenizer=tokenizer,
            batch_size=batch_size,
            max_length=max_length,
            num_workers=num_workers,
            **kwargs,
        )
    else:
        # Fallback to placeholder for unimplemented datasets
        print(f"WARNING: {dataset_name} not yet implemented, using placeholder data")
        return create_placeholder_data(
            args=type('Args', (), {
                'candidates_per_query': 5,
                'max_length': max_length,
                'seed': 42,
                'batch_size': batch_size,
            })(),
            tokenizer=tokenizer,
            device='cpu',  # Placeholder doesn't use device
        )


def _load_esnli(
    tokenizer,
    batch_size: int = 8,
    max_length: int = 256,
    data_dir: str = 'data/raw/e-snli/normalized',
    num_workers: int = 0,
    **kwargs,
) -> Tuple[DataLoader, DataLoader]:
    """Load E-SNLI dataset."""
    print(f"Loading E-SNLI from {data_dir}")

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

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_ranking_batch,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_ranking_batch,
        num_workers=num_workers,
    )

    return train_loader, val_loader


def _load_chaosnli(
    tokenizer,
    batch_size: int = 8,
    max_length: int = 256,
    data_path: str = 'data/raw/chaosnli/chaosNLI_v1.0/chaosNLI_snli.jsonl',
    num_workers: int = 0,
    **kwargs,
) -> Tuple[DataLoader, DataLoader]:
    """Load ChaosNLI dataset."""
    print(f"Loading ChaosNLI from {data_path}")

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

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_ranking_batch,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_ranking_batch,
        num_workers=num_workers,
    )

    return train_loader, val_loader


# ============================================================================
# Main loading function
# ============================================================================

def load_ranking_data(
    dataset_name: str,
    tokenizer,
    batch_size: int = 8,
    max_length: int = 256,
    num_workers: int = 0,
    **kwargs,
) -> Tuple[DataLoader, DataLoader]:
    """
    Main entry point for loading ranking datasets.

    Args:
        dataset_name: Name of dataset to load
            - 'ds_critique': DS-Critique Bank (fully implemented)
            - 'esnli': e-SNLI (implemented)
            - 'chaosnli': ChaosNLI (implemented)
            - 'multinli': MultiNLI (placeholder)
            - 'delta_nli': Delta-NLI (placeholder)
            - 'winowhy': WinoWhy (placeholder)
        tokenizer: Tokenizer to use
        batch_size: Batch size for dataloaders
        max_length: Max sequence length
        num_workers: Number of workers for dataloaders
        **kwargs: Dataset-specific arguments

    Returns:
        (train_loader, val_loader)

    Example:
        >>> from transformers import AutoTokenizer
        >>> tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        >>> train_loader, val_loader = load_ranking_data(
        ...     dataset_name='ds_critique',
        ...     tokenizer=tokenizer,
        ...     batch_size=8,
        ... )
    """
    dataset_name = _normalize_dataset_name(dataset_name)

    if dataset_name == 'ds_critique':
        ds_kwargs = dict(kwargs)
        candidates_per_query = ds_kwargs.pop("candidates_per_query", 5)
        return _load_ds_critique(
            tokenizer=tokenizer,
            batch_size=batch_size,
            max_length=max_length,
            candidates_per_query=candidates_per_query,
            num_workers=num_workers,
            **ds_kwargs,
        )
    elif dataset_name in ['esnli', 'chaosnli', 'multinli', 'delta_nli', 'winowhy']:
        return _load_nli(
            dataset_name=dataset_name,
            tokenizer=tokenizer,
            batch_size=batch_size,
            max_length=max_length,
            num_workers=num_workers,
            **kwargs,
        )
    else:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Supported: ['ds_critique', 'esnli', 'chaosnli', 'multinli', 'delta_nli', 'winowhy']"
        )


def get_data_loaders(args, tokenizer) -> Tuple[DataLoader, DataLoader]:
    """
    Adapter expected by `src.train`: route from argparse args to loaders.
    """
    return load_ranking_data(
        dataset_name=args.dataset,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        num_workers=args.num_workers,
        candidates_per_query=args.candidates_per_query,
    )


# ============================================================================
# CLI for testing
# ============================================================================

if __name__ == "__main__":
    import argparse
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(description="Test data loader bridge")
    parser.add_argument("--dataset", type=str, default="ds_critique",
                        choices=['ds_critique', 'esnli', 'chaosnli', 'multinli', 'delta_nli', 'winowhy'])
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--num_batches", type=int, default=2,
                        help="Number of batches to test")
    args = parser.parse_args()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # Load data
    print(f"\nTesting loader for: {args.dataset}")
    train_loader, val_loader = load_ranking_data(
        dataset_name=args.dataset,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    # Test a few batches
    print(f"\nTesting train loader ({len(train_loader)} batches)...")
    for i, batch in enumerate(train_loader):
        if i >= args.num_batches:
            break
        print(f"  Batch {i+1}:")
        print(f"    input_ids shape: {batch['input_ids'].shape}")
        print(f"    attention_mask shape: {batch['attention_mask'].shape}")
        print(f"    scores shape: {batch['scores'].shape}")
        print(f"    scores range: {batch['scores'].min():.3f} - {batch['scores'].max():.3f}")

    print(f"\n✓ Loader test passed!")
