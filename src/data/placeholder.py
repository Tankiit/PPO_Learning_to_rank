"""
Placeholder data for testing the training pipeline.

DELETE THIS once you wire up your actual dataset loaders:
  - esnli_loader.py
  - multinli_loader.py  
  - delta_nli_loader.py
  - winowhy_loader.py
  - ds_critique_loader.py
  - graded_nli_builder.py
  - heuristic_scorer.py
"""

import torch
from torch.utils.data import Dataset, DataLoader


class PlaceholderRankingDataset(Dataset):
    """
    Generates fake ranking data to test the pipeline.
    
    Each item: k candidates per query with quality scores [4,3,2,1,0] → [1.0, 0.75, 0.5, 0.25, 0.0]
    """
    
    def __init__(self, tokenizer, n_queries=200, k=5, max_length=128, seed=42):
        self.tokenizer = tokenizer
        self.k = k
        self.max_length = max_length
        
        torch.manual_seed(seed)
        
        # Fake queries and candidates
        self.data = []
        for i in range(n_queries):
            query = f"Why does premise {i} entail hypothesis {i}?"
            candidates = [
                f"Gold explanation: The premise directly supports because {i}.",
                f"Good explanation: They are related due to {i}.",
                f"Fair explanation: The premise supports it.",
                f"Poor explanation: This contradicts because different.",
                f"Nonsense: Quantum penguin migration umbrella.",
            ]
            scores = torch.tensor([1.0, 0.75, 0.5, 0.25, 0.0])
            
            # Tokenize all candidates
            texts = [f"{query} {c}" for c in candidates]
            enc = tokenizer(
                texts,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=max_length,
            )
            
            self.data.append({
                "input_ids": enc["input_ids"],        # [k, seq_len]
                "attention_mask": enc["attention_mask"],  # [k, seq_len]
                "scores": scores,                     # [k]
            })
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


def collate_ranking(batch):
    """Collate ranking batches: flatten k candidates into batch dim."""
    input_ids = torch.cat([b["input_ids"] for b in batch], dim=0)       # [B*k, seq]
    attention_mask = torch.cat([b["attention_mask"] for b in batch], dim=0)
    scores = torch.stack([b["scores"] for b in batch], dim=0)           # [B, k]
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "scores": scores,
    }


def create_placeholder_data(args, tokenizer, device):
    """Create train and val dataloaders with fake data."""
    train_ds = PlaceholderRankingDataset(
        tokenizer, n_queries=100, k=args.candidates_per_query,
        max_length=args.max_length, seed=args.seed,
    )
    val_ds = PlaceholderRankingDataset(
        tokenizer, n_queries=50, k=args.candidates_per_query,
        max_length=args.max_length, seed=args.seed + 1,
    )
    
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=True, collate_fn=collate_ranking,
        num_workers=0,  # 0 for placeholder
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size,
        shuffle=False, collate_fn=collate_ranking,
        num_workers=0,
    )
    
    return train_loader, val_loader
