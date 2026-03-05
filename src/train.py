"""
Main Training Script for Ranking Reward Models.

Trains a ranking reward model with configurable loss function, backbone, dataset, and seed.
This is the primary entry point for all experiments in the paper.

Usage:
    # Table 2: Loss function comparison (5 seeds)
    python -m src.train --model roberta-base --loss listnet --dataset multinli --seed 42
    python -m src.train --model roberta-base --loss mse --dataset multinli --seed 42
    
    # Table 1: Model comparison
    python -m src.train --model roberta-base --loss listnet --dataset multinli --seed 42
    python -m src.train --model microsoft/deberta-v3-base --loss listnet --dataset multinli --seed 42
    python -m src.train --model mistralai/Mistral-7B-v0.1 --loss listnet --dataset multinli --seed 42 --quantize_4bit
    
    # Table 3: Data creation ablation
    python -m src.train --model roberta-base --loss listnet --dataset multinli --data_method heuristic --seed 42
    python -m src.train --model roberta-base --loss listnet --dataset multinli --data_method overlap --seed 42
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from tqdm import tqdm

from src.models.ranking_reward_model import RankingRewardModel
from src.losses.ranking_losses import get_loss_function
from src.evaluation.metrics import compute_ranking_metrics
from src.data.loader_bridge import get_data_loaders


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(description="Train ranking reward model")
    
    # Model
    parser.add_argument("--model", type=str, default="roberta-base",
                        help="HuggingFace model name")
    parser.add_argument("--quantize_4bit", action="store_true",
                        help="4-bit NF4 quantization (for 7B decoders)")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--pooling", type=str, default="attention",
                        choices=["mean", "cls", "max", "attention", "last"],
                        help="Pooling strategy. 'attention' recommended for encoders.")
    
    # Loss
    parser.add_argument("--loss", type=str, default="listnet",
                        choices=["mse", "binary", "ranknet", "listnet", "approxndcg", "lambdarank"])
    parser.add_argument("--loss_temperature", type=float, default=1.0,
                        help="Temperature for ListNet softmax")
    
    # Data
    parser.add_argument("--dataset", type=str, default="multinli",
                        choices=["esnli", "chaosnli", "multinli", "deltanli", "delta_nli", "delta-nli", "winowhy", "ds_critique"])
    parser.add_argument("--data_method", type=str, default="heuristic",
                        choices=["heuristic", "graded_delta", "overlap"],
                        help="Data creation method (Table 3 ablation)")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--candidates_per_query", type=int, default=5,
                        help="Number of candidates (quality levels) per query")
    
    # Training
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--gradient_clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=42)
    
    # Output
    parser.add_argument("--output_dir", type=str, default="results/default")
    parser.add_argument("--save_best", action="store_true", default=True)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--eval_every_epoch", type=int, default=1)
    
    # Hardware
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num_workers", type=int, default=4)
    
    return parser.parse_args()


def train_one_epoch(model, dataloader, loss_fn, optimizer, scheduler, device, args):
    """
    Train for one epoch.
    
    TODO: Adapt this to your actual data format.
    Each batch should contain:
        - input_ids: [batch * k, seq_len]  (k candidates per query, flattened)
        - attention_mask: [batch * k, seq_len]
        - scores: [batch, k]  (ground-truth quality scores)
    """
    model.train()
    total_loss = 0.0
    n_batches = 0
    
    for batch in dataloader:
        # === ADAPT THIS TO YOUR DATA FORMAT ===
        # Option A: If batch has pre-tokenized tensors
        input_ids = batch["input_ids"].to(device)          # [B*k, seq_len]
        attention_mask = batch["attention_mask"].to(device)  # [B*k, seq_len]
        true_scores = batch["scores"].to(device)            # [B, k]
        
        batch_size = true_scores.shape[0]
        k = true_scores.shape[1]
        
        # Forward: score each candidate
        pred_scores_flat = model(input_ids, attention_mask)  # [B*k, 1]
        pred_scores = pred_scores_flat.view(batch_size, k)   # [B, k]
        
        # Compute ranking loss
        loss = loss_fn(pred_scores, true_scores)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        
        if args.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
        
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, dataloader, loss_fn, device, args):
    """
    Evaluate on validation set.
    Returns dict with loss and all ranking metrics.
    """
    model.eval()
    all_pred_scores = []
    all_true_scores = []
    total_loss = 0.0
    n_batches = 0
    
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        true_scores = batch["scores"].to(device)
        
        batch_size = true_scores.shape[0]
        k = true_scores.shape[1]
        
        pred_scores_flat = model(input_ids, attention_mask)
        pred_scores = pred_scores_flat.view(batch_size, k)
        
        loss = loss_fn(pred_scores, true_scores)
        total_loss += loss.item()
        n_batches += 1
        
        all_pred_scores.append(pred_scores.cpu())
        all_true_scores.append(true_scores.cpu())
    
    all_pred = torch.cat(all_pred_scores, dim=0)  # [N, k]
    all_true = torch.cat(all_true_scores, dim=0)  # [N, k]
    
    # Compute ranking metrics
    metrics = compute_ranking_metrics(all_pred, all_true)
    metrics["loss"] = total_loss / max(n_batches, 1)
    
    # Score separation analysis (critical for PPO viability)
    score_std = all_pred.std(dim=-1).mean().item()
    true_std = all_true.std(dim=-1).mean().item()
    metrics["score_std"] = score_std
    metrics["separation_ratio"] = score_std / max(true_std, 1e-8)
    metrics["score_range"] = (all_pred.max(dim=-1).values - all_pred.min(dim=-1).values).mean().item()
    
    return metrics


def main():
    args = parse_args()
    
    # Seed
    set_seed(args.seed)
    
    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    # Output dir
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(output_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    
    print(f"{'='*60}")
    print(f"Training: {args.model} | Loss: {args.loss} | Seed: {args.seed}")
    print(f"Dataset: {args.dataset} | Method: {args.data_method}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}")
    
    # =========================================================================
    # Model
    # =========================================================================
    model = RankingRewardModel(
        model_name=args.model,
        dropout=args.dropout,
        quantize_4bit=args.quantize_4bit,
        pooling=args.pooling,
    )
    if not args.quantize_4bit:
        model = model.to(device)
    
    # Count trainable params
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")
    
    # =========================================================================
    # Loss
    # =========================================================================
    loss_kwargs = {}
    if args.loss == "listnet":
        loss_kwargs["temperature"] = args.loss_temperature
    loss_fn = get_loss_function(args.loss, **loss_kwargs)
    
    # =========================================================================
    # Data
    # =========================================================================
    train_loader, val_loader = get_data_loaders(args, model.tokenizer)
    
    # =========================================================================
    # Optimizer
    # =========================================================================
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    
    # Linear warmup scheduler
    total_steps = args.epochs * len(train_loader)
    scheduler = LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=min(args.warmup_steps, total_steps),
    )
    
    # =========================================================================
    # Training loop
    # =========================================================================
    best_ndcg = 0.0
    patience_counter = 0
    history = []
    
    start_time = time.time()
    
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        
        # Train
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, scheduler, device, args)
        
        # Evaluate
        if epoch % args.eval_every_epoch == 0 or epoch == args.epochs:
            val_metrics = evaluate(model, val_loader, loss_fn, device, args)
            
            epoch_time = time.time() - epoch_start
            
            record = {
                "epoch": epoch,
                "train_loss": train_loss,
                "epoch_time": epoch_time,
                **val_metrics,
            }
            history.append(record)
            
            # Print
            print(
                f"Epoch {epoch:3d} | "
                f"Loss: {train_loss:.6f} | "
                f"NDCG@5: {val_metrics.get('ndcg@5', 0):.4f} | "
                f"Sep: {val_metrics.get('separation_ratio', 0):.4f} | "
                f"Spearman: {val_metrics.get('spearman', 0):.4f} | "
                f"Time: {epoch_time:.1f}s"
            )
            
            # Save best
            ndcg5 = val_metrics.get("ndcg@5", 0)
            if ndcg5 > best_ndcg:
                best_ndcg = ndcg5
                patience_counter = 0
                if args.save_best:
                    torch.save({
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "config": vars(args),
                        "metrics": val_metrics,
                    }, output_dir / "best_model.pt")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch} (patience={args.patience})")
                break
    
    total_time = time.time() - start_time
    
    # =========================================================================
    # Save results
    # =========================================================================
    results = {
        "config": vars(args),
        "best_ndcg5": best_ndcg,
        "total_time_seconds": total_time,
        "total_time_hours": total_time / 3600,
        "history": history,
        "final_metrics": history[-1] if history else {},
    }
    
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Done! Best NDCG@5: {best_ndcg:.4f} | Total time: {total_time/3600:.2f}h")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
