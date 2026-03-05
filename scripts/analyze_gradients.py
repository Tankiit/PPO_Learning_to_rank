"""
Gradient Magnitude Analysis.

Computes and compares gradient magnitudes for MSE vs ListNet during training.
Produces Figure X for the paper showing WHY ListNet gives stronger PPO signals.

Three analyses:
  1. Gradient norms per epoch (MSE vs ListNet)
  2. Advantage estimate distributions under each reward model
  3. Signal-to-noise ratio comparison

Usage:
    python -m scripts.analyze_gradients \
        --mse_model results/multi_seed/loss_comparison/mse/seed_42/best_model.pt \
        --listnet_model results/multi_seed/loss_comparison/listnet/seed_42/best_model.pt \
        --output figures/gradient_analysis.pdf
"""

import argparse
import json
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# TODO: Import your actual model and data loading
# from src.models.ranking_reward_model import RankingRewardModel
# from src.data.graded_nli_builder import GradedNLIDataset


def compute_gradient_norms_during_training(
    model, train_loader, loss_fn, optimizer, device, n_epochs=10
):
    """
    Track gradient norms during training.
    
    Returns:
        list of dicts, one per epoch: {
            "epoch": int,
            "grad_norm_mean": float,
            "grad_norm_std": float,
            "loss": float,
        }
    
    TODO: Wire to your actual training loop. The key computation is:
    
        loss.backward()
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
    """
    history = []
    
    for epoch in range(n_epochs):
        epoch_grads = []
        epoch_loss = 0.0
        n_batches = 0
        
        model.train()
        for batch in train_loader:
            # TODO: Adapt to your batch format
            # input_ids = batch["input_ids"].to(device)
            # attention_mask = batch["attention_mask"].to(device)
            # true_scores = batch["scores"].to(device)
            
            # pred_scores = model(input_ids, attention_mask).view(batch_size, k)
            # loss = loss_fn(pred_scores, true_scores)
            
            optimizer.zero_grad()
            # loss.backward()
            
            # Compute gradient norm
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            total_norm = total_norm ** 0.5
            epoch_grads.append(total_norm)
            
            optimizer.step()
            # epoch_loss += loss.item()
            n_batches += 1
        
        history.append({
            "epoch": epoch + 1,
            "grad_norm_mean": float(np.mean(epoch_grads)) if epoch_grads else 0,
            "grad_norm_std": float(np.std(epoch_grads)) if epoch_grads else 0,
            "loss": epoch_loss / max(n_batches, 1),
        })
    
    return history


def compute_advantage_distributions(reward_model, eval_data, device):
    """
    Compute distribution of PPO advantage estimates under a reward model.
    
    For each query with k candidates:
        reward_i = reward_model(query, candidate_i)
        advantage_i = reward_i - mean(all rewards for this query)
    
    Returns:
        dict with advantages array and statistics
    
    TODO: Wire to your data format.
    """
    reward_model.eval()
    all_advantages = []
    all_ranges = []
    
    with torch.no_grad():
        for query_data in eval_data:
            # TODO: Get rewards for all candidates of one query
            # rewards = reward_model.score_candidates(query_text, candidate_texts)
            
            # Placeholder — replace with actual computation
            rewards = torch.randn(5)  # REPLACE THIS
            
            mean_reward = rewards.mean()
            advantages = rewards - mean_reward
            all_advantages.extend(advantages.tolist())
            all_ranges.append((rewards.max() - rewards.min()).item())
    
    adv = np.array(all_advantages)
    ranges = np.array(all_ranges)
    
    return {
        "advantages": adv,
        "ranges": ranges,
        "adv_mean_abs": float(np.mean(np.abs(adv))),
        "adv_std": float(np.std(adv)),
        "signal_to_noise": float(np.mean(np.abs(adv)) / (np.std(adv) + 1e-8)),
        "range_mean": float(np.mean(ranges)),
        "range_std": float(np.std(ranges)),
    }


def plot_gradient_comparison(mse_history, listnet_history, output_path):
    """
    Create the gradient analysis figure for the paper.
    
    Layout: 1 row × 3 panels
        Panel A: Gradient norms over epochs
        Panel B: Advantage distributions (histogram)
        Panel C: Signal-to-noise ratio bar chart
    """
    fig = plt.figure(figsize=(14, 4))
    gs = GridSpec(1, 3, figure=fig, wspace=0.35)
    
    # --- Panel A: Gradient norms ---
    ax1 = fig.add_subplot(gs[0, 0])
    
    epochs = [h["epoch"] for h in mse_history]
    mse_norms = [h["grad_norm_mean"] for h in mse_history]
    listnet_norms = [h["grad_norm_mean"] for h in listnet_history]
    
    ax1.plot(epochs, mse_norms, 'r-o', label='MSE', markersize=4)
    ax1.plot(epochs, listnet_norms, 'b-s', label='ListNet', markersize=4)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Mean Gradient Norm')
    ax1.set_title('(a) Gradient Magnitude During Training')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # --- Panel B: Advantage distributions ---
    ax2 = fig.add_subplot(gs[0, 1])
    
    # TODO: Replace with actual advantage data
    # For now, use illustrative data matching paper claims
    mse_advantages = np.random.normal(0, 0.02, size=1000)  # Narrow — noise-dominated
    listnet_advantages = np.random.normal(0, 0.22, size=1000)  # Wide — clear signal
    
    ax2.hist(mse_advantages, bins=50, alpha=0.6, color='red', label='MSE (σ≈0.02)', density=True)
    ax2.hist(listnet_advantages, bins=50, alpha=0.6, color='blue', label='ListNet (σ≈0.22)', density=True)
    ax2.set_xlabel('Advantage Estimate')
    ax2.set_ylabel('Density')
    ax2.set_title('(b) PPO Advantage Distributions')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # --- Panel C: Signal-to-noise ---
    ax3 = fig.add_subplot(gs[0, 2])
    
    # TODO: Replace with actual SNR data
    methods = ['MSE', 'Binary', 'RankNet', 'ApproxNDCG', 'ListNet']
    snr_values = [0.4, 0.8, 2.1, 3.5, 11.0]  # Illustrative — replace
    colors = ['#d62728', '#ff7f0e', '#bcbd22', '#17becf', '#1f77b4']
    
    bars = ax3.bar(methods, snr_values, color=colors, edgecolor='black', linewidth=0.5)
    ax3.set_ylabel('Signal-to-Noise Ratio')
    ax3.set_title('(c) Advantage Signal-to-Noise')
    ax3.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='SNR = 1 (noise floor)')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Annotate the ListNet bar
    ax3.annotate(f'{snr_values[-1]:.1f}×', xy=(4, snr_values[-1]),
                 ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mse_model", type=str, default=None,
                        help="Path to MSE-trained model checkpoint")
    parser.add_argument("--listnet_model", type=str, default=None,
                        help="Path to ListNet-trained model checkpoint")
    parser.add_argument("--output", type=str, default="figures/gradient_analysis.pdf")
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    
    # TODO: Load models and compute actual gradient norms
    # For now, generate the figure structure with placeholder data
    
    print("NOTE: Using placeholder data. Wire up model loading to get real values.")
    
    # Placeholder training histories — REPLACE with actual computation
    mse_history = [{"epoch": i+1, "grad_norm_mean": 0.01 * (1 + 0.5 * np.random.randn()),
                    "grad_norm_std": 0.003, "loss": 0.05 / (i+1)}
                   for i in range(10)]
    listnet_history = [{"epoch": i+1, "grad_norm_mean": 0.15 * (1 + 0.3 * np.random.randn()),
                        "grad_norm_std": 0.02, "loss": 0.1 / (i+1)}
                       for i in range(10)]
    
    plot_gradient_comparison(mse_history, listnet_history, args.output)


if __name__ == "__main__":
    import os
    main()
