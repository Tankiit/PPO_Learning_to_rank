#!/usr/bin/env python3
"""Visualize TensorBoard logs from different training runs"""

import os
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import pandas as pd
from pathlib import Path

def load_tensorboard_logs(log_dir):
    """Load TensorBoard event files and extract metrics"""
    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()

    data = {}

    # Get all scalar tags
    tags = ea.Tags()['scalars']

    for tag in tags:
        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]
        data[tag] = {'steps': steps, 'values': values}

    return data

def find_tensorboard_runs():
    """Find all TensorBoard log directories"""
    runs = {}
    models_dir = Path('/home/mukherjee/research/PPO_expts/models')

    for run_dir in models_dir.iterdir():
        if run_dir.is_dir():
            tb_dir = run_dir / 'tensorboard_logs'
            if tb_dir.exists():
                # Find the event file
                event_files = list(tb_dir.glob('events.out.tfevents.*'))
                if event_files:
                    runs[run_dir.name] = str(tb_dir)

    return runs

def plot_training_metrics(runs_data, output_dir='visualizations'):
    """Create comparison plots for all runs"""
    os.makedirs(output_dir, exist_ok=True)

    # Define metrics to plot
    metric_groups = {
        'Loss': ['Loss/train_batch', 'Loss/train_epoch'],
        'NDCG': ['Metrics/val_ndcg@1', 'Metrics/val_ndcg@3', 'Metrics/val_ndcg@5'],
        'Ranking': ['Metrics/val_map', 'Metrics/val_mrr'],
        'Correlation': ['Metrics/val_kendall_tau', 'Metrics/val_spearman']
    }

    for group_name, metrics in metric_groups.items():
        fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 4*len(metrics)))
        if len(metrics) == 1:
            axes = [axes]

        for idx, metric in enumerate(metrics):
            ax = axes[idx]

            for run_name, data in runs_data.items():
                if metric in data:
                    steps = data[metric]['steps']
                    values = data[metric]['values']
                    ax.plot(steps, values, label=run_name, marker='o', markersize=3, alpha=0.7)

            ax.set_xlabel('Step' if 'batch' in metric else 'Epoch', fontsize=11)
            ax.set_ylabel('Value', fontsize=11)
            ax.set_title(metric.replace('_', ' ').title(), fontsize=13, fontweight='bold')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{group_name.lower()}_comparison.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved {group_name} comparison plot")

def create_summary_table(runs_data, output_dir='visualizations'):
    """Create a summary table of final metrics"""
    summary = []

    for run_name, data in runs_data.items():
        row = {'Run': run_name}

        # Get final values for key metrics
        metrics_to_extract = [
            'Loss/train_epoch',
            'Metrics/val_ndcg@1',
            'Metrics/val_ndcg@3',
            'Metrics/val_ndcg@5',
            'Metrics/val_map',
            'Metrics/val_spearman'
        ]

        for metric in metrics_to_extract:
            if metric in data and data[metric]['values']:
                # Get the final value
                final_value = data[metric]['values'][-1]
                metric_name = metric.split('/')[-1].replace('_', ' ').title()
                row[metric_name] = f"{final_value:.4f}"

        summary.append(row)

    df = pd.DataFrame(summary)

    # Save to CSV
    csv_path = os.path.join(output_dir, 'training_summary.csv')
    df.to_csv(csv_path, index=False)
    print(f"✅ Saved summary table to {csv_path}")

    # Print to console
    print("\n" + "="*80)
    print("TRAINING RUNS SUMMARY")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80)

    return df

def plot_learning_curve(runs_data, output_dir='visualizations'):
    """Plot learning curves for training loss"""
    plt.figure(figsize=(14, 6))

    for run_name, data in runs_data.items():
        if 'Loss/train_batch' in data:
            steps = data['Loss/train_batch']['steps']
            values = data['Loss/train_batch']['values']

            # Smooth the curve using moving average
            window = min(50, len(values) // 10)
            if window > 1:
                smoothed = pd.Series(values).rolling(window=window, center=True).mean()
                plt.plot(steps, smoothed, label=f'{run_name} (smoothed)', linewidth=2)
                plt.plot(steps, values, alpha=0.2, linewidth=0.5)
            else:
                plt.plot(steps, values, label=run_name, linewidth=2)

    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel('Training Loss', fontsize=12)
    plt.title('Learning Curves Comparison', fontsize=16, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    save_path = os.path.join(output_dir, 'learning_curves.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved learning curves to {save_path}")

def main():
    print("="*80)
    print("TENSORBOARD LOG VISUALIZATION")
    print("="*80)

    # Find all runs
    print("\nSearching for TensorBoard logs...")
    runs = find_tensorboard_runs()

    if not runs:
        print("❌ No TensorBoard logs found!")
        return

    print(f"✅ Found {len(runs)} training runs:")
    for run_name, log_dir in runs.items():
        print(f"   - {run_name}: {log_dir}")

    # Load data from all runs
    print("\nLoading TensorBoard data...")
    runs_data = {}
    for run_name, log_dir in runs.items():
        try:
            data = load_tensorboard_logs(log_dir)
            runs_data[run_name] = data
            print(f"✅ Loaded {run_name}: {len(data)} metrics")
        except Exception as e:
            print(f"⚠️  Failed to load {run_name}: {e}")

    if not runs_data:
        print("❌ No data could be loaded!")
        return

    # Create output directory
    output_dir = 'visualizations/tensorboard_comparison'
    os.makedirs(output_dir, exist_ok=True)

    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_learning_curve(runs_data, output_dir)
    plot_training_metrics(runs_data, output_dir)

    # Create summary table
    print("\nCreating summary table...")
    create_summary_table(runs_data, output_dir)

    print(f"\n✅ All visualizations saved to: {output_dir}")
    print(f"\nTo view TensorBoard interactively, run:")
    print(f"   tensorboard --logdir=models/")

if __name__ == "__main__":
    main()
