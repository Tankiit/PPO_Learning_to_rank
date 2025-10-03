#!/usr/bin/env python3
"""Analyze training logs and extract key metrics"""

import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def parse_training_log(log_file):
    """Parse training log file and extract metrics"""
    with open(log_file, 'r') as f:
        content = f.read()

    # Extract configuration
    config = {}
    config_patterns = {
        'base_model': r'Base model: (.+)',
        'batch_size': r'Batch size: (\d+)',
        'learning_rate': r'Learning rate: ([\d.e-]+)',
        'num_epochs': r'Num epochs: (\d+)',
    }

    for key, pattern in config_patterns.items():
        match = re.search(pattern, content)
        if match:
            config[key] = match.group(1)

    # Extract epoch results
    epochs = []
    epoch_pattern = r'Epoch (\d+)/(\d+).*?Train Loss: ([\d.]+).*?Val NDCG@1: ([\d.]+).*?Val NDCG@3: ([\d.]+).*?Val NDCG@5: ([\d.]+).*?Val MAP: ([\d.]+).*?Val Spearman: ([\d.]+)'

    for match in re.finditer(epoch_pattern, content, re.DOTALL):
        epochs.append({
            'epoch': int(match.group(1)),
            'total_epochs': int(match.group(2)),
            'train_loss': float(match.group(3)),
            'val_ndcg@1': float(match.group(4)),
            'val_ndcg@3': float(match.group(5)),
            'val_ndcg@5': float(match.group(6)),
            'val_map': float(match.group(7)),
            'val_spearman': float(match.group(8))
        })

    return config, epochs

def create_training_report(log_files, output_dir='visualizations/training_analysis'):
    """Create comprehensive training analysis report"""
    import os
    os.makedirs(output_dir, exist_ok=True)

    all_runs = {}

    # Parse all log files
    for log_file in log_files:
        log_path = Path(log_file)
        run_name = log_path.stem

        try:
            config, epochs = parse_training_log(log_file)
            if epochs:
                all_runs[run_name] = {
                    'config': config,
                    'epochs': epochs,
                    'df': pd.DataFrame(epochs)
                }
                print(f"✅ Parsed {run_name}: {len(epochs)} epochs")
            else:
                print(f"⚠️  No epochs found in {run_name}")
        except Exception as e:
            print(f"❌ Error parsing {log_file}: {e}")

    if not all_runs:
        print("No valid training data found!")
        return

    # Create comparison plots
    plot_metrics_comparison(all_runs, output_dir)
    create_summary_tables(all_runs, output_dir)

    return all_runs

def plot_metrics_comparison(all_runs, output_dir):
    """Create comparison plots for all metrics"""

    metrics = ['train_loss', 'val_ndcg@1', 'val_ndcg@3', 'val_ndcg@5', 'val_map', 'val_spearman']
    metric_labels = {
        'train_loss': 'Training Loss',
        'val_ndcg@1': 'Validation NDCG@1',
        'val_ndcg@3': 'Validation NDCG@3',
        'val_ndcg@5': 'Validation NDCG@5',
        'val_map': 'Validation MAP',
        'val_spearman': 'Validation Spearman'
    }

    # Plot each metric
    for metric in metrics:
        plt.figure(figsize=(12, 6))

        for run_name, run_data in all_runs.items():
            df = run_data['df']
            if metric in df.columns:
                plt.plot(df['epoch'], df[metric], marker='o', label=run_name, linewidth=2)

        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel(metric_labels.get(metric, metric), fontsize=12)
        plt.title(f'{metric_labels.get(metric, metric)} Across Training Runs', fontsize=14, fontweight='bold')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plt.savefig(f'{output_dir}/{metric}.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved {metric} plot")

    # Create combined plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for idx, metric in enumerate(metrics):
        ax = axes[idx]

        for run_name, run_data in all_runs.items():
            df = run_data['df']
            if metric in df.columns:
                ax.plot(df['epoch'], df[metric], marker='o', label=run_name, linewidth=2, markersize=4)

        ax.set_xlabel('Epoch', fontsize=10)
        ax.set_ylabel(metric_labels.get(metric, metric), fontsize=10)
        ax.set_title(metric_labels.get(metric, metric), fontsize=11, fontweight='bold')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/all_metrics_combined.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved combined metrics plot")

def create_summary_tables(all_runs, output_dir):
    """Create summary tables"""

    # Final metrics table
    summary_data = []

    for run_name, run_data in all_runs.items():
        df = run_data['df']
        config = run_data['config']

        if not df.empty:
            final_epoch = df.iloc[-1]
            best_epoch = df.loc[df['val_ndcg@5'].idxmax()]

            summary_data.append({
                'Run': run_name,
                'Epochs': len(df),
                'Batch Size': config.get('batch_size', 'N/A'),
                'Learning Rate': config.get('learning_rate', 'N/A'),
                'Final Train Loss': f"{final_epoch['train_loss']:.4f}",
                'Final NDCG@5': f"{final_epoch['val_ndcg@5']:.4f}",
                'Best NDCG@5': f"{best_epoch['val_ndcg@5']:.4f}",
                'Best Epoch': int(best_epoch['epoch']),
                'Final MAP': f"{final_epoch['val_map']:.4f}",
                'Final Spearman': f"{final_epoch['val_spearman']:.4f}"
            })

    summary_df = pd.DataFrame(summary_data)

    # Save to CSV
    csv_path = f'{output_dir}/runs_summary.csv'
    summary_df.to_csv(csv_path, index=False)
    print(f"✅ Saved summary to {csv_path}")

    # Print to console
    print("\n" + "="*100)
    print("TRAINING RUNS SUMMARY")
    print("="*100)
    print(summary_df.to_string(index=False))
    print("="*100)

    # Best performance comparison
    print("\n" + "="*100)
    print("BEST PERFORMANCE COMPARISON")
    print("="*100)

    for run_name, run_data in all_runs.items():
        df = run_data['df']
        if not df.empty:
            best_epoch = df.loc[df['val_ndcg@5'].idxmax()]
            print(f"\n{run_name}:")
            print(f"  Best Epoch: {int(best_epoch['epoch'])}")
            print(f"  NDCG@1: {best_epoch['val_ndcg@1']:.4f}")
            print(f"  NDCG@3: {best_epoch['val_ndcg@3']:.4f}")
            print(f"  NDCG@5: {best_epoch['val_ndcg@5']:.4f}")
            print(f"  MAP: {best_epoch['val_map']:.4f}")
            print(f"  Spearman: {best_epoch['val_spearman']:.4f}")

    print("="*100)

def main():
    print("="*100)
    print("TRAINING LOG ANALYSIS")
    print("="*100)

    # Find all training log files
    log_files = [
        'training_log.txt',
        'training_log_new.txt',
        'training_log_50epochs.txt'
    ]

    # Filter existing files
    existing_logs = [f for f in log_files if Path(f).exists()]

    print(f"\nFound {len(existing_logs)} training log files:")
    for log in existing_logs:
        size = Path(log).stat().st_size / 1024  # KB
        print(f"  - {log} ({size:.1f} KB)")

    print("\nParsing logs...")
    all_runs = create_training_report(existing_logs)

    print(f"\n✅ Analysis complete! Check visualizations/training_analysis/ for results")

if __name__ == "__main__":
    main()
