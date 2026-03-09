#!/usr/bin/env python3
import json
import os

results = {}
experiments = {
    'MSE': 'results/ds_critique_mse_50ep',
    'ListNet': 'results/ds_critique_listnet_50ep',
    'RankNet': 'results/ds_critique_ranknet_50ep',
    'ApproxNDCG': 'results/ds_critique_approxndcg_50ep',
    'LambdaRank': 'results/ds_critique_lambdarank_50ep',
}

for name, path in experiments.items():
    try:
        with open(f'{path}/results.json') as f:
            data = json.load(f)
        results[name] = {
            'best_ndcg5': data['best_ndcg5'],
            'epochs': len(data['history']),
            'final_loss': data['history'][-1]['train_loss'],
            'best_spearman': max(abs(h.get('spearman', 0)) for h in data['history']),
            'best_separation': max(h.get('separation_ratio', 0) for h in data['history']),
            'time_hours': data['total_time_hours']
        }
    except Exception as e:
        results[name] = {'error': str(e)}

# Print comparison table
print("=" * 90)
print("LOSS FUNCTION COMPARISON (DS-Critique Bank Dataset)")
print("=" * 90)
print(f"{'Loss':<12} {'NDCG@5':>10} {'Epochs':>8} {'Spearman':>10} {'Separation':>12} {'Time(h)':>8}")
print("-" * 90)

for name in ['MSE', 'ListNet', 'RankNet', 'ApproxNDCG', 'LambdaRank']:
    r = results.get(name, {})
    if 'error' in r:
        print(f"{name:<12} ERROR: {r['error']}")
    else:
        print(f"{name:<12} {r['best_ndcg5']:>10.4f} {r['epochs']:>8} {r['best_spearman']:>10.4f} {r['best_separation']:>12.4f} {r['time_hours']:>8.2f}")

print("=" * 90)

# Find best
best_ndcg = max(results.items(), key=lambda x: x[1].get('best_ndcg5', 0) if isinstance(x[1], dict) else 0)
print(f"\nBest NDCG@5: {best_ndcg[0]} ({best_ndcg[1]['best_ndcg5']:.4f})")
