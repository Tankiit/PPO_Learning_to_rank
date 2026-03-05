"""
Aggregate multi-seed results into paper-ready format.

Reads results.json from each seed directory, computes:
  - mean ± std for each metric
  - 95% bootstrap confidence intervals
  - Paired significance tests (ListNet vs each baseline)

Outputs:
  - aggregated_results.json — machine-readable
  - paper_tables.txt — LaTeX-ready tables

Usage:
    python -m scripts.aggregate_results --input results/multi_seed
"""

import argparse
import json
import os
from pathlib import Path
from collections import defaultdict

import numpy as np
from src.evaluation.metrics import bootstrap_ci, paired_bootstrap_test


def load_seed_results(experiment_dir: Path) -> dict:
    """Load results from all seed subdirectories."""
    results = {}
    
    for condition_dir in sorted(experiment_dir.iterdir()):
        if not condition_dir.is_dir():
            continue
        
        condition_name = condition_dir.name
        seed_results = []
        
        for seed_dir in sorted(condition_dir.iterdir()):
            if not seed_dir.is_dir():
                continue
            
            results_file = seed_dir / "results.json"
            if results_file.exists():
                with open(results_file) as f:
                    data = json.load(f)
                seed_results.append(data.get("final_metrics", {}))
        
        if seed_results:
            results[condition_name] = seed_results
    
    return results


def aggregate_condition(seed_results: list) -> dict:
    """Aggregate metrics across seeds for one condition."""
    all_metrics = defaultdict(list)
    
    for r in seed_results:
        for k, v in r.items():
            if isinstance(v, (int, float)):
                all_metrics[k].append(v)
    
    agg = {}
    for k, values in all_metrics.items():
        arr = np.array(values)
        ci_lo, ci_hi = bootstrap_ci(values)
        agg[k] = {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "ci_95": [ci_lo, ci_hi],
            "n_seeds": len(values),
            "values": values,
        }
    
    return agg


def significance_tests(results: dict, baseline: str = "listnet") -> dict:
    """Run paired bootstrap tests: baseline vs each other condition."""
    if baseline not in results:
        return {}
    
    tests = {}
    baseline_metrics = results[baseline]
    
    for condition, condition_metrics in results.items():
        if condition == baseline:
            continue
        
        tests[f"{baseline}_vs_{condition}"] = {}
        
        for metric in ["ndcg@5", "map", "spearman", "separation_ratio"]:
            if metric in baseline_metrics and metric in condition_metrics:
                base_vals = baseline_metrics[metric]["values"]
                cond_vals = condition_metrics[metric]["values"]
                
                if len(base_vals) == len(cond_vals) and len(base_vals) > 1:
                    diff, p_val = paired_bootstrap_test(base_vals, cond_vals)
                    tests[f"{baseline}_vs_{condition}"][metric] = {
                        "diff": diff,
                        "p_value": p_val,
                        "significant_005": p_val < 0.05,
                        "significant_001": p_val < 0.01,
                    }
    
    return tests


def format_latex_table_losses(results: dict) -> str:
    """Generate LaTeX for Table 2 (loss function comparison)."""
    lines = [
        r"\begin{table}[t]",
        r"\centering\small",
        r"\caption{Loss function comparison (RoBERTa-base, 5 seeds). "
        r"$\dagger$: significantly different from ListNet ($p < 0.01$).}",
        r"\begin{tabular}{@{}lcccc@{}}",
        r"\toprule",
        r"\textbf{Loss} & \textbf{NDCG@5} & \textbf{Spearman $\rho$} & "
        r"\textbf{Sep. Ratio} & \textbf{Score Range} \\",
        r"\midrule",
    ]
    
    order = ["mse", "binary", "ranknet", "approxndcg", "listnet"]
    
    for loss in order:
        if loss not in results:
            continue
        
        r = results[loss]
        
        def fmt(metric):
            if metric in r:
                return f"{r[metric]['mean']:.3f}$\\pm${r[metric]['std']:.3f}"
            return "—"
        
        bold = r"\textbf{" if loss == "listnet" else ""
        end_bold = "}" if loss == "listnet" else ""
        
        name_map = {
            "mse": "MSE Regression",
            "binary": "Binary (DPO)",
            "ranknet": "RankNet",
            "approxndcg": "ApproxNDCG",
            "listnet": r"\textbf{ListNet (ours)}",
        }
        
        lines.append(
            f"{name_map[loss]} & {bold}{fmt('ndcg@5')}{end_bold} & "
            f"{bold}{fmt('spearman')}{end_bold} & "
            f"{bold}{fmt('separation_ratio')}{end_bold} & "
            f"{bold}{fmt('score_range')}{end_bold} \\\\"
        )
    
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="results/multi_seed")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    output_dir = Path(args.output) if args.output else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Aggregating results from: {input_dir}")
    
    # Process each experiment
    for exp_dir in sorted(input_dir.iterdir()):
        if not exp_dir.is_dir() or exp_dir.name.startswith("."):
            continue
        
        exp_name = exp_dir.name
        print(f"\n{'='*60}")
        print(f"Experiment: {exp_name}")
        print(f"{'='*60}")
        
        # Load all seed results
        raw_results = load_seed_results(exp_dir)
        
        if not raw_results:
            print("  No results found, skipping.")
            continue
        
        # Aggregate
        aggregated = {}
        for condition, seeds in raw_results.items():
            aggregated[condition] = aggregate_condition(seeds)
            n = len(seeds)
            print(f"\n  {condition} ({n} seeds):")
            for metric in ["ndcg@5", "map", "spearman", "separation_ratio"]:
                if metric in aggregated[condition]:
                    m = aggregated[condition][metric]
                    print(f"    {metric}: {m['mean']:.4f} ± {m['std']:.4f}  "
                          f"[{m['ci_95'][0]:.4f}, {m['ci_95'][1]:.4f}]")
        
        # Significance tests
        sig_tests = significance_tests(aggregated)
        if sig_tests:
            print(f"\n  Significance tests (vs listnet):")
            for pair, metrics in sig_tests.items():
                for metric, result in metrics.items():
                    star = "***" if result["significant_001"] else ("*" if result["significant_005"] else "ns")
                    print(f"    {pair} | {metric}: diff={result['diff']:+.4f}, p={result['p_value']:.4f} {star}")
        
        # Save
        save_data = {
            "experiment": exp_name,
            "aggregated": {k: {mk: {kk: vv for kk, vv in mv.items() if kk != "values"}
                               for mk, mv in v.items()}
                          for k, v in aggregated.items()},
            "significance_tests": sig_tests,
        }
        
        with open(output_dir / f"{exp_name}_aggregated.json", "w") as f:
            json.dump(save_data, f, indent=2)
        
        # Generate LaTeX table for loss comparison
        if exp_name == "loss_comparison":
            latex = format_latex_table_losses(aggregated)
            with open(output_dir / "table2_losses.tex", "w") as f:
                f.write(latex)
            print(f"\n  LaTeX table saved to: {output_dir}/table2_losses.tex")
    
    print(f"\n{'='*60}")
    print(f"All aggregated results saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
