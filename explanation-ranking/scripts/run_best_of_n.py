"""Block B runner: best-of-N for every reward model, then Block C mechanism.

Loads each trained reward model from results/loss_comparison/<loss>_seed<seed>/,
builds candidate sets (graded by default; --source generated for robustness),
scores, and writes best_of_n.json per model + a mechanism_correlation.json.

Usage:
    python scripts/run_best_of_n.py --source graded
    python scripts/run_best_of_n.py --source generated --gen_path data/generated_candidates.jsonl
"""

import argparse
import glob
import json
import os

from explrank.eval import best_of_n_eval, mechanism_correlation


def build_candidate_sets(source: str, **kw):
    # MIGRATE: graded -> from explrank.data.graded_dataset (group by qid);
    #          generated -> read JSONL of policy-LM samples + gold scoring.
    raise NotImplementedError("wire candidate loaders (graded / generated)")


def load_score_fn(model_dir: str):
    # MIGRATE: load EncoderRewardModel/LLMJudge checkpoint -> score_fn(qid, texts)
    raise NotImplementedError("load reward model checkpoint")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results/loss_comparison")
    ap.add_argument("--source", choices=["graded", "generated"], default="graded")
    ap.add_argument("--gen_path", default=None)
    ap.add_argument("--n_values", type=int, nargs="+", default=[2, 4, 8, 16])
    args = ap.parse_args()

    cand = build_candidate_sets(args.source, gen_path=args.gen_path)

    per_model = []
    for model_dir in glob.glob(os.path.join(args.results_dir, "*")):
        mpath = os.path.join(model_dir, "metrics.json")
        if not os.path.exists(mpath):
            continue
        with open(mpath) as f:
            base = json.load(f)
        score_fn = load_score_fn(model_dir)
        bon = best_of_n_eval(cand, score_fn, n_values=args.n_values)
        with open(os.path.join(model_dir, f"best_of_n_{args.source}.json"), "w") as f:
            json.dump(bon, f, indent=2)
        per_model.append({**base, **bon})

    mech = mechanism_correlation(per_model, payoff_key=f"payoff@{max(args.n_values)}")
    out = os.path.join(args.results_dir, f"mechanism_{args.source}.json")
    with open(out, "w") as f:
        json.dump(mech, f, indent=2)
    print(f"Mechanism analysis written to {out}")
    print(f"separation_beats_ndcg: {mech['separation_beats_ndcg']}")


if __name__ == "__main__":
    main()
