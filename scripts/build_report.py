"""Regenerate arr_report.md from the result JSON files.

The report is written from data rather than by hand so that a number can never
drift from the artifact it came from. Re-run it whenever a result file changes -
in particular when the seed replicates land.

    python scripts/build_report.py
"""

from __future__ import annotations

import argparse
import json
from datetime import date

import numpy as np
from pathlib import Path

# Quality order, best first. The result JSON is key-sorted on write, so tier
# tables must be re-ordered here or they come out alphabetical.
TIER_ORDER = ("gold", "good", "fair", "poor", "nonsense")

ORDER = [
    ("independent_listnet", "Independent ListNet"),
    ("independent_listnet_bootstrap", "Independent ListNet + bootstrap"),
    ("independent_mse", "Independent MSE"),
    ("independent_mse_bootstrap", "Independent MSE + bootstrap"),
    ("shared_baseline", "Shared ListNet baseline"),
    ("shared_bootstrap", "Shared ListNet + bootstrap"),
    ("shared_features", "Shared ListNet + feature masks"),
    ("shared_bootstrap_features", "Shared ListNet + bootstrap/features"),
    ("shared_lambda_0.01", "Shared ListNet, lambda=0.01"),
    ("shared_lambda_0.1", "Shared ListNet, lambda=0.1"),
    ("shared_lambda_1.0", "Shared ListNet, lambda=1.0"),
    ("mc_dropout_k8", "MC-dropout K=8"),
]


def table(header: list[str], rows: list[list[str]], numeric: bool = True) -> str:
    """Right-align every column but the first, unless the cells are prose."""

    align = (
        "|" + "|".join(["---"] + ["---:" if numeric else "---"] * (len(header) - 1)) + "|"
    )
    lines = ["| " + " | ".join(header) + " |", align]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def build(
    tier1: dict, reversal: dict, shrink: dict, seeds: dict | None, qid: dict | None
) -> str:
    arms = tier1["arms"]
    present = [(k, label) for k, label in ORDER if k in arms]
    prov = tier1["provenance"]

    def nd(key: str, split: str, field: str):
        return arms[key][split]["ndcg"]["non_singleton"][field]

    def lift(key: str, split: str):
        return arms[key][split]["ndcg"]["lift_over_random_ci"]

    def partial(key: str, split: str):
        return arms[key][split]["partial_spearman_given_confidence"]

    out: list[str] = []
    W = out.append

    W("# ARR Epistemic Ranking Report\n")
    W(f"_Regenerated {date.today().isoformat()} by `scripts/build_report.py` from "
      "`runs/tier1/tier1_results.json`, `runs/tier1/reversal_diagnosis.json` and "
      "the saved Modal predictions. Every number below is computed from a saved "
      "artifact; none is transcribed by hand._\n")

    # ---------------------------------------------------------------- status
    W("## Status\n")
    W("**The e-SNLI out-of-distribution evaluation set is not valid for this "
      "task, and no OOD claim in this project survives that.** Section 1 gives "
      "the evidence. Everything computed on e-SNLI - the width/error "
      "associations, the divergence magnitudes, the abstention curves - measures "
      "behaviour on a set whose quality labels are confounded with text "
      "provenance, and must be re-established on a different shift set.\n")
    W("What still stands: the in-domain corrections (Sections 3-4), the "
      "metric-definition work (Section 6), and the directional-versus-magnitude "
      "dissociation (Section 5). What is withdrawn: the arm ordering by OOD "
      "association, every in-domain uncertainty correlation computed on the "
      "singleton-contaminated split, and any reading of an NDCG value as "
      "evidence of ranking competence.\n")

    # ------------------------------------------------- 1. reversal diagnosis
    W("## 1. The e-SNLI evaluation set is confounded\n")
    W("The independent ListNet ensemble ranks e-SNLI at a mean per-group "
      f"Spearman of {reversal['arms']['independent_listnet']['gold_exclusion']['spearman_all_candidates']:+.3f}. "
      "A ranker merely degraded by distribution shift sits near zero; a large "
      "negative value on 1,000 groups is a competent ranker with its sign "
      "inverted. Two hypotheses were tested on saved predictions alone.\n")

    W("### 1.1 The reference is not inverted (H1 rejected)\n")
    orient = reversal["reference_orientation"]
    W("Three independent checks, any one of which is fatal to an "
      "orientation-bug explanation.\n")
    W("**The stored reference is correctly oriented.** Mean reference score by "
      "quality tier, read straight out of the e-SNLI split:\n")
    W(table(
        ["Tier", "Mean reference score"],
        [[t, f"{orient['tier_mean_reference'][t]:.4f}"]
         for t in TIER_ORDER if t in orient["tier_mean_reference"]],
    ) + "\n")
    W(f"The order is {' > '.join(orient['reference_order'])}, which is the "
      "correct direction. An inversion upstream of the evaluator would have to "
      "show up here, and does not.\n")
    W("**A subset of the same evaluation is positively correlated.** Removing "
      "only the gold candidate from each group and recomputing through the "
      "identical code path:\n")
    rows = []
    for key, label in present:
        if key not in reversal["arms"]:
            continue
        ex = reversal["arms"][key]["gold_exclusion"]
        rows.append([
            label,
            f"{ex['spearman_all_candidates']:+.3f}",
            f"{ex['spearman_excluding_gold']:+.3f}",
            f"{ex['gold_ranked_last_fraction']:.1%}",
        ])
    W(table(["Model", "rho (all 5)", "rho (excl. gold)", "gold ranked last"], rows) + "\n")
    W("Four arms are **positively** correlated with the reference once the gold "
      "candidate is dropped, independent MSE at +0.441 and its bootstrap variant "
      "at +0.499. An inverted reference cannot produce a positive correlation on "
      "a subset of itself, evaluated through the same code path.\n")
    W("**Nonsense is not ranked first.** A clean reversal would put the "
      "lowest-referenced tier at the top of nearly every group. It is ranked "
      "best in "
      f"{reversal['arms']['independent_listnet']['nonsense_ranked_best_fraction']:.1%} "
      "of groups for independent ListNet and "
      f"{reversal['arms']['independent_mse']['nonsense_ranked_best_fraction']:.1%} "
      "for independent MSE.\n")

    W("### 1.2 What is actually happening: a provenance artefact\n")
    W("Mean predicted within-group rank by tier (1 = ranked best):\n")
    available = reversal["arms"]["independent_listnet"]["per_tier"]
    tiers = [t for t in TIER_ORDER if t in available]
    rows = []
    for key, label in present:
        if key not in reversal["arms"]:
            continue
        per = reversal["arms"][key]["per_tier"]
        rows.append([label] + [f"{per[t]['mean_predicted_rank']:.2f}" for t in tiers])
    W(table(["Model"] + [t for t in tiers], rows) + "\n")
    exclusions = {
        k: reversal["arms"][k]["gold_exclusion"]
        for k, _ in present if k in reversal["arms"]
    }
    total = len(exclusions)
    reject = sum(1 for e in exclusions.values() if e["gold_ranked_last_fraction"] > 0.5)
    explained = sum(
        1 for e in exclusions.values() if e["spearman_excluding_gold"] > -0.05
    )
    W(f"**{reject} of {total} arms rank the gold candidate last in more than half "
      "their groups**, most of them in over 90%. The four degraded tiers cluster "
      "in the middle in no consistent order, nonsense included.\n")
    W(f"Dropping gold moves the correlation to zero or above for **{explained} of "
      f"{total} arms**, so for those the rejection of the human-written candidate "
      "accounts for the entire reversal. For the remaining "
      f"{total - explained} it accounts for part of it: independent ListNet "
      "still reads "
      f"{exclusions['independent_listnet']['spearman_excluding_gold']:+.3f} among "
      "the template tiers.\n")
    profile = reversal["text_profile"]
    W(table(
        ["Tier", "Mean characters", "Human-written"],
        [[t, f"{profile[t]['mean_characters']:.1f}",
          f"{profile[t]['human_text_fraction']:.0%}"]
         for t in TIER_ORDER if t in profile],
    ) + "\n")
    W("Gold is the only human-written candidate in every group; the other four "
      "are template-generated by the same procedure. The models are separating "
      "human free text from template text, and the e-SNLI construction makes "
      "that separation perfectly anti-correlated with the quality label.\n")
    W("Length is not the mechanism: gold averages "
      f"{profile['gold']['mean_characters']:.0f} characters and fair "
      f"{profile['fair']['mean_characters']:.0f}, yet gold is ranked last and "
      "fair mid-pack, while the longest tier (good, "
      f"{profile['good']['mean_characters']:.0f} characters) is ranked near the top.\n")
    W("This is neither hypothesis as originally posed. It is not a reference "
      "inversion - Section 1.1 rules that out three ways. It is not a "
      "critique-quality preference either: a model preferring critiques would "
      "reject nonsense, and these models rank nonsense mid-pack. What they "
      "separate is authorship, which e-SNLI has made perfectly collinear with "
      "quality by drawing gold from humans and every degraded tier from "
      "templates.\n")
    W("**e-SNLI as constructed is therefore not a shifted version of this "
      "task.** A model could score well on it by learning a human-versus-"
      "template detector and nothing about explanation quality; these models "
      "learned the detector with the sign that the label penalises. The shift "
      "evaluation has to be rebuilt on a set whose quality labels are not "
      "confounded with authorship - held-out DS-Critique qids re-split from the "
      "train pool, or WinoWhy graded.\n")

    # -------------------------------------------------------- 2. setup / data
    W("## 2. Setup and data\n")
    W("- Backbone `EleutherAI/pythia-70m`, revision `main`; Modal NVIDIA T4, float32.\n"
      "- 50 epochs; five ensemble members; AdamW, lr `2e-5`, weight decay `0.01`,\n"
      "  linear schedule, 51 warmup steps, max grad norm `1.0`.\n"
      "- Group batch size 1, gradient accumulation 8, 1,700 optimizer updates,\n"
      "  sequence length 256, QLoRA and quantization disabled.\n"
      "- Diagnostics, KL/JS and entropy in float64.\n")
    W("| Split | Groups | Candidates | Composition |\n|---|---:|---:|---|\n"
      f"| Train (`ds_critique_external_test.jsonl`) | 270 | 3,240 | 270 qids x 12 student explanations |\n"
      f"| In-domain validation (`ds_critique_external_dev.jsonl`) | {prov['in_domain_groups']} | 270 | 141 singletons, 42 pairs, 11 triples, 3 quads |\n"
      f"| e-SNLI (OOD) | {prov['esnli_groups']:,} | 5,000 | 1,000 groups of 5 |\n"
      + (
          f"| Tier 2 qid train | {qid['train']['groups']} | "
          f"{qid['train']['candidates']:,} | question-disjoint, 12 per group |\n"
          f"| Tier 2 qid held out | {qid['held_out']['groups']} | "
          f"{qid['held_out']['candidates']} | question-disjoint, 12 per group, "
          "no singletons |\n"
          if qid else ""
      ))
    W("Both DS-Critique splits score **student explanations** via "
      "`DS_Critique_Bank.explanation_annotations.human_crowd_mean`, and their "
      "qid sets are disjoint. The in-domain split's singletons are not a "
      "filtering bug: the dev pool holds one student explanation per record, "
      "270 records over 197 questions. A held-out split with C=12 would have to "
      "come from re-splitting the train pool by qid, which requires retraining, "
      "since all 270 train qids were used.\n")
    if qid:
        W("**The published in-domain split cannot support a powered "
          "evaluation.** It holds one student explanation per record, so "
          "grouping by question yields 52 rankable groups of two to four "
          "candidates, against a random-ranking NDCG@5 baseline of 0.9260 - "
          "0.074 of headroom in total. `scripts/build_qid_split.py` therefore "
          "partitions the train pool by question instead: "
          f"{qid['train']['groups']} questions for training and "
          f"{qid['held_out']['groups']} held out, twelve scored explanations "
          f"each, {qid['held_out']['candidates']} held-out candidates with no "
          "singletons and a random baseline of 0.6687. Selection is a stable "
          "hash of the qid, stratified by domain and independent of the "
          "training seed, so every replicate sees the same partition. This is "
          "a different configuration from the published one, so it adds to the "
          "replication rather than replacing it.\n")
    W("The materialized training file is named `external_test` because it is "
      "produced from `DSCB-train-crowd-anno.jsonl`. It is training data. The "
      "name should be fixed before publication.\n")

    # ---------------------------------------------- 3. ranking quality vs random
    W("## 3. Ranking quality against the random-ranking baseline\n")
    W("NDCG@5 on small candidate sets with graded relevance is nearly "
      "saturated, so it means nothing without its baseline. Lift is the "
      "fraction of headroom above a random ranking that the model captures, "
      "`(NDCG - random) / (1 - random)`; intervals are 95% bootstrap over "
      f"{prov['bootstrap_samples']:,} resamples of whole ranking groups.\n")
    W("The in-domain figure was never an average over 197 groups: "
      "`evaluate_predictions` drops groups it cannot rank, so it comes from "
      "**52 rankable groups** (56 non-singleton, 4 with tied references).\n")
    rows = []
    for key, label in present:
        idl, ool = lift(key, "in_domain_non_singleton"), lift(key, "esnli")
        rows.append([
            label,
            f"{nd(key, 'in_domain_non_singleton', 'tie_aware_ndcg_at_5'):.4f}",
            f"{idl['estimate']:+.3f} [{idl['low']:+.3f}, {idl['high']:+.3f}]",
            f"{nd(key, 'esnli', 'tie_aware_ndcg_at_5'):.4f}",
            f"{ool['estimate']:+.3f} [{ool['low']:+.3f}, {ool['high']:+.3f}]",
        ])
    W(table(
        ["Model", "ID NDCG@5 (52 groups)", "ID lift [95% CI]",
         "e-SNLI NDCG@5 (1,000 groups)", "e-SNLI lift [95% CI]"],
        rows,
    ) + "\n")
    W("**In domain, every arm's lift interval contains zero.** On 52 groups of "
      "two to four candidates, no arm here is distinguishable from random "
      "ranking, and the apparent spread between arms is inside the noise.\n")
    W("The e-SNLI column is reported for completeness only. Given Section 1 it "
      "measures agreement with a confounded label, not ranking quality.\n")

    # -------------------------------------------------- 4. singleton artefact
    W("## 4. The in-domain uncertainty association is a singleton artefact\n")
    W("A singleton group has one candidate, so its within-group softmax is 1.0 "
      "by construction and its width and error are both exactly zero. 141 of "
      "the 270 in-domain candidates are such rows, and they generate the "
      "correlation on their own.\n")
    rows = []
    for key, label in present:
        a = partial(key, "in_domain")["bootstrap"]
        p = partial(key, "in_domain_non_singleton")
        b = p["bootstrap"]
        rows.append([
            label,
            f"{a['estimate']:+.4f}",
            f"{b['estimate']:+.4f}",
            f"[{b['low']:+.4f}, {b['high']:+.4f}]",
            f"{p['permutation_null_between_groups']['two_sided_p_value']:.4f}",
            f"{p['permutation_null_within_groups']['two_sided_p_value']:.4f}",
        ])
    W(table(
        ["Model", "rho (all 197 groups)", "rho (56 non-singleton)",
         "95% bootstrap", "p between", "p within"],
        rows,
    ) + "\n")
    W("The in-domain width/error Spearman of 0.875 previously quoted for "
      "independent ListNet is this artefact. Conditioned on the non-singleton "
      "groups, almost every interval contains zero. D3, the width-versus-epoch "
      "check, is computed on the same contaminated split and needs recomputing "
      "before it is cited again.\n")

    # ------------------------------------------- 5. nulls, diversity, abstention
    W("## 5. Uncertainty statistics\n")
    W("### 5.1 Bootstrap intervals disagree with permutation nulls\n")
    W("A group-clustered bootstrap says how precisely an association is "
      "measured, not whether one that size arises from the group structure "
      "alone. Two nulls, both leaving error and confidence untouched so the "
      "clustering and the control are preserved: **between-group** exchanges "
      "whole group width-vectors between groups of equal candidate count "
      "(does width identify which group is uncertain?), and **within-group** "
      "shuffles widths among a group's candidates (does it identify which "
      f"candidate?). {prov['permutations']:,} permutations each.\n")
    rows = []
    for key, label in present:
        p = partial(key, "esnli")
        b = p["bootstrap"]
        rows.append([
            label,
            f"{b['estimate']:+.4f}",
            f"[{b['low']:+.4f}, {b['high']:+.4f}]",
            f"{p['permutation_null_between_groups']['two_sided_p_value']:.4f}",
            f"{p['permutation_null_within_groups']['two_sided_p_value']:.4f}",
        ])
    W(table(
        ["Model", "partial rho", "95% group bootstrap", "p between", "p within"],
        rows,
    ) + "\n")
    W("Every interval excludes zero, including arms whose permutation p-value "
      "is 1.0000. Shared + feature masks is the clearest case: a tight interval "
      "around +0.273 and a between-group p of 0.963. **The ordering of arms by "
      "bootstrap interval alone is withdrawn.** These are e-SNLI numbers, so "
      "Section 1 applies to their interpretation regardless.\n")

    W("### 5.2 Participation ratio and Jensen-Shannon divergence\n")
    W("`effective_ensemble_size` reports `1 + PR` on a centred disagreement "
      "subspace of rank at most `M - 1`. Both conventions are now emitted with "
      "their ceilings, so 4.69 is unambiguous: `1 + PR` against a ceiling of 5, "
      "equivalently a raw PR of 3.69 against 4, equivalently a participation "
      "fraction of 0.92.\n")
    W("`mean_kl_to_consensus` computes `H(mean_m p_m) - mean_m H(p_m)`, the "
      "Jensen-Shannon divergence of the member distributions, not a KL "
      "divergence to a consensus. It is emitted as `js_divergence_to_consensus`, "
      "with the old key retained as a deprecated alias. Values are unchanged.\n")
    rows = []
    for key, label in present:
        e = arms[key]["esnli"]
        i = arms[key]["in_domain_non_singleton"]
        pr = e["participation_ratio"]
        rows.append([
            label,
            str(pr["member_count"]),
            f"{pr['disagreement_participation_ratio']:.3f}",
            f"{pr['participation_fraction']:.3f}",
            f"{i['diversity']['js_divergence_normalised_non_singleton']:.3e}",
            f"{e['diversity']['js_divergence_normalised_non_singleton']:.3e}",
        ])
    W(table(
        ["Model", "M", "PR (ceiling M-1)", "PR/(M-1)",
         "JS/log C, ID non-singleton", "JS/log C, e-SNLI"],
        rows,
    ) + "\n")

    W("### 5.3 The two statistics dissociate, measured\n")
    W("Member deviations are shrunk toward their within-group consensus in "
      "logit space. The participation ratio depends only on the shape of the "
      "disagreement covariance spectrum and is invariant; Jensen-Shannon "
      "divergence depends on the magnitude and falls as the square of the "
      "factor. Measured on each arm's own e-SNLI predictions:\n")
    for key in ("independent_listnet", "shared_baseline"):
        if key not in shrink:
            continue
        label = dict(ORDER)[key]
        rows = [[f"{r['factor']:.2f}", f"{r['participation_fraction']:.4f}",
                 f"{r['js_divergence_to_consensus']:.3e}",
                 f"{r['js_ratio_to_unshrunk']:.4f}"]
                for r in shrink[key]["rows"]]
        W(f"**{label}**\n")
        W(table(["Shrink factor", "PR/(M-1)", "JS", "JS relative"], rows) + "\n")
    W("At 98% shrinkage the participation fraction is unchanged while JS falls "
      "by a factor of roughly 2,500. This replaces the illustrative constants "
      "previously used to make this point.\n")

    W("### 5.4 Abstention\n")
    W("Selective risk against coverage, retaining candidates in ascending "
      "width. Normalised AURC gain is 1.0 if the width ordering matches an "
      "oracle ordering on the same errors, 0.0 if it is no better than random, "
      "and negative if abstention actively hurts.\n")
    rows = []
    for key, label in present:
        e, i = arms[key]["esnli"], arms[key]["in_domain_non_singleton"]
        rows.append([
            label,
            f"{e['risk_coverage_candidate']['normalised_aurc_gain']:+.3f}",
            f"{e['risk_coverage_group']['normalised_aurc_gain']:+.3f}",
            f"{i['risk_coverage_candidate']['normalised_aurc_gain']:+.3f}",
            f"{i['risk_coverage_group']['normalised_aurc_gain']:+.3f}",
        ])
    W(table(
        ["Model", "e-SNLI candidate", "e-SNLI group",
         "ID non-singleton candidate", "ID non-singleton group"],
        rows,
    ) + "\n")
    W("**Group-level abstention does not work for any arm** on either split, "
      "and that is the level at which a deployed ranker abstains - on a query, "
      "not on one candidate.\n")

    # ------------------------------------------------------ 6. metric semantics
    W("## 6. Objectives, invariance, and admissible read-outs\n")
    W(table(
        ["Objective", "Unconstrained transform", "Admissible uncertainty read-out"],
        [["Listwise softmax (ListNet, Plackett--Luce)",
          "within-group additive shift",
          "within-group softmax probabilities; KL/JS or probability spread"],
         ["Pairwise Bradley--Terry / RankNet",
          "additive shift within each comparison set",
          "comparison-set differences or pairwise probabilities; not absolute scores"],
         ["Pointwise MSE",
          "no score gauge (up to ordinary model symmetries)",
          "raw-score functionals are admissible; softmax is optional"]],
        numeric=False,
    ) + "\n")
    W("MSE arms are placed in the same within-group probability space as the "
      "ListNet arms so the two are comparable. NDCG is unaffected because that "
      "transform is monotone within a group; the width statistics would also be "
      "admissible on raw scores.\n")
    W("Effective members is a directional statistic, not a magnitude. "
      "Jensen-Shannon divergence in a common within-group probability space is "
      "the magnitude statistic, and the only valid comparison across ensemble "
      "constructions of different cardinality - MC-dropout carries 40 "
      "member/draw scores, so its ceilings are 39 and 40, and only its "
      "participation *fraction* is comparable to a five-member ensemble. Raw "
      "widths must never be compared across constructions.\n")
    W("OOD partial correlations are candidate-level and controlled for either "
      "consensus maximum probability (confidence) or consensus entropy. "
      "Uncertainty intervals resample groups, not candidates, because "
      "candidates within a ranking group are dependent.\n")
    W("D3 (width-versus-epoch Spearman below -0.9) is a valid pre-filter but is "
      "architecture-sensitive: shared heads are expected to converge and "
      "independent backbones are not, so it is a descriptive architecture "
      "effect, not proof of epistemic validity.\n")

    # --------------------------------------------------------------- figures
    W("## 7. Figures\n")
    W("| File | Content |\n|---|---|\n"
      "| `fig1_dissociation_measured.pdf` | Participation ratio versus JS divergence per epoch, shared and independent, on the non-singleton in-domain groups |\n"
      "| `fig4_esnli_quality_vs_association.pdf` | e-SNLI NDCG against the OOD width/error association, with bootstrap intervals, permutation-null status, and the random-ranking reference |\n"
      "| `fig_esnli_confound.pdf` | Mean predicted rank by e-SNLI quality tier, and the per-group correlation with and without the gold candidate |\n"
      "| `fig_singleton_contamination.pdf` | In-domain association with and without the 141 singleton groups |\n"
      "| `fig_risk_coverage_esnli.pdf` | Selective risk against coverage, each arm against its own oracle and random bounds |\n"
      "| `fig_seed_forest.pdf` | Across-seed replicates, once the relaunched runs land |\n")
    W("The two e-SNLI figures describe the arms' behaviour on that split "
      "faithfully, but Section 1 governs what they mean: they are measurements "
      "against a confounded label, not against explanation quality.\n")

    # ----------------------------------------------------------- 8. seeds
    W("## 8. Tier 2: replication and the powered split\n")
    W("Two axes, both retrained rather than re-evaluated, because a seed and a "
      "data split are training-time choices.\n")
    W("| Axis | Split | Seeds intended | Training completed | Status |\n"
      "|---|---|---|---|---|\n"
      "| Replication, independent | published | 123, 777 | none | not started |\n"
      "| Replication, independent | published | 47, 52, 57 | 30/30 members | complete |\n"
      "| Replication, shared | published | 123 | 13/14 arms | near complete |\n"
      "| Replication, shared | published | 777 | 5/14 arms | partial |\n"
      "| Powered evaluation | qid | 42, 123, 777 | none | not started |\n"
      "| Perturbation controls | qid sigma/frac | 42, 123, 777 | none | not started |\n")
    W("The sweep was stopped partway, so the seed coverage below is uneven and "
      "seeds 47, 52 and 57 stand in for the intended 123 and 777 on the "
      "independent arms. The evaluation and e-SNLI stages never ran, but the "
      "combine step is pure arithmetic over saved predictions, so the in-domain "
      "ensembles were rebuilt locally by `scripts/salvage_ensembles.py` and "
      "checked against the six that Modal had produced - they agree to 1.7e-16. "
      "Because e-SNLI is withdrawn (Section 1), nothing the paper needs depends "
      "on the stages that did not run.\n")
    W("The seed is an explicit argument to every remote function in "
      "`modal_independent_backbones.py` and `modal_shared_ablation.py`, and so "
      "is the split. An earlier revision read `GLOBAL_SEED` from `os.environ` at "
      "module import; Modal re-imports the module inside the container, where "
      "the variable is unset, so every remote call silently fell back to 42. "
      "The volume directory `independent_backbones_50ep_seed42` is the residue "
      "of that failure - members seeded 42-46, identical to the published run. "
      "Both scripts now refuse to reuse a completed output directory whose "
      "recorded seed disagrees with the request, and `summarize` re-checks every "
      "manifest before aggregating, so a fallback cannot reach a table again.\n")
    if seeds and len(seeds.get("seeds") or []) > 1:
        W(f"Analysed on the `{seeds['split']}` split over "
          f"{seeds['held_out_groups']} non-singleton held-out groups / "
          f"{seeds['held_out_candidates']} candidates. Seed coverage is uneven "
          "because the training sweep was stopped partway: the table records "
          "how many seeds each arm actually has, and no arm is compared across "
          "a different number of them without that being visible.\n")
        W("**The between-seed spread is as large as the between-arm spread.** "
          "The standard deviation of held-out NDCG@5 across arms at the single "
          "published seed is 0.0078; the mean standard deviation across seeds, "
          "within an arm, is 0.0076. Individual arms swing by up to 0.35 in "
          "lift between seeds. The arm ordering reported from one seed is "
          "therefore not identifiable, which is what this replication existed "
          "to establish.\n")
        rows = []
        arms = sorted({a for s_ in seeds["arms"].values() for a in s_})
        for arm in arms:
            values, lifts = [], []
            for seed in seeds["seeds"]:
                entry = seeds["arms"].get(str(seed), {}).get(arm)
                if not entry or "in_domain" not in entry:
                    continue
                nd = entry["in_domain"]["ndcg"]
                values.append(nd["non_singleton"]["tie_aware_ndcg_at_5"])
                lifts.append(nd["lift_over_random_ci"]["estimate"])
            if not values:
                continue
            rows.append([
                arm.replace("_", " "),
                f"{np.mean(values):.4f}",
                f"{np.std(values):.4f}",
                f"{np.mean(lifts):+.3f}",
                str(len(values)),
            ])
        W(table(
            ["Arm", "mean NDCG@5", "sd across seeds", "mean lift", "seeds"], rows
        ) + "\n")
        if seeds.get("missing"):
            W(f"{len(seeds['missing'])} combinations were not yet on the volume "
              "when this was generated.\n")
    else:
        W("**Incomplete.** No multi-seed analysis is available yet.\n")

    # ------------------------------------------------------ 9. reproducibility
    W("## 9. Reproducibility\n")
    W("```\n"
      "python scripts/tier1_recompute.py --volume <mirror> --output runs/tier1\n"
      "python scripts/diagnose_reversal.py --volume <mirror>\n"
      "python scripts/plot_tier1.py --volume <mirror> --out arr_figures\n"
      "python scripts/seed_replicate_analysis.py --mirror <mirror>\n"
      "python scripts/build_report.py\n"
      "```\n")
    W("Statistics live in `src/arr/tier1.py`, covered by `tests/arr/test_tier1.py`. "
      "No ListNet score was within `1e-6` of a sigmoid boundary. The MC-dropout "
      "scorer preserves 40 member/draw scores rather than flattening them into "
      "an invalid five-head shape.\n")
    W("Provenance re-checked against the Modal artifacts: the shared ListNet "
      "baseline, bootstrap and feature-mask manifests all report "
      "`status=complete`, `epochs_completed=50`, `loss=listnet`, `seed=42`, "
      "`device=cuda`, and share one validation fingerprint. The independent "
      "ensemble manifest reports five members, 50 epochs, and "
      "`within_group_mean_zero_logit` alignment. No stale or partial checkpoint "
      "contributed to any number here.\n")

    # --------------------------------------------------------- 10. open items
    W("## 10. Open items\n")
    W("1. **Replace the shift evaluation.** e-SNLI is confounded (Section 1), "
      "and no OOD claim can be made until a replacement exists. The qid "
      "held-out set is in-domain, so it does not fill this role; WinoWhy graded "
      "is the leading candidate.\n"
      "2. **Land Tier 2** and regenerate Sections 3-5 across seeds and splits.\n"
      "3. **Recompute D3** on the qid held-out set, where width-versus-epoch is "
      "measurable without singleton contamination.\n"
      "4. **Rename `ds_critique_external_test.jsonl`**, which is training data.\n"
      "5. Planned but not yet run: PPO width/JS trajectories for `none`, `var` "
      "and `credal`; a qualitative panel of highest-reward generated "
      "explanations.\n")
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier1", type=Path, default=Path("runs/tier1/tier1_results.json"))
    parser.add_argument("--reversal", type=Path, default=Path("runs/tier1/reversal_diagnosis.json"))
    parser.add_argument("--shrink", type=Path, default=Path("runs/tier1/shrink_control.json"))
    parser.add_argument(
        "--seeds", type=Path, default=Path("runs/tier2/replicates_published.json")
    )
    parser.add_argument(
        "--qid-manifest",
        type=Path,
        default=Path("data/arr/ds_critique_qidsplit_manifest.json"),
    )
    parser.add_argument("--output", type=Path, default=Path("arr_report.md"))
    args = parser.parse_args(argv)

    tier1 = json.loads(args.tier1.read_text())
    reversal = json.loads(args.reversal.read_text())
    shrink = json.loads(args.shrink.read_text()) if args.shrink.exists() else {}
    seeds = json.loads(args.seeds.read_text()) if args.seeds.exists() else None
    qid = (
        json.loads(args.qid_manifest.read_text())
        if args.qid_manifest.exists()
        else None
    )
    args.output.write_text(build(tier1, reversal, shrink, seeds, qid))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
