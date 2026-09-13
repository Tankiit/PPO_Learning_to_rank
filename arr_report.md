# ARR Epistemic Ranking Report

_Regenerated 2026-09-03 by `scripts/build_report.py` from `runs/tier1/tier1_results.json`, `runs/tier1/reversal_diagnosis.json` and the saved Modal predictions. Every number below is computed from a saved artifact; none is transcribed by hand._

## Status

**The e-SNLI out-of-distribution evaluation set is not valid for this task, and no OOD claim in this project survives that.** Section 1 gives the evidence. Everything computed on e-SNLI - the width/error associations, the divergence magnitudes, the abstention curves - measures behaviour on a set whose quality labels are confounded with text provenance, and must be re-established on a different shift set.

What still stands: the in-domain corrections (Sections 3-4), the metric-definition work (Section 6), and the directional-versus-magnitude dissociation (Section 5). What is withdrawn: the arm ordering by OOD association, every in-domain uncertainty correlation computed on the singleton-contaminated split, and any reading of an NDCG value as evidence of ranking competence.

## 1. The e-SNLI evaluation set is confounded

The independent ListNet ensemble ranks e-SNLI at a mean per-group Spearman of -0.703. A ranker merely degraded by distribution shift sits near zero; a large negative value on 1,000 groups is a competent ranker with its sign inverted. Two hypotheses were tested on saved predictions alone.

### 1.1 The reference is not inverted (H1 rejected)

Three independent checks, any one of which is fatal to an orientation-bug explanation.

**The stored reference is correctly oriented.** Mean reference score by quality tier, read straight out of the e-SNLI split:

| Tier | Mean reference score |
|---|---:|
| gold | 0.8532 |
| good | 0.6794 |
| fair | 0.4998 |
| poor | 0.2973 |
| nonsense | 0.1541 |

The order is gold > good > fair > poor > nonsense, which is the correct direction. An inversion upstream of the evaluator would have to show up here, and does not.

**A subset of the same evaluation is positively correlated.** Removing only the gold candidate from each group and recomputing through the identical code path:

| Model | rho (all 5) | rho (excl. gold) | gold ranked last |
|---|---:|---:|---:|
| Independent ListNet | -0.703 | -0.453 | 97.1% |
| Independent ListNet + bootstrap | -0.445 | +0.005 | 94.7% |
| Independent MSE | -0.227 | +0.441 | 95.0% |
| Independent MSE + bootstrap | -0.124 | +0.499 | 75.6% |
| Shared ListNet baseline | -0.356 | +0.065 | 84.1% |
| Shared ListNet + bootstrap | -0.806 | -0.668 | 96.9% |
| Shared ListNet + feature masks | -0.426 | -0.240 | 55.9% |
| Shared ListNet + bootstrap/features | -0.468 | -0.018 | 95.3% |
| Shared ListNet, lambda=0.01 | -0.442 | +0.042 | 97.6% |
| Shared ListNet, lambda=0.1 | -0.012 | -0.325 | 4.7% |
| Shared ListNet, lambda=1.0 | -0.211 | +0.441 | 92.9% |
| MC-dropout K=8 | -0.372 | +0.026 | 83.6% |

Four arms are **positively** correlated with the reference once the gold candidate is dropped, independent MSE at +0.441 and its bootstrap variant at +0.499. An inverted reference cannot produce a positive correlation on a subset of itself, evaluated through the same code path.

**Nonsense is not ranked first.** A clean reversal would put the lowest-referenced tier at the top of nearly every group. It is ranked best in 14.5% of groups for independent ListNet and 0.0% for independent MSE.

### 1.2 What is actually happening: a provenance artefact

Mean predicted within-group rank by tier (1 = ranked best):

| Model | gold | good | fair | poor | nonsense |
|---|---:|---:|---:|---:|---:|
| Independent ListNet | 4.95 | 3.36 | 3.05 | 1.20 | 2.43 |
| Independent ListNet + bootstrap | 4.90 | 2.12 | 3.53 | 1.47 | 2.98 |
| Independent MSE | 4.93 | 1.77 | 2.83 | 1.45 | 4.03 |
| Independent MSE + bootstrap | 4.74 | 1.53 | 2.98 | 1.50 | 4.24 |
| Shared ListNet baseline | 4.68 | 1.56 | 3.36 | 3.80 | 1.60 |
| Shared ListNet + bootstrap | 4.95 | 3.07 | 3.91 | 1.57 | 1.50 |
| Shared ListNet + feature masks | 4.25 | 1.96 | 4.26 | 3.19 | 1.33 |
| Shared ListNet + bootstrap/features | 4.95 | 1.91 | 4.00 | 1.16 | 2.98 |
| Shared ListNet, lambda=0.01 | 4.94 | 2.74 | 1.73 | 3.32 | 2.27 |
| Shared ListNet, lambda=0.1 | 2.14 | 3.62 | 3.22 | 4.56 | 1.47 |
| Shared ListNet, lambda=1.0 | 4.87 | 1.17 | 3.10 | 2.98 | 2.87 |
| MC-dropout K=8 | 4.65 | 1.63 | 3.41 | 3.75 | 1.55 |

**11 of 12 arms rank the gold candidate last in more than half their groups**, most of them in over 90%. The four degraded tiers cluster in the middle in no consistent order, nonsense included.

Dropping gold moves the correlation to zero or above for **8 of 12 arms**, so for those the rejection of the human-written candidate accounts for the entire reversal. For the remaining 4 it accounts for part of it: independent ListNet still reads -0.453 among the template tiers.

| Tier | Mean characters | Human-written |
|---|---:|---:|
| gold | 68.6 | 100% |
| good | 211.8 | 0% |
| fair | 66.0 | 0% |
| poor | 89.8 | 0% |
| nonsense | 82.0 | 0% |

Gold is the only human-written candidate in every group; the other four are template-generated by the same procedure. The models are separating human free text from template text, and the e-SNLI construction makes that separation perfectly anti-correlated with the quality label.

Length is not the mechanism: gold averages 69 characters and fair 66, yet gold is ranked last and fair mid-pack, while the longest tier (good, 212 characters) is ranked near the top.

This is neither hypothesis as originally posed. It is not a reference inversion - Section 1.1 rules that out three ways. It is not a critique-quality preference either: a model preferring critiques would reject nonsense, and these models rank nonsense mid-pack. What they separate is authorship, which e-SNLI has made perfectly collinear with quality by drawing gold from humans and every degraded tier from templates.

**e-SNLI as constructed is therefore not a shifted version of this task.** A model could score well on it by learning a human-versus-template detector and nothing about explanation quality; these models learned the detector with the sign that the label penalises. The shift evaluation has to be rebuilt on a set whose quality labels are not confounded with authorship - held-out DS-Critique qids re-split from the train pool, or WinoWhy graded.

## 2. Setup and data

- Backbone `EleutherAI/pythia-70m`, revision `main`; Modal NVIDIA T4, float32.
- 50 epochs; five ensemble members; AdamW, lr `2e-5`, weight decay `0.01`,
  linear schedule, 51 warmup steps, max grad norm `1.0`.
- Group batch size 1, gradient accumulation 8, 1,700 optimizer updates,
  sequence length 256, QLoRA and quantization disabled.
- Diagnostics, KL/JS and entropy in float64.

| Split | Groups | Candidates | Composition |
|---|---:|---:|---|
| Train (`ds_critique_external_test.jsonl`) | 270 | 3,240 | 270 qids x 12 student explanations |
| In-domain validation (`ds_critique_external_dev.jsonl`) | 197 | 270 | 141 singletons, 42 pairs, 11 triples, 3 quads |
| e-SNLI (OOD) | 1,000 | 5,000 | 1,000 groups of 5 |
| Tier 2 qid train | 216 | 2,592 | question-disjoint, 12 per group |
| Tier 2 qid held out | 54 | 648 | question-disjoint, 12 per group, no singletons |

Both DS-Critique splits score **student explanations** via `DS_Critique_Bank.explanation_annotations.human_crowd_mean`, and their qid sets are disjoint. The in-domain split's singletons are not a filtering bug: the dev pool holds one student explanation per record, 270 records over 197 questions. A held-out split with C=12 would have to come from re-splitting the train pool by qid, which requires retraining, since all 270 train qids were used.

**The published in-domain split cannot support a powered evaluation.** It holds one student explanation per record, so grouping by question yields 52 rankable groups of two to four candidates, against a random-ranking NDCG@5 baseline of 0.9260 - 0.074 of headroom in total. `scripts/build_qid_split.py` therefore partitions the train pool by question instead: 216 questions for training and 54 held out, twelve scored explanations each, 648 held-out candidates with no singletons and a random baseline of 0.6687. Selection is a stable hash of the qid, stratified by domain and independent of the training seed, so every replicate sees the same partition. This is a different configuration from the published one, so it adds to the replication rather than replacing it.

The materialized training file is named `external_test` because it is produced from `DSCB-train-crowd-anno.jsonl`. It is training data. The name should be fixed before publication.

## 3. Ranking quality against the random-ranking baseline

NDCG@5 on small candidate sets with graded relevance is nearly saturated, so it means nothing without its baseline. Lift is the fraction of headroom above a random ranking that the model captures, `(NDCG - random) / (1 - random)`; intervals are 95% bootstrap over 3,000 resamples of whole ranking groups.

The in-domain figure was never an average over 197 groups: `evaluate_predictions` drops groups it cannot rank, so it comes from **52 rankable groups** (56 non-singleton, 4 with tied references).

| Model | ID NDCG@5 (52 groups) | ID lift [95% CI] | e-SNLI NDCG@5 (1,000 groups) | e-SNLI lift [95% CI] |
|---|---:|---:|---:|---:|
| Independent ListNet | 0.9327 | +0.090 [-0.190, +0.372] | 0.7207 | -0.642 [-0.656, -0.628] |
| Independent ListNet + bootstrap | 0.9272 | +0.016 [-0.272, +0.294] | 0.7703 | -0.350 [-0.372, -0.328] |
| Independent MSE | 0.9401 | +0.191 [-0.073, +0.444] | 0.7971 | -0.193 [-0.212, -0.173] |
| Independent MSE + bootstrap | 0.9304 | +0.060 [-0.219, +0.337] | 0.8126 | -0.101 [-0.122, -0.081] |
| Shared ListNet baseline | 0.9258 | -0.002 [-0.264, +0.274] | 0.7897 | -0.236 [-0.264, -0.209] |
| Shared ListNet + bootstrap | 0.9276 | +0.022 [-0.229, +0.287] | 0.6966 | -0.784 [-0.796, -0.772] |
| Shared ListNet + feature masks | 0.9367 | +0.145 [-0.120, +0.393] | 0.7589 | -0.418 [-0.445, -0.391] |
| Shared ListNet + bootstrap/features | 0.9242 | -0.024 [-0.295, +0.247] | 0.7638 | -0.388 [-0.405, -0.372] |
| Shared ListNet, lambda=0.01 | 0.9317 | +0.077 [-0.200, +0.362] | 0.7794 | -0.297 [-0.320, -0.274] |
| Shared ListNet, lambda=0.1 | 0.9169 | -0.123 [-0.383, +0.162] | 0.8182 | -0.069 [-0.103, -0.034] |
| Shared ListNet, lambda=1.0 | 0.9372 | +0.151 [-0.119, +0.425] | 0.8394 | +0.056 [+0.036, +0.075] |
| MC-dropout K=8 | 0.9403 | +0.193 [-0.076, +0.450] | 0.7836 | -0.272 [-0.300, -0.245] |

**In domain, every arm's lift interval contains zero.** On 52 groups of two to four candidates, no arm here is distinguishable from random ranking, and the apparent spread between arms is inside the noise.

The e-SNLI column is reported for completeness only. Given Section 1 it measures agreement with a confounded label, not ranking quality.

## 4. The in-domain uncertainty association is a singleton artefact

A singleton group has one candidate, so its within-group softmax is 1.0 by construction and its width and error are both exactly zero. 141 of the 270 in-domain candidates are such rows, and they generate the correlation on their own.

| Model | rho (all 197 groups) | rho (56 non-singleton) | 95% bootstrap | p between | p within |
|---|---:|---:|---:|---:|---:|
| Independent ListNet | +0.6459 | +0.1142 | [-0.1419, +0.3689] | 0.3743 | 0.0720 |
| Independent ListNet + bootstrap | +0.7308 | +0.3841 | [+0.1687, +0.5621] | 0.0015 | 0.6167 |
| Independent MSE | +0.5203 | -0.1446 | [-0.3528, +0.0977] | 0.1799 | 0.8211 |
| Independent MSE + bootstrap | +0.5469 | -0.0824 | [-0.3156, +0.1691] | 0.4563 | 0.6972 |
| Shared ListNet baseline | +0.6307 | +0.0943 | [-0.1492, +0.3324] | 0.4468 | 0.9965 |
| Shared ListNet + bootstrap | +0.6530 | +0.0527 | [-0.2074, +0.3013] | 0.6557 | 0.2334 |
| Shared ListNet + feature masks | +0.6113 | -0.0457 | [-0.2790, +0.1938] | 0.7531 | 0.9370 |
| Shared ListNet + bootstrap/features | +0.6875 | +0.1552 | [-0.1109, +0.4082] | 0.1974 | 0.4513 |
| Shared ListNet, lambda=0.01 | +0.5855 | -0.0087 | [-0.2480, +0.2269] | 0.9340 | 0.8316 |
| Shared ListNet, lambda=0.1 | +0.6199 | -0.0028 | [-0.2422, +0.2466] | 0.9860 | 0.9125 |
| Shared ListNet, lambda=1.0 | +0.6952 | +0.1668 | [-0.0672, +0.3949] | 0.1999 | 0.5482 |
| MC-dropout K=8 | +0.6350 | -0.0333 | [-0.2539, +0.2057] | 0.7741 | 0.5772 |

The in-domain width/error Spearman of 0.875 previously quoted for independent ListNet is this artefact. Conditioned on the non-singleton groups, almost every interval contains zero. D3, the width-versus-epoch check, is computed on the same contaminated split and needs recomputing before it is cited again.

## 5. Uncertainty statistics

### 5.1 Bootstrap intervals disagree with permutation nulls

A group-clustered bootstrap says how precisely an association is measured, not whether one that size arises from the group structure alone. Two nulls, both leaving error and confidence untouched so the clustering and the control are preserved: **between-group** exchanges whole group width-vectors between groups of equal candidate count (does width identify which group is uncertain?), and **within-group** shuffles widths among a group's candidates (does it identify which candidate?). 2,000 permutations each.

| Model | partial rho | 95% group bootstrap | p between | p within |
|---|---:|---:|---:|---:|
| Independent ListNet | +0.4433 | [+0.4230, +0.4631] | 0.0005 | 0.0005 |
| Independent ListNet + bootstrap | -0.0326 | [-0.0562, -0.0075] | 0.9675 | 0.0695 |
| Independent MSE | -0.2884 | [-0.3082, -0.2689] | 1.0000 | 0.0005 |
| Independent MSE + bootstrap | +0.1801 | [+0.1668, +0.1933] | 1.0000 | 0.0005 |
| Shared ListNet baseline | +0.1609 | [+0.1422, +0.1800] | 0.1074 | 0.0005 |
| Shared ListNet + bootstrap | +0.3330 | [+0.3138, +0.3536] | 0.0025 | 0.0005 |
| Shared ListNet + feature masks | +0.2730 | [+0.2483, +0.2969] | 0.9630 | 0.0005 |
| Shared ListNet + bootstrap/features | -0.0348 | [-0.0618, -0.0074] | 1.0000 | 0.0290 |
| Shared ListNet, lambda=0.01 | -0.0283 | [-0.0563, -0.0016] | 1.0000 | 0.0550 |
| Shared ListNet, lambda=0.1 | +0.3117 | [+0.2922, +0.3326] | 0.0015 | 0.0005 |
| Shared ListNet, lambda=1.0 | -0.1769 | [-0.2030, -0.1513] | 0.7531 | 0.0005 |
| MC-dropout K=8 | -0.1566 | [-0.1823, -0.1310] | 0.0640 | 0.0005 |

Every interval excludes zero, including arms whose permutation p-value is 1.0000. Shared + feature masks is the clearest case: a tight interval around +0.273 and a between-group p of 0.963. **The ordering of arms by bootstrap interval alone is withdrawn.** These are e-SNLI numbers, so Section 1 applies to their interpretation regardless.

### 5.2 Participation ratio and Jensen-Shannon divergence

`effective_ensemble_size` reports `1 + PR` on a centred disagreement subspace of rank at most `M - 1`. Both conventions are now emitted with their ceilings, so 4.69 is unambiguous: `1 + PR` against a ceiling of 5, equivalently a raw PR of 3.69 against 4, equivalently a participation fraction of 0.92.

`mean_kl_to_consensus` computes `H(mean_m p_m) - mean_m H(p_m)`, the Jensen-Shannon divergence of the member distributions, not a KL divergence to a consensus. It is emitted as `js_divergence_to_consensus`, with the old key retained as a deprecated alias. Values are unchanged.

| Model | M | PR (ceiling M-1) | PR/(M-1) | JS/log C, ID non-singleton | JS/log C, e-SNLI |
|---|---:|---:|---:|---:|---:|
| Independent ListNet | 5 | 2.887 | 0.722 | 4.309e-03 | 3.681e-03 |
| Independent ListNet + bootstrap | 5 | 3.050 | 0.763 | 4.474e-03 | 3.013e-03 |
| Independent MSE | 5 | 2.823 | 0.706 | 1.002e-01 | 5.308e-02 |
| Independent MSE + bootstrap | 5 | 1.494 | 0.374 | 1.099e-01 | 8.343e-02 |
| Shared ListNet baseline | 5 | 2.387 | 0.597 | 4.167e-05 | 1.120e-04 |
| Shared ListNet + bootstrap | 5 | 2.592 | 0.648 | 5.259e-05 | 1.840e-04 |
| Shared ListNet + feature masks | 5 | 3.210 | 0.802 | 1.254e-04 | 7.607e-04 |
| Shared ListNet + bootstrap/features | 5 | 3.410 | 0.852 | 1.645e-04 | 4.791e-04 |
| Shared ListNet, lambda=0.01 | 5 | 3.561 | 0.890 | 4.115e-05 | 2.290e-04 |
| Shared ListNet, lambda=0.1 | 5 | 3.020 | 0.755 | 4.610e-05 | 4.183e-02 |
| Shared ListNet, lambda=1.0 | 5 | 3.384 | 0.846 | 6.276e-05 | 6.546e-05 |
| MC-dropout K=8 | 40 | 30.098 | 0.772 | 4.236e-03 | 3.181e-03 |

### 5.3 The two statistics dissociate, measured

Member deviations are shrunk toward their within-group consensus in logit space. The participation ratio depends only on the shape of the disagreement covariance spectrum and is invariant; Jensen-Shannon divergence depends on the magnitude and falls as the square of the factor. Measured on each arm's own e-SNLI predictions:

**Independent ListNet**

| Shrink factor | PR/(M-1) | JS | JS relative |
|---|---:|---:|---:|
| 1.00 | 0.7217 | 5.925e-03 | 1.0000 |
| 0.50 | 0.7323 | 1.479e-03 | 0.2496 |
| 0.10 | 0.7430 | 5.858e-05 | 0.0099 |
| 0.02 | 0.7452 | 2.337e-06 | 0.0004 |

**Shared ListNet baseline**

| Shrink factor | PR/(M-1) | JS | JS relative |
|---|---:|---:|---:|
| 1.00 | 0.5967 | 1.803e-04 | 1.0000 |
| 0.50 | 0.5969 | 4.513e-05 | 0.2503 |
| 0.10 | 0.5970 | 1.807e-06 | 0.0100 |
| 0.02 | 0.5971 | 7.228e-08 | 0.0004 |

At 98% shrinkage the participation fraction is unchanged while JS falls by a factor of roughly 2,500. This replaces the illustrative constants previously used to make this point.

### 5.4 Abstention

Selective risk against coverage, retaining candidates in ascending width. Normalised AURC gain is 1.0 if the width ordering matches an oracle ordering on the same errors, 0.0 if it is no better than random, and negative if abstention actively hurts.

| Model | e-SNLI candidate | e-SNLI group | ID non-singleton candidate | ID non-singleton group |
|---|---:|---:|---:|---:|
| Independent ListNet | +0.423 | +0.050 | +0.222 | +0.031 |
| Independent ListNet + bootstrap | -0.016 | +0.068 | +0.475 | +0.313 |
| Independent MSE | -0.671 | +0.057 | -0.107 | -0.206 |
| Independent MSE + bootstrap | +0.087 | +0.060 | +0.088 | +0.085 |
| Shared ListNet baseline | +0.265 | -0.144 | +0.218 | -0.015 |
| Shared ListNet + bootstrap | +0.322 | -0.104 | +0.172 | -0.131 |
| Shared ListNet + feature masks | +0.200 | -0.173 | +0.211 | -0.083 |
| Shared ListNet + bootstrap/features | -0.039 | +0.068 | +0.260 | -0.092 |
| Shared ListNet, lambda=0.01 | -0.115 | -0.021 | +0.235 | +0.026 |
| Shared ListNet, lambda=0.1 | +0.365 | -0.064 | +0.233 | +0.083 |
| Shared ListNet, lambda=1.0 | -0.289 | +0.001 | +0.371 | +0.051 |
| MC-dropout K=8 | -0.343 | -0.013 | +0.080 | -0.280 |

**Group-level abstention does not work for any arm** on either split, and that is the level at which a deployed ranker abstains - on a query, not on one candidate.

## 6. Objectives, invariance, and admissible read-outs

| Objective | Unconstrained transform | Admissible uncertainty read-out |
|---|---|---|
| Listwise softmax (ListNet, Plackett--Luce) | within-group additive shift | within-group softmax probabilities; KL/JS or probability spread |
| Pairwise Bradley--Terry / RankNet | additive shift within each comparison set | comparison-set differences or pairwise probabilities; not absolute scores |
| Pointwise MSE | no score gauge (up to ordinary model symmetries) | raw-score functionals are admissible; softmax is optional |

MSE arms are placed in the same within-group probability space as the ListNet arms so the two are comparable. NDCG is unaffected because that transform is monotone within a group; the width statistics would also be admissible on raw scores.

Effective members is a directional statistic, not a magnitude. Jensen-Shannon divergence in a common within-group probability space is the magnitude statistic, and the only valid comparison across ensemble constructions of different cardinality - MC-dropout carries 40 member/draw scores, so its ceilings are 39 and 40, and only its participation *fraction* is comparable to a five-member ensemble. Raw widths must never be compared across constructions.

OOD partial correlations are candidate-level and controlled for either consensus maximum probability (confidence) or consensus entropy. Uncertainty intervals resample groups, not candidates, because candidates within a ranking group are dependent.

D3 (width-versus-epoch Spearman below -0.9) is a valid pre-filter but is architecture-sensitive: shared heads are expected to converge and independent backbones are not, so it is a descriptive architecture effect, not proof of epistemic validity.

## 7. Figures

| File | Content |
|---|---|
| `fig1_dissociation_measured.pdf` | Participation ratio versus JS divergence per epoch, shared and independent, on the non-singleton in-domain groups |
| `fig4_esnli_quality_vs_association.pdf` | e-SNLI NDCG against the OOD width/error association, with bootstrap intervals, permutation-null status, and the random-ranking reference |
| `fig_esnli_confound.pdf` | Mean predicted rank by e-SNLI quality tier, and the per-group correlation with and without the gold candidate |
| `fig_singleton_contamination.pdf` | In-domain association with and without the 141 singleton groups |
| `fig_risk_coverage_esnli.pdf` | Selective risk against coverage, each arm against its own oracle and random bounds |
| `fig_seed_forest.pdf` | Across-seed replicates, once the relaunched runs land |

The two e-SNLI figures describe the arms' behaviour on that split faithfully, but Section 1 governs what they mean: they are measurements against a confounded label, not against explanation quality.

## 8. Tier 2: replication and the powered split

Two axes, both retrained rather than re-evaluated, because a seed and a data split are training-time choices.

| Axis | Split | Seeds intended | Training completed | Status |
|---|---|---|---|---|
| Replication, independent | published | 123, 777 | none | not started |
| Replication, independent | published | 47, 52, 57 | 30/30 members | complete |
| Replication, shared | published | 123 | 13/14 arms | near complete |
| Replication, shared | published | 777 | 5/14 arms | partial |
| Powered evaluation | qid | 42, 123, 777 | none | not started |
| Perturbation controls | qid sigma/frac | 42, 123, 777 | none | not started |

The sweep was stopped partway, so the seed coverage below is uneven and seeds 47, 52 and 57 stand in for the intended 123 and 777 on the independent arms. The evaluation and e-SNLI stages never ran, but the combine step is pure arithmetic over saved predictions, so the in-domain ensembles were rebuilt locally by `scripts/salvage_ensembles.py` and checked against the six that Modal had produced - they agree to 1.7e-16. Because e-SNLI is withdrawn (Section 1), nothing the paper needs depends on the stages that did not run.

The seed is an explicit argument to every remote function in `modal_independent_backbones.py` and `modal_shared_ablation.py`, and so is the split. An earlier revision read `GLOBAL_SEED` from `os.environ` at module import; Modal re-imports the module inside the container, where the variable is unset, so every remote call silently fell back to 42. The volume directory `independent_backbones_50ep_seed42` is the residue of that failure - members seeded 42-46, identical to the published run. Both scripts now refuse to reuse a completed output directory whose recorded seed disagrees with the request, and `summarize` re-checks every manifest before aggregating, so a fallback cannot reach a table again.

Analysed on the `published` split over 56 non-singleton held-out groups / 129 candidates. Seed coverage is uneven because the training sweep was stopped partway: the table records how many seeds each arm actually has, and no arm is compared across a different number of them without that being visible.

**The between-seed spread is as large as the between-arm spread.** The standard deviation of held-out NDCG@5 across arms at the single published seed is 0.0078; the mean standard deviation across seeds, within an arm, is 0.0076. Individual arms swing by up to 0.35 in lift between seeds. The arm ordering reported from one seed is therefore not identifiable, which is what this replication existed to establish.

| Arm | mean NDCG@5 | sd across seeds | mean lift | seeds |
|---|---:|---:|---:|---:|
| independent listnet | 0.9338 | 0.0020 | +0.105 | 4 |
| independent listnet bootstrap | 0.9260 | 0.0036 | -0.000 | 4 |
| independent mse | 0.9401 | 0.0000 | +0.191 | 1 |
| independent mse bootstrap | 0.9304 | 0.0000 | +0.060 | 1 |
| shared listnet baseline | 0.9250 | 0.0009 | -0.014 | 2 |
| shared listnet bootstrap | 0.9201 | 0.0075 | -0.079 | 2 |
| shared listnet bootstrap features | 0.9311 | 0.0068 | +0.069 | 2 |
| shared listnet features | 0.9383 | 0.0016 | +0.166 | 2 |
| shared listnet lambda 0p01 | 0.9298 | 0.0019 | +0.051 | 2 |
| shared listnet lambda 0p1 | 0.9280 | 0.0112 | +0.028 | 2 |
| shared listnet lambda 1 | 0.9292 | 0.0079 | +0.044 | 2 |
| shared mse baseline | 0.9286 | 0.0035 | +0.035 | 3 |
| shared mse bootstrap | 0.9360 | 0.0085 | +0.135 | 3 |
| shared mse bootstrap features | 0.9306 | 0.0108 | +0.063 | 3 |
| shared mse features | 0.9308 | 0.0053 | +0.065 | 3 |
| shared mse lambda 0p01 | 0.9273 | 0.0059 | +0.017 | 3 |
| shared mse lambda 0p1 | 0.9310 | 0.0092 | +0.068 | 2 |
| shared mse lambda 1 | 0.9189 | 0.0000 | -0.095 | 1 |

156 combinations were not yet on the volume when this was generated.

## 9. Reproducibility

```
python scripts/tier1_recompute.py --volume <mirror> --output runs/tier1
python scripts/diagnose_reversal.py --volume <mirror>
python scripts/plot_tier1.py --volume <mirror> --out arr_figures
python scripts/seed_replicate_analysis.py --mirror <mirror>
python scripts/build_report.py
```

Statistics live in `src/arr/tier1.py`, covered by `tests/arr/test_tier1.py`. No ListNet score was within `1e-6` of a sigmoid boundary. The MC-dropout scorer preserves 40 member/draw scores rather than flattening them into an invalid five-head shape.

Provenance re-checked against the Modal artifacts: the shared ListNet baseline, bootstrap and feature-mask manifests all report `status=complete`, `epochs_completed=50`, `loss=listnet`, `seed=42`, `device=cuda`, and share one validation fingerprint. The independent ensemble manifest reports five members, 50 epochs, and `within_group_mean_zero_logit` alignment. No stale or partial checkpoint contributed to any number here.

## 10. Open items

1. **Replace the shift evaluation.** e-SNLI is confounded (Section 1), and no OOD claim can be made until a replacement exists. The qid held-out set is in-domain, so it does not fill this role; WinoWhy graded is the leading candidate.
2. **Land Tier 2** and regenerate Sections 3-5 across seeds and splits.
3. **Recompute D3** on the qid held-out set, where width-versus-epoch is measurable without singleton contamination.
4. **Rename `ds_critique_external_test.jsonl`**, which is training data.
5. Planned but not yet run: PPO width/JS trajectories for `none`, `var` and `credal`; a qualitative panel of highest-reward generated explanations.

## 11. Things to do tracker

### 11.1 Paper checklist

- [ ] Finalize fixed protocol: backbone, splits, optimization budget, evaluation rules.
- [ ] Train matched models under ListNet, ListMLE, and MSE with equal search and compute.
- [ ] Run multi-seed training for each condition.
- [ ] Store and version checkpoints, per-member predictions, and split manifests.
- [ ] Run R1 identifiability tests: loss-preserving transforms, raw-score vs probability-space uncertainty.
- [ ] Run R2 non-degeneracy tests: JS, width, covariance magnitude, participation ratio.
- [ ] Run R3 functional tests: error correlation, risk-coverage, deferral, baseline comparisons.
- [ ] Run R3 controls: permutation/null tests for group-structure artifacts.
- [ ] Run R4 exposure tests: withhold, dose-increase, re-expose with matched controls.
- [ ] Build final summary table: pass/fail for R1, R2, R3, R4 by method.
- [ ] Report confidence intervals and seed variance for all key claims.
- [ ] Separate in-domain conclusions from shift conclusions.
- [ ] Restrict claims to tested objectives, architectures, datasets, and uncertainty definitions.

### 11.2 Project board view

| Item | Status | Deliverable |
|---|---|---|
| Protocol lock | Todo | One-page protocol spec |
| Matched training runs | Todo | ListNet/ListMLE/MSE checkpoints |
| Multi-seed runs | Todo | Seed manifest and run log |
| Artifact release | Todo | Predictions and split manifests |
| R1 analysis | Todo | Invariance results table |
| R2 analysis | Todo | Diversity and magnitude table |
| R3 analysis | Todo | Risk-coverage and deferral plots |
| R3 null controls | Todo | Permutation control report |
| R4 interventions | Todo | Exposure sensitivity report |
| Unified R1-R4 outcome table | Todo | Main paper table |
| Uncertainty intervals and seed variance | Todo | Appendix stats table |
| Claim-boundary writeup | Todo | Scope and limitations section |

