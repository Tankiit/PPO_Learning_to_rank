# Controlled uncertainty study: implementation and execution protocol

**Question:** When does disagreement between explanation rankers constitute useful evidence of missing training knowledge?

This is a new DS adaptation protocol. Existing external-transfer files and their results are not adaptation folds. The implementation lives in `src/uncertainty_study`; legacy bounded-score evaluation remains separate.

## Motivation and contribution

Explanation rankers support decisions about which model-generated explanations to present, retain, or inspect. However, strong average ranking performance does not establish when an individual selection is reliable. Uncertainty estimation offers a potential basis for deferring unreliable decisions, but its interpretation is complicated by the ranking objective: disagreement can depend on score representations that leave rankings unchanged, and stochastic variability need not identify errors or reflect missing training knowledge. We investigate four requirements for uncertainty in explanation ranking: invariance to loss-preserving score transformations, measurable disagreement, relevance to ranking errors, and sensitivity to controlled training exposure. Using matched ListNet, ListMLE, and MSE rankers, we test when ensemble disagreement supports selective prediction and when its interpretation as epistemic uncertainty is warranted.

The practical decision is whether to trust an explanation selection or refer it for human review. [Digital Socrates](https://aclanthology.org/2024.acl-long.302/) situates explanation evaluation and critique within NLP. Our contribution is a controlled evaluation framework and its empirical evidence, not three new pretrained language models or an assumed new uncertainty estimator. Each backbone is initialized from the same public pretrained weights and fine-tuned on the same ranking data under each objective. R1 alone does not establish useful uncertainty; R3 and R4 test decision relevance and selective reducibility.

**Real-data requirement:** primary R1–R2 evidence must use stored member predictions from trained rankers on held-out DS-Critique/e-SNLI-derived queries. Synthetic scores remain numerical unit tests only. Offset interventions and contraction controls are applied to those real predictions; actual training trajectories are shown separately.

## Current status (2026-09-13)

Implemented: masked randomized-tie ListMLE; DeBERTa encoder shared heads; identity-output training for all three losses; fixed-update training; validation-regret checkpoint selection; checkpoint member predictions; independent-member joining; single-model MC dropout; R1/R2 numerical diagnostics; query R3 risk curves and controls; question bootstrap comparisons; DS audit, three outer folds, nested doses, paired prompt-matched generator replacement, and R4 contrasts.

Executed: audit of all four local DS source files; fold/intervention construction; synthetic mathematical diagnostics; all-three-loss end-to-end tiny random DeBERTa tests. These smoke tests do **not** demonstrate pretrained-model learning, estimator utility, calibration, or epistemic validity. The pretrained pilot and research matrices have **not** run. Mistral CUDA/QLoRA execution has not been tested in this session.

The local audit found 10,038 source rows, 6,528 generations without human explanation scores, 141 singleton human-scored questions excluded, and **326 ranking questions / 3,369 candidates** retained. No duplicate generation rows or exact question aliases were found under the implemented normalized matching rules. This is not a semantic near-duplicate audit. All four generators had the same minimum matched training support (151 questions across folds), so lexical tie-breaking selected `gpt-3.5-turbo-0613` and `gpt-4-0613`. Selection uses data support only.

Artifacts:

- `data/uncertainty_study/ds/audit.json`: input hashes, inclusion counts, target selection, and exposure balance.
- `data/uncertainty_study/ds/fold-{0,1,2}`: canonical groups, validation/test, nested doses, and exposure conditions.
- `configs/uncertainty_study/ds_pilot/index.json`: three DeBERTa/one-seed pilot configurations.
- `runs/uncertainty_study/diagnostics-final`: numeric R1/R2 diagnostics and figure.

## R1 proposition

Let every member's scores transform as s'[h,q,i] = s[h,q,i] + c[h,q], with finite constants and fixed positive temperature T. Candidate ordering and softmax(s/T) are unchanged. ListNet's log-softmax cross entropy is unchanged. For a fixed ListMLE reference permutation, each term log(sum[j>=i] exp(s[j])) - s[i] is unchanged because the common constant cancels. Thus the randomized-tie expected ListMLE loss is also invariant. Centering each member within query removes the same constants. MSE generally changes: its difference is the mean of 2c(s-y)+c². Raw member covariance/width need not be invariant.

In particular, JS and probability covariance/envelope width are offset invariant. This proves neither calibration nor epistemic meaning. Multiplicative scale and temperature remain separate issues. ListMLE probabilities used here are first-choice probabilities, not distributions over complete permutations.

For p_h(alpha) = mean(p) + alpha*(p_h-mean(p)), covariance is alpha²*C. Its trace contracts by alpha², while PR = trace(C)² / trace(C²) stays constant for nonzero covariance. The implementation reports null PR below trace 1e-24 instead of assigning meaning to numerical degeneracy. The synthetic figure fixes a meaningful PR axis; it does not magnify floating-point noise.

## Data protocol

DS uses only `explanation_annotations.explanation_score`, averaged and divided by five. Automatic critique scores are excluded. Generator and prompt remain separate candidate fields, alongside source locations, annotation payloads, length, and domain. Case/hyphen aliases for Llama generators are normalized. Identical generation observations merge unique annotation payloads and retain every source location; conflicting IDs/question text fail. Exact normalized question aliases share a canonical question before folds.

Three domain-stratified outer question folds use seed 2026. Each non-test pool reserves 15% validation questions. All conditions reuse these files; folds are not independent experimental repetitions. Test candidates are unchanged across interventions. Split checks cover both group IDs and normalized question text.

Exposure uses only eligible training questions/prompts with a target candidate and at least three other-generator candidates. Per prompt, three controls form the absent condition; one target replaces one control in full exposure; a seeded nested half of slots is replaced in partial exposure. Two other candidates remain fixed. All levels retain the same questions, candidate count, prompt count, and update budget. Domain is matched by question. Length and quality are audited, not exactly matched. Their remaining imbalance must be reported and examined before causal interpretation. Candidate probability variances are coupled within questions; report target and other components as well as their difference-in-differences.

The e-SNLI adapter accepts existing canonical constructed groups. Call this an **e-SNLI-derived ranking benchmark**, never human quality rankings. Preparation retains first occurrences of normalized questions in train/validation/test order and logs excluded duplicates. The old loader's expected Arrow cache is absent locally; e-SNLI preparation has not been executed. No third dataset is introduced.

## Training and estimators

DeBERTa-v3-base is fully fine-tuned; Mistral-7B-Instruct-v0.3 uses NF4 QLoRA. Five scalar MLP heads use shared features and dropout rates from .05 to .30. Head capacity and input template are identical across losses within architecture. Identity outputs apply to MSE, ListNet, and ListMLE. Sigmoid is a predefined ablation. All independent and MC reference models also request the scalar MLP head.

The primary seeds are 42, 43, 44. Five heads are one training run. Five independently initialized/fine-tuned members are one ensemble. Reference member seeds are derived from the ensemble repetition seed; join rejects repeated member seeds. MC dropout uses 20 passes of a single model, enabling dropout modules while leaving the model otherwise in evaluation mode. Single-member entropy/margin use the first prespecified member; entropy/margin of mean probabilities are also reported.

ListMLE shuffles candidates before stable target sorting on every call, giving uniform ordering within exact tie blocks. Fixed candidate order does not resolve ties. `matrix --stage ties` generates a sensitivity run averaging eight random tie permutations per loss call; results remain to be collected.

Matched-update doses use 10%, 25%, 50%, 100% nested training questions, paired initialization, fixed validation/test, and the same number of optimizer updates. History records groups/candidates processed and losses. Checkpoints are selected by mean validation top-1 regret with earliest-checkpoint tie-breaking. Extended 10% and 100% endpoints get three times the provisional update budget; **this is a convergence diagnostic, not a guarantee of adequate training**. Inspect learning curves and revise the endpoint budget before freezing the confirmatory protocol.

The YAML's learning rate and 300 updates are provisional pilot values. `matrix --stage hpo` allocates the same three learning rates to every loss/architecture/fold, using the first seed. Choose per architecture/loss/fold using validation only, record choices, and freeze them across dose/exposure conditions. `select-hpo` accepts completed trial directories, rejects incomplete or duplicate learning-rate grids, and writes selected settings into a YAML. Pass that YAML to subsequent `matrix --config` calls to propagate the selected value by architecture/fold/loss. Use separate YAMLs for DS and e-SNLI. The caller must supply the intended first-seed trial grid; a selection is not evidence of convergence. Pin model revisions to resolved commit IDs before research runs (the supplied development config uses `main`).

## Evaluation and evidence

Each prediction file contains member-by-candidate scores, candidate identities, reference scores, generator labels, fingerprint, temperature, and training seed. Validation and test predictions are saved at every requested checkpoint. Use `selection.json` to identify the validation-selected checkpoint; test data never selects a checkpoint.

Primary decision: maximum mean within-query probability. Exact maximum ties use expected regret under uniform selection. Secondary error: 1 minus NDCG@5, averaging DCG over exact prediction ties (including a tie crossing the cutoff). Gains are 2^normalized_score - 1. Mean-score regret is a sensitivity metric; scalar MAE is not a primary metric.

R3 reports JS, entropy, negative top-two margin, raw and centered variance, and single-member baselines; query Spearman; risk at coverages 1/n,...,1; and their discrete mean AURC. Exact uncertainty ties are averaged so file order cannot improve rejection. Random rejection is its exact expected curve; oracle uses ranking regret. Global and domain-by-candidate-count JS permutations preserve the uncertainty marginal. Paired AURC confidence intervals resample questions; the same resample is used for baseline and JS. Training-seed spread must be reported separately. Do not pool overlapping folds or members as independent observations.

`compare` verifies identical question/candidate sets and reference fingerprints. For exposure it reports target/other mean probability variances, target-involving pairwise error (half error for a predicted tie), regret, JS, and target-specific variance reduction with paired question bootstrap intervals. Positive selective reduction supports sensitivity to controlled fine-tuning exposure only when target-related ranking also improves. It does not establish missing pretraining exposure or separate annotator uncertainty. For dose it reports paired JS, covariance-trace, and regret changes.

Analyze e-SNLI ID, DS held-out questions, and e-SNLI-to-DS transfer separately. Transfer is secondary R3 evidence and is not R4 exposure evidence. Low disagreement on well-learned questions is not automatically failure. `figures.trajectories` draws real checkpoint measurements beside a controlled contraction; `figures.exposure_figure` draws the target/control components and contrast intervals. These require actual model predictions; only the synthetic R1/R2 figure currently exists.

## Commands

Run from the repository root. Output directories must be new to prevent accidental overwrite.

```bash
python -m src.uncertainty_study.cli --help
python -m pytest tests/uncertainty_study tests/arr/test_losses.py tests/arr/test_epistemic.py tests/arr/test_gauge_invariance.py -q

python -m src.uncertainty_study.cli prepare-ds data/arr/DSCB-train-crowd-anno.jsonl data/arr/DSCB-dev-crowd-anno.jsonl data/arr/DSCB-train-expert.jsonl data/arr/DSCB-dev-non-anno.jsonl --output data/uncertainty_study/ds-new

python -m src.uncertainty_study.cli prepare-esnli --train PATH_TO_TRAIN --validation PATH_TO_VALIDATION --test PATH_TO_TEST --output data/uncertainty_study/esnli

python -m src.uncertainty_study.cli matrix --dataset ds --data data/uncertainty_study/ds --stage hpo --output runs/uncertainty_study/hpo-configs
python -m src.uncertainty_study.cli train --config configs/uncertainty_study/ds_pilot/0000-core-encoder-listnet-f0-s42-full.json --output runs/uncertainty_study/pilot-listnet
python -m src.uncertainty_study.cli analyze --predictions RUN/test-SELECTED_STEP.json --output RUN/analysis
python -m src.uncertainty_study.cli predict --run RUN --data DS_TEST.jsonl --output RUN/transfer.json
python -m src.uncertainty_study.cli combine MEMBER0.json MEMBER1.json MEMBER2.json MEMBER3.json MEMBER4.json --output ensemble.json
python -m src.uncertainty_study.cli compare --before ABSENT.json --after FULL.json --target gpt-3.5-turbo-0613 --output exposure-comparison.json
```

`matrix` defaults to three pilot runs (DeBERTa, three losses, first seed and DS fold). Request each later stage explicitly. No bulk job launcher starts the matrix automatically. Full-size dose runs reuse core runs. Addition-based exposure, automated length/quality stratified sensitivity analyses, and pooled out-of-fold reporting are outstanding analyses rather than completed results.

## Primary sources

- e-SNLI: https://arxiv.org/abs/1812.01193
- DS-Critique data/schema: https://huggingface.co/datasets/allenai/DS_Critique_Bank
- DeBERTa: https://huggingface.co/microsoft/deberta-v3-base
- Mistral: https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3
- ListMLE: https://icml.cc/Conferences/2008/papers/167.pdf
- Deep ensembles: https://arxiv.org/abs/1612.01474
- MC dropout: https://arxiv.org/abs/1506.02142

## Implementation references and normalization

[PT-Ranking ListNet](https://github.com/wildltr/ptranking/blob/master/ptranking/ltr_adhoc/listwise/listnet.py) uses softmax-normalized relevance targets. Its batch reduction is a sum; ours is a query mean. [PT-Ranking ListMLE](https://github.com/wildltr/ptranking/blob/master/ptranking/ltr_adhoc/listwise/listmle.py) uses tie shuffling and reverse cumulative exponentials; ours uses stable reverse logcumsumexp and a query mean. Check equality after accounting for batch reduction and identical tie permutations. MSE is squared error on normalized human scores; output activation must be held fixed across losses. [TensorFlow Ranking SoftmaxLoss](https://www.tensorflow.org/ranking/api_docs/python/tfr/keras/losses/SoftmaxLoss) uses labels directly as weights; its default is not our ListNet without softmax-transforming the target labels. No TensorFlow runtime parity is claimed merely from inspecting its documentation.
