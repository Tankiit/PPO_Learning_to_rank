from datasets import load_from_disk

ds = load_from_disk("datasets/DSCB_digital_socrates_v1_ppo_100K_ES_DPO")

def extract_scores(examples, key="critiques"):
    scores = []
    cpt = 0
    if key in examples and examples[key] is not None:
        for critique in examples[key]:
            if critique is not None:
                if critique['critique_model'] == "DS-7B":
                    try:
                        score = critique["critique_elements"]["explanation_score"]
                        if score is not None:
                            scores.append(score)
                            cpt += 1
                    except Exception:
                        pass  # ignore parsing errors
    return scores, cpt

all_scores_critiques = []
all_scores_new_critiques = []
total_cpt_critiques = 0
total_cpt_new_critiques = 0

for ex in ds:
    scores1, cpt1 = extract_scores(ex, key="critiques")
    scores2, cpt2 = extract_scores(ex, key="new_critiques")
    all_scores_critiques.extend(scores1)
    all_scores_new_critiques.extend(scores2)
    total_cpt_critiques += cpt1
    total_cpt_new_critiques += cpt2

def mean_or_nan(scores):
    return sum(scores) / len(scores) if len(scores) > 0 else float('nan')

print(f"Average explanation_score in 'critiques': {mean_or_nan(all_scores_critiques):.2f}")
print(f"Average explanation_score in 'new_critiques': {mean_or_nan(all_scores_new_critiques):.2f}")
print(f"Total number of critiques in 'critiques': {total_cpt_critiques}")
print(f"Total number of critiques in 'new_critiques': {total_cpt_new_critiques}")