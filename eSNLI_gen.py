import datasets
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig
)
import torch
from tqdm import tqdm
import evaluate
from peft import PeftModel

# === Model ===
model_name = "meta-llama/Llama-2-7b-chat-hf" 
compute_dtype = torch.float16

model_config = AutoConfig.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    max_new_tokens=64,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    config=model_config,
    device_map="auto",
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# === (Optional) Load a PEFT policy ===
peft_path = "saved_policy_llama2_snli_trl_EarlyS_Reward_CLS-full"
try:
    model = PeftModel.from_pretrained(model, peft_path)
    print(f"Loaded PEFT from: {peft_path}")
except Exception as e:
    print(f"PEFT not loaded ({e}), using base model.")

# === Mapping labels ===
ID2LABEL = {0: "entailment", 1: "neutral", 2: "contradiction"}

def to_label_str(lbl):
    if isinstance(lbl, int):
        return ID2LABEL[int(lbl)]
    s = str(lbl).lower().strip()
    if "entail" in s: return "entailment"
    if "contrad" in s: return "contradiction"
    if "neutral" in s: return "neutral"
    return ID2LABEL[int(lbl)]


# === System prompt / User prompt style ===
def build_gen_prompt(example):
    premise = example["premise"]
    hypothesis = example["hypothesis"]
    label_txt = to_label_str(example["label"])
    
    sys_msg  = "Respond with a short explanation in a maximum of one sentence."
    user_msg = f"If: '{premise}' {label_txt}: '{hypothesis}', why is that true?"
    
    return [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": user_msg}
    ]

# === Load eSNLI ===
raw_ds = load_dataset("esnli")
val_split = raw_ds["test"]

# Filter to keep valid examples
def has_valid_fields(x):
    return bool(x.get("premise")) and bool(x.get("hypothesis"))

val_split = val_split.filter(has_valid_fields)
val_split = val_split.map(lambda x: {"messages": build_gen_prompt(x)})

# === Generation ===
def generate_explanations(batch):
    prompts = [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in batch["messages"]]
    inputs = tokenizer(prompts, padding=True, truncation=True, max_length=256, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False,
        )
    generated = tokenizer.batch_decode(outputs[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return {"gen_explanation": [g.strip().split("\n")[0] for g in generated]}

val_with_gen = val_split.map(generate_explanations, batched=True, batch_size=32)

# === Prepare refs (take explanation_1 as main ref) ===
refs = [ex["explanation_1"] for ex in val_with_gen]
gens = [ex["gen_explanation"] for ex in val_with_gen]

# === Evaluations (BLEU / METEOR / ROUGE / CIDEr / BERTScore) ===
bleu = evaluate.load("bleu")
bleu_result = bleu.compute(predictions=gens, references=[[r] for r in refs])
print("BLEU:", bleu_result)

meteor = evaluate.load("meteor")
meteor_result = meteor.compute(predictions=gens, references=refs)
print("METEOR:", meteor_result)

rouge = evaluate.load("rouge")
rouge_result = rouge.compute(predictions=gens, references=refs)
print("ROUGE:", rouge_result)

try:
    cider = evaluate.load("cider")
    cider_result = cider.compute(predictions=gens, references=[[r] for r in refs])
    print("CIDEr:", cider_result)
except Exception as e:
    print("CIDEr not available:", e)

bertscore = evaluate.load("bertscore")
bertscore_result = bertscore.compute(predictions=gens, references=refs, lang="en")
print("BERTScore F1:", sum(bertscore_result["f1"]) / len(bertscore_result["f1"]))