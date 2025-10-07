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

# torch._dynamo.config.disable = True

model_name = "meta-llama/Llama-3.1-8B-Instruct"
# model_name = "models/Llama-3.1-8B-Instruct"
# model_name = "mistralai/Mistral-7B-Instruct-v0.3"
# model_name = "meta-llama/Llama-2-7b-chat-hf"
# model_name = "models/Llama-2-13b-chat-hf"
# model_name = "Qwen/Qwen3-4B-Instruct-2507"
# model_name = "google/gemma-3-4b-it"
compute_dtype = torch.float16

model_config = AutoConfig.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    max_new_tokens=64,
    # local_files_only=True
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    config=model_config,
    device_map="auto",
    # local_files_only=True,
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# PPO
peft_path = "saved_policy_llama2_snli_trl_EarlyS_Reward_CLS-full"
model = PeftModel.from_pretrained(model, peft_path)

# System prompt for generation
SYSTEM_PROMPT = (
    "You are a helpful assistant. "
    "Given two observations, generate the best possible hypothesis in one sentence that is most strongly supported by those observations."
)

def build_gen_prompt(example):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content":
            f"Observations:\n"
            f"1. {example['observation_1']}\n"
            f"2. {example['observation_2']}\n\n"
            f"Generate the best possible hypothesis in one sentence, supported by these observations."
        }
    ]

# Load ART dataset (validation split for testing)
raw_ds = load_dataset("allenai/art")
val_split = raw_ds["validation"]
# val_split = raw_ds["validation"].select([i for i in range(20)])

# Generate chat prompts
val_split = val_split.map(lambda x: {"messages": build_gen_prompt(x)})

# Batched generation
def generate_hypotheses(batch):
    msgs_batch = batch["messages"]
    prompts = [tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) for msgs in msgs_batch]
    inputs = tokenizer(prompts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False,
        )
    generated = tokenizer.batch_decode(outputs[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(generated)
    # Minimal cleaning to remove unnecessary text
    generated = [g.strip().split("\n")[0] for g in generated]
    return {"gen_hypothesis": generated}

val_with_gen = val_split.map(
    generate_hypotheses,
    batched=True,
    batch_size=8
)

# # To see the result:
# for i in range(3):
#     print("Observations:")
#     print(val_with_gen[i]["observation_1"])
#     print(val_with_gen[i]["observation_2"])
#     print("Generated:", val_with_gen[i]["gen_hypothesis"])
#     print("Hypothesis 1 (ref):", val_with_gen[i]["hypothesis_1"])
#     print("Hypothesis 2 (ref):", val_with_gen[i]["hypothesis_2"])
#     print("-" * 40)


# Prepare refs (take hypothesis_1 as main ref)
refs = [val_with_gen[i][f"hypothesis_{val_with_gen[i]['label']}"] for i in range(len(val_with_gen))]
gens = [val_with_gen[i]["gen_hypothesis"] for i in range(len(val_with_gen))]

# BLEU
bleu = evaluate.load("bleu")
bleu_result = bleu.compute(predictions=gens, references=[[r] for r in refs])
print("BLEU:", bleu_result)

# METEOR
meteor = evaluate.load("meteor")
meteor_result = meteor.compute(predictions=gens, references=refs)
print("METEOR:", meteor_result)

# ROUGE
rouge = evaluate.load("rouge")
rouge_result = rouge.compute(predictions=gens, references=refs)
print("ROUGE:", rouge_result)

# CIDEr (requires installation of 'cider' or 'coco-caption', sometimes tricky, skip if error)
try:
    cider = evaluate.load("cider")
    cider_result = cider.compute(predictions=gens, references=[[r] for r in refs])
    print("CIDEr:", cider_result)
except Exception as e:
    print("CIDEr not available:", e)

# BERTScore
bertscore = evaluate.load("bertscore")
bertscore_result = bertscore.compute(predictions=gens, references=refs, lang="en")
print("BERTScore F1:", sum(bertscore_result["f1"]) / len(bertscore_result["f1"]))