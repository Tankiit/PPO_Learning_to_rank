#!/bin/bash

set -euo pipefail
if [[ "$(hostname)" != "nodeC000" ]]; then
  printf 'Run from nodeC000 after: ssh clustercril; ssh nodec000\n' >&2
  exit 2
fi
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

SOURCE="$ARR_PROJECT_ROOT/data/arr/ds_critique_external_test.jsonl"
QID_TRAIN="$ARR_PROJECT_ROOT/data/arr/ds_critique_qidsplit_train.jsonl"
QID_DEV="$ARR_PROJECT_ROOT/data/arr/ds_critique_qidsplit_dev.jsonl"
QID_MANIFEST="$ARR_PROJECT_ROOT/data/arr/ds_critique_qidsplit_manifest.json"
INFRA_ROOT="$ARR_PROJECT_ROOT/runs/pythia_infra"
MODEL_MANIFEST="$INFRA_ROOT/model_manifest.json"

[[ -f "$SOURCE" ]] || {
  printf 'Missing canonical DS-Critique training groups: %s\n' "$SOURCE" >&2
  exit 2
}
for required in config.json model.safetensors special_tokens_map.json tokenizer.json tokenizer_config.json; do
  [[ -f "$ARR_PYTHIA_MODEL_DIR/$required" ]] || {
    printf 'Incomplete offline Pythia snapshot: missing %s\n' "$ARR_PYTHIA_MODEL_DIR/$required" >&2
    exit 2
  }
done

mkdir -p "$INFRA_ROOT/slurm"
if [[ ! -f "$QID_MANIFEST" ]]; then
  if [[ -e "$QID_TRAIN" || -e "$QID_DEV" ]]; then
    printf 'Refusing an incomplete QID split; archive it before retrying.\n' >&2
    exit 2
  fi
  "$ARR_PYTHON" scripts/build_qid_split.py \
    --source "$SOURCE" --out-dir "$ARR_PROJECT_ROOT/data/arr" \
    --prefix ds_critique_qidsplit --holdout-fraction 0.2 \
    --salt arr-qid-split-v1
fi

"$ARR_PYTHON" - "$QID_MANIFEST" "$QID_TRAIN" "$QID_DEV" <<'PY'
import json
import sys
from pathlib import Path

from src.arr.utils import read_jsonl

manifest_path, train_path, dev_path = map(Path, sys.argv[1:])
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
train = list(read_jsonl(train_path))
dev = list(read_jsonl(dev_path))

expected = {
    "train": {"groups": 216, "candidates": 2592},
    "held_out": {"groups": 54, "candidates": 648},
}
for split, counts in expected.items():
    actual = manifest[split]
    for key, value in counts.items():
        if actual[key] != value:
            raise AssertionError((split, key, actual[key], value))

def qids(groups):
    return {(group.get("metadata") or {}).get("qid") for group in groups}

if len(train) != 216 or len(dev) != 54:
    raise AssertionError((len(train), len(dev)))
if any(len(group["candidates"]) != 12 for group in train + dev):
    raise AssertionError("every QID group must contain exactly 12 candidates")
if qids(train) & qids(dev):
    raise AssertionError("QID leakage between training and held-out data")
if not manifest.get("qid_disjoint"):
    raise AssertionError("manifest does not certify a QID-disjoint split")
print(json.dumps({"qid_split": expected, "qid_disjoint": True}, sort_keys=True))
PY

"$ARR_PYTHON" - "$ARR_PYTHIA_MODEL_DIR" "$ARR_PYTHIA_REVISION" "$MODEL_MANIFEST" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

from transformers import AutoConfig, AutoTokenizer

model_dir, revision, output = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
expected_hashes = {
    "config.json": "002050231a9b1ec3ac77aa6b9b3bbdc4d923f4068a7dd33b8da72a9bd6ad9a43",
    "model.safetensors": "ebfa4e2f18696ebd83716a0d39fe2c025f2ff8483f72a83ca59c475692fc9d15",
    "special_tokens_map.json": "6f50ab5a5a509a1c309d6171f339b196a900dc9c99ad0408ff23bb615fdae7ad",
    "tokenizer.json": "c24618a1b3e6a38167beff1c72cffd126c3a66254347304b50547d12c5f25624",
    "tokenizer_config.json": "70e38394e494931c6f773ba41e19460dd4436526b852207367f04341b4066d3f",
}
actual_hashes = {}
for name, expected in expected_hashes.items():
    digest = hashlib.sha256((model_dir / name).read_bytes()).hexdigest()
    if digest != expected:
        raise AssertionError(f"hash mismatch for {name}: {digest}")
    actual_hashes[name] = digest

config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
if config.model_type != "gpt_neox" or config.hidden_size != 512 or config.num_hidden_layers != 6:
    raise AssertionError(config.to_dict())
manifest = {
    "status": "complete",
    "repo_id": "EleutherAI/pythia-70m",
    "revision": revision,
    "model_dir": str(model_dir),
    "model_type": config.model_type,
    "hidden_size": config.hidden_size,
    "layers": config.num_hidden_layers,
    "tokenizer_size": len(tokenizer),
    "sha256": actual_hashes,
    "offline": True,
}
output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(manifest, sort_keys=True))
PY

printf 'Pythia infrastructure inputs are ready.\n'
printf 'Model manifest: %s\n' "$MODEL_MANIFEST"
printf 'QID split manifest: %s\n' "$QID_MANIFEST"
