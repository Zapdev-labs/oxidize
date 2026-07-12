#!/usr/bin/env bash
# Merge the trained LoRA adapter into the base model and upload merged safetensors.
set -euo pipefail

OUT_DIR="${OUT_DIR:-$HOME/models/qwen35-9b-agent}"
HF_DIR="${HF_DIR:-$OUT_DIR/hf}"
ADAPTER="${ADAPTER:-$OUT_DIR/unsloth-lora/adapter}"
MERGED="${MERGED:-$OUT_DIR/merged}"
BASE_MODEL="${HF_BASE_MODEL:-empero-ai/Qwythos-9B-Claude-Mythos-5-1M}"
MERGED_REPO="${MERGED_REPO:-freakyskittle/qwen35-9b-self-train-merged}"
LOG="${MERGE_LOG:-$OUT_DIR/merge-upload.log}"

if [[ -z "${HF_TOKEN:-}" ]]; then echo "HF_TOKEN required"; exit 1; fi

mkdir -p "$MERGED"

python3 - <<PY
import os
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

hf_dir = "${HF_DIR}"
adapter = "${ADAPTER}"
merged = "${MERGED}"
base_model = "${BASE_MODEL}"

print("loading base model (bf16)...", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    hf_dir, torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True
)
print("loading adapter...", flush=True)
model = PeftModel.from_pretrained(model, adapter)
print("merging...", flush=True)
model = model.merge_and_unload()
print("saving merged safetensors...", flush=True)
model.save_pretrained(merged, safe_serialization=True, max_shard_size="5GB")

tok = AutoTokenizer.from_pretrained(adapter, trust_remote_code=True)
tok.save_pretrained(merged)

# stamp real base_model into config for provenance
import json
cfg = Path(merged) / "config.json"
if cfg.is_file():
    d = json.loads(cfg.read_text())
    d.setdefault("_name_or_path", base_model)
    cfg.write_text(json.dumps(d, indent=2))
print("merged ->", merged, flush=True)
PY

echo "==> uploading merged model to $MERGED_REPO"
python3 - <<PY
import os
from huggingface_hub import HfApi, create_repo, upload_folder

token = os.environ["HF_TOKEN"]
repo = "${MERGED_REPO}"
merged = "${MERGED}"

create_repo(repo, repo_type="model", private=True, exist_ok=True, token=token)
api = HfApi(token=token)
api.update_repo_settings(repo, private=True, repo_type="model", token=token)
upload_folder(
    folder_path=merged,
    repo_id=repo,
    repo_type="model",
    token=token,
    commit_message="Merged Qwen3.5-9B + self-train LoRA (GPU SFT)",
)
print("uploaded merged ->", "https://huggingface.co/" + repo, flush=True)
PY
