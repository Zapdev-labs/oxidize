#!/usr/bin/env bash
# Upload finished LoRA adapter + synthetic data to a private HuggingFace model repo.
set -euo pipefail

OUT_DIR="${OUT_DIR:-$HOME/models/qwen35-9b-agent}"
TRAIN_OUT="${TRAIN_OUT:-$OUT_DIR/self-train-out}"
HF_REPO="${HF_REPO:-freakyskittle/qwen35-9b-self-train-lora}"
POLL_SEC="${UPLOAD_POLL_SEC:-300}"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN required"
  exit 1
fi

python3 - <<PY
from huggingface_hub import HfApi, create_repo
import os
token = os.environ["HF_TOKEN"]
repo = os.environ.get("HF_REPO", "freakyskittle/qwen35-9b-self-train-lora")
create_repo(repo, repo_type="model", private=True, exist_ok=True, token=token)
api = HfApi(token=token)
api.update_repo_settings(repo, private=True, repo_type="model", token=token)
print(f"repo ready (private): https://huggingface.co/{repo}")
PY

wait_for_done() {
  local log="$OUT_DIR/self-train.log"
  local pidfile="$OUT_DIR/self-train.pid"
  if [[ -n "${TRAIN_LOG:-}" ]]; then log="$TRAIN_LOG"; fi
  if [[ -n "${TRAIN_PIDFILE:-}" ]]; then pidfile="$TRAIN_PIDFILE"; fi
  while true; do
    if grep -qE "oxidize-finetuning self-train: finished|^done: |unsloth-train: finished" "$log" 2>/dev/null; then
      return 0
    fi
    if [[ -f "$pidfile" ]]; then
      pid=$(cat "$pidfile")
      if ! kill -0 "$pid" 2>/dev/null; then
        if ! grep -qE "finished|^done: " "$log" 2>/dev/null; then
          echo "training process exited without finish marker — uploading latest checkpoint anyway"
        fi
        return 0
      fi
    fi
    sleep "$POLL_SEC"
  done
}

echo "==> waiting for self-train to finish..."
wait_for_done

echo "==> uploading adapter folder"
python3 - <<PY
import os, re
from pathlib import Path
from huggingface_hub import HfApi, upload_folder

token = os.environ["HF_TOKEN"]
repo = os.environ.get("HF_REPO", "freakyskittle/qwen35-9b-self-train-lora")
base_model = os.environ.get("HF_BASE_MODEL", "empero-ai/Qwythos-9B-Claude-Mythos-5-1M")
out = Path(os.environ.get("TRAIN_OUT", Path.home() / "models/qwen35-9b-agent/self-train-out"))
api = HfApi(token=token)

def fix_readme(adapter_dir: Path) -> None:
    readme = adapter_dir / "README.md"
    if not readme.is_file():
        return
    text = readme.read_text()
    text = re.sub(r"base_model:\s*[^\n]+", f"base_model: {base_model}", text)
    text = text.replace("/root/models/qwen35-9b-agent/hf", base_model)
    readme.write_text(text)

if (out / "adapter").is_dir():
    fix_readme(out / "adapter")
    upload_folder(
        folder_path=str(out / "adapter"),
        path_in_repo="adapter",
        repo_id=repo,
        repo_type="model",
        token=token,
        commit_message="oxidize self-train LoRA adapter",
    )
elif (out / "adapter_manifest.json").is_file():
    api.upload_file(
        path_or_fileobj=str(out / "adapter_manifest.json"),
        path_in_repo="adapter_manifest.json",
        repo_id=repo,
        repo_type="model",
        token=token,
        commit_message="oxidize-c GPU finetune LoRA adapter",
    )

for extra in ("synthetic.jsonl", "metrics.csv", "self_train_state.json"):
    p = out / extra
    if p.is_file():
        api.upload_file(
            path_or_fileobj=str(p),
            path_in_repo=extra,
            repo_id=repo,
            repo_type="model",
            token=token,
            commit_message=f"upload {extra}",
        )

print(f"uploaded -> https://huggingface.co/{repo}")
PY
