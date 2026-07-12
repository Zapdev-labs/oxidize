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
  while true; do
    if grep -q "oxidize-finetuning self-train: finished" "$log" 2>/dev/null; then
      return 0
    fi
    if [[ -f "$OUT_DIR/self-train.pid" ]]; then
      pid=$(cat "$OUT_DIR/self-train.pid")
      if ! kill -0 "$pid" 2>/dev/null; then
        if ! grep -q "finished" "$log" 2>/dev/null; then
          echo "training process exited without finish marker" >&2
          return 1
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
import os
from pathlib import Path
from huggingface_hub import HfApi, upload_folder

token = os.environ["HF_TOKEN"]
repo = os.environ.get("HF_REPO", "freakyskittle/qwen35-9b-self-train-lora")
out = Path(os.environ.get("TRAIN_OUT", Path.home() / "models/qwen35-9b-agent/self-train-out"))
api = HfApi(token=token)

if (out / "adapter").is_dir():
    upload_folder(
        folder_path=str(out / "adapter"),
        path_in_repo="adapter",
        repo_id=repo,
        repo_type="model",
        token=token,
        commit_message="oxidize self-train LoRA adapter",
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
