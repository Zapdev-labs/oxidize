#!/usr/bin/env bash
set -euo pipefail

export KIMI_CALIB="${KIMI_CALIB:-/data/kimi-k2/calib-corpus-mixed.jsonl}"
export KIMI_PRUNE_MODE="${KIMI_PRUNE_MODE:-deep}"
export KIMI_PRUNE_RATIO="${KIMI_PRUNE_RATIO:-0.3}"

ROOT="/data/kimi-k2"
PY="$ROOT/.venv/bin/python"
PIPE="$ROOT/kimi_k2_ai2_pipeline.sh"

download_model() {
  local repo="$1"
  local out="$2"
  "$PY" - "$repo" "$out" <<'PY'
import sys
from huggingface_hub import snapshot_download

repo, out = sys.argv[1], sys.argv[2]
print(f"snapshot_download repo={repo} out={out}", flush=True)
path = snapshot_download(
    repo_id=repo,
    local_dir=out,
    resume_download=True,
    max_workers=8,
)
print(f"downloaded {repo} -> {path}", flush=True)
PY
}

test -f "$ROOT/checkpoints/k2.7-code/config.json"
download_model moonshotai/Kimi-K2.6 "$ROOT/checkpoints/k2.6"

"$PIPE" verify-arch
du -sh "$ROOT/checkpoints/k2.7-code" "$ROOT/checkpoints/k2.6"

"$PIPE" merge
test -f "$ROOT/k2-merged/config.json"
CONFIRM_DELETE=1 "$PIPE" cleanup-sources

"$PIPE" prune
test -d "$ROOT/k2-merged-pruned"
CONFIRM_DELETE=1 "$PIPE" cleanup-merged

"$PIPE" gguf
"$PIPE" smoke
