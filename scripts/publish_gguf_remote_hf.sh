#!/usr/bin/env bash
# Stream-publish GGUFs from a remote host to HF (one file at a time; avoids filling /tmp).
# Usage: scripts/publish_gguf_remote_hf.sh [ssh-host] [remote-glob] [hf-repo]
set -euo pipefail

HOST="${1:-ai@192.168.1.121}"
REMOTE_GLOB="${2:-~/models/gemma4-31b-al/gemma-4-31B-it-*.gguf}"
HF_REPO="${3:-freakyskittle/gemma-4-31B-it-AL-GGUF}"
STAGING="${STAGING:-$(pwd)/.hf-staging}"

mkdir -p "$STAGING"
trap 'rm -rf "$STAGING"' EXIT

# Smallest first so we validate the pipeline early
mapfile -t REMOTE_FILES < <(ssh "$HOST" bash -s -- "$REMOTE_GLOB" <<'REMOTE_LIST'
set -euo pipefail
pattern=${1/#\~/$HOME}
compgen -G "$pattern" | while IFS= read -r file; do
  stat -c '%s\t%n' -- "$file"
done | sort -n | cut -f2-
REMOTE_LIST
)

echo "==> creating private repo $HF_REPO"
python3 scripts/publish_gguf_hf.py --repo "$HF_REPO" --private --files /dev/null 2>/dev/null || true

python3 - <<PY
from huggingface_hub import HfApi, create_repo
import os
token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
api = HfApi(token=token)
create_repo("$HF_REPO", repo_type="model", private=True, exist_ok=True, token=token)
api.update_repo_settings("$HF_REPO", private=True, repo_type="model", token=token)
print("repo ready (private)")
PY

for remote in "${REMOTE_FILES[@]}"; do
  name=$(basename "$remote")
  local="$STAGING/$name"
  echo "==> scp $name"
  scp "$HOST:$remote" "$local"
  echo "==> upload $name"
  python3 scripts/publish_gguf_hf.py --repo "$HF_REPO" --private --files "$local"
  rm -f "$local"
done

# README once at end
python3 scripts/publish_gguf_hf.py --repo "$HF_REPO" --private --files /dev/null 2>/dev/null || \
python3 -c "
from huggingface_hub import HfApi
import os
api = HfApi(token=os.environ.get('HF_TOKEN') or os.environ.get('HUGGINGFACE_HUB_TOKEN'))
readme = open('scripts/publish_gguf_hf.py').read()  # noop fallback
"

echo "==> published private repo: https://huggingface.co/$HF_REPO"
