#!/usr/bin/env bash
# Download Qwen3-Coder-30B safetensors, convert to BF16 GGUF, quantize to AL5.
# Usage: scripts/qwen30_bf16_al5_remote.sh [ssh-host]
set -euo pipefail

HOST="${1:-ai@192.168.1.121}"
REPO="${2:-~/oxidize}"
HF_DIR="${3:-~/models/qwen3coder30b-hf}"
OUT_DIR="${4:-~/models/qwen3coder30b}"
BF16="${OUT_DIR}/Qwen3-Coder-30B-A3B-Instruct-bf16.gguf"
AL5="${OUT_DIR}/Qwen3-Coder-30B-A3B-Instruct-al5.gguf"
THREADS="${BENCH_THREADS:-96}"

ssh "$HOST" bash -s -- "$REPO" "$HF_DIR" "$OUT_DIR" "$BF16" "$AL5" "$THREADS" <<'REMOTE'
set -euo pipefail
REPO=$1 HF_DIR=$2 OUT_DIR=$3 BF16=$4 AL5=$5 THREADS=$6
REPO="${REPO/#\~/$HOME}" HF_DIR="${HF_DIR/#\~/$HOME}"
OUT_DIR="${OUT_DIR/#\~/$HOME}" BF16="${BF16/#\~/$HOME}" AL5="${AL5/#\~/$HOME}"
CONVERT="$REPO/bin/oxidize-convert"
QUANTIZE="$REPO/bin/oxidize-quantize"
mkdir -p "$HF_DIR" "$OUT_DIR"

if ! python3 -c 'import huggingface_hub' 2>/dev/null; then
  echo "==> installing huggingface_hub"
  python3 -m pip install --user --break-system-packages -q huggingface_hub hf_transfer
fi
export PATH="$HOME/.local/bin:$PATH"
export HF_HUB_ENABLE_HF_TRANSFER=1

if python3 - "$HF_DIR" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
index = root / "model.safetensors.index.json"
if not (root / "config.json").is_file() or not index.is_file():
    raise SystemExit(1)
weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
if not all((root / shard).is_file() for shard in set(weight_map.values())):
    raise SystemExit(1)
PY
then
  echo "==> complete safetensors checkpoint already present in $HF_DIR"
else
  echo "==> downloading Qwen/Qwen3-Coder-30B-A3B-Instruct safetensors (~61GB)"
  hf download Qwen/Qwen3-Coder-30B-A3B-Instruct \
    --local-dir "$HF_DIR" \
    --include "config.json" "*.safetensors" "*.json" "tokenizer*" "*.txt" "merges.txt" "vocab.json"
fi

if [[ ! -f "$BF16" ]]; then
  echo "==> oxidize-convert safetensors -> BF16 GGUF"
  /usr/bin/time -f "convert wall=%e s" \
    "$CONVERT" --input "$HF_DIR" --output "$BF16" --arch qwen3
  ls -lh "$BF16"
else
  echo "==> BF16 GGUF exists: $(ls -lh "$BF16")"
fi

echo "==> oxidize-quantize BF16 -> AL5 (threads=$THREADS)"
/usr/bin/time -f "al5 quant wall=%e s cpu=%P" \
  "$QUANTIZE" --input "$BF16" --output "$AL5" --target AL5 --threads "$THREADS"
ls -lh "$BF16" "$AL5"
echo "==> done"
REMOTE
