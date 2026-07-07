#!/usr/bin/env bash
# Download Gemma 4 31B BF16 GGUF shards, requant to Oxidize AL family, optional HF publish.
#
# Usage:
#   scripts/gemma4_31b_al_remote.sh [ssh-host]
#   PUBLISH=1 HF_REPO=freakyskittle/gemma-4-31B-it-AL-GGUF scripts/gemma4_31b_al_remote.sh
#
set -euo pipefail

HOST="${1:-ai@192.168.1.121}"
REPO="${2:-~/oxidize}"
SRC_REPO="unsloth/gemma-4-31B-it-GGUF"
OUT_DIR="${OUT_DIR:-~/models/gemma4-31b-al}"
THREADS="${QUANT_THREADS:-96}"
HF_REPO="${HF_REPO:-freakyskittle/gemma-4-31B-it-AL-GGUF}"
PUBLISH="${PUBLISH:-0}"
QUANTS="${QUANTS:-AL5 AL6 AL8 AL5_XS}"

echo "==> host=$HOST threads=$THREADS quants=$QUANTS publish=$PUBLISH"

ssh "$HOST" bash -s -- "$REPO" "$OUT_DIR" "$THREADS" "$PUBLISH" "$HF_REPO" "$QUANTS" <<'REMOTE'
set -euo pipefail
REPO=$1 OUT_DIR=$2 THREADS=$3 PUBLISH=$4 HF_REPO=$5
QUANTS=$6
SRC_REPO="unsloth/gemma-4-31B-it-GGUF"
REPO="${REPO/#\~/$HOME}"
OUT_DIR="${OUT_DIR/#\~/$HOME}"
mkdir -p "$OUT_DIR"

export PATH="$HOME/.local/bin:$PATH"
export HF_HUB_ENABLE_HF_TRANSFER=1

QZ="${QZ:-$REPO/bin/oxidize-quantize}"
if [[ ! -x "$QZ" ]]; then
  echo "missing $QZ" >&2
  exit 1
fi

if ! python3 -c 'import huggingface_hub' 2>/dev/null; then
  python3 -m pip install --user --break-system-packages -q huggingface_hub hf_transfer
fi

BF16_DIR="$OUT_DIR/bf16-src"
mkdir -p "$BF16_DIR"

echo "==> downloading BF16 shards (lossless source)"
for shard in \
  "BF16/gemma-4-31B-it-BF16-00001-of-00002.gguf" \
  "BF16/gemma-4-31B-it-BF16-00002-of-00002.gguf"; do
  hf download "$SRC_REPO" "$shard" --local-dir "$BF16_DIR"
done

SRC="$BF16_DIR/BF16/gemma-4-31B-it-BF16-00001-of-00002.gguf"
[[ -f "$SRC" ]] || { echo "missing BF16 shard 1" >&2; exit 1; }

quant_one() {
  local target=$1
  local out="$OUT_DIR/gemma-4-31B-it-${target}.gguf"
  if [[ -f "$out" ]]; then
    echo "==> skip $target (exists)"
    return 0
  fi
  echo "==> quant BF16 -> $target"
  /usr/bin/time -f "${target} wall=%e s cpu=%P" \
    "$QZ" --input "$SRC" --output "$out" --target "$target" --threads "$THREADS"
  ls -lh "$out"
}

# Primary quant first (best size/quality tradeoff for inference)
for q in $QUANTS; do
  quant_one "$q"
done

echo "==> outputs:"
ls -lh "$OUT_DIR"/*.gguf 2>/dev/null || true

if [[ "$PUBLISH" == "1" ]]; then
  echo "==> publish to $HF_REPO from remote (private)"
  echo "    Set HF_TOKEN on remote or pass via: HF_TOKEN=... scripts/gemma4_31b_al_remote.sh"
fi
REMOTE

if [[ "$PUBLISH" == "1" ]]; then
  echo "==> publishing from remote host (no local disk copy)..."
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  TOKEN="${HF_TOKEN:-$(cat "${HOME}/.cache/huggingface/token" 2>/dev/null || true)}"
  if [[ -z "$TOKEN" ]]; then
    echo "HF_TOKEN required for publish" >&2
    exit 1
  fi
  scp "$SCRIPT_DIR/publish_gguf_hf.py" "$HOST:~/oxidize/bin/publish_gguf_hf.py"
  ssh "$HOST" bash -s -- "$HF_REPO" "$OUT_DIR" <<REMOTE_PUB
set -euo pipefail
HF_REPO=\$1
OUT_DIR=\${2/#\\~/\$HOME}
export HF_TOKEN='$TOKEN'
python3 ~/oxidize/bin/publish_gguf_hf.py --repo "\$HF_REPO" --private \
  --files "\$OUT_DIR/gemma-4-31B-it-AL5.gguf" \
          "\$OUT_DIR/gemma-4-31B-it-AL6.gguf" \
          "\$OUT_DIR/gemma-4-31B-it-AL8.gguf" \
          "\$OUT_DIR/gemma-4-31B-it-AL5_XS.gguf"
REMOTE_PUB
fi

echo "==> done"
