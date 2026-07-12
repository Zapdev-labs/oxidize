#!/usr/bin/env bash
# Convert tencent/Hy3 -> GGUF (hy_v3 via llama.cpp fork), extend to 1M YaRN, AL5 quant.
#
# Usage:
#   scripts/hy3_1m_al_remote.sh [ssh-host]
#   HF_TOKEN=... scripts/hy3_1m_al_remote.sh
#
# Requires ~600 GB for BF16 download + convert, ~80-120 GB for AL5 output.
set -euo pipefail

HOST="${1:-ai@192.168.1.121}"
REPO="${REPO:-~/oxidize}"
OUT_DIR="${OUT_DIR:-~/models/hy3-1m-al}"
THREADS="${QUANT_THREADS:-96}"
CTX="${CTX:-1048576}"
ORIG_CTX="${ORIG_CTX:-262144}"
HF_MODEL="${HF_MODEL:-tencent/Hy3}"
HF_TOKEN="${HF_TOKEN:-}"
LLAMA_BRANCH="${LLAMA_BRANCH:-hy3-mtp}"
LLAMA_REPO="${LLAMA_REPO:-https://github.com/satindergrewal/llama.cpp.git}"

echo "==> host=$HOST ctx=$CTX orig=$ORIG_CTX threads=$THREADS"

ssh "$HOST" bash -s -- "$REPO" "$OUT_DIR" "$THREADS" "$CTX" "$ORIG_CTX" "$HF_MODEL" "$LLAMA_BRANCH" "$LLAMA_REPO" "$HF_TOKEN" <<'REMOTE'
set -euo pipefail
REPO=$1 OUT_DIR=$2 THREADS=$3 CTX=$4 ORIG_CTX=$5 HF_MODEL=$6 LLAMA_BRANCH=$7 LLAMA_REPO=$8
export HF_TOKEN=$9
REPO="${REPO/#\~/$HOME}"
OUT_DIR="${OUT_DIR/#\~/$HOME}"
mkdir -p "$OUT_DIR"

export PATH="$HOME/.local/bin:$PATH"
export HF_HUB_ENABLE_HF_TRANSFER=1

QZ="${QZ:-$REPO/bin/oxidize-quantize}"
if [[ ! -x "$QZ" ]]; then
  echo "missing $QZ — build oxidize-quantize on remote first" >&2
  exit 1
fi

if ! python3 -c 'import huggingface_hub' 2>/dev/null; then
  python3 -m pip install --user --break-system-packages -q huggingface_hub hf_transfer
fi

LLAMA_DIR="$OUT_DIR/llama.cpp-hy3"
HF_DIR="$OUT_DIR/hf-src"
F16_GGUF="$OUT_DIR/hy3-f16.gguf"
AL5_OUT="$OUT_DIR/Hy3-1M-AL5.gguf"

if [[ ! -d "$LLAMA_DIR/.git" ]]; then
  echo "==> clone llama.cpp ($LLAMA_BRANCH) for hy_v3 conversion"
  git clone --branch "$LLAMA_BRANCH" --depth 1 "$LLAMA_REPO" "$LLAMA_DIR"
fi

if [[ ! -f "$HF_DIR/config.json" ]]; then
  echo "==> download $HF_MODEL (safetensors, ~600 GB)"
  hf download "$HF_MODEL" --local-dir "$HF_DIR"
fi

if [[ ! -f "$F16_GGUF" ]]; then
  echo "==> convert HF -> F16 GGUF (hy_v3)"
  python3 "$LLAMA_DIR/convert_hf_to_gguf.py" "$HF_DIR" \
    --outfile "$F16_GGUF" \
    --outtype f16
  ls -lh "$F16_GGUF"
fi

if [[ ! -f "$AL5_OUT" ]]; then
  echo "==> quant F16 -> AL5 with 1M YaRN metadata"
  /usr/bin/time -f "AL5 wall=%e s cpu=%P" \
    "$QZ" \
    --input "$F16_GGUF" \
    --output "$AL5_OUT" \
    --target AL5 \
    --threads "$THREADS" \
    --context-length "$CTX" \
    --yarn-orig-ctx "$ORIG_CTX"
  ls -lh "$AL5_OUT"
else
  echo "==> skip AL5 (exists)"
fi

echo "==> done: $AL5_OUT"
REMOTE
