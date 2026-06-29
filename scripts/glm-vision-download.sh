#!/usr/bin/env bash
# Download GLM-4.5V vision tower + mmproj for multimodal (pairs with GLM-5.2 text).
set -euo pipefail

HF="${HF:-$HOME/.local/bin/hf}"
OUT="${OUT:-$HOME/models/glm-vision}"
REPO="${REPO:-mradermacher/GLM-4.5V-GGUF}"
QUANT="${QUANT:-Q4_K_M}"
MMPROJ="${MMPROJ:-mmproj-Q8_0}"

mkdir -p "$OUT"
cd "$OUT"

echo "Downloading ${REPO} (${QUANT} + ${MMPROJ}) -> $OUT"

"$HF" download "$REPO" \
  --local-dir "$OUT/repo" \
  "*${QUANT}*.gguf" "*${MMPROJ}*.gguf"

echo "Done: $OUT/repo"
