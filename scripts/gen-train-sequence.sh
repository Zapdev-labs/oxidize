#!/usr/bin/env bash
# Fast→quality 3-stage generator training (~3–4 hours on 761 clips @ 72×128).
set -euo pipefail

BIN="${OXIDIZE_TRAIN_BIN:-./target/release/oxidize-train}"
DATA="${OXIDIZE_DATA:-$HOME/tt-data/videos}"
CACHE="${OXIDIZE_CACHE:-$HOME/tt-data/frames-gen-full}"
LOG="${OXIDIZE_LOG:-$HOME/tt-data/gen-sequence.log}"
STAGE1="${OXIDIZE_STAGE1:-$HOME/tt-data/oxidize-video-generator-v2.json}"

exec >>"$LOG" 2>&1
echo "=== gen-train sequence $(date -Is) ==="

wait_for() {
  local path="$1"
  while [[ ! -f "$path" ]]; do
    if ! pgrep -f "oxidize-train gen-train" >/dev/null 2>&1; then
      echo "no gen-train running and $path missing — abort"
      exit 1
    fi
    echo "waiting for $path …"
    sleep 60
  done
}

if [[ ! -f "$STAGE1" ]]; then
  wait_for "$STAGE1"
fi
echo "stage1 ready: $STAGE1"

echo "--- stage 2: quality fine-tune (80 epochs, lr=4e-4, rollout=0.18) ---"
"$BIN" gen-train \
  --data "$DATA" \
  --cache "$CACHE" \
  --init-from "$STAGE1" \
  --epochs 80 \
  --batch-size 64 \
  --learning-rate 0.0004 \
  --self-rollout 0.18 \
  --context-dropout 0.15 \
  --context-noise 0.07 \
  --out "$HOME/tt-data/stage2-quality.json"

echo "--- stage 3: polish (50 epochs, lr=1.5e-4, rollout=0.22) ---"
"$BIN" gen-train \
  --data "$DATA" \
  --cache "$CACHE" \
  --init-from "$HOME/tt-data/stage2-quality.json" \
  --epochs 50 \
  --batch-size 64 \
  --learning-rate 0.00015 \
  --self-rollout 0.22 \
  --context-dropout 0.12 \
  --context-noise 0.05 \
  --out "$HOME/tt-data/oxidize-video-generator-v3.json"

echo "--- generate synthetic sample ---"
"$BIN" generate \
  --model "$HOME/tt-data/oxidize-video-generator-v3.json" \
  --data "$DATA" \
  --cache "$CACHE" \
  --mode synthetic \
  --output-frames 64 \
  --temperature 0.08 \
  --out "$HOME/tt-data/generated-v3.mp4"

echo "=== sequence complete $(date -Is) ==="
