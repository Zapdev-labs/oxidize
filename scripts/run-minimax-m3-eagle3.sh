#!/usr/bin/env bash
# Run MiniMax-M3 + EAGLE3 draft on ai@192.168.1.68 via oxidize.
set -euo pipefail

HOST="${OXIDIZE_AI_HOST:-ai@192.168.1.68}"
PASS="${OXIDIZE_AI_PASS:-machine}"

TARGET_GLOB="${TARGET_GLOB:-/home/ai/models/minimax-m3/target/UD-IQ1_M/MiniMax-M3-UD-IQ1_M-00001-of-00004.gguf}"
DRAFT="${DRAFT:-/home/ai/models/minimax-m3/eagle3/draft}"
OXIDIZE="${OXIDIZE:-/home/ai/oxidize/oxidize-c/oxidize-c}"
PROMPT="${PROMPT:-Hello, what is 2+2?}"
MAX_TOKENS="${MAX_TOKENS:-32}"
DRAFT_TOKENS="${DRAFT_TOKENS:-3}"

run_remote() {
  sshpass -p "$PASS" ssh -o StrictHostKeyChecking=no "$HOST" "$@"
}

run_remote "test -f '$TARGET_GLOB' || { echo 'target GGUF missing — wait for hf download'; exit 1; }"
run_remote "test -f '$DRAFT/model.safetensors' || { echo 'EAGLE3 draft missing'; exit 1; }"

run_remote "export OMP_NUM_THREADS=32; numactl --interleave=all $OXIDIZE prompt \
  --model '$TARGET_GLOB' \
  --prompt '$PROMPT' \
  --n-predict $MAX_TOKENS \
  --ctx 4096 \
  --no-auto \
  --threads 32 2>&1"
