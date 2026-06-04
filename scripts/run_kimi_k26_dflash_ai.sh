#!/usr/bin/env bash
# Run Kimi K2.6 + DFlash speculative generation on the ai host (Tailscale: ai@100.126.251.122).
# Requires: sshpass, SSH access (password or key). Do not commit credentials.
set -euo pipefail

AI_HOST="${AI_HOST:-ai@100.126.251.122}"
SSH_PASS="${SSH_PASS:-}"
PROMPT="${PROMPT:-The capital of France is}"
MAX_TOKENS="${MAX_TOKENS:-16}"
DRAFT_TOKENS="${DRAFT_TOKENS:-4}"

TARGET="${TARGET:-/home/ai/gguf-out/kimi-k2.6-base.Q8_0.gguf}"
DRAFT="${DRAFT:-/home/ai/Kimi-K2.6-DFlash-baseinit-Q4_0.gguf}"
CLI="${CLI:-/home/ai/oxidize/target/release/oxidize-cli}"

REMOTE_CMD=$(cat <<EOF
set -euo pipefail
exec ${CLI} \\
  --model ${TARGET} \\
  --draft-model ${DRAFT} \\
  --tokenizer-model ${TARGET} \\
  --layer-wise \\
  --layer-cache 2 \\
  --cpu-optimized \\
  --ctx-size 512 \\
  --max-tokens ${MAX_TOKENS} \\
  --draft-tokens ${DRAFT_TOKENS} \\
  --prompt $(printf '%q' "${PROMPT}")
EOF
)

if [[ -n "${SSH_PASS}" ]]; then
  exec sshpass -p "${SSH_PASS}" ssh -o StrictHostKeyChecking=no "${AI_HOST}" "${REMOTE_CMD}"
else
  exec ssh -o StrictHostKeyChecking=no "${AI_HOST}" "${REMOTE_CMD}"
fi
