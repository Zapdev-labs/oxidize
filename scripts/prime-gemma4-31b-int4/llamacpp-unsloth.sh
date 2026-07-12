#!/usr/bin/env bash
# Alternative path: Unsloth UD-Q4_K_XL GGUF + llama.cpp MTP on GPU 0,
# while GPU 1 runs a vLLM replica. Useful when vLLM compressed-tensors regresses.
set -euo pipefail

HF_HOME="${HF_HOME:-/data/hf-cache}"
GGUF_DIR="${GGUF_DIR:-${HF_HOME}/unsloth-gemma4-31b-qat-gguf}"
LLAMA_BIN="${LLAMA_BIN:-./llama.cpp/build/bin/llama-server}"

if [[ ! -x "$LLAMA_BIN" ]]; then
  echo "Build llama.cpp first or set LLAMA_BIN" >&2
  exit 1
fi
for model in "${GGUF_DIR}/UD-Q4_K_XL.gguf" "${GGUF_DIR}/UD-Q4_K_XL-assistant.gguf"; do
  [[ -f "$model" ]] || { echo "missing model: $model" >&2; exit 1; }
done

CUDA_VISIBLE_DEVICES=0 "$LLAMA_BIN" \
  -m "${GGUF_DIR}/UD-Q4_K_XL.gguf" \
  --model-draft "${GGUF_DIR}/UD-Q4_K_XL-assistant.gguf" \
  --spec-type draft-mtp \
  --spec-draft-n-max 8 \
  -ngl 99 \
  -c 16384 \
  -b 512 \
  --host 0.0.0.0 \
  --port 8081 &
server_pid=$!
sleep 2
if ! kill -0 "$server_pid" 2>/dev/null; then
  set +e
  wait "$server_pid"
  status=$?
  set -e
  exit "$status"
fi

echo "llama.cpp Unsloth QAT on :8081 (GPU 0)"
echo "Pair with vLLM on GPU 1: CUDA_VISIBLE_DEVICES=1 TIER=alpha ..."
