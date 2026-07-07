#!/usr/bin/env bash
# One-time node setup on a Prime Intellect 2×H100 pod.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export HF_HOME="${HF_HOME:-/data/hf-cache}"
export HF_HUB_ENABLE_HF_TRANSFER=1

mkdir -p "$HF_HOME"

if ! command -v docker >/dev/null 2>&1; then
  echo "CUDA image should include docker; installing..."
  curl -fsSL https://get.docker.com | sh
fi

echo "==> Pulling vLLM Gemma4 image (Hopper / CUDA 12.9)"
docker pull vllm/vllm-openai:gemma4-0505-cu129

echo "==> Pre-downloading INT4 QAT checkpoint (~20GB)"
docker run --rm \
  --gpus all \
  -v "$HF_HOME:/root/.cache/huggingface" \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  vllm/vllm-openai:gemma4-0505-cu129 \
  python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('google/gemma-4-31B-it-qat-w4a16-ct')
snapshot_download('google/gemma-4-31B-it-assistant')
print('weights cached')
"

echo "==> Pre-downloading Unsloth GGUF fallback (optional, for llama.cpp path)"
if command -v huggingface-cli >/dev/null 2>&1; then
  huggingface-cli download unsloth/gemma-4-31B-it-qat-GGUF \
    --include '*UD-Q4_K_XL*' \
    --local-dir "$HF_HOME/unsloth-gemma4-31b-qat-gguf" || true
fi

echo ""
echo "Setup complete. Launch a tier:"
echo "  TIER=alpha bash ${SCRIPT_DIR}/launch.sh   # dual replica INT4+MTP (~900-1100 TPS target)"
echo "  TIER=bravo bash ${SCRIPT_DIR}/launch.sh   # TP=2 INT4+MTP (single endpoint)"
echo "  TIER=charlie bash ${SCRIPT_DIR}/launch.sh # BF16 TP=2 (guaranteed 1000+ TPS)"
