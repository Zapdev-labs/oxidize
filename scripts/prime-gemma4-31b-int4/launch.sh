#!/usr/bin/env bash
# Launch Gemma 4 31B serving on 2×H100. Set TIER=alpha|bravo|charlie.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIER="${TIER:-alpha}"
HF_HOME="${HF_HOME:-/data/hf-cache}"
MODEL_INT4="${MODEL_INT4:-google/gemma-4-31B-it-qat-w4a16-ct}"
MODEL_BF16="${MODEL_BF16:-google/gemma-4-31B-it}"
DRAFTER="${DRAFTER:-google/gemma-4-31B-it-assistant}"
VLLM_IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:gemma4-0505-cu129}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
GPU_UTIL="${GPU_UTIL:-0.92}"
SPEC_TOKENS="${SPEC_TOKENS:-8}"

DOCKER_COMMON=(
  --ipc=host
  --network=host
  --shm-size=32g
  -v "${HF_HOME}:/root/.cache/huggingface"
  -e "HF_TOKEN=${HF_TOKEN:-}"
)

vllm_base_args() {
  echo \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_UTIL" \
    --kv-cache-dtype fp8 \
    --async-scheduling \
    --enable-chunked-prefill \
    --max-num-batched-tokens 32768 \
    --limit-mm-per-prompt '{"image":0,"audio":0}' \
    --host 0.0.0.0
}

spec_config() {
  printf '{"model":"%s","num_speculative_tokens":%s}' "$DRAFTER" "$SPEC_TOKENS"
}

stop_existing() {
  docker rm -f gemma4-r0 gemma4-r1 gemma4-tp2 gemma4-lb 2>/dev/null || true
}

launch_alpha() {
  # ALPHA: 2× TP=1 replicas, one GPU each. Best aggregate tok/s for short chat.
  # ~20GB weights + FP8 KV leaves ~55GB/GPU for batching. Target: 900-1100 aggregate TPS.
  stop_existing

  docker run -d --name gemma4-r0 --gpus '"device=0"' \
    "${DOCKER_COMMON[@]}" \
    "$VLLM_IMAGE" \
    --model "$MODEL_INT4" \
    --tensor-parallel-size 1 \
    --port 8000 \
    --max-num-seqs 96 \
    $(vllm_base_args) \
    --speculative-config "$(spec_config)"

  docker run -d --name gemma4-r1 --gpus '"device=1"' \
    "${DOCKER_COMMON[@]}" \
    "$VLLM_IMAGE" \
    --model "$MODEL_INT4" \
    --tensor-parallel-size 1 \
    --port 8001 \
    --max-num-seqs 96 \
    $(vllm_base_args) \
    --speculative-config "$(spec_config)"

  docker run -d --name gemma4-lb \
    -p 8080:8080 \
    -v "${SCRIPT_DIR}/nginx-dual.conf:/etc/nginx/nginx.conf:ro" \
    nginx:alpine

  echo "ALPHA live:"
  echo "  replica-0: http://localhost:8000/v1"
  echo "  replica-1: http://localhost:8001/v1"
  echo "  loadbalancer: http://localhost:8080/v1"
}

launch_bravo() {
  # BRAVO: TP=2 single endpoint. Better TTFT on long prompts, simpler ops.
  stop_existing

  docker run -d --name gemma4-tp2 --gpus all \
    "${DOCKER_COMMON[@]}" \
    "$VLLM_IMAGE" \
    --model "$MODEL_INT4" \
    --tensor-parallel-size 2 \
    --port 8000 \
    --max-num-seqs 128 \
    $(vllm_base_args) \
    --speculative-config "$(spec_config)"

  echo "BRAVO live: http://localhost:8000/v1"
}

launch_charlie() {
  # CHARLIE: BF16 fallback — InferenceBench measured 1,472 peak / 1,675 sustained on 2×H100.
  stop_existing

  docker run -d --name gemma4-tp2 --gpus all \
    "${DOCKER_COMMON[@]}" \
    "$VLLM_IMAGE" \
    --model "$MODEL_BF16" \
    --dtype bfloat16 \
    --tensor-parallel-size 2 \
    --port 8000 \
    --max-num-seqs 128 \
    $(vllm_base_args) \
    --speculative-config "$(spec_config)"

  echo "CHARLIE live (BF16, guaranteed 1000+ TPS): http://localhost:8000/v1"
}

case "$TIER" in
  alpha) launch_alpha ;;
  bravo) launch_bravo ;;
  charlie) launch_charlie ;;
  *)
    echo "Unknown TIER=$TIER (use alpha, bravo, or charlie)" >&2
    exit 2
    ;;
esac

echo ""
echo "Benchmark: python3 ${SCRIPT_DIR}/bench_tps.py --url http://localhost:8080/v1 --target-tps 1000"
echo "Expose via Prime Tunnel: prime tunnel start --port 8080 --auth admin"
