#!/usr/bin/env bash
# Benchmark oxidize-cpp on the remote NUMA inference box.
set -euo pipefail

HOST="${OXIDIZE_AI_HOST:-ai@192.168.1.132}"
PASS="${OXIDIZE_AI_PASS:-}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CMAKE_REMOTE="${OXIDIZE_REMOTE_CMAKE:-cmake}"
REMOTE_DIR="${OXIDIZE_REMOTE_DIR:-/home/ai/oxidize-cpp-perf-ulw}"

SMALL_MODEL="${OXIDIZE_SMALL_MODEL:-/home/ai/models/llama31-8b/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf}"
MEDIUM_MODEL="${OXIDIZE_MEDIUM_MODEL:-/home/ai/models/mixtral-8x22b/Mixtral-8x22B-Instruct-v0.1.Q4_K_M-00001-of-00002.gguf}"
BIG_MODEL="${OXIDIZE_BIG_MODEL:-/home/ai/models/kimi-k2.75/pruned-compact-oxidize-q4/k2-merged-oxidize-Q4_K_M.gguf}"

die() {
  echo "bench-ai-box: $*" >&2
  exit 2
}

shell_quote() {
  printf '%q' "$1"
}

validate_remote_dir() {
  [[ "$REMOTE_DIR" == /* ]] || die "OXIDIZE_REMOTE_DIR must be absolute"
  [[ "$REMOTE_DIR" != *[[:space:]]* ]] || die "OXIDIZE_REMOTE_DIR must not contain whitespace"
  case "$REMOTE_DIR" in
    /|/home|/home/ai|/home/ai/oxidize) die "refusing unsafe OXIDIZE_REMOTE_DIR=$REMOTE_DIR" ;;
  esac
  [[ "$REMOTE_DIR" == /home/ai/oxidize-cpp* ]] ||
    die "OXIDIZE_REMOTE_DIR must live under /home/ai/oxidize-cpp*"
}

run_remote() {
  if [[ -n "$PASS" ]]; then
    SSHPASS="$PASS" sshpass -e ssh -o StrictHostKeyChecking=accept-new "$HOST" "$@"
  else
    ssh -o StrictHostKeyChecking=accept-new "$HOST" "$@"
  fi
}

sync_cpp() {
  validate_remote_dir
  local remote_dir_q
  remote_dir_q="$(shell_quote "$REMOTE_DIR")"
  run_remote "mkdir -p $remote_dir_q"
  echo "bench-ai-box: syncing oxidize-cpp to $HOST:$REMOTE_DIR/"
  if [[ -n "$PASS" ]]; then
    SSHPASS="$PASS" sshpass -e rsync -az --delete \
      -e "ssh -o StrictHostKeyChecking=accept-new" \
      "$ROOT/oxidize-cpp/" "$HOST:$REMOTE_DIR/"
  else
    rsync -az --delete -e "ssh -o StrictHostKeyChecking=accept-new" \
      "$ROOT/oxidize-cpp/" "$HOST:$REMOTE_DIR/"
  fi
}

build_cpp() {
  validate_remote_dir
  local remote_dir_q cmake_q
  remote_dir_q="$(shell_quote "$REMOTE_DIR")"
  cmake_q="$(shell_quote "$CMAKE_REMOTE")"
  run_remote "cd $remote_dir_q && rm -rf build && $cmake_q -B build -DCMAKE_BUILD_TYPE=Release -DOXIDIZE_CUDA=OFF -DOXIDIZE_ROCM=OFF && $cmake_q --build build -j32 --target oxidize-cpp"
}

plan_model() {
  local model="${1:?model path required}"
  local build_dir_q model_q
  build_dir_q="$(shell_quote "$REMOTE_DIR/build")"
  model_q="$(shell_quote "$model")"
  run_remote "cd $build_dir_q && ./oxidize-cpp --model $model_q --print-plan --json"
}

bench_small() {
  local build_dir_q model_q
  build_dir_q="$(shell_quote "$REMOTE_DIR/build")"
  model_q="$(shell_quote "$SMALL_MODEL")"
  run_remote "cd $build_dir_q && ./oxidize-cpp \
    --model $model_q \
    --tokens '1,2,3,4,5,6,7,8,9,10' --max-tokens '${OXIDIZE_SMALL_TOKENS:-64}' \
    --auto --json"
}

bench_medium() {
  local build_dir_q model_q
  build_dir_q="$(shell_quote "$REMOTE_DIR/build")"
  model_q="$(shell_quote "$MEDIUM_MODEL")"
  run_remote "cd $build_dir_q && ./oxidize-cpp \
    --model $model_q \
    --tokens '1,2,3,4,5,6,7,8' --max-tokens '${OXIDIZE_MEDIUM_TOKENS:-16}' \
    --auto --json"
}

bench_big() {
  local build_dir_q model_q
  build_dir_q="$(shell_quote "$REMOTE_DIR/build")"
  model_q="$(shell_quote "$BIG_MODEL")"
  run_remote "cd $build_dir_q && timeout \${OXIDIZE_BIG_TIMEOUT:-1800} ./oxidize-cpp \
    --model $model_q \
    --tokens '1,2,3,4' --max-tokens 1 \
    --auto --json"
}

case "${1:-all}" in
  sync)   sync_cpp ;;
  build)  build_cpp ;;
  small)  bench_small ;;
  medium) bench_medium ;;
  big)    bench_big ;;
  plan-small)  plan_model "$SMALL_MODEL" ;;
  plan-medium) plan_model "$MEDIUM_MODEL" ;;
  plan-big)    plan_model "$BIG_MODEL" ;;
  all)
    sync_cpp
    build_cpp
    echo "=== Small >5B (Llama 3.1 8B, --auto) ==="
    bench_small
    echo "=== Medium >35B (Mixtral 8x22B, --auto) ==="
    bench_medium || true
    echo "=== Big >500B-class (Kimi/K2, --auto) ==="
    bench_big || true
    ;;
  *) echo "usage: $0 {sync|build|small|medium|big|plan-small|plan-medium|plan-big|all}"; exit 1 ;;
esac
