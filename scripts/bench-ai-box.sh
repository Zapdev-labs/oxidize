#!/usr/bin/env bash
# Benchmark oxidize-cpp on ai@192.168.1.68 (dual Xeon Silver 4110, 123GB, 2 NUMA).
set -euo pipefail

HOST="${OXIDIZE_AI_HOST:-ai@192.168.1.68}"
PASS="${OXIDIZE_AI_PASS:-machine}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CMAKE_REMOTE='/home/ai/.local/lib/python3.14/site-packages/cmake/data/bin/cmake'

run_remote() {
  sshpass -p "$PASS" ssh -o StrictHostKeyChecking=no "$HOST" "$@"
}

sync_cpp() {
  run_remote "mkdir -p ~/oxidize/oxidize-cpp"
  sshpass -p "$PASS" rsync -az --delete -e "ssh -o StrictHostKeyChecking=no" \
    "$ROOT/oxidize-cpp/" "$HOST:~/oxidize/oxidize-cpp/"
}

build_cpp() {
  run_remote "cd ~/oxidize/oxidize-cpp && $CMAKE_REMOTE -B build -DCMAKE_BUILD_TYPE=Release && $CMAKE_REMOTE --build build -j32 --target oxidize-cpp"
}

bench_qwen() {
  local extra="${1:-}"
  run_remote "cd ~/oxidize/oxidize-cpp/build && ./oxidize-cpp \
    --model ~/models/qwen2.5-0.5b-instruct-q4_0.gguf \
    --tokens '1,2,3,4,5,6,7,8,9,10' --max-tokens 64 \
    --numa interleave --threads 32 --json $extra"
}

bench_glm() {
  run_remote "cd ~/oxidize/oxidize-cpp-glm/build && ./oxidize-cpp \
    --model ~/models/glm-5.2/target/UD-IQ1_M/GLM-5.2-UD-IQ1_M-00001-of-00006.gguf \
    --tokens '1,2,3,4,5' --max-tokens 8 --json"
}

case "${1:-all}" in
  sync)   sync_cpp ;;
  build)  build_cpp ;;
  qwen)   bench_qwen ;;
  glm)    bench_glm ;;
  all)
    sync_cpp
    build_cpp
    echo "=== Qwen 0.5B (oxidize-cpp, NUMA interleave) ==="
    bench_qwen
    echo "=== GLM-5.2 shard (oxidize-cpp-glm) ==="
    bench_glm || true
    ;;
  *) echo "usage: $0 {sync|build|qwen|glm|all}"; exit 1 ;;
esac
