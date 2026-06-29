#!/usr/bin/env bash
# Benchmark oxidize-cpp on the remote NUMA inference box.
# P0 defaults: Qwen 0.5B uses --numa single --threads 16; large models use --auto.
set -euo pipefail

HOST="${OXIDIZE_AI_HOST:-ai@192.168.1.132}"
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
    --numa single --threads 16 --json $extra"
}

bench_qwen_auto() {
  local extra="${1:-}"
  run_remote "cd ~/oxidize/oxidize-cpp/build && ./oxidize-cpp \
    --model ~/models/qwen2.5-0.5b-instruct-q4_0.gguf \
    --tokens '1,2,3,4,5,6,7,8,9,10' --max-tokens 64 \
    --auto --json $extra"
}

bench_glm() {
  run_remote "cd ~/oxidize/oxidize-cpp-glm/build && ./oxidize-cpp \
    --model ~/models/glm-5.2/target/UD-IQ1_M/GLM-5.2-UD-IQ1_M-00001-of-00006.gguf \
    --tokens '1,2,3,4,5' --max-tokens 8 --auto --json"
}

case "${1:-all}" in
  sync)   sync_cpp ;;
  build)  build_cpp ;;
  qwen)   bench_qwen ;;
  qwen-auto) bench_qwen_auto ;;
  glm)    bench_glm ;;
  plan)
    run_remote "cd ~/oxidize/oxidize-cpp/build && ./oxidize-cpp \
      --model ~/models/qwen2.5-0.5b-instruct-q4_0.gguf --print-plan --json"
    ;;
  all)
    sync_cpp
    build_cpp
    echo "=== Qwen 0.5B (oxidize-cpp, NUMA single / 16 threads) ==="
    bench_qwen
    echo "=== Qwen 0.5B (--auto) ==="
    bench_qwen_auto || true
    echo "=== GLM-5.2 shard (oxidize-cpp-glm, --auto) ==="
    bench_glm || true
    ;;
  *) echo "usage: $0 {sync|build|qwen|qwen-auto|glm|plan|all}"; exit 1 ;;
esac
