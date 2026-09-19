#!/usr/bin/env bash
# Benchmark oxidize-c on the remote NUMA inference box.
# P0 defaults: Qwen 0.5B uses --numa single --threads 16; large models use --auto.
set -euo pipefail

HOST="${OXIDIZE_AI_HOST:-ai@192.168.1.132}"
PASS="${OXIDIZE_AI_PASS:-machine}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

run_remote() {
  sshpass -p "$PASS" ssh -o StrictHostKeyChecking=no "$HOST" "$@"
}

sync_c() {
  run_remote "mkdir -p ~/oxidize/oxidize-c"
  sshpass -p "$PASS" rsync -az --delete -e "ssh -o StrictHostKeyChecking=no" \
    --exclude 'oxidize-c' --exclude 'test_oc' --exclude '*.o' \
    "$ROOT/oxidize-c/" "$HOST:~/oxidize/oxidize-c/"
}

build_c() {
  run_remote "make -C ~/oxidize/oxidize-c clean build CFLAGS='-O3 -march=native'"
}

bench_qwen() {
  local extra="${1:-}"
  run_remote "cd ~/oxidize/oxidize-c && ./oxidize-c \
    --model ~/models/qwen2.5-0.5b-instruct-q4_0.gguf \
    --prompt 'The speed of light is' --max-tokens 64 \
    --numa single --threads 16 --json $extra"
}

bench_qwen_auto() {
  local extra="${1:-}"
  run_remote "cd ~/oxidize/oxidize-c && ./oxidize-c \
    --model ~/models/qwen2.5-0.5b-instruct-q4_0.gguf \
    --prompt 'The speed of light is' --max-tokens 64 \
    --auto --json $extra"
}

case "${1:-all}" in
  sync)   sync_c ;;
  build)  build_c ;;
  qwen)   bench_qwen ;;
  qwen-auto) bench_qwen_auto ;;
  plan)
    run_remote "cd ~/oxidize/oxidize-c && ./oxidize-c \
      --model ~/models/qwen2.5-0.5b-instruct-q4_0.gguf --print-plan --json"
    ;;
  all)
    sync_c
    build_c
    echo "=== Qwen 0.5B (oxidize-c, NUMA single / 16 threads) ==="
    bench_qwen
    echo "=== Qwen 0.5B (--auto) ==="
    bench_qwen_auto || true
    ;;
  *) echo "usage: $0 {sync|build|qwen|qwen-auto|plan|all}"; exit 1 ;;
esac
