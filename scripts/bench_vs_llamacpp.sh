#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/bench_vs_llamacpp.sh --model PATH --llamacpp-bin PATH [options]

Options:
  --oxidize-bin PATH       oxidize-bench binary or command (default: cargo run --release -p oxidize-cli --bin oxidize-bench --)
  --prompt-tokens N        prompt/prefill token count (default: 512)
  --decode-tokens N        decode token count (default: 128)
  --iterations N           oxidize-bench iterations (default: 3)
  --prompt TEXT            prompt passed to llama.cpp (default: synthetic prompt)
  --min-prefill-tps N      fail if oxidize prefill throughput is below N tok/s
  --min-decode-tps N       fail if oxidize decode throughput is below N tok/s
USAGE
}

model=""
llamacpp_bin=""
oxidize_bin="cargo run --release -p oxidize-cli --bin oxidize-bench --"
prompt_tokens=512
decode_tokens=128
iterations=3
prompt=""
min_prefill_tps=""
min_decode_tps=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) model="${2:?}"; shift 2 ;;
    --llamacpp-bin) llamacpp_bin="${2:?}"; shift 2 ;;
    --oxidize-bin) oxidize_bin="${2:?}"; shift 2 ;;
    --prompt-tokens) prompt_tokens="${2:?}"; shift 2 ;;
    --decode-tokens) decode_tokens="${2:?}"; shift 2 ;;
    --iterations) iterations="${2:?}"; shift 2 ;;
    --prompt) prompt="${2:?}"; shift 2 ;;
    --min-prefill-tps) min_prefill_tps="${2:?}"; shift 2 ;;
    --min-decode-tps) min_decode_tps="${2:?}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$model" || -z "$llamacpp_bin" ]]; then
  usage >&2
  exit 2
fi

if [[ ! -f "$model" ]]; then
  echo "model not found: $model" >&2
  exit 2
fi

if [[ -z "$prompt" ]]; then
  prompt="Write a concise technical explanation of quantized transformer inference."
fi

run_oxidize() {
  local mode="$1"
  local tokens="$2"
  local min_tps="$3"
  local -a gate=()
  if [[ -n "$min_tps" ]]; then
    gate=(--min-throughput "$min_tps")
  fi

  # shellcheck disable=SC2086
  $oxidize_bin --model "$model" --engine inference --mode "$mode" \
    --prompt-tokens "$tokens" --draft-tokens "$tokens" --iterations "$iterations" "${gate[@]}"
}

run_llamacpp() {
  local prompt_count="$1"
  local predict_count="$2"

  "$llamacpp_bin" -m "$model" -p "$prompt" -n "$predict_count" --prompt-cache /tmp/oxidize-llamacpp-prompt-cache \
    2>&1 | tee /tmp/oxidize-llamacpp-bench.log

  if ! grep -E "prompt eval|eval time" /tmp/oxidize-llamacpp-bench.log >/dev/null; then
    echo "warning: llama.cpp output did not include expected timing lines" >&2
  fi

  echo "llama.cpp requested prompt tokens: $prompt_count"
  echo "llama.cpp requested decode tokens: $predict_count"
}

echo "=== oxidize prefill ==="
run_oxidize pp "$prompt_tokens" "$min_prefill_tps"

echo
echo "=== oxidize decode ==="
run_oxidize decode "$decode_tokens" "$min_decode_tps"

echo
echo "=== llama.cpp ==="
run_llamacpp "$prompt_tokens" "$decode_tokens"
