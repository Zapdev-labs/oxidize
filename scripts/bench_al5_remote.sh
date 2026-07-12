#!/usr/bin/env bash
# Requantize an F16 GGUF to Q4_0 vs AL5 and benchmark decode on a NUMA box.
# Usage: scripts/bench_al5_remote.sh [ssh-host] [f16-gguf-path]
set -euo pipefail

HOST="${1:-ai@192.168.1.132}"
MODEL="${2:-~/models/qwen25-3b/Qwen2.5-3B-Instruct-f16.gguf}"
REPO="${3:-~/oxidize}"
OUT="/tmp/al5-bench-$$"
THREADS="${BENCH_THREADS:-48}"
DECODE_TOKENS="${BENCH_DECODE_TOKENS:-64}"

echo "==> host=$HOST model=$MODEL repo=$REPO"

ssh "$HOST" bash -s -- "$MODEL" "$REPO" "$OUT" "$THREADS" "$DECODE_TOKENS" <<'REMOTE'
set -euo pipefail
MODEL=$1
REPO=$2
OUT=$3
THREADS=$4
DECODE_TOKENS=$5

MODEL="${MODEL/#\~/$HOME}"
REPO="${REPO/#\~/$HOME}"
mkdir -p "$OUT"

cd "$REPO"
QZ="${QZ:-$REPO/target/release/oxidize-quantize}"
OX="${OX:-$REPO/oxidize-cpp/build/oxidize-cpp}"

if [[ ! -x "$QZ" ]]; then
  echo "missing $QZ — build locally: cargo build -p oxidize-quantize --release" >&2
  exit 1
fi
if [[ ! -x "$OX" ]]; then
  echo "==> building oxidize-cpp on host (cmake only)"
  rm -rf oxidize-cpp/build
  cmake -S oxidize-cpp -B oxidize-cpp/build -DCMAKE_BUILD_TYPE=Release >/dev/null
  cmake --build oxidize-cpp/build -j"$(nproc)" --target oxidize-cpp
fi

echo "==> requant F16 -> Q4_0"
/usr/bin/time -f 'q40 quant wall=%e s' \
  "$QZ" --input "$MODEL" --output "$OUT/q40.gguf" --source F16 --target Q4_0 --threads "$THREADS"

echo "==> requant F16 -> AL5"
/usr/bin/time -f 'al5 quant wall=%e s' \
  "$QZ" --input "$MODEL" --output "$OUT/al5.gguf" --source F16 --target AL5 --threads "$THREADS"

echo "==> requant F16 -> AL8"
/usr/bin/time -f 'al8 quant wall=%e s' \
  "$QZ" --input "$MODEL" --output "$OUT/al8.gguf" --source F16 --target AL8 --threads "$THREADS"

echo "==> requant F16 -> AL6"
/usr/bin/time -f 'al6 quant wall=%e s' \
  "$QZ" --input "$MODEL" --output "$OUT/al6.gguf" --source F16 --target AL6 --threads "$THREADS"

echo "==> requant F16 -> AL5_XS"
/usr/bin/time -f 'al5_xs quant wall=%e s' \
  "$QZ" --input "$MODEL" --output "$OUT/al5_xs.gguf" --source F16 --target AL5_XS --threads "$THREADS"

ls -lh "$OUT"/*.gguf
echo "==> F16 source: $(ls -lh "$MODEL" | awk '{print $5}')"

bench_decode() {
  local label=$1
  local gguf=$2
  echo "==> decode bench $label"
  local output
  if ! output=$(/usr/bin/time -f "${label} wall=%e s" \
    "$OX" --model "$gguf" --prompt "The speed of light is" --max-tokens "$DECODE_TOKENS" \
    --threads 16 --no-auto 2>&1); then
    printf '%s\n' "$output" >&2
    return 1
  fi
  printf '%s\n' "$output" | grep -iE "tok/s|tokens|benchmark|error|light|text:" || true
}

bench_decode q40 "$OUT/q40.gguf"
bench_decode al8 "$OUT/al8.gguf"
bench_decode al5 "$OUT/al5.gguf"
bench_decode al6 "$OUT/al6.gguf"
bench_decode al5_xs "$OUT/al5_xs.gguf"

echo "==> done out=$OUT"
REMOTE
