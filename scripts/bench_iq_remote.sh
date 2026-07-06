#!/usr/bin/env bash
# IQ4 (XS/NL) decode benchmark: oxidize (Rust) vs oxidize-cpp on a remote NUMA host.
# Usage: scripts/bench_iq_remote.sh [ssh-host] [repo-on-remote]
set -euo pipefail

HOST="${1:-ai@192.168.1.121}"
REMOTE_REPO="${2:-~/oxidize}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
THREADS="${BENCH_THREADS:-16}"
DECODE_TOKENS="${BENCH_DECODE_TOKENS:-64}"
PROMPT_TOKENS="${BENCH_PROMPT_TOKENS:-1,2,3,4,5,6,7,8,9,10}"

echo "==> sync workspace -> ${HOST}:${REMOTE_REPO}"
ssh "$HOST" "mkdir -p $REMOTE_REPO"
rsync -az \
  --exclude 'target' \
  --exclude 'oxidize-cpp/build' \
  --exclude 'dist' \
  --exclude '.git' \
  "$ROOT/" "$HOST:${REMOTE_REPO}/"

echo "==> remote build + benchmark (threads=${THREADS}, decode=${DECODE_TOKENS})"
ssh "$HOST" bash -s -- "$REMOTE_REPO" "$THREADS" "$DECODE_TOKENS" "$PROMPT_TOKENS" <<'REMOTE'
set -euo pipefail
REPO=$1
THREADS=$2
DECODE_TOKENS=$3
PROMPT_TOKENS=$4
REPO="${REPO/#\~/$HOME}"
OUT="/tmp/iq-bench-oxidize"
mkdir -p "$OUT"
cd "$REPO"

if ! command -v cargo >/dev/null 2>&1; then
  if [[ -x "$HOME/.cargo/bin/cargo" ]]; then
    export PATH="$HOME/.cargo/bin:$PATH"
  else
    echo "==> installing Rust (rustup) on remote"
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable --profile minimal
    export PATH="$HOME/.cargo/bin:$PATH"
  fi
fi
# shellcheck disable=SC1090
[[ -f "$HOME/.cargo/env" ]] && source "$HOME/.cargo/env"

echo "==> building oxidize (release) + oxidize-cpp"
if grep -q '^release:' Makefile 2>/dev/null; then
  make release 2>&1 | tail -5
else
  cargo build -p oxidize-cli -p oxidize-quantize --release 2>&1 | tail -5
fi
rm -rf oxidize-cpp/build
cmake -S oxidize-cpp -B oxidize-cpp/build -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build oxidize-cpp/build -j"$(nproc)" --target oxidize-cpp 2>&1 | tail -3
make -C oxidize-c oxidize-c CFLAGS="-O3 -march=native -fopenmp" 2>&1 | tail -3

RUST_BENCH=./target/release/oxidize-bench
RUST_CLI=./target/release/oxidize
QZ=./target/release/oxidize-quantize
OX=./oxidize-cpp/build/oxidize-cpp
OC=./oxidize-c/oxidize-c

find_iq_gguf() {
  local f
  for pat in IQ4_XS IQ4_NL IQ4; do
    f=$(find "$HOME/models" -type f -iname "*${pat}*.gguf" 2>/dev/null | head -1 || true)
    if [[ -n "${f:-}" ]]; then
      echo "$f"
      return 0
    fi
  done
  python3 - <<'PY'
import mmap, os, struct, sys

def gguf_has_iq(path: str) -> bool:
    iq = {20, 23}  # IQ4_NL, IQ4_XS
    try:
        with open(path, "rb") as f:
            if f.read(4) != b"GGUF":
                return False
            f.seek(0)
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    except OSError:
        return False
    off = 4
    version = struct.unpack_from("<I", mm, off)[0]
    off += 4
    if version not in (2, 3):
        return False
    tensor_count = struct.unpack_from("<Q", mm, off)[0]
    off += 8
    meta_count = struct.unpack_from("<Q", mm, off)[0]
    off += 8

    def read_str(o):
        n = struct.unpack_from("<Q", mm, o)[0]
        o += 8
        return mm[o : o + n].split(b"\x00", 1)[0], o + n

    for _ in range(meta_count):
        _, off = read_str(off)
        vtype = struct.unpack_from("<I", mm, off)[0]
        off += 4
        if vtype == 8:
            n = struct.unpack_from("<Q", mm, off)[0]
            off += 8
            for _ in range(n):
                _, off = read_str(off)
        elif vtype in (0, 1, 2, 3, 4, 5, 6, 7):
            off += {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1}[vtype]
        elif vtype in (10, 11):
            off += 8
        else:
            off += 8

    for _ in range(tensor_count):
        _, off = read_str(off)
        n_dims = struct.unpack_from("<I", mm, off)[0]
        off += 4 + 8 * n_dims
        ggml_type = struct.unpack_from("<I", mm, off)[0]
        off += 4
        off += 8
        if ggml_type in iq:
            return True
    return False

for root, _dirs, files in os.walk(os.path.expanduser("~/models")):
    for name in sorted(files):
        if not name.endswith(".gguf"):
            continue
        path = os.path.join(root, name)
        try:
            if os.path.getsize(path) > 30_000_000_000:
                continue
        except OSError:
            continue
        try:
            if gguf_has_iq(path):
                print(path)
                sys.exit(0)
        except Exception:
            continue
sys.exit(1)
PY
}

MODEL=""
if MODEL=$(find_iq_gguf); then
  echo "==> using existing IQ GGUF: $MODEL"
else
  F16="${IQ_F16_SOURCE:-$HOME/models/qwen05b/Qwen2.5-0.5B-Instruct-f16.gguf}"
  if [[ ! -f "$F16" ]]; then
    F16=$(find "$HOME/models" -type f -iname '*f16*.gguf' 2>/dev/null | head -1 || true)
  fi
  if [[ -z "${F16:-}" || ! -f "$F16" ]]; then
    echo "error: no IQ model and no F16 source for requant" >&2
    exit 1
  fi
  MODEL="$OUT/iq4_nl.gguf"
  echo "==> requant F16 -> IQ4_NL ($F16)"
  /usr/bin/time -f 'iq4_nl quant wall=%e s' \
    "$QZ" --input "$F16" --output "$MODEL" --source F16 --target IQ4_NL --threads "$THREADS"
fi

bench_rust() {
  export RAYON_NUM_THREADS="$THREADS"
  "$RUST_BENCH" --model "$MODEL" --mode decode --draft-tokens "$DECODE_TOKENS" \
    --iterations 2 --engine standard 2>&1
}

bench_cpp() {
  "$OX" --model "$MODEL" --tokens "$PROMPT_TOKENS" --max-tokens "$DECODE_TOKENS" \
    --threads "$THREADS" --no-auto --json 2>/dev/null
}

bench_oc() {
  "$OC" --model "$MODEL" --prompt "The speed of light is" \
    --max-tokens "$DECODE_TOKENS" --threads "$THREADS" 2>&1
}

echo "==> oxidize (Rust) decode bench"
RUST_OUT=$(bench_rust | tee "$OUT/rust.log")
RUST_TPS=$(echo "$RUST_OUT" | grep -oE 'Throughput: [0-9.]+ tok/s' | tail -1 | awk '{print $2}')
RUST_TPS=${RUST_TPS:-$(echo "$RUST_OUT" | grep -oE 'speed=[0-9.]+' | tail -1 | cut -d= -f2)}

echo "==> oxidize-cpp decode bench"
CPP_JSON=$(bench_cpp | tee "$OUT/cpp.json")
CPP_TPS=$(python3 -c "import json,sys; print(json.load(sys.stdin)['decode_tps'])" <<<"$CPP_JSON")

echo "==> oxidize-c decode bench"
OC_OUT=$(bench_oc | tee "$OUT/oc.log")
OC_TPS=$(echo "$OC_OUT" | grep -oE '[0-9]+\.[0-9]+ tok/s' | tail -1 | awk '{print $1}')
OC_TPS=${OC_TPS:-0}

RUST_TPS=${RUST_TPS:-0}
CPP_TPS=${CPP_TPS:-0}
RATIO=$(python3 -c "r=float('$RUST_TPS'); c=float('$CPP_TPS'); print(f'{r/c:.3f}' if c>0 else 'n/a')")
OC_RATIO=$(python3 -c "c=float('$OC_TPS'); r=float('$CPP_TPS'); print(f'{c/r:.3f}' if r>0 else 'n/a')")

echo ""
echo "================ IQ decode benchmark ================"
printf '%-18s %14s %14s\n' "engine" "decode_tok/s" "notes"
printf '%-18s %14s %14s\n' "oxidize (Rust)" "$RUST_TPS" "oxidize-bench decode"
printf '%-18s %14s %14s\n' "oxidize-cpp" "$CPP_TPS" "json decode_tps"
printf '%-18s %14s %14s\n' "oxidize-c" "$OC_TPS" "CLI tok/s line"
printf '%-18s %14s %14s\n' "rust/cpp ratio" "$RATIO" ""
printf '%-18s %14s %14s\n' "oc/cpp ratio" "$OC_RATIO" ""
echo "model: $MODEL"
echo "threads: $THREADS  decode_tokens: $DECODE_TOKENS"
echo "artifacts: $OUT"
REMOTE
