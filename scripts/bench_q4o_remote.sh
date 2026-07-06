#!/usr/bin/env bash
# Requantize an F16 GGUF to Q4_0 vs Q4_O and benchmark decode on a NUMA box.
# Usage: scripts/bench_q4o_remote.sh [ssh-host] [f16-gguf-path]
set -euo pipefail

HOST="${1:-ai@192.168.1.121}"
MODEL="${2:-~/models/qwen05b/Qwen2.5-0.5B-Instruct-f16.gguf}"
REPO="${3:-~/oxidize}"
OUT="/tmp/q4o-bench-$$"
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
echo "==> building oxidize-quantize + oxidize-cpp (release)"
cargo build -p oxidize-quantize -p oxidize-core --release 2>&1 | tail -3
cmake -S oxidize-cpp -B oxidize-cpp/build -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build oxidize-cpp/build -j"$(nproc)" --target oxidize-cpp 2>&1 | tail -3

QZ=./target/release/oxidize-quantize
OX=./oxidize-cpp/build/oxidize-cpp

echo "==> requant F16 -> Q4_0"
/usr/bin/time -f 'q40 quant wall=%e s' \
  "$QZ" --input "$MODEL" --output "$OUT/q40.gguf" --source F16 --target Q4_0 --threads "$THREADS"

echo "==> requant F16 -> Q4_O"
/usr/bin/time -f 'q4o quant wall=%e s' \
  "$QZ" --input "$MODEL" --output "$OUT/q4o.gguf" --source F16 --target Q4_O --threads "$THREADS"

ls -lh "$OUT"/*.gguf

mse_layer() {
  local gguf=$1
  python3 - "$gguf" <<'PY'
import struct, sys, mmap, math
path = sys.argv[1]
with open(path, "rb") as f:
    magic = f.read(4)
    if magic != b"GGUF":
        raise SystemExit("not gguf")
    f.seek(0)
    data = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

def read_u64(off):
    return struct.unpack_from("<Q", data, off)[0]

def read_u32(off):
    return struct.unpack_from("<I", data, off)[0]

def read_str(off):
    n = read_u64(off)
    off += 8
    s = bytes(data[off:off+n])
  # strip null padding from gguf strings
    return s.split(b"\x00", 1)[0].decode(), off + n

off = 4
version = read_u32(off); off += 4
tensor_count = read_u64(off); off += 8
meta_count = read_u64(off); off += 8
for _ in range(meta_count):
    k, off = read_str(off)
    vtype = read_u32(off); off += 4
    if vtype == 8:
        n = read_u64(off); off += 8
        for _ in range(n):
            _, off = read_str(off)
    elif vtype in (0,1,2,3,4,5,6,7):
        off += {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1}[vtype]
    elif vtype in (10,11):
        off += 8
    elif vtype == 9:
        raise SystemExit("nested arrays unsupported")
    else:
        off += 8

infos = []
for _ in range(tensor_count):
    name, off = read_str(off)
    n_dims = read_u32(off); off += 4
    dims = [read_u64(off + 8*i) for i in range(n_dims)]
    off += 8*n_dims
    ggml_type = read_u32(off); off += 4
    offset = read_u64(off); off += 8
    infos.append((name, dims, ggml_type, offset))

alignment = 32
for k, vtype in []:
    pass
# find general.alignment if present - skip, use 32

data_base = (off + alignment - 1) // alignment * alignment

# pick first 2d weight tensor
pick = None
for name, dims, t, rel in infos:
    if len(dims) == 2 and "weight" in name:
        pick = (name, dims, t, rel)
        break
if not pick:
    raise SystemExit("no 2d weight tensor")

name, dims, t, rel = pick
n = dims[0]*dims[1]
# ggml q4 block = 18 bytes / 32 values for types 2 and 240
if t not in (2, 240):
    raise SystemExit(f"unexpected type {t}")
blocks = n // 32
raw = data[data_base+rel:data_base+rel+blocks*18]

def dequant_q4(raw):
    out = [0.0]*n
    for b in range(blocks):
        d = struct.unpack("<e", raw[b*18:b*18+2])[0]
        for i in range(16):
            byte = raw[b*18+2+i]
            lo = (byte & 0xF) - 8
            hi = (byte >> 4) - 8
            out[b*32+i] = lo*d
            out[b*32+16+i] = hi*d
    return out

# We only have quantized bytes here; report mean |dequant| as sanity
vals = dequant_q4(raw)
print(f"{path}: tensor={name} type={t} mean_abs={sum(abs(v) for v in vals)/len(vals):.6f}")
PY
}

echo "==> tensor sanity"
mse_layer "$OUT/q40.gguf"
mse_layer "$OUT/q4o.gguf"

bench_decode() {
  local label=$1
  local gguf=$2
  echo "==> decode bench $label"
  /usr/bin/time -f "${label} wall=%e s" \
    "$OX" --model "$gguf" --prompt "The speed of light is" --max-tokens "$DECODE_TOKENS" \
    --threads 16 --no-auto 2>&1 | rg -i "tok/s|tokens|benchmark|error" || true
}

bench_decode q40 "$OUT/q40.gguf"
bench_decode q4o "$OUT/q4o.gguf"

echo "==> done out=$OUT"
REMOTE
