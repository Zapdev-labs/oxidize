#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LIBOXK="$ROOT/liboxk"
OUT="$ROOT/dist/liboxk"

mkdir -p "$OUT"
make -C "$LIBOXK" clean liboxk.so liboxk.a
cp "$LIBOXK/liboxk.so" "$OUT/"
cp "$LIBOXK/liboxk.a" "$OUT/"
cp "$LIBOXK/include/oxk.h" "$OUT/"
echo "Built $OUT/liboxk.so"
