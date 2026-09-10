#!/usr/bin/env bash
# Compare oxidize-c vs ik_llama.cpp prefill/decode on .121 (or local).
# Usage: MODEL=... DRAFT=... OX=... IK=... ./scripts/dflash_speed_gate.sh
set -euo pipefail

MODEL="${MODEL:-/home/ai/models/qwen38-27b-ablit/gguf/Qwen3.8-27B-ABLITERATED-Q4_0_R8-MTP.gguf}"
DRAFT="${DRAFT:-/home/ai/models/dflash-qwen38-ablit/Qwen3.8-27B-ABLITERATED-DFlash-F16.gguf}"
OX="${OX:-/home/ai/oxidize-c-dflash/oxidize-c}"
IK="${IK:-/home/ai/ik_llama.cpp/build/bin/llama-cli}"
THREADS="${THREADS:-16}"
CTX="${CTX:-512}"
PROMPT_N="${PROMPT_N:-256}"
GEN_N="${GEN_N:-32}"
PROMPT="${PROMPT:-Say hello in one short sentence.}"
OUT="${OUT:-/tmp/dflash-speed-gate}"
mkdir -p "$OUT"

run_ox() {
  local name="$1"
  shift
  echo "===== oxidize-c $name ====="
  "$OX" --model "$MODEL" --prompt "$PROMPT" --n-predict "$GEN_N" --ctx "$CTX" \
    --threads "$THREADS" --numa single --temperature 0 --repeat-penalty 1.0 \
    "$@" 2>"$OUT/${name}.err" | tee "$OUT/${name}.out" || true
  grep -E "prefill:|speed:|dflash:|generated " "$OUT/${name}.err" || true
}

echo "===== oxidize-c tokenize ====="
"$OX" tokenize --model "$MODEL" --prompt "$PROMPT" 2>"$OUT/tok.err" | tee "$OUT/tok.out" || true

run_ox greedy --spec-type none
run_ox mtp --spec-type mtp
if [[ -f "$DRAFT" ]]; then
  run_ox dflash --spec-type dflash --draft-model "$DRAFT"
fi

if [[ -x "$IK" ]]; then
  echo "===== ik_llama greedy ====="
  numactl --cpunodebind=0 --membind=0 "$IK" -m "$MODEL" -p "$PROMPT" -n "$GEN_N" \
    -c "$CTX" -t "$THREADS" -ngl 0 --temp 0 --no-warmup \
    2>"$OUT/ik.err" | tee "$OUT/ik.out" || true
  grep -E "eval time|prompt eval|Hello|error" "$OUT/ik.err" "$OUT/ik.out" | tail -20 || true
fi
