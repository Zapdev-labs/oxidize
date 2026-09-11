#!/usr/bin/env bash
# Speculative-decoding speed gate for the oxidize-c CLI (on .121 or local).
#
# Runs a greedy baseline (--spec-type none) and a speculative run
# (--spec-type "$SPEC_TYPE") with identical settings, parses the decode
# throughput each run prints ("<X> tok/s (<N> tokens in <T>s)" on stderr),
# and exits non-zero unless speculative/baseline > MIN_SPEEDUP.
# Optionally prints an ik_llama.cpp greedy run for reference (not gated).
#
# Usage: MODEL=... OX=... [SPEC_TYPE=mtp|dspark] [MIN_SPEEDUP=1.0] [IK=...] \
#        ./scripts/dflash_speed_gate.sh
#
# Exit codes: 0 = gate passed, 1 = gate failed or a run errored,
#             2 = gate cannot be evaluated (bad config / unsupported mode).
set -euo pipefail
# Make failures inside $(...) (e.g. run_ox) abort too.
shopt -s inherit_errexit

MODEL="${MODEL:-/home/ai/models/qwen38-27b-ablit/gguf/Qwen3.8-27B-ABLITERATED-Q4_0_R8-MTP.gguf}"
OX="${OX:-/home/ai/oxidize-c-dflash/oxidize-c}"
IK="${IK:-/home/ai/ik_llama.cpp/build/bin/llama-cli}"
SPEC_TYPE="${SPEC_TYPE:-dflash}"
MIN_SPEEDUP="${MIN_SPEEDUP:-1.0}"
THREADS="${THREADS:-16}"
CTX="${CTX:-512}"
GEN_N="${GEN_N:-32}"
PROMPT="${PROMPT:-Say hello in one short sentence.}"
OUT="${OUT:-/tmp/dflash-speed-gate}"

die() {
  local code="$1"
  shift
  echo "error: $*" >&2
  exit "$code"
}

case "$SPEC_TYPE" in
  mtp | dspark) ;;
  dflash)
    # The oxidize-c CLI has no DFlash decode path: --draft-model is parsed
    # (src/cli/args.c) but never read by the generation loop, and --spec-type
    # only distinguishes none | mtp | dspark (any other value, including
    # "dflash", silently runs DSpark). A "dflash" run would therefore measure
    # DSpark, not DFlash, so the gate cannot be evaluated.
    die 2 "oxidize-c has no DFlash speculative mode (--draft-model is ignored);" \
      "cannot evaluate a DFlash gate. Set SPEC_TYPE=mtp or SPEC_TYPE=dspark to gate the in-GGUF MTP paths."
    ;;
  *) die 2 "unsupported SPEC_TYPE '$SPEC_TYPE' (expected mtp or dspark)" ;;
esac

[[ -x "$OX" ]] || die 2 "oxidize-c binary not found or not executable: $OX"
[[ -f "$MODEL" ]] || die 2 "model not found: $MODEL"
awk -v s="$MIN_SPEEDUP" 'BEGIN { exit !(s ~ /^[0-9]+(\.[0-9]+)?$/) }' \
  || die 2 "MIN_SPEEDUP must be a non-negative number, got '$MIN_SPEEDUP'"

mkdir -p "$OUT"

# Run oxidize-c and print the decode tok/s it reports on stdout.
# Progress and logs go to stderr so the caller can capture the number.
run_ox() {
  local name="$1"
  shift
  echo "===== oxidize-c $name =====" >&2
  "$OX" --model "$MODEL" --prompt "$PROMPT" --n-predict "$GEN_N" --ctx "$CTX" \
    --threads "$THREADS" --numa single --temperature 0 --repeat-penalty 1.0 \
    "$@" 2>"$OUT/${name}.err" | tee "$OUT/${name}.out" >&2
  grep -E "prefill:|speed:|generated |tok/s \(" "$OUT/${name}.err" >&2 \
    || echo "warning: no timing lines in $OUT/${name}.err" >&2
  local tps
  tps="$(sed -nE 's/^([0-9]+(\.[0-9]+)?) tok\/s \(.*/\1/p' "$OUT/${name}.err" | tail -n 1)"
  [[ -n "$tps" ]] || die 1 "could not parse decode tok/s for '$name' (see $OUT/${name}.err)"
  echo "$tps"
}

echo "===== oxidize-c tokenize ====="
"$OX" tokenize --model "$MODEL" --prompt "$PROMPT" 2>"$OUT/tok.err" | tee "$OUT/tok.out"

base_tps="$(run_ox greedy --spec-type none)"
spec_tps="$(run_ox "$SPEC_TYPE" --spec-type "$SPEC_TYPE")"

# The CLI only takes the speculative path when the model carries an MTP/nextn
# head; without one --spec-type silently falls back to greedy and the gate
# would compare greedy against greedy.
grep -q "MTP/nextn block loaded" "$OUT/${SPEC_TYPE}.err" \
  || die 2 "model has no MTP/nextn head, so --spec-type $SPEC_TYPE ran plain greedy; cannot evaluate the gate"

echo "===== gate ====="
echo "baseline (greedy): ${base_tps} tok/s"
echo "${SPEC_TYPE}: ${spec_tps} tok/s"
if awk -v b="$base_tps" -v s="$spec_tps" -v m="$MIN_SPEEDUP" \
  'BEGIN { r = (b > 0) ? s / b : 0; printf "speedup: %.3fx (required > %sx)\n", r, m; exit !(b > 0 && r > m) }'; then
  verdict=0
  echo "PASS"
else
  verdict=1
  echo "FAIL: ${SPEC_TYPE} is not faster than greedy by more than ${MIN_SPEEDUP}x" >&2
fi

# Reference only: runs after the verdict is computed and never changes it.
if [[ -x "$IK" ]]; then
  if ! command -v numactl >/dev/null 2>&1; then
    echo "warning: numactl not found, skipping ik_llama reference run" >&2
  else
    echo "===== ik_llama greedy (reference, not gated) ====="
    if numactl --cpunodebind=0 --membind=0 "$IK" -m "$MODEL" -p "$PROMPT" -n "$GEN_N" \
      -c "$CTX" -t "$THREADS" -ngl 0 --temp 0 --no-warmup \
      2>"$OUT/ik.err" | tee "$OUT/ik.out"; then
      grep -E "eval time|prompt eval" "$OUT/ik.err" "$OUT/ik.out" | tail -20 \
        || echo "warning: no timing lines in ik_llama output" >&2
    else
      echo "warning: ik_llama reference run failed (see $OUT/ik.err)" >&2
    fi
  fi
fi

exit "$verdict"
