#!/bin/sh
# Build-independent smoke test: run the corpus -> vocab -> tokens -> train ->
# sample -> score pipeline end to end and check that the tools reject bad input.
# Usage: tools/smoke.sh [bindir]
set -eu
BIN=${1:-./bin}
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
PROMPT='AAPL 1h close 228 RSI 32 ATR 9. Setup: opening range breakout. ACTION='

fail() {
    echo "smoke: $1" >&2
    exit 1
}

# Each tool must create its own output parent directory.
"$BIN/gen_corpus" 2000 "$WORK/data/corpus.txt" >/dev/null 2>&1
[ -s "$WORK/data/corpus.txt" ] || fail "gen_corpus wrote no corpus"

"$BIN/bpe" train "$WORK/data/corpus.txt" "$WORK/data/vocab.bin" 512 >/dev/null 2>&1
"$BIN/bpe" encode "$WORK/data/corpus.txt" "$WORK/data/vocab.bin" "$WORK/data/tokens.bin" >/dev/null 2>&1
[ -s "$WORK/data/tokens.bin" ] || fail "bpe encode wrote no tokens"

OMP_NUM_THREADS=2 "$BIN/oxidize-training" train --tokens "$WORK/data/tokens.bin" \
    --out "$WORK/out/model.bin" --steps 20 --batch 2 --seq 64 \
    --n-layer 2 --n-head 4 --n-embd 128 >/dev/null 2>&1
[ -s "$WORK/out/model.bin" ] || fail "train wrote no checkpoint"

# With --prompt ending in ACTION=, the forced first token must open one of the
# three actions rather than land mid-word.
out=$(OMP_NUM_THREADS=2 "$BIN/oxidize-training" sample --ckpt "$WORK/out/model.bin" \
    --vocab "$WORK/data/vocab.bin" --prompt "$PROMPT" --tokens 4 --temp 0.2 --seed 1)
cont=$(printf '%s\n' "$out" | awk 'f{print} /^---$/{f=1}' | head -1)
cont=${cont#*ACTION=}
case $cont in
B* | S* | H*) ;;
*) fail "forced action token opened with '${cont}', want B/S/H" ;;
esac

printf '%s\n' "$PROMPT" | OMP_NUM_THREADS=2 "$BIN/oxidize-training" score \
    --ckpt "$WORK/out/model.bin" --vocab "$WORK/data/vocab.bin" |
    grep -qE '^(BUY|SELL|HOLD) ' || fail "score did not emit an action"

expect_fail() {
    what=$1
    shift
    if "$@" >/dev/null 2>&1; then
        fail "$what was accepted"
    fi
}

expect_fail "non-numeric doc count" "$BIN/gen_corpus" abc "$WORK/bad.txt"
expect_fail "zero doc count" "$BIN/gen_corpus" 0 "$WORK/bad.txt"
expect_fail "bpe train without paths" "$BIN/bpe" train
expect_fail "oversized vocab" "$BIN/bpe" train "$WORK/data/corpus.txt" "$WORK/bad.bin" 99999999
expect_fail "truncated checkpoint" "$BIN/oxidize-training" sample \
    --ckpt "$WORK/data/vocab.bin" --vocab "$WORK/data/vocab.bin"
expect_fail "traversing --date" "$BIN/oxidize-training" picks --ckpt "$WORK/out/model.bin" \
    --vocab "$WORK/data/vocab.bin" --data-dir "$WORK/data" --date ../../etc/passwd

if [ -x "$BIN/gemm_bench" ]; then
    expect_fail "zero gemm iterations" "$BIN/gemm_bench" 8 8 8 0
    expect_fail "negative gemm dimension" "$BIN/gemm_bench" 8 -8 8
fi

echo "smoke: ok"
