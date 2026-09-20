#!/usr/bin/env bash
# Prune GLM-5.2 shards with oxidize-c (magnitude sparsity).
set -euo pipefail

INPUT_DIR="${INPUT_DIR:-$HOME/models/glm-5.2/target/UD-IQ1_M}"
OUTPUT_DIR="${OUTPUT_DIR:-$HOME/models/glm-5.2/pruned/magnitude-25}"
PRUNE_BIN="${PRUNE_BIN:-$HOME/oxidize/oxidize-c/oxidize-c}"
SPARSITY="${SPARSITY:-0.25}"
METHOD="${METHOD:-magnitude}"
DRY_RUN="${DRY_RUN:-0}"

mkdir -p "$OUTPUT_DIR" "${OUTPUT_DIR}/logs"

shopt -s nullglob
if [[ -n "${SHARD_ONLY:-}" ]]; then
  shards=("$INPUT_DIR"/*"${SHARD_ONLY}"*.gguf)
else
  shards=("$INPUT_DIR"/GLM-5.2-UD-IQ1_M-*-of-*.gguf)
fi
if ((${#shards[@]} == 0)); then
  echo "No GLM-5.2 shards in $INPUT_DIR" >&2
  exit 1
fi

echo "Pruning ${#shards[@]} shards -> $OUTPUT_DIR (${METHOD} sparsity=${SPARSITY})"

for shard in "${shards[@]}"; do
  base=$(basename "$shard")
  out="$OUTPUT_DIR/$base"
  log="$OUTPUT_DIR/logs/${base%.gguf}.log"
  if [[ -f "$out" ]]; then
    echo "skip existing $base"
    continue
  fi
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "dry-run: $PRUNE_BIN prune --model $shard --output $out --strategy $METHOD --sparsity $SPARSITY"
    continue
  fi
  echo "==> $base"
  "$PRUNE_BIN" prune --model "$shard" --output "$out" --strategy "$METHOD" --sparsity "$SPARSITY" 2>&1 | tee "$log"
done

echo "Done. Output: $OUTPUT_DIR"
