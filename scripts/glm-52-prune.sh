#!/usr/bin/env bash
# Prune GLM-5.2 UD-IQ1_M shards: magnitude 25% + requant Q4_K_M (text weights preserved).
set -euo pipefail

INPUT_DIR="${INPUT_DIR:-$HOME/models/glm-5.2/target/UD-IQ1_M}"
OUTPUT_DIR="${OUTPUT_DIR:-$HOME/models/glm-5.2/pruned/Q4_K_M-mag25}"
PRUNE_BIN="${PRUNE_BIN:-$HOME/oxidize/target/release/oxidize-prune}"
SPARSITY="${SPARSITY:-0.25}"
METHOD="${METHOD:-magnitude}"
JOINT_QUANT="${JOINT_QUANT:-Q4_K_M}"
DRY_RUN="${DRY_RUN:-0}"
RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-2}"
export RAYON_NUM_THREADS

mkdir -p "$OUTPUT_DIR" "${OUTPUT_DIR}/logs"

KEEP_FLAGS=(
  --keep-name token_embd
  --keep-name output
  --keep-name norm
  --keep-name rope
  --keep-name attn
  --keep-name shexp
  --keep-name indexer
)

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

echo "Pruning ${#shards[@]} shards -> $OUTPUT_DIR (${METHOD} sparsity=${SPARSITY} -> ${JOINT_QUANT})"

for shard in "${shards[@]}"; do
  base=$(basename "$shard")
  out="$OUTPUT_DIR/$base"
  log="$OUTPUT_DIR/logs/${base%.gguf}.log"
  if [[ -f "$out" ]]; then
    echo "skip existing $base"
    continue
  fi
  args=(
    --input "$shard"
    --output "$out"
    --method "$METHOD"
    --sparsity "$SPARSITY"
    --joint-quantize "$JOINT_QUANT"
    "${KEEP_FLAGS[@]}"
  )
  if [[ "$DRY_RUN" == "1" ]]; then
    args+=(--dry-run)
  fi
  echo "==> $base"
  "$PRUNE_BIN" "${args[@]}" 2>&1 | tee "$log"
done

echo "Done. Output: $OUTPUT_DIR"
