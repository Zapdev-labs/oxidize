#!/usr/bin/env bash
# Wait for MiniMax-M3 IQ1_M download, then run EAGLE3 speculative decode.
set -euo pipefail

TARGET_DIR="${TARGET_DIR:-/home/ai/models/minimax-m3/target/UD-IQ1_M}"
DRAFT="${DRAFT:-/home/ai/models/minimax-m3/eagle3/draft}"
OXIDIZE="${OXIDIZE:-/home/ai/oxidize/target/release/oxidize}"
FIRST_SHARD="${FIRST_SHARD:-MiniMax-M3-UD-IQ1_M-00001-of-00004.gguf}"
# Shard 1 is metadata-only (~8 MB); data shards are tens of GB each.
declare -A MIN_SHARD_BYTES=(
  [00001]=5000000
  [00002]=40000000000
  [00003]=40000000000
  [00004]=20000000000
)

log() { echo "[$(date -Iseconds)] $*"; }

wait_for_shard() {
  local id="$1"
  local min="${MIN_SHARD_BYTES[$id]}"
  local s="$TARGET_DIR/MiniMax-M3-UD-IQ1_M-${id}-of-00004.gguf"
  while true; do
    if [[ -f "$s" ]]; then
      local sz
      sz=$(stat -c%s "$s" 2>/dev/null || echo 0)
      if (( sz >= min )); then
        log "shard $id ready: $(numfmt --to=iec "$sz")"
        return 0
      fi
      log "shard $id partial: $(numfmt --to=iec "$sz") (need >= $(numfmt --to=iec "$min"))"
    else
      log "shard $id: missing"
    fi
    sleep 20
  done
}

wait_for_shards() {
  for id in 00001 00002 00003 00004; do
    wait_for_shard "$id"
  done
}

run_bench() {
  local model="$TARGET_DIR/$FIRST_SHARD"
  log "warming page cache (mmap read)..."
  head -c 1G "$model" >/dev/null 2>&1 || true

  log "=== EAGLE3 run (draft_tokens=3, 32 threads, NUMA interleave) ==="
  export OMP_NUM_THREADS=32
  numactl --interleave=all "$OXIDIZE" run "$model" \
    --draft-model="$DRAFT" \
    --draft-tokens=3 \
    --prompt="The capital of France is" \
    --max-tokens=32 \
    --no-api \
    --no-auto \
    --threads=32 \
    --cpu-optimized \
    --ctx-size=4096 \
    2>&1 | tee "/home/ai/models/minimax-m3/logs/eagle3-run-$(date +%Y%m%d-%H%M%S).log"
}

mkdir -p /home/ai/models/minimax-m3/logs
wait_for_shards
run_bench
