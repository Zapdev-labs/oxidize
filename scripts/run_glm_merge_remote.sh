#!/usr/bin/env bash
set -euo pipefail

export HF_TOKEN="${HF_TOKEN:?set HF_TOKEN}"
export PATH="$HOME/.venvs/merge/bin:$HOME/.local/bin:$PATH"

A_DIR="$HOME/models/GLM-5.1"
B_CACHE="$HOME/models/GLM-5.2-cache"
OUT_DIR="$HOME/models/GLM-5.1-5.2-merged"
LOG="$HOME/work/glm-merge.log"

log() { echo "[$(date -Iseconds)] $*" | tee -a "$LOG"; }

log "waiting for GLM-5.1 download (282 weight shards, ~1.5TB)..."
while true; do
  n=$(find "$A_DIR" -name 'model-*.safetensors' 2>/dev/null | wc -l)
  partial=$(find "$A_DIR" -name '*.incomplete' 2>/dev/null | wc -l)
  bytes=$(du -sb "$A_DIR" 2>/dev/null | awk '{print $1}')
  gib=$((bytes / 1024 / 1024 / 1024))
  log "GLM-5.1: ${n}/282 shards complete, ${partial} in-flight, ${gib} GiB"
  if [[ "$n" -ge 282 ]]; then
    break
  fi
  if ! pgrep -f 'hf download zai-org/GLM-5.1' >/dev/null 2>&1; then
    if [[ "$n" -lt 282 ]]; then
      log "download process ended with $n shards — restarting"
      nohup hf download zai-org/GLM-5.1 --local-dir "$A_DIR" >> "$HOME/work/glm51-download.log" 2>&1 &
      sleep 30
      continue
    fi
  fi
  sleep 120
done

log "validating with stream-merge dry-run..."
python3 "$HOME/work/glm_stream_merge.py" \
  --a-dir "$A_DIR" \
  --b-repo zai-org/GLM-5.2 \
  --b-cache "$B_CACHE" \
  --output "$OUT_DIR" \
  --dry-run 2>&1 | tee -a "$LOG"

log "starting streamed SLERP merge - 5.1 base plus 5.2 blend..."
python3 "$HOME/work/glm_stream_merge.py" \
  --a-dir "$A_DIR" \
  --b-repo zai-org/GLM-5.2 \
  --b-cache "$B_CACHE" \
  --output "$OUT_DIR" \
  --progress-file "$HOME/work/glm-merge-progress.json" \
  --method slerp \
  --attention-t 0.35 \
  --mlp-t 0.55 \
  --other-t 0.45 \
  2>&1 | tee -a "$LOG"

log "copying tokenizer/config from GLM-5.2..."
for f in config.json tokenizer.json tokenizer_config.json generation_config.json chat_template.jinja; do
  python3 -c "
from huggingface_hub import hf_hub_download
import shutil, os
p = hf_hub_download('zai-org/GLM-5.2', '$f', local_dir='$OUT_DIR', token=os.environ['HF_TOKEN'])
shutil.copy(p, '$OUT_DIR/$f')
" 2>/dev/null || true
done

log "merge complete: $OUT_DIR"
du -sh "$OUT_DIR" | tee -a "$LOG"
df -h /home | tee -a "$LOG"
