#!/usr/bin/env bash
# GPU LoRA finetune via oxidize-c (CUDA SSM + attention forward on A100).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${OXIDIZE_REPO:-$HOME/oxidize}"
OUT_DIR="${OUT_DIR:-$HOME/models/qwen35-9b-agent}"
MODEL="${TRAIN_MODEL:-$OUT_DIR/qwen35-9b-instruct-q4_k_m.gguf}"
DATASET="${TRAIN_DATASET:-$OUT_DIR/seed_coding_agent.jsonl}"
TRAIN_OUT="${TRAIN_OUT:-$OUT_DIR/lora-out}"
LOG="${TRAIN_LOG:-$OUT_DIR/gpu-finetune.log}"
PIDFILE="${TRAIN_PIDFILE:-$OUT_DIR/gpu-finetune.pid}"
BIN="$REPO/oxidize-c/oxidize-c-cuda"

THREADS="${TRAIN_THREADS:-$(nproc)}"
EPOCHS="${EPOCHS_PER_ROUND:-2}"
WINDOW="${TRAIN_WINDOW:-64}"
TOKENS_STEP="${TRAIN_TOKENS_PER_STEP:-256}"
MAX_SEQ="${TRAIN_MAX_SEQ_LEN:-512}"
LORA_RANK="${LORA_RANK:-32}"
LR="${LEARNING_RATE:-1.5e-4}"

export OMP_NUM_THREADS="$THREADS"
export OC_NO_GRAPH=1

mkdir -p "$OUT_DIR" "$TRAIN_OUT" "$(dirname "$LOG")"

if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "gpu-finetune already running pid=$(cat "$PIDFILE")"
  exit 0
fi

if [[ ! -x "$BIN" ]]; then
  echo "missing $BIN — run: cd $REPO/oxidize-c && make cuda"
  exit 1
fi

# Stop CPU-only rust self-train if still running
if [[ -f "$OUT_DIR/self-train.pid" ]]; then
  old=$(cat "$OUT_DIR/self-train.pid" 2>/dev/null || true)
  if [[ -n "$old" ]] && kill -0 "$old" 2>/dev/null; then
    echo "stopping CPU self-train pid=$old"
    kill "$old" 2>/dev/null || true
    sleep 2
    kill -9 "$old" 2>/dev/null || true
  fi
fi

echo "==> launching oxidize-c GPU finetune"
echo "    model=$MODEL"
echo "    dataset=$DATASET ($(wc -l < "$DATASET") rows)"
echo "    threads=$THREADS window=$WINDOW max_seq=$MAX_SEQ"
echo "    epochs=$EPOCHS log=$LOG"

setsid "$BIN" finetune \
  --model "$MODEL" \
  --data "$DATASET" \
  --out "$TRAIN_OUT" \
  --epochs "$EPOCHS" \
  --lr "$LR" \
  --rank "$LORA_RANK" \
  --alpha 64 \
  --max-seq-len "$MAX_SEQ" \
  --window "$WINDOW" \
  --tokens-per-step "$TOKENS_STEP" \
  --warmup 10 \
  >> "$LOG" 2>&1 &

echo $! > "$PIDFILE"
disown || true

sleep 3
echo "gpu-finetune pid=$(cat "$PIDFILE")"
tail -20 "$LOG" || true

if [[ -n "${HF_TOKEN:-}" ]] && [[ -n "${HF_REPO:-}" ]]; then
  echo "==> scheduling HF upload watcher"
  TRAIN_OUT="$TRAIN_OUT" TRAIN_LOG="$LOG" TRAIN_PIDFILE="$PIDFILE" \
    nohup bash "$SCRIPT_DIR/upload-hf.sh" >> "$OUT_DIR/upload-hf.log" 2>&1 &
fi
