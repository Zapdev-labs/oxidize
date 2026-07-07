#!/usr/bin/env bash
# Launch self-train on Prime pod with turbo CPU settings (22 vCPU A100 node).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${OXIDIZE_REPO:-$HOME/oxidize}"
OUT_DIR="${OUT_DIR:-$HOME/models/qwen35-9b-agent}"
MODEL="${TRAIN_MODEL:-$OUT_DIR/qwen35-9b-instruct-q8_0.gguf}"
DATASET="${TRAIN_DATASET:-$OUT_DIR/seed_coding_agent.jsonl}"
TRAIN_OUT="${TRAIN_OUT:-$OUT_DIR/self-train-out}"
LOG="${TRAIN_LOG:-$OUT_DIR/self-train.log}"
PIDFILE="${TRAIN_PIDFILE:-$OUT_DIR/self-train.pid}"
FINETUNE="$REPO/target/release/oxidize-finetuning"

THREADS="${TRAIN_THREADS:-$(nproc)}"
ROUNDS="${SELF_TRAIN_ROUNDS:-5}"
PROMPTS="${PROMPTS_PER_ROUND:-12}"
EPOCHS="${EPOCHS_PER_ROUND:-2}"
WINDOW="${TRAIN_WINDOW:-256}"
TOKENS_STEP="${TRAIN_TOKENS_PER_STEP:-1024}"
MAX_SEQ="${TRAIN_MAX_SEQ_LEN:-4096}"
LORA_RANK="${LORA_RANK:-32}"
LR="${LEARNING_RATE:-1.5e-4}"

export PATH="$HOME/.cargo/bin:$PATH"
export RAYON_NUM_THREADS="$THREADS"

mkdir -p "$OUT_DIR" "$(dirname "$LOG")"

if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "self-train already running pid=$(cat "$PIDFILE")"
  exit 0
fi

if [[ ! -x "$FINETUNE" ]]; then
  echo "missing $FINETUNE — run setup-node.sh first"
  exit 1
fi

echo "==> launching self-train"
echo "    model=$MODEL"
echo "    dataset=$DATASET ($(wc -l < "$DATASET") rows)"
echo "    threads=$THREADS window=$WINDOW tokens/step=$TOKENS_STEP"
echo "    rounds=$ROUNDS log=$LOG"

setsid "$FINETUNE" --threads "$THREADS" self-train \
  --model "$MODEL" \
  --dataset "$DATASET" \
  --output "$TRAIN_OUT" \
  --rounds "$ROUNDS" \
  --prompts-per-round "$PROMPTS" \
  --epochs-per-round "$EPOCHS" \
  --lora-rank "$LORA_RANK" \
  --learning-rate "$LR" \
  --max-seq-len "$MAX_SEQ" \
  --window "$WINDOW" \
  --tokens-per-step "$TOKENS_STEP" \
  --max-new-tokens 256 \
  --temperature 0.7 \
  --self-critique \
  --checkpoint-every 100 \
  --eval-split 0.05 \
  >> "$LOG" 2>&1 &

echo $! > "$PIDFILE"
disown || true

sleep 2
echo "self-train pid=$(cat "$PIDFILE")"
tail -20 "$LOG" || true

if [[ -n "${HF_TOKEN:-}" ]] && [[ -n "${HF_REPO:-}" ]]; then
  echo "==> scheduling HF upload watcher"
  nohup bash "$SCRIPT_DIR/upload-hf.sh" >> "$OUT_DIR/upload-hf.log" 2>&1 &
fi
