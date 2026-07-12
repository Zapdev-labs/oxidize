#!/usr/bin/env bash
# BF16 LoRA SFT on A100 via Unsloth (true GPU training for Qwen3.5 hybrid).
set -euo pipefail

OUT_DIR="${OUT_DIR:-$HOME/models/qwen35-9b-agent}"
HF_DIR="${HF_DIR:-$OUT_DIR/hf}"
DATASET="${TRAIN_DATASET:-$OUT_DIR/seed_coding_agent.jsonl}"
TRAIN_OUT="${TRAIN_OUT:-$OUT_DIR/unsloth-lora}"
LOG="${TRAIN_LOG:-$OUT_DIR/unsloth-train.log}"
PIDFILE="${TRAIN_PIDFILE:-$OUT_DIR/unsloth-train.pid}"
MODEL_ID="${HF_MODEL_ID:-empero-ai/Qwythos-9B-Claude-Mythos-5-1M}"

EPOCHS="${EPOCHS_PER_ROUND:-1}"
MAX_SEQ="${TRAIN_MAX_SEQ_LEN:-2048}"
LORA_RANK="${LORA_RANK:-32}"
LR="${LEARNING_RATE:-2e-4}"

mkdir -p "$OUT_DIR" "$TRAIN_OUT" "$(dirname "$LOG")"

if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "unsloth train already running pid=$(cat "$PIDFILE")"
  exit 0
fi

# Stop CPU/GPU oxidize trainers
for pf in "$OUT_DIR/self-train.pid" "$OUT_DIR/gpu-finetune.pid"; do
  if [[ -f "$pf" ]]; then
    pid=$(cat "$pf" 2>/dev/null || true)
    kill "$pid" 2>/dev/null || true
  fi
done

if [[ ! -f "$HF_DIR/config.json" ]]; then
  echo "==> fetching config.json for $MODEL_ID"
  python3 - <<PY
from huggingface_hub import hf_hub_download
import os, shutil
token = os.environ.get("HF_TOKEN")
path = hf_hub_download(repo_id="${MODEL_ID}", filename="config.json", token=token)
shutil.copy(path, "${HF_DIR}/config.json")
print("config.json -> ${HF_DIR}/config.json")
PY
fi

pip install -q "unsloth" "transformers>=5.0.0" peft datasets trl accelerate 2>/dev/null || \
  pip3 install -q "unsloth" "transformers>=5.0.0" peft datasets trl accelerate

echo "==> launching Unsloth GPU LoRA"
setsid python3 - <<PY >> "$LOG" 2>&1 &
import os, json
from pathlib import Path
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig

hf_dir = Path("${HF_DIR}")
out = Path("${TRAIN_OUT}")
out.mkdir(parents=True, exist_ok=True)
data_path = "${DATASET}"

max_seq = int("${MAX_SEQ}")
rank = int("${LORA_RANK}")
lr = float("${LR}")
epochs = float("${EPOCHS}")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=str(hf_dir),
    max_seq_length=max_seq,
    load_in_4bit=False,
    load_in_16bit=True,
    full_finetuning=False,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=rank,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=rank * 2,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
    max_seq_length=max_seq,
)

ds = load_dataset("json", data_files=data_path, split="train")

def fmt(example):
    return {"text": example["text"]}

ds = ds.map(fmt)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=ds,
    args=SFTConfig(
        output_dir=str(out),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=epochs,
        learning_rate=lr,
        logging_steps=10,
        save_steps=200,
        warmup_steps=20,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        bf16=True,
        max_seq_length=max_seq,
        dataset_text_field="text",
        report_to="none",
    ),
)

print("unsloth-train: starting GPU SFT", flush=True)
trainer.train()
model.save_pretrained(str(out / "adapter"))
tokenizer.save_pretrained(str(out / "adapter"))
(out / "done.json").write_text(json.dumps({"status": "finished"}))
print("unsloth-train: finished", flush=True)
PY

echo $! > "$PIDFILE"
disown || true
sleep 5
echo "unsloth-train pid=$(cat "$PIDFILE")"
tail -20 "$LOG" || true

if [[ -n "${HF_TOKEN:-}" ]] && [[ -n "${HF_REPO:-}" ]]; then
  TRAIN_OUT="$TRAIN_OUT" TRAIN_LOG="$LOG" TRAIN_PIDFILE="$PIDFILE" \
    nohup bash "$(dirname "$0")/upload-hf.sh" >> "$OUT_DIR/upload-unsloth.log" 2>&1 &
fi
