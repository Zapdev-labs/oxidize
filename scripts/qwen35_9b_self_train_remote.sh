#!/usr/bin/env bash
# Convert empero-ai/Qwythos-9B-Claude-Mythos-5-1M safetensors -> neutral GGUF, then self-train SFT.
# Usage: scripts/qwen35_9b_self_train_remote.sh [ssh-host]
set -euo pipefail

HOST="${1:-ai@192.168.1.121}"

ssh "$HOST" bash -s <<'REMOTE'
set -euo pipefail

REPO="$HOME/oxidize"
HF_ID="empero-ai/Qwythos-9B-Claude-Mythos-5-1M"
HF_DIR="$HOME/models/qwen35-9b-agent/hf"
OUT_DIR="$HOME/models/qwen35-9b-agent"
F16="$OUT_DIR/qwen35-9b-instruct-f16.gguf"
DATASET="$OUT_DIR/seed_coding_agent.jsonl"
TRAIN_OUT="$OUT_DIR/self-train-out"
LOG="$OUT_DIR/self-train.log"
THREADS="${TRAIN_THREADS:-48}"
ROUNDS="${SELF_TRAIN_ROUNDS:-5}"
PROMPTS="${PROMPTS_PER_ROUND:-12}"
EPOCHS="${EPOCHS_PER_ROUND:-2}"

export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
export HF_HUB_ENABLE_HF_TRANSFER=1

mkdir -p "$HF_DIR" "$OUT_DIR" "$(dirname "$LOG")"

if ! python3 -c 'import huggingface_hub, datasets' 2>/dev/null; then
  echo "==> installing huggingface_hub + datasets"
  python3 -m pip install --user --break-system-packages -q huggingface_hub hf_transfer datasets
fi

CONVERT="$REPO/target/release/oxidize-convert"
FINETUNE="$REPO/target/release/oxidize-finetuning"
[[ -x "$CONVERT" ]] || CONVERT="$REPO/bin/oxidize-convert"

if [[ ! -x "$CONVERT" ]]; then
  echo "==> building oxidize-convert (release)"
  cd "$REPO"
  cargo build --release -p oxidize-convert
fi
if [[ ! -x "$FINETUNE" ]]; then
  echo "==> building oxidize-finetuning (release)"
  cd "$REPO"
  cargo build --release -p oxidize-finetuning
fi

if [[ ! -f "$HF_DIR/config.json" ]] || [[ ! -f "$HF_DIR/model.safetensors" ]]; then
  echo "==> downloading $HF_ID safetensors (~18GB)"
  hf download "$HF_ID" \
    --local-dir "$HF_DIR" \
    --include "config.json" "model.safetensors" "tokenizer.json" "tokenizer_config.json" \
              "chat_template.jinja" "generation_config.json" "preprocessor_config.json"
else
  echo "==> HF weights already in $HF_DIR"
fi

echo "==> stripping branding from HF chat template + metadata"
python3 - <<'PY'
import json, re
from pathlib import Path

hf = Path.home() / "models/qwen35-9b-agent/hf"

# Neutralize chat template (remove Qwythos / Empero identity injection).
for name in ("chat_template.jinja",):
    p = hf / name
    if not p.exists():
        continue
    text = p.read_text(encoding="utf-8")
    text = re.sub(
        r"\{%- set qwythos_identity = .*?%\}\s*",
        "",
        text,
        flags=re.DOTALL,
    )
    text = text.replace("qwythos_identity", '""')
    text = re.sub(r"\+\s*qwythos_identity", "", text)
    text = re.sub(r"\\n\\n'\s*\+\s*qwythos_identity", "", text)
    text = re.sub(r"Qwythos|Empero AI|empero\.org", "", text, flags=re.IGNORECASE)
    p.write_text(text, encoding="utf-8")

cfg = hf / "config.json"
if cfg.exists():
    data = json.loads(cfg.read_text(encoding="utf-8"))
    data.pop("_name_or_path", None)
    cfg.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

tok_cfg = hf / "tokenizer_config.json"
if tok_cfg.exists():
    data = json.loads(tok_cfg.read_text(encoding="utf-8"))
    if "chat_template" in data and isinstance(data["chat_template"], str):
        t = data["chat_template"]
        t = re.sub(r"\{%- set qwythos_identity = .*?%\}\s*", "", t, flags=re.DOTALL)
        t = re.sub(r"Qwythos|Empero AI|empero\.org", "", t, flags=re.IGNORECASE)
        data["chat_template"] = t
    tok_cfg.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

print("branding stripped from HF sidecar files")
PY

if [[ ! -f "$F16" ]]; then
  echo "==> oxidize-convert safetensors -> F16 GGUF (neutral name)"
  /usr/bin/time -f "convert wall=%e s" \
    "$CONVERT" --input "$HF_DIR" --output "$F16" --arch qwen3_5
  ls -lh "$F16"
else
  echo "==> F16 GGUF exists: $(ls -lh "$F16")"
fi

echo "==> scrub GGUF metadata branding"
python3 - <<'PY'
import re, struct, sys
from pathlib import Path

path = Path.home() / "models/qwen35-9b-agent/qwen35-9b-instruct-f16.gguf"
data = bytearray(path.read_bytes())

BRANDS = re.compile(
    rb"Qwythos|Empero[\x00-\xff]{0,4}AI|empero\.org|Claude-Mythos",
    re.IGNORECASE,
)
NEUTRAL = {
    b"Qwythos-9B-Claude-Mythos-5-1M": b"qwen3.5-9b-instruct",
    b"Qwythos-9B": b"qwen3.5-9b-instruct",
    b"empero-ai/Qwythos-9B-Claude-Mythos-5-1M": b"qwen3.5-9b-instruct",
}

changed = 0
for old, new in NEUTRAL.items():
    if old in data:
        data = data.replace(old, new.ljust(len(old), b"\x00")[: len(old)])
        changed += 1

# Best-effort: zero-out remaining brand tokens inside string payloads only.
for m in BRANDS.finditer(bytes(data)):
    span = slice(m.start(), m.end())
    data[span] = b" " * (m.end() - m.start())
    changed += 1

path.write_bytes(data)
print(f"gguf metadata scrub: {changed} replacements")
PY

if [[ ! -f "$DATASET" ]] || [[ "$(wc -l < "$DATASET")" -lt 100 ]]; then
  echo "==> building seed coding-agent JSONL (Nexlab/fable5 + SWE trajectories)"
  python3 - <<'PY'
import json
from pathlib import Path

from datasets import load_dataset

out = Path.home() / "models/qwen35-9b-agent/seed_coding_agent.jsonl"
out.parent.mkdir(parents=True, exist_ok=True)
IM_END = "<|im_end|>"

def messages_to_text(msgs):
    parts = []
    for m in msgs:
        role = m.get("role", "user")
        content = m.get("content", "")
        if isinstance(content, list):
            content = " ".join(
                x.get("text", "") if isinstance(x, dict) else str(x) for x in content
            )
        parts.append(f"<|im_start|>{role}\n{content}{IM_END}\n")
    return "".join(parts)

def normalize(row):
    if isinstance(row.get("messages"), list) and row["messages"]:
        return {"text": messages_to_text(row["messages"])}
    if isinstance(row.get("conversations"), list) and row["conversations"]:
        return {"text": messages_to_text(row["conversations"])}
    if isinstance(row.get("trajectory"), list) and row["trajectory"]:
        return {"text": messages_to_text(row["trajectory"])}
    instruction = row.get("instruction", "")
    inp = row.get("input", "")
    out = row.get("output", "")
    if instruction or out:
        user = instruction if not inp else f"{instruction}\n{inp}"
        return {
            "text": f"<|im_start|>user\n{user}{IM_END}\n<|im_start|>assistant\n{out}{IM_END}\n"
        }
    return None

sources = [
    ("Nexlab/fable5-agentic-coding-sft", "train", 1200),
    ("TIGER-Lab/SWE-Next-SFT-Trajectories", "train", 400),
    ("nvidia/Open-SWE-Traces", "train", 400),
]

written = 0
with out.open("w", encoding="utf-8") as f:
    for ds_id, split, limit in sources:
        if written >= 2000:
            break
        try:
            ds = load_dataset(ds_id, split=split, streaming=True)
        except Exception as e:
            print(f"skip {ds_id}: {e}")
            continue
        n = 0
        for row in ds:
            rec = normalize(row)
            if not rec or len(rec["text"]) < 40:
                continue
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1
            n += 1
            if n >= limit or written >= 2000:
                break
        print(f"{ds_id}: +{n} (total {written})")

if written < 50:
    fallback = [
        "Implement a Rust function that finds the longest palindromic substring.",
        "Debug a race condition in an async Tokio service with shared state.",
        "Write pytest tests for a FastAPI endpoint that uploads files.",
        "Refactor a monolithic Python script into modules with clear interfaces.",
        "Explain how to add LoRA adapters to a transformer LM head only.",
    ]
    with out.open("a", encoding="utf-8") as f:
        for p in fallback:
            f.write(
                json.dumps(
                    {
                        "text": f"<|im_start|>user\n{p}{IM_END}\n<|im_start|>assistant\n"
                        f"Here is a careful answer to: {p}{IM_END}\n"
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            written += 1

print(f"wrote {written} rows -> {out}")
PY
else
  echo "==> seed dataset exists ($(wc -l < "$DATASET") rows)"
fi

echo "==> starting self-train (rounds=$ROUNDS prompts/round=$PROMPTS epochs/round=$EPOCHS threads=$THREADS)"
echo "    log: $LOG"

nohup "$FINETUNE" --threads "$THREADS" self-train \
  --model "$F16" \
  --dataset "$DATASET" \
  --output "$TRAIN_OUT" \
  --rounds "$ROUNDS" \
  --prompts-per-round "$PROMPTS" \
  --epochs-per-round "$EPOCHS" \
  --lora-rank 32 \
  --learning-rate 1.5e-4 \
  --max-seq-len 4096 \
  --window 128 \
  --tokens-per-step 512 \
  --max-new-tokens 256 \
  --temperature 0.7 \
  --self-critique \
  --checkpoint-every 100 \
  --eval-split 0.05 \
  > "$LOG" 2>&1 &

echo "self-train pid=$!"
sleep 2
tail -20 "$LOG" || true
echo "==> monitor: ssh ai@192.168.1.121 tail -f $LOG"
REMOTE
