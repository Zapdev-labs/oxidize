#!/usr/bin/env bash
# One-time setup on a Prime A100 pod: Rust toolchain, oxidize build, HF model, seed data.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${OXIDIZE_REPO:-$HOME/oxidize}"
HF_ID="${HF_MODEL:-empero-ai/Qwythos-9B-Claude-Mythos-5-1M}"
OUT_DIR="${OUT_DIR:-$HOME/models/qwen35-9b-agent}"
HF_DIR="$OUT_DIR/hf"
F16="$OUT_DIR/qwen35-9b-instruct-f16.gguf"
Q8="$OUT_DIR/qwen35-9b-instruct-q8_0.gguf"
DATASET="$OUT_DIR/seed_coding_agent.jsonl"
THREADS="${TRAIN_THREADS:-$(nproc)}"

export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export RUSTUP_HOME="$HOME/.rustup"
export CARGO_HOME="$HOME/.cargo"

mkdir -p "$OUT_DIR" "$HF_HOME"

echo "==> system deps"
sudo apt-get update -qq
sudo apt-get install -y -qq build-essential pkg-config libssl-dev git curl python3 python3-pip

if ! command -v cargo >/dev/null 2>&1; then
  echo "==> installing Rust stable"
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
fi

if ! python3 -c 'import huggingface_hub, datasets' 2>/dev/null; then
  python3 -m pip install --user -q huggingface_hub hf_transfer datasets
fi

if [[ ! -d "$REPO/.git" ]]; then
  echo "==> cloning oxidize"
  git clone --depth 1 -b oxidize-c-gemma-bench \
    https://github.com/Zapdev-labs/oxidize.git "$REPO" || \
  git clone --depth 1 https://github.com/Zapdev-labs/oxidize.git "$REPO"
fi

cd "$REPO"
echo "==> building oxidize-convert + oxidize-quantize + oxidize-finetuning (release)"
cargo build --release -p oxidize-convert -p oxidize-quantize -p oxidize-finetuning

CONVERT="$REPO/target/release/oxidize-convert"
QUANTIZE="$REPO/target/release/oxidize-quantize"

if [[ ! -f "$HF_DIR/model.safetensors" ]]; then
  echo "==> downloading $HF_ID (~18GB safetensors)"
  hf download "$HF_ID" \
    --local-dir "$HF_DIR" \
    --include "config.json" "model.safetensors" "tokenizer.json" "tokenizer_config.json" \
              "chat_template.jinja" "generation_config.json"
fi

echo "==> strip branding from HF sidecars"
python3 "$SCRIPT_DIR/strip_branding.py" "$HF_DIR"

if [[ ! -f "$F16" ]]; then
  echo "==> convert safetensors -> F16 GGUF"
  /usr/bin/time -f "convert wall=%e s" \
    "$CONVERT" --input "$HF_DIR" --output "$F16" --arch qwen3_5
fi

if [[ ! -f "$Q8" ]]; then
  echo "==> quantize F16 -> Q8_0 (faster CPU SFT, minimal quality loss)"
  /usr/bin/time -f "q8 quant wall=%e s" \
    "$QUANTIZE" --input "$F16" --output "$Q8" --target Q8_0 --threads "$THREADS"
fi

python3 "$SCRIPT_DIR/strip_branding.py" --gguf "$F16" "$Q8"

if [[ ! -f "$DATASET" ]] || [[ "$(wc -l < "$DATASET")" -lt 100 ]]; then
  echo "==> building seed coding-agent JSONL"
  python3 "$SCRIPT_DIR/build_seed_dataset.py" "$DATASET"
fi

echo ""
echo "Setup complete."
echo "  base (train): $Q8"
echo "  dataset:      $DATASET ($(wc -l < "$DATASET") rows)"
echo "  finetune:     $REPO/target/release/oxidize-finetuning"
echo ""
echo "Launch:"
echo "  HF_TOKEN=... HF_REPO=freakyskittle/qwen35-9b-self-train-lora bash $SCRIPT_DIR/run-self-train.sh"
