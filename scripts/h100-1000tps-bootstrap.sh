#!/usr/bin/env bash
# Bootstrap a Prime Intellect 2xH100 pod for the oxidize-c Gemma 4 31B
# 1000 TPS benchmark. Run ON the pod (ubuntu + CUDA image).
set -euo pipefail

MODEL_DIR=${MODEL_DIR:-$HOME/models}
REPO_DIR=${REPO_DIR:-$HOME/oxidize}
BRANCH=${BRANCH:-c-port-1000tps}

mkdir -p "$MODEL_DIR"

# Model + MTP draft head (parallel)
dl() { # url dest
  [ -f "$2" ] || curl -L --retry 5 -C - "$1" -o "$2.part" && mv -f "$2.part" "$2" 2>/dev/null || true
}
# NOTE: the freakyskittle AL5_XS upload is broken (degenerate output, verified
# 2026-07-11 against llama.cpp). Download the official Q4_K_M and requant to
# AL5_XS on the pod instead.
dl "https://huggingface.co/unsloth/gemma-4-31B-it-GGUF/resolve/main/gemma-4-31B-it-Q4_K_M.gguf" \
   "$MODEL_DIR/gemma-4-31B-it-Q4_K_M.gguf" &
dl "https://huggingface.co/unsloth/gemma-4-31B-it-GGUF/resolve/main/MTP/mtp-gemma-4-31B-it-BF16.gguf" \
   "$MODEL_DIR/mtp-gemma-4-31B-it-BF16.gguf" &

# Toolchain
command -v gcc >/dev/null || { apt-get update && apt-get install -y build-essential; }
nvidia-smi
nvcc --version || { echo "need CUDA toolkit image"; exit 1; }

wait  # downloads

cd "$REPO_DIR/oxidize-c"
make clean && make cuda -j && make requant

[ -f "$MODEL_DIR/gemma-4-31B-it-AL5_XS.gguf" ] || \
  ./oxidize-c-requant --input "$MODEL_DIR/gemma-4-31B-it-Q4_K_M.gguf" \
    --output "$MODEL_DIR/gemma-4-31B-it-AL5_XS.gguf" --target AL5_XS

# Baseline single-GPU, then measure
./oxidize-c-cuda --model "$MODEL_DIR/gemma-4-31B-it-AL5_XS.gguf" \
  --prompt "Explain how a transformer decoder generates text." \
  --max-tokens 256 --bench | tee /tmp/bench-1gpu.txt
