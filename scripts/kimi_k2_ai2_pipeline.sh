#!/usr/bin/env bash
set -euo pipefail

# Kimi-K2.6 + Kimi-K2.7-Code merge/prune/GGUF pipeline for ai-2.
#
# Usage:
#   scripts/kimi_k2_ai2_pipeline.sh probe
#   scripts/kimi_k2_ai2_pipeline.sh prep
#   HF_TOKEN=... scripts/kimi_k2_ai2_pipeline.sh download
#   scripts/kimi_k2_ai2_pipeline.sh merge
#   scripts/kimi_k2_ai2_pipeline.sh eval-merge
#   scripts/kimi_k2_ai2_pipeline.sh prune
#   scripts/kimi_k2_ai2_pipeline.sh eval-prune
#   scripts/kimi_k2_ai2_pipeline.sh gguf
#   scripts/kimi_k2_ai2_pipeline.sh smoke
#
# Destructive cleanup is opt-in:
#   CONFIRM_DELETE=1 scripts/kimi_k2_ai2_pipeline.sh cleanup-sources
#   CONFIRM_DELETE=1 scripts/kimi_k2_ai2_pipeline.sh cleanup-merged

ROOT="${KIMI_ROOT:-/data/kimi-k2}"
SRC_CODE="${KIMI_K27_DIR:-$ROOT/checkpoints/k2.7-code}"
SRC_BASE="${KIMI_K26_DIR:-$ROOT/checkpoints/k2.6}"
MERGED="${KIMI_MERGED_DIR:-$ROOT/k2-merged}"
PRUNED="${KIMI_PRUNED_DIR:-$ROOT/k2-merged-pruned}"
LLAMA_CPP="${LLAMA_CPP_DIR:-$ROOT/llama.cpp}"
OXIDIZE="${OXIDIZE_DIR:-$ROOT/oxidize-oxk}"
VENV="${KIMI_VENV:-$ROOT/.venv}"
CALIB="${KIMI_CALIB:-$ROOT/calib-corpus-mixed}"
LOG_DIR="$ROOT/logs"
MERGE_CONFIG="$ROOT/merge-config.yaml"
ROUTING_STATS="$ROOT/routing-stats.json"
POST_MERGE_EVAL="$ROOT/eval-post-merge.json"
POST_PRUNE_EVAL="$ROOT/eval-post-prune.json"
BF16_GGUF="$ROOT/k2-merged-pruned-bf16.gguf"
Q8_GGUF="$ROOT/k2-merged-Q8_0.gguf"
Q4_GGUF="$ROOT/k2-merged-Q4_K_M.gguf"

export ROOT SRC_CODE SRC_BASE MERGED PRUNED LLAMA_CPP OXIDIZE VENV CALIB LOG_DIR \
  MERGE_CONFIG ROUTING_STATS POST_MERGE_EVAL POST_PRUNE_EVAL BF16_GGUF Q8_GGUF Q4_GGUF

mkdir -p "$ROOT" "$ROOT/checkpoints" "$LOG_DIR"

# Non-login SSH shells do not automatically see rustup's PATH update.
# Source it early so prep is idempotent after the first Rust install.
# shellcheck disable=SC1091
[ -f "$HOME/.cargo/env" ] && . "$HOME/.cargo/env"

log() { printf '[%(%Y-%m-%dT%H:%M:%S%z)T] %s\n' -1 "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
need() { command -v "$1" >/dev/null 2>&1 || die "missing required command: $1"; }

run_logged() {
  local name="$1"; shift
  log "running $name"
  "$@" 2>&1 | tee "$LOG_DIR/$name.log"
}

uv_bin() {
  if command -v uv >/dev/null 2>&1; then
    command -v uv
  elif [ -x "$HOME/.local/bin/uv" ]; then
    printf '%s\n' "$HOME/.local/bin/uv"
  else
    die "uv is not installed; run the prep stage first"
  fi
}

py() {
  "$(uv_bin)" run --python "$VENV/bin/python" python "$@"
}

probe() {
  log "host: $(hostname)"
  df -h /data 2>/dev/null || df -h "$ROOT"
  free -h
  python3 --version || true
  command -v hf || true
  command -v cmake || true
  command -v git || true
  command -v cargo || true
  command -v uv || true
}

prep() {
  need git
  need cmake
  need curl

  if ! command -v uv >/dev/null 2>&1 && [ ! -x "$HOME/.local/bin/uv" ]; then
    log "installing uv into ~/.local/bin"
    curl -LsSf https://astral.sh/uv/install.sh | sh
  fi
  local uv; uv="$(uv_bin)"

  if [ ! -x "$VENV/bin/python" ]; then
    log "creating Python 3.11 virtualenv with uv"
    "$uv" python install 3.11
    "$uv" venv --python 3.11 "$VENV"
  fi

  log "installing Python tooling"
  "$uv" pip install --python "$VENV/bin/python" \
    'mergekit[lazy]' huggingface_hub safetensors lm-eval datasets sentencepiece protobuf accelerate

  if [ ! -d "$LLAMA_CPP/.git" ]; then
    git clone https://github.com/ggml-org/llama.cpp "$LLAMA_CPP"
  else
    git -C "$LLAMA_CPP" pull --ff-only
  fi
  cmake -S "$LLAMA_CPP" -B "$LLAMA_CPP/build" -DGGML_NATIVE=ON -DLLAMA_CURL=ON
  cmake --build "$LLAMA_CPP/build" --config Release -j"$(nproc)"

  if [ -d "$OXIDIZE/.git" ]; then
    git -C "$OXIDIZE" pull --ff-only || true
  elif [ -d "$OXIDIZE" ]; then
    log "using existing non-git oxidize workspace at $OXIDIZE"
  else
    git clone https://github.com/Zapdev-labs/oxidize "$OXIDIZE" || \
      git clone https://github.com/Zapdev-labs/oxidize-oxk "$OXIDIZE"
  fi

  if ! command -v cargo >/dev/null 2>&1; then
    log "cargo not found; installing Rust with rustup"
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    # shellcheck disable=SC1091
    [ -f "$HOME/.cargo/env" ] && . "$HOME/.cargo/env"
  fi

  if command -v cargo >/dev/null 2>&1; then
    if command -v sfw >/dev/null 2>&1; then
      (cd "$OXIDIZE" && sfw cargo build --release -p oxidize-core -p oxidize-quantize)
      (cd "$OXIDIZE" && sfw cargo build --release -p oxidize-cli) || \
        log "oxidize-cli build failed; core/quantize are available, inspect CLI before smoke"
    else
      (cd "$OXIDIZE" && cargo build --release -p oxidize-core -p oxidize-quantize)
      (cd "$OXIDIZE" && cargo build --release -p oxidize-cli) || \
        log "oxidize-cli build failed; core/quantize are available, inspect CLI before smoke"
    fi
  else
    log "cargo not found; skipping oxidize build until Rust is installed"
  fi

  if [ ! -d "$ROOT/snapprune/.git" ]; then
    git clone https://github.com/Zapdev-labs/snapprune "$ROOT/snapprune" || \
      log "snapprune clone failed (private repo or missing auth); prune stage remains blocked"
  fi
  if [ -d "$ROOT/snapprune" ]; then
    if [ -f "$ROOT/snapprune/pyproject.toml" ] || [ -f "$ROOT/snapprune/setup.py" ]; then
      "$uv" pip install --python "$VENV/bin/python" -e "$ROOT/snapprune"
    elif [ -f "$ROOT/snapprune/python/pyproject.toml" ] || [ -f "$ROOT/snapprune/python/setup.py" ]; then
      "$uv" pip install --python "$VENV/bin/python" -e "$ROOT/snapprune/python"
    else
      log "snapprune has no Python package at repo root; skipping pip install"
    fi
    if [ -f "$ROOT/snapprune/rust/Cargo.toml" ] && command -v cargo >/dev/null 2>&1; then
      if command -v sfw >/dev/null 2>&1; then
        sfw cargo build --release --manifest-path "$ROOT/snapprune/rust/Cargo.toml" -p snapprune-cli
      else
        cargo build --release --manifest-path "$ROOT/snapprune/rust/Cargo.toml" -p snapprune-cli
      fi
    fi
  fi
}

download() {
  [ -n "${HF_TOKEN:-}" ] && "$VENV/bin/hf" auth login --token "$HF_TOKEN" || true
  run_logged download-k27 "$VENV/bin/hf" download moonshotai/Kimi-K2.7-Code --local-dir "$SRC_CODE"
  run_logged download-k26 "$VENV/bin/hf" download moonshotai/Kimi-K2.6 --local-dir "$SRC_BASE"
  verify_arch
  du -sh "$SRC_CODE" "$SRC_BASE"
}

verify_arch() {
  py - <<'PY'
import json, os, sys
code = os.environ.get('SRC_CODE')
base = os.environ.get('SRC_BASE')
if not code or not base:
    code = '/data/kimi-k2/checkpoints/k2.7-code'
    base = '/data/kimi-k2/checkpoints/k2.6'
a = json.load(open(os.path.join(code, 'config.json')))
b = json.load(open(os.path.join(base, 'config.json')))
keys = [
    'model_type', 'num_hidden_layers', 'num_experts', 'n_routed_experts',
    'num_experts_per_tok', 'n_group', 'topk_group', 'n_shared_experts',
    'hidden_size', 'moe_intermediate_size', 'intermediate_size', 'vocab_size'
]
bad = False
for k in keys:
    av, bv = a.get(k), b.get(k)
    ok = av == bv
    print(('OK ' if ok else 'BAD') + f' {k}: {av!r} vs {bv!r}')
    bad |= not ok and k not in {'model_type'}
if bad:
    raise SystemExit('architecture mismatch; refusing to merge')
PY
}

write_merge_config() {
  cat > "$MERGE_CONFIG" <<YAML
slices:
  - sources:
      - { model: $SRC_CODE, layer_range: [0, 61] }
      - { model: $SRC_BASE, layer_range: [0, 61] }
merge_method: slerp
base_model: $SRC_CODE
parameters:
  t:
    - { filter: self_attn, value: 0.3 }
    - { filter: mlp,       value: 0.5 }
    - { value: 0.4 }
dtype: bfloat16
YAML
  log "wrote $MERGE_CONFIG"
}

merge() {
  [ -d "$SRC_CODE" ] || die "missing $SRC_CODE; run download first"
  [ -d "$SRC_BASE" ] || die "missing $SRC_BASE; run download first"
  write_merge_config
  run_logged mergekit "$VENV/bin/mergekit-yaml" "$MERGE_CONFIG" "$MERGED" \
    --lazy-unpickle --allow-crimes --trust-remote-code --out-shard-size 5B --low-cpu-memory
}

eval_merge() {
  [ -d "$MERGED" ] || die "missing $MERGED; run merge first"
  run_logged eval-post-merge "$VENV/bin/python" -m lm_eval \
    --model hf --model_args "pretrained=$MERGED" \
    --tasks wikitext \
    --output_path "$POST_MERGE_EVAL"
}

prune() {
  [ -d "$MERGED" ] || die "missing $MERGED; run merge first"
  [ -e "$CALIB" ] || die "missing calibration corpus at $CALIB"
  command -v snapprune >/dev/null 2>&1 || [ -x "$VENV/bin/snapprune" ] || [ -x "$ROOT/snapprune/rust/target/release/snapprune" ] || die "snapprune CLI not available"
  local snap="snapprune"; [ -x "$VENV/bin/snapprune" ] && snap="$VENV/bin/snapprune"
  [ -x "$ROOT/snapprune/rust/target/release/snapprune" ] && snap="$ROOT/snapprune/rust/target/release/snapprune"
  local mode="${KIMI_PRUNE_MODE:-deep}"
  local ratio="${KIMI_PRUNE_RATIO:-0.3}"
  case "$mode" in
    deep)
      run_logged snapprune-deep "$snap" deep "$MERGED" \
        --calib-data "$CALIB" --ratio "$ratio" --output "$PRUNED"
      ;;
    swift)
      run_logged snapprune-swift "$snap" swift "$MERGED" \
        --calib-data "$CALIB" --calib-samples "${KIMI_CALIB_SAMPLES:-512}" \
        --ratio "$ratio" --output "$PRUNED"
      ;;
    flash)
      run_logged snapprune-flash "$snap" flash "$MERGED" --ratio "$ratio" --output "$PRUNED"
      ;;
    *) die "unknown KIMI_PRUNE_MODE=$mode (expected deep, swift, or flash)" ;;
  esac
}

eval_prune() {
  [ -d "$PRUNED" ] || die "missing $PRUNED; run prune first"
  run_logged eval-post-prune "$VENV/bin/python" -m lm_eval \
    --model hf --model_args "pretrained=$PRUNED" \
    --tasks wikitext \
    --output_path "$POST_PRUNE_EVAL"
}

gguf() {
  [ -d "$PRUNED" ] || die "missing $PRUNED; run prune first"
  run_logged convert-gguf "$VENV/bin/python" "$LLAMA_CPP/convert_hf_to_gguf.py" \
    "$PRUNED" --outfile "$BF16_GGUF" --outtype bf16
  run_logged quantize-q8 "$LLAMA_CPP/build/bin/llama-quantize" "$BF16_GGUF" "$Q8_GGUF" Q8_0
  run_logged quantize-q4 "$LLAMA_CPP/build/bin/llama-quantize" "$Q8_GGUF" "$Q4_GGUF" Q4_K_M
}

smoke() {
  [ -f "$Q4_GGUF" ] || die "missing $Q4_GGUF; run gguf first"
  run_logged llama-smoke "$LLAMA_CPP/build/bin/llama-cli" -m "$Q4_GGUF" \
    -p 'write quicksort in rust' -n 200
  if [ -x "$OXIDIZE/target/release/oxidize" ]; then
    run_logged oxidize-smoke "$OXIDIZE/target/release/oxidize" run "$Q4_GGUF" \
      --no-api --prompt 'write quicksort in rust'
  fi
}

cleanup_sources() {
  [ "${CONFIRM_DELETE:-0}" = "1" ] || die "set CONFIRM_DELETE=1 to delete source checkpoints"
  rm -rf "$SRC_CODE" "$SRC_BASE"
  df -h /data 2>/dev/null || df -h "$ROOT"
}

cleanup_merged() {
  [ "${CONFIRM_DELETE:-0}" = "1" ] || die "set CONFIRM_DELETE=1 to delete merged bf16 checkpoint"
  rm -rf "$MERGED"
  df -h /data 2>/dev/null || df -h "$ROOT"
}

case "${1:-probe}" in
  probe) probe ;;
  prep) prep ;;
  download) download ;;
  verify-arch) verify_arch ;;
  merge-config) write_merge_config ;;
  merge) merge ;;
  eval-merge) eval_merge ;;
  prune) prune ;;
  eval-prune) eval_prune ;;
  gguf) gguf ;;
  smoke) smoke ;;
  cleanup-sources) cleanup_sources ;;
  cleanup-merged) cleanup_merged ;;
  all) prep; download; merge; eval_merge; prune; eval_prune; gguf; smoke ;;
  *) die "unknown stage: $1" ;;
esac
