#!/usr/bin/env bash
# Download GLM-5.2 + EAGLE3 draft into separate oxidize-ready directories.
# Requires: huggingface-cli (`pip install -U huggingface_hub[cli]`)
# Auth: export HF_TOKEN=hf_...  (or run `hf auth login --token $HF_TOKEN`)

set -euo pipefail

ROOT="${GLM52_ROOT:-$HOME/models/glm-5.2}"
TARGET_REPO="${GLM52_TARGET_REPO:-unsloth/GLM-5.2-GGUF}"
TARGET_QUANT="${GLM52_QUANT:-UD-IQ1_M}"
EAGLE3_DRAFT_REPO="${GLM52_EAGLE3_REPO:-AQ-MedAI/GLM-5.1-eagle3}"
OXIDIZE_REPO="${OXIDIZE_REPO:-$HOME/oxidize}"
HF_BIN="${HF_BIN:-$HOME/.venvs/hf/bin/hf}"

export PATH="$(dirname "$HF_BIN"):$HOME/.cargo/bin:$PATH"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "error: set HF_TOKEN before running (hf auth login --token \$HF_TOKEN)" >&2
  exit 1
fi

export HF_HUB_ENABLE_HF_TRANSFER=1
HF_BIN="${HF_BIN:-$HOME/.venvs/hf/bin/hf}"
if [[ ! -x "$HF_BIN" ]]; then
  HF_BIN="$(command -v hf || true)"
fi
if [[ -z "$HF_BIN" ]]; then
  echo "error: hf CLI not found; install with: python3 -m venv ~/.venvs/hf && ~/.venvs/hf/bin/pip install huggingface_hub hf_transfer" >&2
  exit 1
fi

"$HF_BIN" auth login --token "$HF_TOKEN" --add-to-git-credential 2>/dev/null || true

mkdir -p "$ROOT"/{target,eagle3/draft,quantspec,logs}

echo "==> downloading EAGLE3 draft (background): $EAGLE3_DRAFT_REPO -> $ROOT/eagle3/draft"
"$HF_BIN" download "$EAGLE3_DRAFT_REPO" \
  --local-dir "$ROOT/eagle3/draft" \
  >"$ROOT/logs/eagle3-download.log" 2>&1 &
EAGLE3_PID=$!

echo "==> downloading target GGUF: $TARGET_REPO ($TARGET_QUANT) -> $ROOT/target"
"$HF_BIN" download "$TARGET_REPO" \
  --include "${TARGET_QUANT}/*" \
  --local-dir "$ROOT/target"

wait "$EAGLE3_PID" || {
  echo "warning: EAGLE3 draft download failed; see $ROOT/logs/eagle3-download.log" >&2
}

OXIDIZE_SCRIPTS="${OXIDIZE_REPO}/scripts/glm-5.2-eagle3/prepare-draft.py"
if [[ -f "$OXIDIZE_SCRIPTS" && -f "$ROOT/eagle3/draft/config.json" ]]; then
  echo "==> patching EAGLE3 draft for GLM-5.2 extract layers"
  python3 "$OXIDIZE_SCRIPTS" --draft-dir "$ROOT/eagle3/draft"
fi

ln -sfn "$ROOT/target" "$ROOT/eagle3/target"
ln -sfn "$ROOT/target" "$ROOT/quantspec/target"

cat >"$ROOT/manifest.json" <<EOF
{
  "model": "GLM-5.2",
  "target_repo": "$TARGET_REPO",
  "target_quant": "$TARGET_QUANT",
  "target_dir": "$ROOT/target",
  "eagle3": {
    "draft_repo": "$EAGLE3_DRAFT_REPO",
    "draft_dir": "$ROOT/eagle3/draft",
    "target_link": "$ROOT/eagle3/target",
    "oxidize_run": "oxidize run $ROOT/eagle3/target/${TARGET_QUANT}/*.gguf --draft-model $ROOT/eagle3/draft/model.safetensors --draft-tokens 4 --tokenizer-model $ROOT/eagle3/target/${TARGET_QUANT}"
  },
  "quantspec": {
    "target_link": "$ROOT/quantspec/target",
    "oxidize_run": "oxidize run $ROOT/quantspec/target/${TARGET_QUANT}/*.gguf --quantspec --draft-tokens 6"
  }
}
EOF

if [[ -d "$OXIDIZE_REPO" ]]; then
  echo "==> building oxidize in $OXIDIZE_REPO"
  (cd "$OXIDIZE_REPO" && make build)
  echo "oxidize binary: $OXIDIZE_REPO/target/release/oxidize"
fi

echo "==> done. Layout:"
find "$ROOT" -maxdepth 3 -type d | sort
echo "manifest: $ROOT/manifest.json"
