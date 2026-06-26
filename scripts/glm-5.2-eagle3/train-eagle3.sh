#!/usr/bin/env bash
# Train a native GLM-5.2 EAGLE3 draft with SpecForge (GPU required).
# Example: run on Modal B200/H100 or any CUDA host with ~80GB+ VRAM for GLM-5.2 target.

set -euo pipefail

TARGET_REPO="${GLM52_TARGET_REPO:-zai-org/GLM-5.2}"
DRAFT_OUT="${GLM52_EAGLE3_OUT:-$HOME/models/glm-5.2/eagle3/draft-trained}"
SPECFORGE_REPO="${SPECFORGE_REPO:-$HOME/SpecForge}"
HF_BIN="${HF_BIN:-$(command -v hf || true)}"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "error: export HF_TOKEN before training" >&2
  exit 1
fi

if [[ ! -d "$SPECFORGE_REPO/.git" ]] || [[ ! -f "$SPECFORGE_REPO/pyproject.toml" ]]; then
  echo "==> cloning SpecForge"
  git clone https://github.com/sgl-project/SpecForge.git "$SPECFORGE_REPO"
fi

if [[ ! -f "$SPECFORGE_REPO/pyproject.toml" ]]; then
  echo "error: $SPECFORGE_REPO does not look like a SpecForge checkout" >&2
  exit 1
fi

mkdir -p "$DRAFT_OUT"

cat <<EOF
GLM-5.2 EAGLE3 training scaffold
================================
Target:  $TARGET_REPO
Output:  $DRAFT_OUT
Trainer: $SPECFORGE_REPO

No upstream SpecForge recipe ships for GLM-5.2 yet. Start from the closest
DeepSeek-V3 / Qwen3 EAGLE3 scripts and set:
  - hidden_size=6144, num_hidden_layers=78
  - extract_layers=[2,39,75]
  - vocab_size=154880, draft_vocab_size=32000

After training, copy config.json + model.safetensors into:
  $DRAFT_OUT

Then run:
  python3 scripts/glm-5.2-eagle3/prepare-draft.py --draft-dir $DRAFT_OUT
EOF

if [[ -n "$HF_BIN" ]]; then
  echo "==> prefetching target config for layer hints"
  "$HF_BIN" download "$TARGET_REPO" config.json --local-dir "$DRAFT_OUT/target-config"
fi

echo "See SpecForge docs: https://github.com/sgl-project/SpecForge"
