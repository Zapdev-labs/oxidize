#!/usr/bin/env python3
"""Prepare a GLM-5.2 EAGLE3 draft directory from the GLM-5.1 eagle3 bootstrap weights."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

DEFAULT_DRAFT_REPO = "AQ-MedAI/GLM-5.1-eagle3"
GLM52_TARGET_LAYERS = 78
GLM52_EXTRACT_LAYERS = [2, 39, 75]
GLM52_VOCAB = 154_880


def default_extract_layers(layer_count: int) -> list[int]:
    if layer_count < 4:
        return [0, 0, 0]
    return [2, layer_count // 2, layer_count - 3]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--draft-dir",
        type=Path,
        required=True,
        help="Directory containing model.safetensors + config.json",
    )
    parser.add_argument(
        "--target-layers",
        type=int,
        default=GLM52_TARGET_LAYERS,
        help="GLM-5.2 backbone layer count (excludes MTP head)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write patched config.json here (default: draft-dir/config.json)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional manifest.json path",
    )
    args = parser.parse_args()

    draft_dir = args.draft_dir.expanduser().resolve()
    config_path = draft_dir / "config.json"
    if not config_path.is_file():
        raise SystemExit(f"missing config.json in {draft_dir}")
    if not (draft_dir / "model.safetensors").is_file():
        raise SystemExit(f"missing model.safetensors in {draft_dir}")

    config = json.loads(config_path.read_text())
    extract_layers = default_extract_layers(args.target_layers)
    config["extract_layers"] = extract_layers
    config["vocab_size"] = GLM52_VOCAB
    config.setdefault("draft_vocab_size", config.get("draft_vocab_size", 32_000))
    config["glm52_bootstrap"] = {
        "source_repo": DEFAULT_DRAFT_REPO,
        "target_layers": args.target_layers,
        "extract_layers": extract_layers,
        "note": "GLM-5.1 EAGLE3 weights bootstrapped for GLM-5.2; train a native draft for best quality",
    }

    out_config = args.output or config_path
    out_config.parent.mkdir(parents=True, exist_ok=True)
    if out_config != config_path:
        shutil.copy2(config_path, out_config.with_suffix(".orig.json"))
    out_config.write_text(json.dumps(config, indent=2) + "\n")

    manifest = {
        "model": "GLM-5.2",
        "draft_source": DEFAULT_DRAFT_REPO,
        "draft_dir": str(draft_dir),
        "target_layers": args.target_layers,
        "extract_layers": extract_layers,
        "vocab_size": GLM52_VOCAB,
        "oxidize_run": (
            f"oxidize run <target.gguf> --draft-model {draft_dir} "
            f"--draft-tokens 4 --prompt \"Hello\""
        ),
    }
    manifest_path = args.manifest or (draft_dir / "glm-5.2-eagle3.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"patched {out_config}")
    print(f"extract_layers={extract_layers}")
    print(f"manifest {manifest_path}")


if __name__ == "__main__":
    main()
