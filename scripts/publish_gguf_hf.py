#!/usr/bin/env python3
"""Upload GGUF files to Hugging Face Hub."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

OXIDIZE_GH = "https://github.com/Zapdev-labs/oxidize"


def build_readme(repo: str, base_model: str) -> str:
    return f"""---
license: apache-2.0
license_link: https://ai.google.dev/gemma/docs/gemma_4_license
base_model: {base_model}
tags:
  - gguf
  - gemma4
  - gemma
  - oxidize
  - al-quant
  - text-generation
library_name: gguf
pipeline_tag: text-generation
---

# Gemma 4 31B — Oxidize AL Quants

**[Oxidize]({OXIDIZE_GH})** AL-family GGUF quants of
[google/gemma-4-31B-it](https://huggingface.co/google/gemma-4-31B-it).

Quantized with [`oxidize-quantize`]({OXIDIZE_GH}/tree/master/oxidize-quantize)
from Unsloth BF16 source weights (`unsloth/gemma-4-31B-it-GGUF`) — lossless
requant at the weight level (no quality loss from a prior quant).

## Files

| File | Type | ~Size | Bits | When to use |
|------|------|-------|------|-------------|
| `gemma-4-31B-it-AL5.gguf` | AL5 | 17 GB | ~4.5 | **Default** — same block size as Q4_0, lower RMSE |
| `gemma-4-31B-it-AL6.gguf` | AL6 | 20 GB | ~5.5 | Better quality than AL5, still compact |
| `gemma-4-31B-it-AL8.gguf` | AL8 | 31 GB | 8 | Near-F16 quality, largest AL variant |
| `gemma-4-31B-it-AL5_XS.gguf` | AL5_XS | 13 GB | ~3 | Smallest; tight VRAM (e.g. 16 GB GPU) |

### What is AL?

Oxidize **AL** quants are custom ggml types (IDs 240–243) that use **multi-seed
MSE-optimal per-block scales** at the same block layouts as Q4_0 / Q5_0 / Q8_0.
AL5 decodes identically to Q4_0 bitstream layout but encodes with lower
reconstruction error. Implementation:
[`al_family.rs`]({OXIDIZE_GH}/blob/master/oxidize-core/src/compute/quantization/al_family.rs).

## Quick start — Oxidize-patched llama.cpp

AL quant IDs 240–243 are not supported by upstream llama.cpp. This command
requires a fork patched with Oxidize's custom AL formats.

```bash
huggingface-cli download {repo} gemma-4-31B-it-AL5.gguf --local-dir .

./llama-cli -m gemma-4-31B-it-AL5.gguf \\
  -p "<|turn>user\\nHello<turn|>\\n<|turn>model\\n<|channel>final<|message|>" \\
  -n 256 --temp 0.7
```

## Quick start — oxidize-cpp (CPU)

```bash
git clone {OXIDIZE_GH}.git && cd oxidize/oxidize-cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j

./build/oxidize-cpp --model gemma-4-31B-it-AL5.gguf \\
  --prompt "Hello" --max-tokens 128 --auto
```

## Quick start — oxidize-c (CUDA)

```bash
git clone {OXIDIZE_GH}.git && cd oxidize/oxidize-c
make cuda

./oxidize-c-cuda --model gemma-4-31B-it-AL5.gguf \\
  --prompt "<|turn>user\\nHello<turn|>\\n<|turn>model\\n" --max-tokens 128
```

## Reproduce these quants

```bash
cargo build -p oxidize-quantize --release
./target/release/oxidize-quantize \\
  --input gemma-4-31B-it-BF16-00001-of-00002.gguf \\
  --output gemma-4-31B-it-AL5.gguf \\
  --target AL5 --threads 96
```

Remote pipeline:
[`scripts/gemma4_31b_al_remote.sh`]({OXIDIZE_GH}/blob/master/scripts/gemma4_31b_al_remote.sh)

## Links

- **Oxidize**: {OXIDIZE_GH}
- **Base model**: https://huggingface.co/google/gemma-4-31B-it
- **Source BF16 GGUF**: https://huggingface.co/unsloth/gemma-4-31B-it-GGUF

## License

Apache 2.0 (per [Gemma 4 license](https://ai.google.dev/gemma/docs/gemma_4_license)).
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish GGUF files to Hugging Face")
    parser.add_argument(
        "--repo", required=True, help="HF repo id, e.g. user/gemma-4-31B-it-AL-GGUF"
    )
    parser.add_argument("--files", nargs="*", default=[], help="GGUF paths to upload")
    parser.add_argument(
        "--readme-only",
        action="store_true",
        help="Update README.md only (no weight upload)",
    )
    parser.add_argument(
        "--private",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Create/update repo as private (default: true)",
    )
    parser.add_argument(
        "--base-model",
        default="google/gemma-4-31B-it",
        help="base_model metadata tag",
    )
    args = parser.parse_args()

    files = [Path(file) for file in args.files]
    if not args.readme_only and not files:
        raise SystemExit("no --files given (use --readme-only to update README only)")
    missing = [path for path in files if not path.is_file()]
    if missing:
        raise SystemExit(f"missing: {missing[0]}")

    from huggingface_hub import HfApi, create_repo

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    api = HfApi(token=token)

    create_repo(
        args.repo,
        repo_type="model",
        private=args.private,
        exist_ok=True,
        token=token,
    )
    if args.private:
        api.update_repo_settings(
            args.repo, private=True, repo_type="model", token=token
        )

    readme = build_readme(args.repo, args.base_model)
    api.upload_file(
        path_or_fileobj=readme.encode(),
        path_in_repo="README.md",
        repo_id=args.repo,
        repo_type="model",
        commit_message="Update model card",
        token=token,
    )

    if args.readme_only:
        print(
            f"readme updated ({'private' if args.private else 'public'}): https://huggingface.co/{args.repo}"
        )
        return

    for path in files:
        print(f"uploading {path.name} ({path.stat().st_size / 1e9:.2f} GB)...")
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=path.name,
            repo_id=args.repo,
            repo_type="model",
            commit_message=f"Add {path.name}",
            token=token,
        )
        print(f"  done: https://huggingface.co/{args.repo}/blob/main/{path.name}")

    print(
        f"published ({'private' if args.private else 'public'}): https://huggingface.co/{args.repo}"
    )


if __name__ == "__main__":
    main()
