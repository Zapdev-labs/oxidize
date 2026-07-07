#!/usr/bin/env python3
"""Stream coding-agent HF datasets into oxidize-finetuning JSONL."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from datasets import load_dataset

IM_END = "<|im_end|>"


def messages_to_text(msgs: list[dict]) -> str:
    parts: list[str] = []
    for m in msgs:
        role = m.get("role", "user")
        content = m.get("content", "")
        if isinstance(content, list):
            content = " ".join(
                x.get("text", "") if isinstance(x, dict) else str(x) for x in content
            )
        parts.append(f"<|im_start|>{role}\n{content}{IM_END}\n")
    return "".join(parts)


def normalize(row: dict) -> dict | None:
    for key in ("messages", "conversations", "trajectory"):
        val = row.get(key)
        if isinstance(val, list) and val:
            return {"text": messages_to_text(val)}
    instruction = row.get("instruction", "")
    inp = row.get("input", "")
    out = row.get("output", "")
    if instruction or out:
        user = instruction if not inp else f"{instruction}\n{inp}"
        return {
            "text": f"<|im_start|>user\n{user}{IM_END}\n<|im_start|>assistant\n{out}{IM_END}\n"
        }
    return None


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("seed_coding_agent.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)

    sources = [
        ("Nexlab/fable5-agentic-coding-sft", "train", 1200),
        ("TIGER-Lab/SWE-Next-SFT-Trajectories", "train", 400),
        ("nvidia/Open-SWE-Traces", "openhands", 400),
    ]

    written = 0
    with out.open("w", encoding="utf-8") as f:
        for ds_id, split, limit in sources:
            if written >= 2000:
                break
            try:
                ds = load_dataset(ds_id, split=split, streaming=True)
            except Exception as exc:
                print(f"skip {ds_id}/{split}: {exc}")
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
            print(f"{ds_id}/{split}: +{n} (total {written})")

    if written < 50:
        fallback = [
            "Implement a Rust function that finds the longest palindromic substring.",
            "Debug a race condition in an async Tokio service with shared state.",
            "Write pytest tests for a FastAPI endpoint that uploads files.",
        ]
        with out.open("a", encoding="utf-8") as f:
            for p in fallback:
                f.write(
                    json.dumps(
                        {
                            "text": f"<|im_start|>user\n{p}{IM_END}\n"
                            f"<|im_start|>assistant\nHere is a careful answer to: {p}{IM_END}\n"
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                written += 1

    print(f"wrote {written} rows -> {out}")


if __name__ == "__main__":
    main()
