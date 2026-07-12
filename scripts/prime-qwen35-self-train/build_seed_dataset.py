#!/usr/bin/env python3
"""Stream coding-agent HF datasets into oxidize-finetuning JSONL."""

from __future__ import annotations

import json
import sys
from pathlib import Path

IM_END = "<|im_end|>"
ALLOWED_ROLES = {"user", "assistant", "system", "tool"}
ROLE_ALIASES = {"human": "user", "gpt": "assistant"}
CONTROL_TOKENS = ("<|im_start|>", "<|im_end|>")


def clean_content(value: str) -> str:
    content = value.strip()
    for token in CONTROL_TOKENS:
        content = content.replace(token, "")
    return content


def messages_to_text(msgs: list[dict]) -> str:
    parts: list[str] = []
    for m in msgs:
        if not isinstance(m, dict):
            continue
        raw_role = m.get("role", m.get("from", "user"))
        role = ROLE_ALIASES.get(
            str(raw_role).strip().lower(), str(raw_role).strip().lower()
        )
        if role not in ALLOWED_ROLES:
            continue
        content = m.get(
            "content",
            m.get("value", m.get("text", m.get("action", m.get("observation", "")))),
        )
        if isinstance(content, list):
            content = " ".join(
                x.get("text", "") if isinstance(x, dict) else str(x) for x in content
            )
        content = clean_content(str(content))
        if content:
            parts.append(f"<|im_start|>{role}\n{content}{IM_END}\n")
    return "".join(parts)


def normalize(row: dict) -> dict | None:
    for key in ("messages", "conversations", "trajectory"):
        val = row.get(key)
        if isinstance(val, list) and val:
            text = messages_to_text(val)
            if text:
                return {"text": text}
    instruction = row.get("instruction", "")
    inp = row.get("input", "")
    out = row.get("output", "")
    if instruction or out:
        user = instruction if not inp else f"{instruction}\n{inp}"
        user = clean_content(str(user))
        assistant = clean_content(str(out))
        return {
            "text": f"<|im_start|>user\n{user}{IM_END}\n<|im_start|>assistant\n{assistant}{IM_END}\n"
        }
    return None


def main() -> None:
    from datasets import load_dataset

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
        out.unlink(missing_ok=True)
        raise SystemExit(
            f"only collected {written} rows; refusing to create a training dataset"
        )

    print(f"wrote {written} rows -> {out}")


if __name__ == "__main__":
    main()
