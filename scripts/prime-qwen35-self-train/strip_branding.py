#!/usr/bin/env python3
"""Remove Qwythos/Empero branding from HF sidecars and GGUF string metadata."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

BRAND_RE = re.compile(rb"Qwythos|Empero[\x00-\xff]{0,4}AI|empero\.org|Claude-Mythos", re.I)
NEUTRAL = {
    b"Qwythos-9B-Claude-Mythos-5-1M": b"qwen3.5-9b-instruct",
    b"Qwythos-9B": b"qwen3.5-9b-instruct",
    b"empero-ai/Qwythos-9B-Claude-Mythos-5-1M": b"qwen3.5-9b-instruct",
}


def strip_hf_dir(hf: Path) -> None:
    for name in ("chat_template.jinja",):
        p = hf / name
        if not p.exists():
            continue
        text = p.read_text(encoding="utf-8")
        text = re.sub(r"\{%- set qwythos_identity = .*?%\}\s*", "", text, flags=re.S)
        text = text.replace("qwythos_identity", '""')
        text = re.sub(r"\+\s*qwythos_identity", "", text)
        text = re.sub(r"Qwythos|Empero AI|empero\.org", "", text, flags=re.I)
        p.write_text(text, encoding="utf-8")

    cfg = hf / "config.json"
    if cfg.exists():
        data = json.loads(cfg.read_text(encoding="utf-8"))
        data.pop("_name_or_path", None)
        cfg.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

    tok = hf / "tokenizer_config.json"
    if tok.exists():
        data = json.loads(tok.read_text(encoding="utf-8"))
        if isinstance(data.get("chat_template"), str):
            t = data["chat_template"]
            t = re.sub(r"\{%- set qwythos_identity = .*?%\}\s*", "", t, flags=re.S)
            t = re.sub(r"Qwythos|Empero AI|empero\.org", "", t, flags=re.I)
            data["chat_template"] = t
        tok.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def strip_gguf(path: Path, scan_bytes: int = 32 * 1024 * 1024) -> int:
    """Scrub branding strings from GGUF metadata/header region only (fast, low RAM)."""
    size = path.stat().st_size
    head_len = min(scan_bytes, size)
    with path.open("r+b") as f:
        head = bytearray(f.read(head_len))
        changed = 0
        for old, new in NEUTRAL.items():
            if old in head:
                repl = new.ljust(len(old), b"\x00")[: len(old)]
                head = head.replace(old, repl)
                changed += 1
        for m in list(BRAND_RE.finditer(bytes(head))):
            head[m.start() : m.end()] = b" " * (m.end() - m.start())
            changed += 1
        if changed:
            f.seek(0)
            f.write(head)
    return changed


def main() -> None:
    if len(sys.argv) >= 2 and sys.argv[1] == "--gguf":
        for p in sys.argv[2:]:
            n = strip_gguf(Path(p))
            print(f"scrubbed {p}: {n} replacements")
        return

    hf = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "models/qwen35-9b-agent/hf"
    strip_hf_dir(hf)
    print(f"branding stripped from {hf}")


if __name__ == "__main__":
    main()
