#!/usr/bin/env python3
"""Remove Qwythos/Empero branding from HF sidecars and GGUF string metadata."""

from __future__ import annotations

import json
import mmap
import re
import struct
import sys
from pathlib import Path

BRAND_RE = re.compile(
    rb"Qwythos|Empero[\x00-\xff]{0,4}AI|empero\.org|Claude-Mythos", re.I
)
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


def gguf_header_end(data: mmap.mmap) -> int:
    if data[:4] != b"GGUF":
        raise ValueError("not a GGUF file")
    version = struct.unpack_from("<I", data, 4)[0]
    if version not in (2, 3):
        raise ValueError(f"unsupported GGUF version: {version}")
    tensor_count, metadata_count = struct.unpack_from("<QQ", data, 8)
    offset = 24

    def skip_string(position: int) -> int:
        length = struct.unpack_from("<Q", data, position)[0]
        return position + 8 + length

    def skip_value(value_type: int, position: int, depth: int = 0) -> int:
        if depth > 32:
            raise ValueError("GGUF metadata nesting exceeds 32 levels")
        if value_type == 8:
            return skip_string(position)
        if value_type == 9:
            element_type = struct.unpack_from("<I", data, position)[0]
            count = struct.unpack_from("<Q", data, position + 4)[0]
            position += 12
            for _ in range(count):
                position = skip_value(element_type, position, depth + 1)
            return position
        sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
        return position + sizes[value_type]

    for _ in range(metadata_count):
        offset = skip_string(offset)
        value_type = struct.unpack_from("<I", data, offset)[0]
        offset = skip_value(value_type, offset + 4)

    for _ in range(tensor_count):
        offset = skip_string(offset)
        dimensions = struct.unpack_from("<I", data, offset)[0]
        offset += 4 + dimensions * 8 + 4 + 8
    if offset > len(data):
        raise ValueError("GGUF header extends past end of file")
    return offset


def strip_gguf(path: Path) -> int:
    changed = 0
    with path.open("r+b") as file:
        with mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_WRITE) as data:
            header_end = gguf_header_end(data)
            for old, new in NEUTRAL.items():
                replacement = new.ljust(len(old), b"\x00")[: len(old)]
                position = data.find(old, 0, header_end)
                while position >= 0:
                    data[position : position + len(old)] = replacement
                    changed += 1
                    position = data.find(old, position + len(old), header_end)
            for match in BRAND_RE.finditer(bytes(data[:header_end])):
                data[match.start() : match.end()] = b" " * (match.end() - match.start())
                changed += 1
            data.flush()
    return changed


def main() -> None:
    if len(sys.argv) >= 2 and sys.argv[1] == "--gguf":
        for p in sys.argv[2:]:
            n = strip_gguf(Path(p))
            print(f"scrubbed {p}: {n} replacements")
        return

    hf = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path.home() / "models/qwen35-9b-agent/hf"
    )
    strip_hf_dir(hf)
    print(f"branding stripped from {hf}")


if __name__ == "__main__":
    main()
