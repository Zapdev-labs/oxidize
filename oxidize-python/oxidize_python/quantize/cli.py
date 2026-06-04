"""oxidize-quantize CLI mirroring oxidize-quantize (Rust)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from oxidize_python.core.quantization.types import Type as QuantType
from oxidize_python.internal.gguf.parse import load_file, parse
from oxidize_python.internal.gguf.types import MetadataType, MetadataValue
from oxidize_python.internal.gguf.writer import WriterHeader, encode


def _parse_quant(name: str) -> int:
    key = name.upper().replace("-", "_")
    for member in QuantType:
        if member.name == key:
            return int(member)
    raise argparse.ArgumentTypeError(f"unsupported quantization type: {name}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="oxidize-quantize")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--source", type=_parse_quant)
    p.add_argument("--target", type=_parse_quant)
    p.add_argument("--append-tensor", action="append", default=[])
    ns = p.parse_args(argv)

    inp = Path(ns.input)
    raw = inp.read_bytes()
    file = parse(raw)
    if ns.target is None and not ns.append_tensor:
        print("provide --target or --append-tensor", file=sys.stderr)
        return 1

    body_start = file.data_section_start
    body = raw[body_start:]
    if ns.target is not None:
        meta = dict(file.metadata)
        meta["general.quantization_version"] = MetadataValue(type=MetadataType.UINT32, uint64=2)
        header = WriterHeader(
            version=file.version,
            metadata=meta,
            tensors=file.tensor_infos,
            alignment=file.alignment,
            data_section_start=body_start,
        )
        out = encode(header, body)
    else:
        header = WriterHeader(
            version=file.version,
            metadata=file.metadata,
            tensors=file.tensor_infos,
            alignment=file.alignment,
        )
        out = encode(header, body)

    Path(ns.output).write_bytes(out)
    _ = load_file(ns.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
