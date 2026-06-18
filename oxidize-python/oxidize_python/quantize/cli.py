"""oxidize-quantize CLI mirroring oxidize-quantize (Rust)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from oxidize_python.core.quantization.dequant_k import dequantize
from oxidize_python.core.quantization.quantize import quantize_scalar
from oxidize_python.core.quantization.types import Type, quantized_size
from oxidize_python.internal.gguf.parse import load_file, parse
from oxidize_python.internal.gguf.tensor_size import tensor_byte_size, tensor_element_count
from oxidize_python.internal.gguf.types import MetadataType, MetadataValue
from oxidize_python.internal.gguf.writer import WriterHeader, encode


def _parse_quant(name: str) -> Type:
    key = name.upper().replace("-", "_")
    for member in Type:
        if member.name == key:
            return member
    raise argparse.ArgumentTypeError(f"unsupported quantization type: {name}")


def _ggml_type_id(t: Type) -> int:
    return int(t)


def _requantize_body(
    raw: bytes,
    file,
    source: Type | None,
    target: Type,
) -> bytes:
    body = bytearray()
    align = file.alignment or 32
    for tensor in file.tensor_infos:
        elems = tensor_element_count(tensor.dimensions)
        src_size = tensor_byte_size(tensor.ggml_type, elems)
        start = file.data_section_start + tensor.relative_offset
        tensor_bytes = raw[start : start + src_size]
        try:
            src_type = Type(tensor.ggml_type)
        except ValueError:
            src_type = Type.F32
        if source is not None:
            src_type = source
        can_quantize = len(tensor.dimensions) >= 2 and src_type in (Type.F32, Type.F16)
        if can_quantize and target not in (Type.F32, Type.F16):
            f32 = [0.0] * elems
            dequantize(src_type, tensor_bytes, f32)
            dst_size = quantized_size(target, elems)
            out_bytes = bytearray(dst_size)
            quantize_scalar(target, f32, out_bytes, None)
            payload = bytes(out_bytes)
            ggml_type = _ggml_type_id(target)
        else:
            payload = tensor_bytes
            ggml_type = tensor.ggml_type
        pad = (-len(body)) % align
        if pad:
            body.extend(b"\x00" * pad)
        tensor.relative_offset = len(body)
        tensor.ggml_type = ggml_type
        body.extend(payload)
    return bytes(body)


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

    if ns.target is not None:
        body = _requantize_body(raw, file, ns.source, ns.target)
        meta = dict(file.metadata)
        meta["general.quantization_version"] = MetadataValue(type=MetadataType.UINT32, uint64=2)
        meta["general.file_type"] = MetadataValue(
            type=MetadataType.UINT32, uint64=_ggml_type_id(ns.target)
        )
        header = WriterHeader(
            version=file.version,
            metadata=meta,
            tensors=file.tensor_infos,
            alignment=file.alignment,
            data_section_start=0,
        )
        out = encode(header, body)
    else:
        body_start = file.data_section_start
        body = raw[body_start:]
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
