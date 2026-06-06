"""Tests for GGUF parse/write roundtrip."""

from __future__ import annotations

from oxidize_python.internal.gguf.parse import parse
from oxidize_python.internal.gguf.types import MetadataType, MetadataValue, TensorInfo
from oxidize_python.internal.gguf.writer import WriterHeader, encode


def test_encode_minimal_roundtrip() -> None:
    meta = {
        "general.name": MetadataValue(type=MetadataType.STRING, string="test"),
        "general.alignment": MetadataValue(type=MetadataType.UINT32, uint64=32),
    }
    tensors = [
        TensorInfo(
            name="tensor",
            dimensions=[4],
            ggml_type=0,
            relative_offset=0,
        )
    ]
    body = b"\x00\x00\x00\x00" * 4
    raw = encode(
        WriterHeader(version=3, metadata=meta, tensors=tensors, alignment=32),
        body,
    )
    file = parse(raw)
    assert file.version == 3
    assert file.metadata["general.name"].string == "test"
    assert len(file.tensor_infos) == 1


def test_parse_qwen_fixture_if_present() -> None:
    from io import BytesIO

    from oxidize_python.internal.gguf.parse import parse_header
    from oxidize_python.internal.gguf.reader import BinaryReader
    from oxidize_python.testutil import qwen_model_path

    path = qwen_model_path()
    raw = path.read_bytes()
    version, tensor_count, metadata, _ = parse_header(BinaryReader(BytesIO(raw)))
    assert version == 3
    assert tensor_count >= 1
    assert metadata


def test_parse_fixture_if_present() -> None:
    import os

    root = os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "..", "oxidize-core", "tests", "fixtures"
    )
    path = os.path.join(root, "valid-v3.gguf")
    if os.path.isfile(path):
        from io import BytesIO

        from oxidize_python.internal.gguf.parse import parse_header
        from oxidize_python.internal.gguf.reader import BinaryReader

        raw = open(path, "rb").read()
        version, tensor_count, metadata, _ = parse_header(BinaryReader(BytesIO(raw)))
        assert version == 3
        assert tensor_count >= 1
        assert metadata
