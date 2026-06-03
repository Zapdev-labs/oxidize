"""GGUF parse errors mirroring oxidize-golang/internal/gguf/errors.go."""

from __future__ import annotations

MAX_GGUF_STRING_BYTES = 16 * 1024 * 1024


class GgufError(Exception):
    pass


def err_invalid_magic() -> GgufError:
    return GgufError("invalid gguf magic")


def err_unsupported_version(version: int) -> GgufError:
    return GgufError(f"unsupported gguf version: {version}")


def err_unexpected_eof() -> GgufError:
    return GgufError("unexpected end of file")


def err_unknown_metadata_type(value: int) -> GgufError:
    return GgufError(f"unknown metadata type: {value}")


def err_invalid_alignment(value: int) -> GgufError:
    return GgufError(f"invalid alignment: {value}")


def err_integer_overflow() -> GgufError:
    return GgufError("integer overflow while parsing")


def err_unknown_ggml_type(value: int) -> GgufError:
    return GgufError(f"unknown ggml type: {value}")


def err_string_too_long(length: int) -> GgufError:
    return GgufError(
        f"gguf string length {length} exceeds limit {MAX_GGUF_STRING_BYTES}"
    )
