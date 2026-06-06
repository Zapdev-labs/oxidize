"""High-level GGUF helpers mirroring oxidize-golang/core/ggufcore/gguf.go."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

from oxidize_python.core.quantization import types as quant
from oxidize_python.internal.gguf.parse import parse as parse_gguf
from oxidize_python.internal.gguf.errors import GgufError
from oxidize_python.internal.gguf.types import File, MetadataType, MetadataValue, TensorInfo


class ParseError(Exception):
    def __init__(self, err: Exception) -> None:
        super().__init__(f"gguf: {err}")
        self.err = err


QuantizationType = quant.Type


@dataclass
class MetadataArray:
    element_type: MetadataType
    values: list[MetadataValue]


class MappedFile:
    def __init__(self, path: str, raw: bytes, parsed: File) -> None:
        self._mu = threading.RLock()
        self.path = path
        self.bytes = raw
        self.parsed = parsed
        self.closed = False

    @classmethod
    def load_mapped(cls, path: str) -> MappedFile:
        raw = Path(path).read_bytes()
        try:
            parsed = parse_gguf(raw)
        except GgufError as err:
            raise ParseError(err) from err
        except OSError as err:
            raise ParseError(err) from err
        return cls(path, raw, parsed)

    def advise_random_access(self) -> None:
        pass

    def advise_will_need(self) -> None:
        pass

    def advise_huge_pages(self) -> None:
        pass

    def prefault_pages(self) -> None:
        pass

    def tensor_bytes(self, name: str) -> bytes:
        with self._mu:
            for info in self.parsed.tensor_infos:
                if info.name == name:
                    end = info.absolute_offset + _quantized_byte_size(info)
                    if end > len(self.bytes):
                        raise ParseError(ValueError(f"tensor {name} out of bounds"))
                    return self.bytes[info.absolute_offset : end]
        raise ParseError(ValueError(f"tensor {name} not found"))


def architecture(file: File) -> str:
    v = file.metadata.get("general.architecture")
    return v.string if v else ""


def quantization_of(file: File) -> tuple[QuantizationType, bool]:
    if v := file.metadata.get("general.quantization_version"):
        n, ok = v.as_uint64()
        if ok:
            return QuantizationType(int(n)), True
    if v := file.metadata.get("general.file_type"):
        n, ok = v.as_uint64()
        if ok:
            return quant.from_llama_ftype(int(n)), True
    return QuantizationType.F32, False


def mapped_tensor_infos(file: File) -> list[TensorInfo]:
    return list(file.tensor_infos)


def write(path: str, file: File, body: bytes) -> None:
    from oxidize_python.internal.gguf.writer import WriterHeader, encode

    header = WriterHeader(
        version=file.version,
        tensor_count=file.tensor_count,
        metadata_count=len(file.metadata),
        metadata=file.metadata,
        tensors=file.tensor_infos,
        alignment=file.alignment,
        data_section_start=file.data_section_start,
    )
    raw = encode(header, body)
    Path(path).write_bytes(raw)


def _total_elements(dims: list[int]) -> int:
    n = 1
    for d in dims:
        n *= d
    return n


def _element_bytes(ggml_type: int) -> int:
    if ggml_type == 0:
        return 4
    if ggml_type == 1:
        return 2
    return 0


def _gguf_block_size(ggml_type: int) -> int:
    if ggml_type in (0, 1):
        return 0
    if ggml_type in (2, 3, 6, 7, 8):
        return 32
    if ggml_type in (10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 29):
        return 256
    if ggml_type == 40:
        return 64
    return 0


def _block_bytes(ggml_type: int) -> int:
    m = {
        0: 4,
        1: 2,
        2: quant.BLOCK_Q4_0_SIZE,
        3: quant.BLOCK_Q4_1_SIZE,
        6: quant.BLOCK_Q5_0_SIZE,
        7: quant.BLOCK_Q5_1_SIZE,
        8: quant.BLOCK_Q8_0_SIZE,
        10: quant.BLOCK_Q2_K_SIZE,
        11: quant.BLOCK_Q3_K_SIZE,
        12: quant.BLOCK_Q3_K_SIZE,
        13: quant.BLOCK_Q3_K_SIZE,
        14: quant.BLOCK_Q4_K_SIZE,
        19: quant.BLOCK_Q4_K_SIZE,
        15: quant.BLOCK_Q5_K_SIZE,
        16: quant.BLOCK_Q5_K_SIZE,
        17: quant.BLOCK_Q6_K_SIZE,
        18: quant.BLOCK_IQ2_XXS_SIZE,
        20: quant.BLOCK_IQ3_XXS_SIZE,
        21: quant.BLOCK_IQ2_S_SIZE,
        22: quant.BLOCK_IQ4_XS_SIZE,
        23: quant.BLOCK_IQ1_S_SIZE,
        24: quant.BLOCK_NVFP4_SIZE,
        40: quant.BLOCK_NVFP4_SIZE,
        29: quant.BLOCK_IQ1_M_SIZE,
    }
    return m.get(ggml_type, 0)


def _quantized_byte_size(info: TensorInfo) -> int:
    block = _gguf_block_size(info.ggml_type)
    elements = _total_elements(info.dimensions)
    if block == 0:
        return elements * _element_bytes(info.ggml_type)
    blocks = (elements + block - 1) // block
    return blocks * _block_bytes(info.ggml_type)


def load_mapped(path: str) -> MappedFile:
    return MappedFile.load_mapped(path)
