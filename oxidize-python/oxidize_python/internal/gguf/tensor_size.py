"""Tensor byte-size helpers mirroring oxidize-golang/internal/gguf/tensor_size.go."""

from __future__ import annotations

from oxidize_python.internal.gguf.errors import err_integer_overflow, err_unknown_ggml_type


def tensor_element_count(dimensions: list[int]) -> int:
    count = 1
    for dimension in dimensions:
        if dimension == 0:
            return 0
        if count > (1 << 64) - 1 // dimension:
            raise err_integer_overflow()
        count *= dimension
    return count


def tensor_byte_size(ggml_type: int, element_count: int) -> int:
    if element_count == 0:
        return 0
    match ggml_type:
        case 0:
            return multiply_uint64(element_count, 4)
        case 1:
            return multiply_uint64(element_count, 2)
        case 2:
            return quantized_tensor_size(element_count, 32, 18)
        case 3:
            return quantized_tensor_size(element_count, 32, 20)
        case 6:
            return quantized_tensor_size(element_count, 32, 22)
        case 7:
            return quantized_tensor_size(element_count, 32, 24)
        case 8:
            return quantized_tensor_size(element_count, 32, 34)
        case 10:
            return quantized_tensor_size(element_count, 256, 84)
        case 11:
            return quantized_tensor_size(element_count, 256, 110)
        case 12:
            return quantized_tensor_size(element_count, 256, 144)
        case 13:
            return quantized_tensor_size(element_count, 256, 176)
        case 14:
            return quantized_tensor_size(element_count, 256, 210)
        case 15:
            return quantized_tensor_size(element_count, 256, 66)
        case 16:
            return quantized_tensor_size(element_count, 256, 74)
        case 17:
            return quantized_tensor_size(element_count, 256, 98)
        case 18:
            return quantized_tensor_size(element_count, 256, 50)
        case 19:
            return quantized_tensor_size(element_count, 256, 18)
        case 20:
            return quantized_tensor_size(element_count, 256, 110)
        case 21:
            return quantized_tensor_size(element_count, 256, 82)
        case 22:
            return quantized_tensor_size(element_count, 256, 34)
        case 23:
            return quantized_tensor_size(element_count, 256, 56)
        case 24:
            return quantized_tensor_size(element_count, 64, 34)
        case 29:
            return quantized_tensor_size(element_count, 256, 56)
        case 40:
            return quantized_tensor_size(element_count, 64, 34)
        case _:
            raise err_unknown_ggml_type(ggml_type)


def quantized_tensor_size(element_count: int, block_size: int, bytes_per_block: int) -> int:
    blocks = element_count // block_size
    if element_count % block_size != 0:
        blocks += 1
    return multiply_uint64(blocks, bytes_per_block)


def multiply_uint64(left: int, right: int) -> int:
    if left == 0 or right == 0:
        return 0
    if left > (1 << 64) - 1 // right:
        raise err_integer_overflow()
    return left * right
