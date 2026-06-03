import math

from oxidize_python.core.tensor import (
    Tensor,
    gemm_f32,
    gemm_quantized_f32,
    gemv_f32,
    gemv_f32_transposed,
    rms_norm_f32,
)


def test_gemv_f32() -> None:
    matrix = [1.0, 2.0, 3.0, 4.0]
    vector = [5.0, 6.0]
    output = [0.0, 0.0]
    gemv_f32(matrix, 2, 2, vector, output)
    assert abs(output[0] - (1 * 5 + 2 * 6)) < 1e-6
    assert abs(output[1] - (3 * 5 + 4 * 6)) < 1e-6


def test_gemv_f32_transposed() -> None:
    matrix = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    vector = [7.0, 8.0]
    output = [0.0, 0.0, 0.0]
    gemv_f32_transposed(matrix, 2, 3, vector, output)
    want = [1 * 7 + 4 * 8, 2 * 7 + 5 * 8, 3 * 7 + 6 * 8]
    for i, w in enumerate(want):
        assert abs(output[i] - w) < 1e-6


def test_gemm_f32() -> None:
    left = [1.0, 2.0, 3.0, 4.0]
    right = [5.0, 6.0, 7.0, 8.0]
    output = [0.0] * 4
    gemm_f32(left, right, 2, 2, 2, output)
    want = [19.0, 22.0, 43.0, 50.0]
    for i, w in enumerate(want):
        assert abs(output[i] - w) < 1e-6


def test_gemm_quantized_f32_batch() -> None:
    import struct

    qbytes = struct.pack("<ffff", 1.0, 2.0, 1.0, 2.0)

    def dequant(block: bytes | bytearray, out: list[float]) -> None:
        out[0], out[1] = 1.0, 2.0

    right = [1.0, 0.0, 0.0, 1.0]
    output = [0.0, 0.0]
    gemm_quantized_f32(qbytes, dequant, 1, 2, 1, 2, right, output, None)
    assert abs(output[0] - 1.0) < 1e-6
    assert abs(output[1] - 2.0) < 1e-6


def test_rms_norm() -> None:
    input_ = [1.0, 2.0, 3.0, 4.0]
    weight = [1.0, 1.0, 1.0, 1.0]
    output = [0.0] * 4
    rms_norm_f32(input_, weight, output, 1e-6)
    mean = (1.0 + 4.0 + 9.0 + 16.0) / 4.0
    want = 1.0 / math.sqrt(mean + 1e-6)
    assert abs(output[0] - want) < 1e-3


def test_tensor_new() -> None:
    t = Tensor.new([1.0, 2.0, 3.0, 4.0], [2, 2])
    assert t is not None
    assert t.num_elements() == 4
