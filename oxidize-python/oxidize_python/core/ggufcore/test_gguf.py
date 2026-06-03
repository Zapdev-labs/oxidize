from oxidize_python.core.ggufcore.gguf import _quantized_byte_size
from oxidize_python.core.quantization import types as quant
from oxidize_python.internal.gguf.types import File, TensorInfo


def test_architecture_empty() -> None:
    from oxidize_python.core.ggufcore import gguf

    assert gguf.architecture(File(3, 0, {}, [], 32, 0)) == ""


def test_quantized_byte_size_f32() -> None:
    size = _quantized_byte_size(TensorInfo("w", [2, 3], 0, 0))
    assert size == 24


def test_quantized_byte_size_q4_0() -> None:
    size = _quantized_byte_size(TensorInfo("w", [32], 2, 0))
    assert size == quant.BLOCK_Q4_0_SIZE
