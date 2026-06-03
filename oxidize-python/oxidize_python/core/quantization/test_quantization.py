import pytest

from oxidize_python.core.quantization.types import Type, quantized_size


@pytest.mark.parametrize(
    "qt,n,want",
    [
        (Type.F32, 1024, 4096),
        (Type.F16, 1024, 2048),
        (Type.Q4_0, 32, 18),
        (Type.Q8_0, 32, 34),
        (Type.Q2_K, 256, 84),
        (Type.Q6_K, 256, 210),
        (Type.NVFP4, 64, 34),
    ],
)
def test_quantized_size(qt: Type, n: int, want: int) -> None:
    assert quantized_size(qt, n) == want
