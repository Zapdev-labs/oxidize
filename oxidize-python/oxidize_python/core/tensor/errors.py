"""Error types mirroring oxidize_core::compute::tensor::*Error."""


class GemvError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(f"gemv: {message}")


class GemmError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(f"gemm: {message}")


class AttentionError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(f"attention: {message}")


class RopeError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(f"rope: {message}")


class SwiGluError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(f"swiglu: {message}")


class LinearActivationError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(f"linear: {message}")


class RmsNormError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(f"rms_norm: {message}")


class LayerNormError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(f"layer_norm: {message}")


class SoftmaxError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(f"softmax: {message}")
