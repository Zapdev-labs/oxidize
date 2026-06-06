"""ctypes bindings to liboxidize_ffi.so.

Provides:
  - gemv_quantized_rust  — per-GEMV fast path (zero-copy)
  - RustModel            — full model load/forward via Rust (fastest path)
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np


def _find_lib() -> str | None:
    candidates = [
        Path(__file__).parents[3] / "target" / "release" / "liboxidize_ffi.so",
        Path(__file__).parents[3] / "target" / "release" / "liboxidize_ffi.dylib",
        *(Path(d) / name
          for d in os.environ.get("LD_LIBRARY_PATH", "").split(":")
          if d  # Skip empty paths to avoid current-dir loading
          for name in ("liboxidize_ffi.so", "liboxidize_ffi.dylib")),
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return None


_lib: ctypes.CDLL | None = None


def _ensure_loaded() -> bool:
    global _lib
    if _lib is not None:
        return True
    path = _find_lib()
    if path is None:
        return False
    try:
        lib = ctypes.CDLL(path)

        lib.oxidize_gemv_quantized.argtypes = [
            ctypes.c_uint32, ctypes.c_void_p, ctypes.c_size_t,
            ctypes.c_size_t, ctypes.c_size_t,
            ctypes.c_void_p, ctypes.c_void_p,
        ]
        lib.oxidize_gemv_quantized.restype = ctypes.c_int

        lib.oxidize_model_load.argtypes = [ctypes.c_char_p]
        lib.oxidize_model_load.restype = ctypes.c_void_p

        lib.oxidize_model_free.argtypes = [ctypes.c_void_p]
        lib.oxidize_model_free.restype = None

        lib.oxidize_model_vocab_size.argtypes = [ctypes.c_void_p]
        lib.oxidize_model_vocab_size.restype = ctypes.c_uint32

        lib.oxidize_model_forward.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_size_t,
            ctypes.c_void_p, ctypes.c_size_t,
        ]
        lib.oxidize_model_forward.restype = ctypes.c_int

        lib.oxidize_session_new.argtypes = []
        lib.oxidize_session_new.restype = ctypes.c_void_p

        lib.oxidize_session_reset.argtypes = [ctypes.c_void_p]
        lib.oxidize_session_reset.restype = None

        lib.oxidize_session_free.argtypes = [ctypes.c_void_p]
        lib.oxidize_session_free.restype = None

        lib.oxidize_sample_argmax.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        lib.oxidize_sample_argmax.restype = ctypes.c_uint32

        _lib = lib
        return True
    except Exception:
        return False


# ── per-GEMV fast path ────────────────────────────────────────────────────────

_QUANT_TYPES: dict[str, int] = {
    "F32": 0, "F16": 1, "Q4_0": 2, "Q4_1": 3,
    "Q8_0": 6, "Q2_K": 7,
    "Q3_K_S": 8, "Q3_K_M": 9, "Q3_K_L": 10,
    "Q4_K_S": 11, "Q4_K_M": 12,
    "Q5_K_S": 13, "Q5_K_M": 14,
    "Q6_K": 15,
}


def gemv_quantized_rust(
    qbytes: bytes | bytearray,
    quant_type_name: str,
    rows: int,
    cols: int,
    vector: np.ndarray,
    output: np.ndarray,
) -> bool:
    if not _ensure_loaded() or _lib is None:
        return False
    qt = _QUANT_TYPES.get(quant_type_name)
    if qt is None:
        return False
    if rows <= 0 or cols <= 0:
        return False
    # Validate buffers are large enough before handing raw pointers to native
    if vector.shape[0] < cols or output.shape[0] < rows:
        return False
    try:
        from oxidize_python.core.quantization.types import (
            parse_type,
            quantized_size,
        )

        expected_weight_bytes = rows * quantized_size(parse_type(quant_type_name), cols)
    except Exception:
        return False
    if len(qbytes) < expected_weight_bytes:
        return False
    v = np.ascontiguousarray(vector[:cols], dtype=np.float32)
    o = np.ascontiguousarray(output[:rows], dtype=np.float32)
    rc = _lib.oxidize_gemv_quantized(
        qt,
        ctypes.c_char_p(bytes(qbytes) if isinstance(qbytes, bytearray) else qbytes),
        len(qbytes), rows, cols,
        v.ctypes.data_as(ctypes.c_void_p),
        o.ctypes.data_as(ctypes.c_void_p),
    )
    if rc == 0:
        output[:rows] = o
        return True
    return False


# ── full Rust model (fastest path for Python) ─────────────────────────────────

class RustModel:
    """Full model loaded via Rust FFI — bypasses all Python inference code."""

    def __init__(self, path: str) -> None:
        if not _ensure_loaded() or _lib is None:
            raise RuntimeError("liboxidize_ffi.so not found")
        handle = _lib.oxidize_model_load(path.encode())
        if not handle:
            raise RuntimeError(f"Failed to load model from {path}")
        self._handle = handle
        self._vocab_size = int(_lib.oxidize_model_vocab_size(handle))
        self._logits_buf = np.zeros(self._vocab_size, dtype=np.float32)
        self._session = _lib.oxidize_session_new()

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def reset_session(self) -> None:
        _lib.oxidize_session_reset(self._session)

    def forward(self, tokens: list[int]) -> np.ndarray:
        """Run forward pass; returns logits as a numpy float32 array."""
        n = len(tokens)
        tok_arr = (ctypes.c_uint32 * n)(*tokens)
        rc = _lib.oxidize_model_forward(
            self._handle,
            self._session,
            tok_arr,
            n,
            self._logits_buf.ctypes.data_as(ctypes.c_void_p),
            self._vocab_size,
        )
        if rc != 0:
            raise RuntimeError("oxidize_model_forward failed")
        return self._logits_buf

    def sample_argmax(self) -> int:
        return int(_lib.oxidize_sample_argmax(
            self._logits_buf.ctypes.data_as(ctypes.c_void_p),
            self._vocab_size,
        ))

    def close(self) -> None:
        """Deterministic resource cleanup."""
        if hasattr(self, "_session") and self._session:
            _lib.oxidize_session_free(self._session)
            self._session = None
        if hasattr(self, "_handle") and self._handle:
            _lib.oxidize_model_free(self._handle)
            self._handle = None

    def __enter__(self) -> RustModel:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
