"""ctypes bindings to liboxk.so — C++ fused quantized GEMV kernels."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np

_lib: ctypes.CDLL | None = None
_init_done = False


def _find_lib() -> str | None:
    root = Path(__file__).parents[3]
    candidates = [
        root / "dist" / "liboxk" / "liboxk.so",
        root / "liboxk" / "liboxk.so",
        root / "dist" / "liboxk" / "liboxk.dylib",
        root / "liboxk" / "liboxk.dylib",
        *(Path(d) / name
          for d in os.environ.get("LD_LIBRARY_PATH", "").split(":")
          if d
          for name in ("liboxk.so", "liboxk.dylib")),
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return None


def _ensure_loaded() -> bool:
    global _lib, _init_done
    if _lib is not None:
        return True
    path = _find_lib()
    if path is None:
        return False
    try:
        lib = ctypes.CDLL(path)
        lib.oxk_init.argtypes = []
        lib.oxk_init.restype = ctypes.c_int
        lib.oxk_has_avx2.argtypes = []
        lib.oxk_has_avx2.restype = ctypes.c_int
        lib.oxk_gemv_quantized.argtypes = [
            ctypes.c_uint32, ctypes.c_void_p, ctypes.c_size_t,
            ctypes.c_size_t, ctypes.c_size_t,
            ctypes.c_void_p, ctypes.c_void_p,
        ]
        lib.oxk_gemv_quantized.restype = ctypes.c_int
        lib.oxk_dot_f32.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
        lib.oxk_dot_f32.restype = ctypes.c_float
        lib.oxk_rms_norm_f32.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
            ctypes.c_float, ctypes.c_void_p,
        ]
        lib.oxk_rms_norm_f32.restype = ctypes.c_int
        lib.oxk_init()
        _init_done = True
        _lib = lib
        return True
    except OSError:
        return False


_QUANT_TYPES: dict[str, int] = {
    "F32": 0, "F16": 1, "Q4_0": 2, "Q4_1": 3,
    "Q8_0": 6, "Q2_K": 7,
    "Q3_K_S": 8, "Q3_K_M": 9, "Q3_K_L": 10,
    "Q4_K_S": 11, "Q4_K_M": 12,
    "Q5_K_S": 13, "Q5_K_M": 14,
    "Q6_K": 15,
}


def has_avx2() -> bool:
    if not _ensure_loaded() or _lib is None:
        return False
    return bool(_lib.oxk_has_avx2())


def gemv_quantized_oxk(
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
    if vector.shape[0] < cols or output.shape[0] < rows:
        return False
    try:
        from oxidize_python.core.quantization.types import (
            parse_type,
            quantized_size,
        )

        expected = rows * quantized_size(parse_type(quant_type_name), cols)
    except (ImportError, ValueError, KeyError):
        return False
    if len(qbytes) < expected:
        return False
    v = np.ascontiguousarray(vector[:cols], dtype=np.float32)
    o = np.ascontiguousarray(output[:rows], dtype=np.float32)
    raw = bytes(qbytes) if isinstance(qbytes, bytearray) else qbytes
    rc = _lib.oxk_gemv_quantized(
        qt,
        ctypes.c_char_p(raw),
        len(raw),
        rows,
        cols,
        v.ctypes.data_as(ctypes.c_void_p),
        o.ctypes.data_as(ctypes.c_void_p),
    )
    if rc == 0:
        output[:rows] = o
        return True
    return False


def dot_f32(a: np.ndarray, b: np.ndarray) -> float:
    if not _ensure_loaded() or _lib is None:
        return float(np.dot(a, b))
    n = min(a.shape[0], b.shape[0])
    av = np.ascontiguousarray(a[:n], dtype=np.float32)
    bv = np.ascontiguousarray(b[:n], dtype=np.float32)
    return float(
        _lib.oxk_dot_f32(
            av.ctypes.data_as(ctypes.c_void_p),
            bv.ctypes.data_as(ctypes.c_void_p),
            n,
        )
    )


def rms_norm_f32(
    x: np.ndarray,
    weight: np.ndarray,
    eps: float,
    out: np.ndarray,
) -> bool:
    if not _ensure_loaded() or _lib is None:
        return False
    n = x.shape[0]
    if weight.shape[0] < n or out.shape[0] < n:
        return False
    xv = np.ascontiguousarray(x[:n], dtype=np.float32)
    wv = np.ascontiguousarray(weight[:n], dtype=np.float32)
    ov = np.ascontiguousarray(out[:n], dtype=np.float32)
    rc = _lib.oxk_rms_norm_f32(
        xv.ctypes.data_as(ctypes.c_void_p),
        wv.ctypes.data_as(ctypes.c_void_p),
        n,
        ctypes.c_float(eps),
        ov.ctypes.data_as(ctypes.c_void_p),
    )
    if rc == 0:
        out[:n] = ov
        return True
    return False
