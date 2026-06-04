"""SIMD backend detection mirroring oxidize-golang/core/simd/simd.go."""

from __future__ import annotations

import platform
import threading
from enum import IntEnum


class Backend(IntEnum):
    SCALAR = 0
    SSE2 = 1
    AVX = 2
    AVX2 = 3
    AVX512F = 4
    NEON = 5

    def __str__(self) -> str:
        names = {
            Backend.SCALAR: "scalar",
            Backend.SSE2: "sse2",
            Backend.AVX: "avx",
            Backend.AVX2: "avx2",
            Backend.AVX512F: "avx512f",
            Backend.NEON: "neon",
        }
        return names.get(self, "unknown")

    def lane_width_f32(self) -> int:
        match self:
            case Backend.SCALAR:
                return 1
            case Backend.SSE2 | Backend.NEON:
                return 4
            case Backend.AVX | Backend.AVX2:
                return 8
            case Backend.AVX512F:
                return 16
            case _:
                return 1


_detect_once = threading.Lock()
_detected: list[Backend] | None = None
_preferred: Backend | None = None


def _detect() -> None:
    global _detected, _preferred
    _detected = [Backend.SCALAR]
    arch = platform.machine().lower()
    if arch in ("x86_64", "amd64"):
        _detected.extend([Backend.SSE2, Backend.AVX, Backend.AVX2, Backend.AVX512F])
        _preferred = Backend.AVX512F
    elif arch in ("aarch64", "arm64"):
        _detected.append(Backend.NEON)
        _preferred = Backend.NEON
    else:
        _preferred = Backend.SCALAR


def available() -> list[Backend]:
    global _detected
    if _detected is None:
        with _detect_once:
            if _detected is None:
                _detect()
    return list(_detected or [Backend.SCALAR])


def preferred() -> Backend:
    global _preferred
    if _preferred is None:
        with _detect_once:
            if _preferred is None:
                _detect()
    return _preferred or Backend.SCALAR
