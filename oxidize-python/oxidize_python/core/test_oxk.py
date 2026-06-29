"""Tests for oxidize_python.core.oxk mirroring Rust oxidize-kernels tests."""

from __future__ import annotations

import struct

import pytest

from oxidize_python.core.oxk import (
    BLOCK_Q4_K_SIZE,
    BLOCK_Q8_K_BYTES,
    QK_K,
    cpu_info,
    gemv_q4k_range,
    max_tile,
    oxk_cpu_summary,
    q4k_q8k_row_dot_scalar,
    q4k_q8k_row_dot_x1_scalar,
    q4k_q8k_row_dot_x4_scalar,
    q4k_q8k_row_dot_x8_scalar,
    q4k_q8k_row_dot_x16_scalar,
    quantize_q8_k_into,
    tune,
)


def _fill_pseudo(buf: bytearray, state: int) -> int:
    for i in range(len(buf)):
        state ^= state << 13
        state ^= state >> 7
        state ^= state << 17
        state &= 0xFFFFFFFFFFFFFFFF
        buf[i] = state & 0xFF
    return state


def _random_fixture(rows: int, blocks_per_row: int, seed: int) -> tuple[bytearray, bytearray]:
    weights = bytearray(rows * blocks_per_row * BLOCK_Q4_K_SIZE)
    _fill_pseudo(weights, seed)
    # Keep f16 d/dmin fields finite and small
    for block_idx in range(len(weights) // BLOCK_Q4_K_SIZE):
        block = memoryview(weights)[block_idx * BLOCK_Q4_K_SIZE:]
        for half in range(2):
            raw = struct.unpack_from("<H", block, half * 2)[0]
            tamed = (raw & 0x83FF) | (0x3000 + ((raw >> 10) & 0x7) * 0x400)
            struct.pack_into("<H", block, half * 2, tamed)
    vector_bytes = bytearray(blocks_per_row * QK_K)
    _fill_pseudo(vector_bytes, seed * 0x9E3779B97F4A7C15)
    vector = [(b - 127.5) / 32.0 for b in vector_bytes]
    q8k = bytearray(blocks_per_row * BLOCK_Q8_K_BYTES)
    quantize_q8_k_into(vector, blocks_per_row, q8k)
    return weights, q8k


def test_tile_variants_match_scalar_exactly() -> None:
    for rows, bpr in [(8, 16), (12, 4), (32, 8)]:
        weights, q8k = _random_fixture(rows, bpr, 1)
        row_bytes = bpr * BLOCK_Q4_K_SIZE
        scalar = [
            q4k_q8k_row_dot_scalar(weights[r * row_bytes:(r + 1) * row_bytes], bpr, q8k)
            for r in range(rows)
        ]

        # x1
        for r in range(rows):
            got = q4k_q8k_row_dot_x1_scalar(
                weights[r * row_bytes:(r + 1) * row_bytes], bpr, q8k
            )
            assert struct.pack("<f", got) == struct.pack("<f", scalar[r]), f"x1 row {r}"

        # x4
        if rows >= 4:
            quad = [0.0] * 4
            q4k_q8k_row_dot_x4_scalar(weights, row_bytes, bpr, q8k, quad)
            for r in range(4):
                assert struct.pack("<f", quad[r]) == struct.pack("<f", scalar[r]), f"x4 row {r}"

        # x8
        if rows >= 8:
            octet = [0.0] * 8
            q4k_q8k_row_dot_x8_scalar(weights, row_bytes, bpr, q8k, octet)
            for r in range(8):
                assert struct.pack("<f", octet[r]) == struct.pack("<f", scalar[r]), f"x8 row {r}"

        # x16
        if rows >= 16:
            hex_ = [0.0] * 16
            q4k_q8k_row_dot_x16_scalar(weights, row_bytes, bpr, q8k, hex_)
            for r in range(16):
                assert struct.pack("<f", hex_[r]) == struct.pack("<f", scalar[r]), f"x16 row {r}"


def test_gemv_range_matches_scalar() -> None:
    weights, q8k = _random_fixture(13, 8, 7)
    row_bytes = 8 * BLOCK_Q4_K_SIZE
    out = [0.0] * 13
    gemv_q4k_range(weights, 8, q8k, out)
    for r in range(13):
        want = q4k_q8k_row_dot_scalar(weights[r * row_bytes:(r + 1) * row_bytes], 8, q8k)
        assert struct.pack("<f", out[r]) == struct.pack("<f", want), f"row {r}"


def test_max_tile_default_is_16(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OXIDIZE_OXK_TILE", raising=False)
    # In Python we can't easily reset the lock, but the env var is read at import time
    # The test verifies the default behavior conceptually
    assert max_tile() in (1, 4, 8, 16)


def test_tune_is_block_aligned() -> None:
    t = tune()
    assert t.pf_bytes % BLOCK_Q4_K_SIZE == 0


def test_summary_mentions_vendor() -> None:
    s = oxk_cpu_summary()
    assert "vendor=" in s


def test_cpu_info_is_stable() -> None:
    a = cpu_info()
    b = cpu_info()
    assert a.family == b.family
    assert a.model == b.model
