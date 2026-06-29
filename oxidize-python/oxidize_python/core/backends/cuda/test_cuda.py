"""Tests for the CUDA backend port, mirroring oxidize-golang cuda_test.go."""

from __future__ import annotations

import struct

import pytest

import oxidize_python.core.backends.cuda as c


@pytest.fixture(autouse=True)
def _fresh_state():
    c.reset_gpu_state()
    yield
    c.reset_gpu_state()


def _q8_0_block(vals: list[float]) -> bytes:
    amax = max((abs(v) for v in vals), default=0.0)
    d = amax / 127.0 if amax > 0 else 0.0
    out = bytearray()
    out += struct.pack("<e", d)
    inv = 1.0 / d if d else 0.0
    for v in vals:
        out += struct.pack("<b", max(-127, min(127, round(v * inv))))
    return bytes(out)


def test_build_info_detection():
    info = c.build_info()
    assert info.detected_at_build is False
    assert info.cuda_path == ""


def test_pool_allocation_and_reuse():
    st = c.global_gpu_state()
    buf = c.get_f32_buffer(st, 4)
    buf[0] = 9.0
    c.return_f32_buffer(st, buf)
    buf2 = c.get_f32_buffer(st, 4)
    assert buf2 is buf  # reused
    assert buf2[0] == 0.0  # zeroed on checkout


def test_layer_lru_eviction():
    c.set_layer_config(c.CudaLayerConfig(max_resident_layers=2))
    st = c.global_gpu_state()
    for i in range(3):
        c.preload_layer(i, [c.F32Weight(data=[float(i)] * 8)])
    assert 0 not in st.layer_map
    assert 1 in st.layer_map and 2 in st.layer_map


def test_vram_budget_enforcement():
    # One 8-float layer = 32 bytes budget.
    c.set_layer_config(c.CudaLayerConfig(max_vram_bytes=8 * 4))
    c.preload_layer(10, [c.F32Weight(data=[1.0] * 8)])
    c.preload_layer(11, [c.F32Weight(data=[2.0] * 8)])
    assert c.resident_vram_bytes() <= 8 * 4


def test_gemm_f32_correctness():
    # [[1,2],[3,4]] * [[5,6],[7,8]] = [[19,22],[43,50]]
    out = [0.0] * 4
    c.gemm_f32_cuda([1, 2, 3, 4], [5, 6, 7, 8], 2, 2, 2, out)
    assert out == [19, 22, 43, 50]


def test_quant_support_detection():
    assert c.supports_quantized_gpu(c.GgmlType.Q8_0)
    assert c.supports_quantized_gpu(c.GgmlType.Q4_K)
    assert c.supports_quantized_gpu(c.GgmlType.Q6_K)
    assert not c.supports_quantized_gpu(c.GgmlType.F16)


def test_quantized_gemv_dispatch_q8_0():
    rows, cols = 2, 32
    wbytes = b"".join(_q8_0_block([float(r + 1)] * cols) for r in range(rows))
    out = [0.0] * rows
    c.gemv_quantized_cuda(wbytes, int(c.GgmlType.Q8_0), [1.0] * cols, rows, cols, out)
    assert abs(out[0] - 32.0) < 0.5
    assert abs(out[1] - 64.0) < 0.5


def test_activation_buffers_and_rms_norm():
    h = 16
    c.gpu_init_activation_buffers(h, h)
    c.gpu_upload_hidden([1.0] * h)
    c.gpu_rms_norm([1.0] * h, 1e-5)
    out = [0.0] * h
    c.gpu_download_hidden(out)
    # rms of all-ones is 1, so normed ~= 1.
    assert all(abs(v - 1.0) < 1e-3 for v in out)


def test_fused_gpu_attn_rms_and_qkv():
    rows, cols = 2, 32
    wbytes = b"".join(_q8_0_block([float(r + 1)] * cols) for r in range(rows))
    c.gpu_init_activation_buffers(cols, cols)
    c.gpu_upload_hidden([1.0] * cols)
    qo = [0.0] * rows
    ko = [0.0] * rows
    vo = [0.0] * rows
    c.gpu_attn_rms_and_qkv_q4k(
        [1.0] * cols, 1e-5, wbytes, rows, wbytes, rows, wbytes,
        c.GgmlType.Q8_0, cols, qo, ko, vo,
    )
    assert abs(qo[0] - 32.0) < 0.5 and abs(qo[1] - 64.0) < 0.5
    assert qo == ko == vo
    # Resident quant cached once for the shared weight bytes.
    assert c.resident_vram_bytes() == len(wbytes)
