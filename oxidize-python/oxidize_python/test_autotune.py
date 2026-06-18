"""Autotune unit tests."""

from __future__ import annotations

from oxidize_python.core import autotune
from oxidize_python.core.quantization.types import Type


def test_detect_returns_inventory() -> None:
    inv = autotune.detect()
    assert inv.physical_cores >= 1
    assert inv.total_ram_bytes > 0


def test_plan_has_threads() -> None:
    inv = autotune.detect()
    fp = autotune.ModelFingerprint(
        architecture="llama",
        layer_count=32,
        hidden_size=4096,
        num_attention_heads=32,
        num_kv_heads=32,
        head_dim=128,
        intermediate_size=11008,
        vocab_size=32000,
        file_size_bytes=2_000_000_000,
        quant=Type.Q4_0,
        is_moe=False,
        expert_count=0,
        has_mtp=False,
    )
    plan = autotune.plan(inv, fp)
    assert plan.threads >= 1
    assert plan.ctx_size >= 512


def test_overrides_from_plan() -> None:
    inv = autotune.detect()
    fp = autotune.ModelFingerprint(
        architecture="llama",
        layer_count=16,
        hidden_size=2048,
        num_attention_heads=16,
        num_kv_heads=16,
        head_dim=128,
        intermediate_size=5504,
        vocab_size=32000,
        file_size_bytes=500_000_000,
        quant=Type.Q4_0,
        is_moe=False,
        expert_count=0,
        has_mtp=False,
    )
    plan = autotune.plan(inv, fp)
    overrides = autotune.overrides_from_plan(plan)
    assert overrides.threads is not None or overrides.ctx_size is not None
