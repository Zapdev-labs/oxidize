"""Shared helpers for oxidize-python integration tests."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

QWEN_MODEL_FILENAME = "Qwen3-4B-Q4_K_M.gguf"
QWEN_MODEL_ID = "Qwen3-4B-Q4_K_M"


def oxidize_repo_root() -> Path | None:
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "oxidize-core").is_dir() and (parent / "models").is_dir():
            return parent
    return None


def slow_tests_enabled() -> bool:
    return os.environ.get("OXIDIZE_SLOW_TESTS", "") != ""


def require_slow_tests() -> None:
    if not slow_tests_enabled():
        pytest.skip("set OXIDIZE_SLOW_TESTS=1 to run full-model prompt integration")


def qwen_model_path() -> Path:
    root = oxidize_repo_root()
    if root is None:
        pytest.skip("oxidize repo root not found")
    path = root / "models" / QWEN_MODEL_FILENAME
    if not path.is_file():
        pytest.skip(f"qwen model not found at {path}")
    return path


def generation_text(raw: str) -> str:
    text = raw.strip()
    if "generation stats:" in text:
        text = text.split("generation stats:", 1)[0].strip()
    return text


def assert_generation_text(raw: str) -> None:
    text = generation_text(raw)
    assert text, f"expected generated text, got {raw!r}"
    assert "DateFormat" not in text, f"degenerate output: {text!r}"
