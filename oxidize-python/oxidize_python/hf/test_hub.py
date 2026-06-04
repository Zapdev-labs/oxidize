from __future__ import annotations

from oxidize_python.hf.hub import _split_repo_and_file


def test_split_repo_and_file() -> None:
    repo, file = _split_repo_and_file("org/model/file.gguf", "")
    assert repo == "org/model"
    assert file == "file.gguf"


def test_split_repo_explicit_file() -> None:
    repo, file = _split_repo_and_file("org/model", "quant.gguf")
    assert repo == "org/model"
    assert file == "quant.gguf"
