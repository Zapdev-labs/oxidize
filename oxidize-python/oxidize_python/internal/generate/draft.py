"""Draft model loading mirroring oxidize-golang/internal/generate/loader.go."""

from __future__ import annotations

from oxidize_python.core.ggufcore import gguf as ggufcore
from oxidize_python.core.model.loader import LoaderConfig, load_gguf_model_from_path
from oxidize_python.core.model.model import Model


def _hidden_size_from_mapped(mapped) -> int:
    meta = mapped.parsed.metadata
    for key in ("llama.embedding_length", "general.embedding_length", "hidden_size"):
        if key in meta and meta[key].uint64:
            return int(meta[key].uint64)
        if key in meta and meta[key].int32:
            return int(meta[key].int32)
    return 0


def load_draft_from_path(path: str, loader: LoaderConfig, target_hidden: int) -> Model:
    path = path.strip()
    if not path:
        raise ValueError("generate: empty draft model path")
    mapped = ggufcore.load_mapped(path)
    draft_hidden = _hidden_size_from_mapped(mapped)
    if target_hidden > 0 and draft_hidden > 0 and draft_hidden != target_hidden:
        raise ValueError(
            f"generate: draft hidden_size {draft_hidden} != target {target_hidden}"
        )
    return load_gguf_model_from_path(path, loader)
