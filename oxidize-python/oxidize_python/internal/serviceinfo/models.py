"""Model discovery mirroring oxidize-golang/internal/serviceinfo/models.go."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from oxidize_python.internal.gguf.parse import parse as parse_gguf

MIN_GGUF_FULL_VALIDATION_BYTES = 1024


@dataclass
class ModelInfo:
    id: str
    path: str
    version: int
    architecture: str


def default_model_id(models: list[ModelInfo]) -> str:
    if not models:
        return "oxidize-default"
    return models[0].id


def discover_models(dir_path: str) -> list[ModelInfo]:
    if not dir_path.strip():
        return []
    root = Path(dir_path)
    if not root.is_dir():
        return []
    models: list[ModelInfo] = []
    for path in sorted(root.glob("*.gguf")):
        stat = path.stat()
        raw = path.read_bytes()
        if stat.st_size >= MIN_GGUF_FULL_VALIDATION_BYTES:
            parse_gguf(raw)
        parsed = parse_gguf(raw)
        arch = ""
        if v := parsed.metadata.get("general.architecture"):
            arch = v.string
        models.append(
            ModelInfo(
                id=path.stem,
                path=str(path),
                version=parsed.version,
                architecture=arch,
            )
        )
    return sorted(models, key=lambda m: m.id)
