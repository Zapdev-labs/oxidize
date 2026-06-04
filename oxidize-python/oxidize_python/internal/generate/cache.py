"""Process-wide GGUF model cache mirroring oxidize-golang/internal/generate/cache.go."""

from __future__ import annotations

import threading
from dataclasses import dataclass

from oxidize_python.core.backends.factory import BackendConfig, create_compute_backend
from oxidize_python.core.model.inference import InferenceModel
from oxidize_python.core.model.loader import LoaderConfig, load_gguf_model_from_path
from oxidize_python.core.model.model import Model


@dataclass
class CachedModel:
    path: str
    model: Model
    backend: str = "cpu"
    warning: str = ""


class ModelCache:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._items: dict[str, CachedModel] = {}

    def _key(self, path: str, cfg: LoaderConfig) -> str:
        return f"{path}|{cfg.backend}|{cfg.ctx_size}|{cfg.n_gpu_layers}|{cfg.gpus}"

    def load(self, path: str, cfg: LoaderConfig) -> CachedModel:
        key = self._key(path, cfg)
        with self._lock:
            if key in self._items:
                return self._items[key]
        loaded = load_gguf_model_from_path(path, cfg)
        _, effective, warning = create_compute_backend(
            BackendConfig(name=cfg.backend, n_gpu_layers=cfg.n_gpu_layers, gpus=cfg.gpus)
        )
        entry = CachedModel(
            path=path,
            model=loaded,
            backend=effective or cfg.backend,
            warning=warning or "",
        )
        with self._lock:
            self._items[key] = entry
        return entry


default_model_cache = ModelCache()


def inference_from_cache(path: str, cfg: LoaderConfig) -> tuple[InferenceModel, CachedModel]:
    entry = default_model_cache.load(path, cfg)
    if not isinstance(entry.model, InferenceModel):
        raise TypeError(f"expected InferenceModel, got {type(entry.model)}")
    return entry.model, entry
