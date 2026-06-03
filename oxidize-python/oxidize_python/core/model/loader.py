"""Model loading mirroring oxidize-golang/core/model/loader.go."""

from __future__ import annotations

import math
import os
import tempfile
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from oxidize_python.core.ggufcore.gguf import load_mapped
from oxidize_python.core.model.inference_config import load_inference_from_gguf
from oxidize_python.core.model.model import Architecture, Model, Session, Token


class ModelSource(Protocol):
    pass


@dataclass
class FileSource:
    path: str


@dataclass
class MemorySource:
    name: str
    data: bytes


@dataclass
class HFSource:
    repo: str
    revision: str = ""


@dataclass
class LoaderConfig:
    preferred_dtype: str = ""
    max_memory_budget: int = 8 << 30
    allow_fallback: bool = True


def new_loader_config() -> LoaderConfig:
    return LoaderConfig()


@dataclass
class LoadMetrics:
    bytes_read: int = 0
    elapsed_millis: int = 0
    layer_count: int = 0


class ModelLoader:
    def __init__(self, source: ModelSource, config: LoaderConfig) -> None:
        self.source = source
        self.config = config
        self.metrics = LoadMetrics()

    def load(self) -> Model:
        if isinstance(self.source, FileSource):
            return load_gguf_model_from_path(self.source.path, self.config)
        if isinstance(self.source, MemorySource):
            return load_gguf_model_from_bytes(self.source.data, self.config)
        if isinstance(self.source, HFSource):
            return load_gguf_model_from_hf(self.source.repo, self.source.revision, self.config)
        raise ValueError(f"loader: unsupported source type {type(self.source)}")


class ModelType(StrEnum):
    GGUF = "gguf"
    SAFETENSORS = "safetensors"
    ONNX = "onnx"
    PYTORCH = "pytorch"


def detect_model_type(path: str) -> ModelType:
    lower = path.lower()
    if lower.endswith(".gguf"):
        return ModelType.GGUF
    if lower.endswith(".safetensors"):
        return ModelType.SAFETENSORS
    if lower.endswith(".onnx"):
        return ModelType.ONNX
    if lower.endswith(".pt") or lower.endswith(".pth"):
        return ModelType.PYTORCH
    return ModelType.GGUF


def guess_arch_from_path(path: str) -> Architecture:
    lower = path.lower()
    pairs = (
        ("llama", Architecture.LLAMA),
        ("mistral", Architecture.MISTRAL),
        ("mixtral", Architecture.MIXTRAL),
        ("qwen", Architecture.QWEN),
        ("gemma", Architecture.GEMMA),
        ("phi", Architecture.PHI),
        ("falcon", Architecture.FALCON),
        ("deepseek", Architecture.DEEPSEEK),
        ("gpt2", Architecture.GPT2),
        ("gptj", Architecture.GPTJ),
        ("gptneox", Architecture.GPT_NEOX),
    )
    for key, arch in pairs:
        if key in lower:
            return arch
    return Architecture.LLAMA


def load_gguf_model_from_path(path: str, config: LoaderConfig) -> Model:
    if not path:
        raise ValueError("loader: empty path")
    if path.lower().endswith(".gguf"):
        mapped = load_mapped(path)
        return load_inference_from_gguf(mapped)
    if not Path(path).exists():
        raise FileNotFoundError(f"loader: {path}")
    raise ValueError(f"loader: unsupported model path {path!r}")


def load_gguf_model_from_bytes(data: bytes, config: LoaderConfig) -> Model:
    if not data:
        raise ValueError("loader: empty bytes")
    with tempfile.NamedTemporaryFile(prefix="oxidize-", suffix=".gguf", delete=False) as tmp:
        path = tmp.name
        tmp.write(data)
    try:
        return load_gguf_model_from_path(path, config)
    finally:
        os.remove(path)


def load_gguf_model_from_hf(repo: str, revision: str, config: LoaderConfig) -> Model:
    if not repo:
        raise ValueError("loader: empty HF repo")
    return load_gguf_model_from_path(repo, config)


@dataclass
class GgufModelLoader:
    path: str
    config: LoaderConfig

    def load(self) -> Model:
        return load_gguf_model_from_path(self.path, self.config)


@dataclass
class BaselineGgufModel:
    path: str = "<baseline>"
    arch: Architecture = Architecture.LLAMA
    layers: int = 0
    hidden: int = 0
    heads: int = 0
    kv_heads: int = 0
    vocab: int = 0

    def forward(self, tokens: list[Token], session: Session) -> list[float]:
        if self.vocab <= 0:
            raise ValueError("baseline: empty vocab")
        return [
            float(math.sin(len(tokens) + i)) * 0.01 for i in range(self.vocab)
        ]

    def vocab_size(self) -> int:
        return self.vocab

    def context_size(self) -> int:
        return 2048

    def layer_count(self) -> int:
        return self.layers

    def param_count(self) -> int:
        return self.layers * self.hidden * self.hidden * 4
