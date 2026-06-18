"""Vision encoder stubs mirroring oxidize-golang/core/vision."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

CLIP_IMAGE_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_IMAGE_STD = (0.26862954, 0.26130258, 0.27577711)


class Modality(StrEnum):
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"


class Error(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(f"vision: {message}")


ErrUnsupportedModality = Error("unsupported modality")


class Preprocessor(Protocol):
    def process(self, raw: bytes, modality: Modality) -> object: ...


class Encoder(Protocol):
    def encode(self, pixels: object) -> list[float]: ...
    def dims(self) -> list[int]: ...


@dataclass
class Config:
    image_size: int = 336
    patch_size: int = 14
    num_patches: int = 0
    num_channels: int = 3
    hidden_size: int = 1024
    num_heads: int = 16
    num_hidden_layers: int = 24
    intermediate_size: int = 4096
    layer_norm_eps: float = 1e-5
    projection_dim: int = 4096
    image_mean: tuple[float, float, float] = CLIP_IMAGE_MEAN
    image_std: tuple[float, float, float] = CLIP_IMAGE_STD
    num_image_tokens: int = 0
    model_name: str = ""
    patch_grid_cols: int = 0
    patch_grid_rows: int = 0

    def patch(self) -> tuple[int, int]:
        if self.patch_size == 0:
            return 0, 0
        cols = rows = self.image_size // self.patch_size
        if self.patch_grid_cols > 0:
            cols = self.patch_grid_cols
        if self.patch_grid_rows > 0:
            rows = self.patch_grid_rows
        return cols, rows


def _with_clip(
    image_size: int,
    patch_size: int,
    hidden: int,
    heads: int,
    layers: int,
    intermediate: int,
    eps: float,
    projection: int,
    name: str,
) -> Config:
    side = image_size // patch_size if patch_size else 0
    return Config(
        image_size=image_size,
        patch_size=patch_size,
        num_patches=side * side,
        hidden_size=hidden,
        num_heads=heads,
        num_hidden_layers=layers,
        intermediate_size=intermediate,
        layer_norm_eps=eps,
        projection_dim=projection,
        num_image_tokens=side * side,
        model_name=name,
    )


def clip_large() -> Config:
    return _with_clip(336, 14, 1024, 16, 24, 4096, 1e-5, 4096, "clip-large")


def llava_15() -> Config:
    return clip_large()


def clip_base() -> Config:
    return _with_clip(224, 14, 768, 12, 12, 3072, 1e-5, 2048, "clip-base")


def qwen_vl() -> Config:
    return _with_clip(448, 14, 1664, 16, 48, 6656, 1e-6, 4096, "qwen-vl")


def default_config() -> Config:
    return clip_large()


@dataclass
class PatchEncoder:
    cfg: Config

    def encode(self, pixels: bytes | list[float]) -> list[float]:
        chw = self._to_chw(pixels)
        cols, rows = self.cfg.patch()
        patch_dim = self.cfg.patch_size * self.cfg.patch_size * self.cfg.num_channels
        out_dim = cols * rows * self.cfg.hidden_size
        out = [0.0] * out_dim
        img = self.cfg.image_size
        for py in range(rows):
            for px in range(cols):
                patch = [0.0] * patch_dim
                self._extract_patch(chw, img, px, py, patch)
                base = (py * cols + px) * self.cfg.hidden_size
                self._project_patch(patch, out[base : base + self.cfg.hidden_size])
        return out

    def dims(self) -> list[int]:
        cols, rows = self.cfg.patch()
        return [1, cols * rows, self.cfg.hidden_size]

    def _to_chw(self, pixels: bytes | list[float]) -> list[float]:
        if isinstance(pixels, list):
            want = self.cfg.num_channels * self.cfg.image_size * self.cfg.image_size
            if len(pixels) < want:
                raise Error("float32 pixels too small")
            return pixels[:want]
        want = 3 * self.cfg.image_size * self.cfg.image_size
        if len(pixels) < want:
            raise Error("byte pixels too small")
        out = [float(b) / 255.0 for b in pixels[:want]]
        for c in range(3):
            mean = self.cfg.image_mean[c]
            std = self.cfg.image_std[c]
            off = c * self.cfg.image_size * self.cfg.image_size
            for i in range(self.cfg.image_size * self.cfg.image_size):
                out[off + i] = (out[off + i] - mean) / std
        return out

    def _extract_patch(
        self, chw: list[float], img: int, px: int, py: int, patch: list[float]
    ) -> None:
        ps = self.cfg.patch_size
        ch = self.cfg.num_channels
        idx = 0
        for c in range(ch):
            plane = c * img * img
            for y in range(ps):
                for x in range(ps):
                    ix = px * ps + x
                    iy = py * ps + y
                    if ix >= img or iy >= img:
                        patch[idx] = 0.0
                    else:
                        patch[idx] = chw[plane + iy * img + ix]
                    idx += 1

    def _project_patch(self, patch: list[float], out: list[float]) -> None:
        if not out:
            return
        mean = sum(patch) / len(patch)
        for i in range(len(out)):
            out[i] = mean * float((i % 7) + 1) * 0.01


@dataclass
class StubEncoder:
    cfg: Config

    def encode(self, pixels: object) -> list[float]:
        if pixels is None:
            raise Error("nil pixels")
        cols, rows = self.cfg.patch()
        return [0.0] * (cols * rows * self.cfg.hidden_size)

    def dims(self) -> list[int]:
        cols, rows = self.cfg.patch()
        return [1, cols * rows, self.cfg.hidden_size]


@dataclass
class StubPreprocessor:
    cfg: Config

    def process(self, raw: bytes, modality: Modality) -> object:
        if modality != Modality.IMAGE:
            raise ErrUnsupportedModality
        return raw


@dataclass
class MultimodalPrompt:
    text: str = ""
    tokens: list[int] = None
    images: list[list[float]] = None

    def __post_init__(self) -> None:
        if self.tokens is None:
            self.tokens = []
        if self.images is None:
            self.images = []


def encode(prompt: MultimodalPrompt, enc: Encoder) -> list[float]:
    out: list[float] = []
    for img in prompt.images:
        encoded = enc.encode(img)
        out.extend(encoded)
    return out
