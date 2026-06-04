"""KV cache mirroring oxidize_core::compute::kv_cache."""

from __future__ import annotations

import json
import struct
import threading
from dataclasses import dataclass, field
from enum import IntEnum, auto
from pathlib import Path


class Quantization(IntEnum):
    ASYMMETRIC = auto()
    TURBOQUANT = auto()


class EvictionStrategy(IntEnum):
    SLIDING_WINDOW = auto()
    STOP_AT_CAPACITY = auto()

    def __str__(self) -> str:
        if self == EvictionStrategy.SLIDING_WINDOW:
            return "sliding_window"
        if self == EvictionStrategy.STOP_AT_CAPACITY:
            return "stop_at_capacity"
        return f"eviction({int(self)})"


@dataclass
class Config:
    layer_count: int = 32
    context_size: int = 2048
    head_count: int = 32
    head_dim: int = 128
    dtype: str = "f16"
    quantization: Quantization = Quantization.ASYMMETRIC
    eviction: EvictionStrategy = EvictionStrategy.SLIDING_WINDOW

    def token_size(self) -> int:
        return self.head_count * self.head_dim * 2

    def layer_size(self) -> int:
        return self.context_size * self.token_size() * 4

    def element_count(self) -> int:
        return self.context_size * self.layer_count * self.token_size()

    def blocks_per_token(self) -> int:
        return (self.token_size() + 31) // 32


class Error(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(f"kv_cache: {message}")


class PersistenceError(Exception):
    def __init__(self, err: Exception) -> None:
        super().__init__(f"kv_cache persistence: {err}")
        self.err = err


@dataclass
class Cache:
    config: Config
    keys: list[list[float]] = field(default_factory=list)
    values: list[list[float]] = field(default_factory=list)
    lengths: list[int] = field(default_factory=list)
    occupied: int = 0
    _mu: threading.RLock = field(default_factory=threading.RLock)

    @classmethod
    def new(cls, config: Config) -> Cache:
        keys = [[] for _ in range(config.layer_count)]
        vals = [[] for _ in range(config.layer_count)]
        for i in range(config.layer_count):
            keys[i] = []
            vals[i] = []
        return cls(
            config=config,
            keys=keys,
            values=vals,
            lengths=[0] * config.layer_count,
        )

    def append(self, layer: int, key: list[float], value: list[float]) -> None:
        if layer < 0 or layer >= self.config.layer_count:
            raise Error(f"layer {layer} out of bounds")
        dim = self.config.head_count * self.config.head_dim
        if len(key) < dim or len(value) < dim:
            raise Error("key/value too small")
        with self._mu:
            if self.lengths[layer] >= self.config.context_size:
                if self.config.eviction == EvictionStrategy.SLIDING_WINDOW:
                    self._evict_oldest(layer)
                else:
                    raise Error("cache full")
            self.keys[layer].extend(key[:dim])
            self.values[layer].extend(value[:dim])
            self.lengths[layer] += 1
            if layer == 0:
                self.occupied += 1

    def get(self, layer: int) -> tuple[list[float], list[float], int]:
        if layer < 0 or layer >= self.config.layer_count:
            raise Error("layer out of bounds")
        with self._mu:
            dim = self.config.head_count * self.config.head_dim
            length = self.lengths[layer]
            keys = list(self.keys[layer][: length * dim])
            vals = list(self.values[layer][: length * dim])
            return keys, vals, length

    def _evict_oldest(self, layer: int) -> None:
        dim = self.config.head_count * self.config.head_dim
        self.keys[layer] = self.keys[layer][dim:]
        self.values[layer] = self.values[layer][dim:]
        self.lengths[layer] -= 1

    def occupied_tokens(self) -> int:
        with self._mu:
            return max(self.lengths, default=0)

    def save_to_disk(self, path: str | Path) -> None:
        path = Path(path)
        with self._mu:
            snap = {"config": self.config.__dict__, "lengths": self.lengths}
            header = json.dumps(snap).encode()
            buf = bytearray()
            buf.extend(struct.pack("<I", len(header)))
            buf.extend(header)
            dim = self.config.head_count * self.config.head_dim
            for layer in range(self.config.layer_count):
                n = self.lengths[layer]
                k = self.keys[layer][: n * dim]
                v = self.values[layer][: n * dim]
                buf.extend(struct.pack("<I", len(k)))
                for x in k:
                    buf.extend(struct.pack("<f", x))
                buf.extend(struct.pack("<I", len(v)))
                for x in v:
                    buf.extend(struct.pack("<f", x))
            path.write_bytes(buf)

    @classmethod
    def load_from_disk(cls, path: str | Path) -> Cache:
        raw = Path(path).read_bytes()
        off = 0
        (hdr_len,) = struct.unpack_from("<I", raw, off)
        off += 4
        snap = json.loads(raw[off : off + hdr_len].decode())
        off += hdr_len
        cfg = Config(
            **{k: v for k, v in snap["config"].items() if k in Config.__dataclass_fields__}
        )
        cache = cls.new(cfg)
        cache.lengths = list(snap["lengths"])
        dim = cfg.head_count * cfg.head_dim
        for layer in range(cfg.layer_count):
            (k_len,) = struct.unpack_from("<I", raw, off)
            off += 4
            keys = [struct.unpack_from("<f", raw, off + i * 4)[0] for i in range(k_len)]
            off += k_len * 4
            (v_len,) = struct.unpack_from("<I", raw, off)
            off += 4
            vals = [struct.unpack_from("<f", raw, off + i * 4)[0] for i in range(v_len)]
            off += v_len * 4
            cache.keys[layer] = keys
            cache.values[layer] = vals
            cache.lengths[layer] = k_len // dim if dim else 0
        cache.occupied = cache.occupied_tokens()
        return cache
