"""Prefix cache mirroring oxidize-golang/core/model/prefix_cache.go."""

from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass

from oxidize_python.core.model.model import Token

PrefixHash = bytes


def compute_prefix_hash(prefix: list[Token]) -> PrefixHash:
    buf = bytearray()
    for t in prefix:
        buf.extend((int(t) & 0xFFFFFFFF).to_bytes(4, "little", signed=False))
    return hashlib.sha1(buf).digest()


@dataclass
class CachedPrefix:
    hash: PrefixHash
    tokens: list[Token]
    length: int
    shared_layers: int


class PrefixCache:
    def __init__(self, limit: int = 32) -> None:
        self._limit = limit if limit > 0 else 32
        self._entries: dict[PrefixHash, CachedPrefix] = {}
        self._mu = threading.Lock()
        self._hits = 0
        self._misses = 0

    def lookup(self, prefix: list[Token]) -> tuple[CachedPrefix | None, bool]:
        h = compute_prefix_hash(prefix)
        with self._mu:
            entry = self._entries.get(h)
            if entry is not None:
                self._hits += 1
                return entry, True
            self._misses += 1
            return None, False

    def insert(self, prefix: list[Token], shared_layers: int) -> CachedPrefix:
        h = compute_prefix_hash(prefix)
        entry = CachedPrefix(
            hash=h,
            tokens=list(prefix),
            length=len(prefix),
            shared_layers=shared_layers,
        )
        with self._mu:
            if len(self._entries) >= self._limit:
                oldest: CachedPrefix | None = None
                for e in self._entries.values():
                    if oldest is None or e.length < oldest.length:
                        oldest = e
                if oldest is not None:
                    del self._entries[oldest.hash]
            self._entries[h] = entry
        return entry

    def clear(self) -> None:
        with self._mu:
            self._entries.clear()

    def stats(self) -> tuple[int, int]:
        with self._mu:
            return self._hits, self._misses

    def hit_rate(self) -> float:
        with self._mu:
            total = self._hits + self._misses
            if total == 0:
                return 0.0
            return self._hits / total


class PrefixMatcher:
    def __init__(self, cache: PrefixCache | None) -> None:
        self.cache = cache

    def longest_match(self, tokens: list[Token]) -> CachedPrefix | None:
        if self.cache is None:
            return None
        for n in range(len(tokens), 0, -1):
            entry, ok = self.cache.lookup(tokens[:n])
            if ok:
                return entry
        return None

    def can_reuse_state(self, prefix: list[Token]) -> bool:
        entry, ok = self.cache.lookup(prefix) if self.cache else (None, False)
        return ok and entry is not None and entry.shared_layers > 0


@dataclass
class PrefixCacheConfig:
    max_entries: int = 64
    min_prefix_length: int = 4
    enable_lru: bool = True


def default_prefix_cache_config() -> PrefixCacheConfig:
    return PrefixCacheConfig()


def build_prefix_cache(cfg: PrefixCacheConfig) -> PrefixCache:
    return PrefixCache(cfg.max_entries)
