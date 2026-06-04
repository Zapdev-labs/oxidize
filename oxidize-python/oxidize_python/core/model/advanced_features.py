"""Advanced samplers and tool registry mirroring oxidize-golang/core/model/advanced_features.go."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol

from oxidize_python.core.model.model import Logits, Token
from oxidize_python.core.model.sampling import (
    NEG_INF_F32,
    MirostatConfig,
    SamplingConfig,
    _min_p,
    _softmax,
    _sorted_indices,
    _top_k,
    _top_p,
    default_sampling_config,
)

TokenType = Token


@dataclass
class XtcSamplerConfig:
    threshold: float = 0.0
    min_keep: int = 1


@dataclass
class DrySamplerConfig:
    multiplier: float = 0.0
    base: float = 0.0
    allowed_length: int = 0
    range: int = 0


@dataclass
class DynamicTempConfig:
    min_temp: float = 0.0
    max_temp: float = 0.0
    exponent: float = 0.0


@dataclass
class NewlinePenalty:
    count: int = 0
    reward: float = 0.0


@dataclass
class SamplerContext:
    history: list[Token] = field(default_factory=list)
    rng_seed: int = 0
    position: int = 0
    temperature: float = 1.0
    custom: dict[str, Any] = field(default_factory=dict)


class SamplerStep(Protocol):
    def apply(self, logits: Logits, ctx: SamplerContext | None) -> Logits: ...


@dataclass
class SamplerChain:
    steps: list[SamplerStep] = field(default_factory=list)

    def add(self, step: SamplerStep) -> None:
        self.steps.append(step)

    def run(self, logits: Logits, ctx: SamplerContext | None) -> Logits:
        for step in self.steps:
            logits = step.apply(logits, ctx)
        return logits


@dataclass
class TemperatureStep:
    temp: float = 1.0

    def apply(self, logits: Logits, _ctx: SamplerContext | None) -> Logits:
        if self.temp == 0 or self.temp == 1.0:
            return logits
        inv = 1.0 / self.temp
        return [v * inv for v in logits]


@dataclass
class TopKStep:
    k: int = 40

    def apply(self, logits: Logits, _ctx: SamplerContext | None) -> Logits:
        return _top_k(logits, self.k)


@dataclass
class TopPStep:
    p: float = 0.9

    def apply(self, logits: Logits, _ctx: SamplerContext | None) -> Logits:
        return _top_p(logits, self.p)


@dataclass
class MinPStep:
    p: float = 0.05

    def apply(self, logits: Logits, _ctx: SamplerContext | None) -> Logits:
        return _min_p(logits, self.p)


@dataclass
class TypicalPStep:
    p: float = 0.9

    def apply(self, logits: Logits, _ctx: SamplerContext | None) -> Logits:
        if self.p <= 0 or not logits:
            return logits
        probs = _softmax(logits)
        entropy = sum(-p * math.log(p) for p in probs if p > 0)
        cands = [(i, abs(-math.log(p + 1e-12) - entropy)) for i, p in enumerate(probs)]
        cands.sort(key=lambda x: x[1])
        cum_prob = 0.0
        keep = [False] * len(probs)
        for i, _ in cands:
            keep[i] = True
            cum_prob += probs[i]
            if cum_prob >= self.p:
                break
        out = list(logits)
        for i, k in enumerate(keep):
            if not k:
                out[i] = NEG_INF_F32
        return out


@dataclass
class TailFreeStep:
    z: float = 0.9

    def apply(self, logits: Logits, _ctx: SamplerContext | None) -> Logits:
        if self.z <= 0 or len(logits) < 3:
            return logits
        idx = _sorted_indices(logits)
        probs = _softmax(logits)
        second_derivs = [
            abs(probs[idx[i + 1]] - 2 * probs[idx[i]] + probs[idx[i - 1]])
            for i in range(1, len(idx) - 1)
        ]
        total = sum(second_derivs)
        keep = [False] * len(idx)
        cum_z = 0.0
        for i, sd in enumerate(second_derivs):
            if total > 0:
                cum_z += sd / total
            if cum_z < self.z:
                keep[i + 1] = True
            else:
                break
        keep[0] = True
        out = [NEG_INF_F32] * len(logits)
        for pos, k in enumerate(idx):
            if keep[pos]:
                out[k] = logits[k]
        return out


@dataclass
class DryStep:
    cfg: DrySamplerConfig = field(default_factory=DrySamplerConfig)

    def apply(self, logits: Logits, ctx: SamplerContext | None) -> Logits:
        if self.cfg.multiplier == 0 or ctx is None or len(ctx.history) < 2:
            return logits
        range_len = self.cfg.range or len(ctx.history)
        range_len = min(range_len, len(ctx.history))
        out = list(logits)
        allowed = self.cfg.allowed_length or 1
        for n in range(allowed, range_len):
            if len(ctx.history) < n * 2:
                continue
            a = ctx.history[-n:]
            b = ctx.history[-2 * n : -n]
            if a != b:
                continue
            for t in a:
                if int(t) < len(out):
                    out[t] -= self.cfg.multiplier * self.cfg.base
        return out


@dataclass
class XtcStep:
    cfg: XtcSamplerConfig = field(default_factory=XtcSamplerConfig)

    def apply(self, logits: Logits, _ctx: SamplerContext | None) -> Logits:
        if self.cfg.threshold <= 0 or not logits:
            return logits
        idx = _sorted_indices(logits)
        probs = _softmax(logits)
        removed = 0
        out = list(logits)
        for i in idx:
            if len(probs) - removed <= self.cfg.min_keep:
                break
            if probs[i] > self.cfg.threshold:
                out[i] = NEG_INF_F32
                removed += 1
        return out


@dataclass
class DynamicTempStep:
    cfg: DynamicTempConfig = field(default_factory=DynamicTempConfig)

    def apply(self, logits: Logits, ctx: SamplerContext | None) -> Logits:
        if self.cfg.min_temp == 0 and self.cfg.max_temp == 0:
            return logits
        if ctx is None:
            return logits
        if self.cfg.exponent == 0:
            ctx.temperature = 1.0
            return logits
        ctx.temperature = self.cfg.min_temp + (self.cfg.max_temp - self.cfg.min_temp) * (
            ctx.position**self.cfg.exponent
        )
        if ctx.temperature == 0 or ctx.temperature == 1.0:
            return logits
        inv = 1.0 / ctx.temperature
        return [v * inv for v in logits]


@dataclass
class NewlinePenaltyStep:
    cfg: NewlinePenalty = field(default_factory=NewlinePenalty)

    def apply(self, logits: Logits, _ctx: SamplerContext | None) -> Logits:
        if self.cfg.count <= 0:
            return logits
        out = list(logits)
        nl = ord("\n")
        if nl < len(out):
            out[nl] += self.cfg.reward * self.cfg.count
        return out


DEFAULT_MIROSTAT_CONFIG = MirostatConfig(tau=5.0, eta=0.1, m=5)


@dataclass
class ToolFunction:
    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


class ToolFormat(StrEnum):
    JSON = "json"
    YAML = "yaml"
    NATIVE = "native"


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolFunction] = {}

    def register(self, tool: ToolFunction) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> tuple[ToolFunction, bool]:
        t = self._tools.get(name)
        return t, t is not None

    def names(self) -> list[str]:
        return sorted(self._tools.keys())


@dataclass
class EngineConfig:
    sampling: SamplingConfig = field(default_factory=default_sampling_config)
    template: str = ""
    stop_tokens: list[Token] = field(default_factory=list)
    tools: list[ToolFunction] = field(default_factory=list)


def new_engine_config() -> EngineConfig:
    return EngineConfig()
