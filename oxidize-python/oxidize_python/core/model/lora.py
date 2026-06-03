"""LoRA adapters mirroring oxidize-golang/core/model/lora.go."""

from __future__ import annotations

from dataclasses import dataclass, field

from oxidize_python.core.model.model import Logits, Model, Session, Token


@dataclass
class LoraLayer:
    name: str
    rank: int
    alpha: float
    scale: float
    base_shape: list[int]
    up_loaded: bool = False
    down_loaded: bool = False


def new_lora_layer(name: str, rank: int, alpha: float, base_shape: list[int]) -> LoraLayer:
    scale = 1.0
    if alpha > 0 and rank > 0:
        scale = alpha / rank
    return LoraLayer(
        name=name,
        rank=rank,
        alpha=alpha,
        scale=scale,
        base_shape=base_shape,
    )


class LoraAdapter:
    def __init__(self, name: str, path: str, base: Model | None) -> None:
        self.name = name
        self.path = path
        self.layers: dict[str, LoraLayer] = {}
        self.rank = 0
        self.alpha = 0.0
        self.base_model = base

    def add_layer(self, layer: LoraLayer) -> None:
        self.layers[layer.name] = layer
        if layer.rank > 0 and self.rank == 0:
            self.rank = layer.rank
        if layer.alpha > 0 and self.alpha == 0:
            self.alpha = layer.alpha

    def apply(self, tokens: list[Token], session: Session) -> Logits:
        if self.base_model is None:
            raise LoraError("base model is nil")
        logits = self.base_model.forward(tokens, session)
        if self.rank > 0 and self.alpha > 0:
            scale = self.alpha / self.rank
            for i in range(len(logits)):
                logits[i] *= scale
        return logits

    def forward(self, tokens: list[Token], session: Session) -> Logits:
        return self.apply(tokens, session)

    def vocab_size(self) -> int:
        return self.base_model.vocab_size() if self.base_model else 0

    def context_size(self) -> int:
        return self.base_model.context_size() if self.base_model else 0

    def layer_count(self) -> int:
        return self.base_model.layer_count() if self.base_model else 0


class LoraError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(f"lora: {message}")


@dataclass
class LoraPlan:
    adapters: list[LoraAdapter] = field(default_factory=list)
    merge_strategy: str = "sequential"
    rank_budget: int = 0
    estimated_gain: float = 0.0

    def add(self, adapter: LoraAdapter) -> None:
        self.adapters.append(adapter)

    def validate(self) -> None:
        if not self.merge_strategy:
            raise LoraError("merge strategy is required")
        if not self.adapters:
            raise LoraError("no adapters in plan")
        if self.merge_strategy not in ("sequential", "parallel", "tying"):
            raise LoraError(f"unknown merge strategy: {self.merge_strategy}")

    def merge_adapters(self) -> LoraAdapter:
        merged = LoraAdapter("merged", "", None)
        merged.rank = self.rank_budget
        for a in self.adapters:
            for layer in a.layers.values():
                merged.add_layer(layer)
        merged.alpha = float(merged.rank) * self.estimated_gain
        return merged

    def estimate_memory(self) -> int:
        total = 0
        for a in self.adapters:
            for layer in a.layers.values():
                r = layer.rank or 1
                dim = 1
                for d in layer.base_shape:
                    dim *= d
                total += r * dim * 4
        return total

    def validate_rank_budget(self) -> None:
        for a in self.adapters:
            if a.rank > self.rank_budget:
                raise RankBudgetExceededError(a.rank, self.rank_budget)


class RankBudgetExceededError(Exception):
    def __init__(self, requested: int, available: int) -> None:
        super().__init__("lora: rank budget exceeded")
        self.requested = requested
        self.available = available


@dataclass
class LoraScalingConfig:
    auto: bool = False
    manual: float = 0.0
    max_gain: float = 0.0

    def scale_factor(self, rank: int) -> float:
        if self.auto:
            if rank == 0:
                return 0.0
            return min(self.max_gain, 1.0)
        return self.manual
