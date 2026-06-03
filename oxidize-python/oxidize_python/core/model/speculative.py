"""Speculative decoding mirroring oxidize-golang/core/model/speculative.go."""

from __future__ import annotations

from dataclasses import dataclass

from oxidize_python.core.model.inference import InferenceModel
from oxidize_python.core.model.llama import LlamaModel
from oxidize_python.core.model.model import Model, Session, Token
from oxidize_python.core.model.sampling import (
    SamplingConfig,
    default_sampling_config,
    speculative_decode,
)


@dataclass
class SpeculativeConfig:
    draft_tokens_per_step: int = 4
    max_new_tokens: int = 128
    sampling: SamplingConfig | None = None
    stop_token: Token | None = None
    strict_mode: bool = False
    min_acceptance_rate: float = 0.3

    def __post_init__(self) -> None:
        if self.sampling is None:
            self.sampling = default_sampling_config()


def default_speculative_config() -> SpeculativeConfig:
    return SpeculativeConfig()


def conservative_speculative_config() -> SpeculativeConfig:
    c = default_speculative_config()
    c.draft_tokens_per_step = 2
    c.strict_mode = True
    return c


def aggressive_speculative_config() -> SpeculativeConfig:
    c = default_speculative_config()
    c.draft_tokens_per_step = 8
    c.min_acceptance_rate = 0.3
    return c


@dataclass
class SpeculativeStats:
    total_draft_tokens: int = 0
    total_accepted_tokens: int = 0
    total_rejected_tokens: int = 0
    draft_forward_passes: int = 0
    target_forward_passes: int = 0
    fallback_tokens: int = 0
    accepted: int = 0
    rejected: int = 0
    total: int = 0

    def acceptance_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return self.accepted / self.total


class SpeculativeError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(f"speculative: {message}")


class SpeculativeDecoder:
    def __init__(
        self,
        draft: Model,
        target: Model,
        session: Session,
        config: SpeculativeConfig,
    ) -> None:
        self.draft = draft
        self.target = target
        self.session = session
        self.config = config
        self.stats = SpeculativeStats()

    def step(self) -> list[Token]:
        if self.config.draft_tokens_per_step <= 0:
            raise SpeculativeError("draft tokens per step must be > 0")
        tokens, accepted = speculative_decode(
            self.draft,
            self.target,
            self.config.draft_tokens_per_step,
            self.session,
        )
        self.stats.total += self.config.draft_tokens_per_step
        if accepted < self.config.draft_tokens_per_step:
            self.stats.rejected += 1
        else:
            self.stats.accepted += self.config.draft_tokens_per_step
        return tokens


def load_draft_model_for_speculative(target: Model) -> Model:
    if isinstance(target, LlamaModel):
        cfg = target.config
        return LlamaModel(
            architecture=cfg.architecture,
            vocab_size=cfg.vocab_size,
            context_size=cfg.context_size,
            layer_count=max(1, cfg.layer_count // 2),
        )
    if isinstance(target, InferenceModel):
        from dataclasses import replace

        cfg = replace(target.config, layer_count=max(1, target.config.layer_count // 2))
        return InferenceModel(config=cfg, storage=target.storage, stack=target.stack)
    return target


@dataclass
class SpeculativeConfigBuilder:
    cfg: SpeculativeConfig | None = None

    def __post_init__(self) -> None:
        if self.cfg is None:
            self.cfg = default_speculative_config()

    def with_draft_tokens(self, n: int) -> SpeculativeConfigBuilder:
        assert self.cfg is not None
        self.cfg.draft_tokens_per_step = n
        return self

    def with_max_new_tokens(self, n: int) -> SpeculativeConfigBuilder:
        assert self.cfg is not None
        self.cfg.max_new_tokens = n
        return self

    def with_strict(self, s: bool) -> SpeculativeConfigBuilder:
        assert self.cfg is not None
        self.cfg.strict_mode = s
        return self

    def build(self) -> SpeculativeConfig:
        assert self.cfg is not None
        return self.cfg
