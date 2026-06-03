"""Token generation streams mirroring oxidize-golang/core/model/generation.go."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

from oxidize_python.core.model.model import Logits, Model, Session, Token
from oxidize_python.core.model.sampling import (
    SamplingConfig,
    default_sampling_config,
    sample,
    speculative_decode,
)


class GenerationError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(f"generation: {message}")

    def is_model_error(self) -> bool:
        return False


ERR_GENERATION_FINISHED = GenerationError("generation finished")


def is_finished(err: BaseException | None) -> bool:
    return err is not None and err is ERR_GENERATION_FINISHED


@dataclass
class GenerationConfig:
    max_new_tokens: int = 128
    stop_token: Token = 2
    stop_sequences: list[list[Token]] | None = None
    prefill_batch_size: int = 1
    sampling: SamplingConfig = field(default_factory=default_sampling_config)
    suppressed_tokens: list[Token] | None = None


def default_generation_config() -> GenerationConfig:
    return GenerationConfig(
        max_new_tokens=128,
        stop_token=2,
        stop_sequences=None,
        prefill_batch_size=1,
        sampling=default_sampling_config(),
        suppressed_tokens=None,
    )


@dataclass
class SpeculativeGenerationConfig:
    generation: GenerationConfig = field(default_factory=default_generation_config)
    draft_tokens_per_step: int = 4


def default_speculative_generation_config() -> SpeculativeGenerationConfig:
    return SpeculativeGenerationConfig(
        generation=default_generation_config(),
        draft_tokens_per_step=4,
    )


class GenerationStream:
    def __init__(self, model: Model, session: Session, config: GenerationConfig) -> None:
        self._mu = threading.Lock()
        self._model = model
        self._session = session
        self._config = config
        self._done = False
        self._tokens: list[Token] = []

    def seed(self, prompt: list[Token]) -> None:
        with self._mu:
            if not self._tokens:
                self._tokens = list(prompt)

    def next(self) -> tuple[Token, bool, GenerationError | None]:
        with self._mu:
            if self._done:
                return 0, True, None
            if self._config.max_new_tokens > 0 and len(self._tokens) >= self._config.max_new_tokens:
                self._done = True
                return 0, True, None
            try:
                logits: Logits = self._model.forward(self._tokens, self._session)
            except Exception as err:
                return 0, True, GenerationError(str(err))
            if not logits:
                return 0, True, GenerationError("empty logits")
            try:
                tok = sample(logits, self._config.sampling, None)
            except Exception as err:
                return 0, True, GenerationError(str(err))
            self._tokens.append(tok)
            if tok == self._config.stop_token:
                self._done = True
                return tok, True, None
            for seq in self._config.stop_sequences or []:
                if not seq:
                    continue
                if len(self._tokens) >= len(seq):
                    tail = self._tokens[-len(seq) :]
                    if tail == seq:
                        self._done = True
                        return tok, True, None
            return tok, False, None


class SpeculativeGenerationStream:
    def __init__(
        self,
        draft: Model,
        target: Model,
        session: Session,
        config: SpeculativeGenerationConfig,
    ) -> None:
        self._mu = threading.Lock()
        self._draft = draft
        self._target = target
        self._session = session
        self._config = config
        self._done = False

    def next(self) -> tuple[Token, bool, GenerationError | None]:
        with self._mu:
            if self._done:
                return 0, True, None
            try:
                tokens, accepted = speculative_decode(
                    self._draft,
                    self._target,
                    self._config.draft_tokens_per_step,
                    self._session,
                )
            except Exception as err:
                return 0, True, GenerationError(str(err))
            if accepted == 0:
                self._done = True
                return 0, True, None
            return tokens[accepted - 1], False, None
