"""Token generation streams mirroring oxidize-golang/core/model/generation.go."""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from enum import Enum, auto

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


class _GenerationState(Enum):
    PREFILL = auto()
    DECODE = auto()
    DONE = auto()


class GenerationStream:
    def __init__(self, model: Model, session: Session, config: GenerationConfig) -> None:
        self._mu = threading.Lock()
        self._model = model
        self._session = session
        self._config = config
        self._state = _GenerationState.PREFILL
        self._done = False
        self._prompt: list[Token] = []
        self._generated = 0
        self._last_token: Token | None = None
        self._max_stop = max((len(s) for s in (config.stop_sequences or [])), default=0)
        self._recent: list[Token] = []

    def seed(self, prompt: list[Token]) -> None:
        with self._mu:
            if not self._prompt and prompt:
                self._prompt = list(prompt)

    def _forward_prefill(self) -> Logits:
        batch = self._config.prefill_batch_size
        if batch <= 0:
            batch = 1
        logits: Logits = []
        for i in range(0, len(self._prompt), batch):
            chunk = self._prompt[i : i + batch]
            logits = self._model.forward(chunk, self._session)
        return logits

    def _suppress_tokens(self, logits: Logits) -> None:
        for tok in self._config.suppressed_tokens or []:
            if int(tok) < len(logits):
                logits[int(tok)] = -math.inf

    def _finish_after_token(self, tok: Token) -> bool:
        if tok == self._config.stop_token:
            return True
        if self._max_stop == 0:
            return False
        self._recent.append(tok)
        if len(self._recent) > self._max_stop:
            self._recent = self._recent[-self._max_stop :]
        for seq in self._config.stop_sequences or []:
            if not seq or len(self._recent) < len(seq):
                continue
            if self._recent[-len(seq) :] == seq:
                return True
        return False

    def next(self) -> tuple[Token, bool, GenerationError | None]:
        with self._mu:
            if self._done or self._state is _GenerationState.DONE:
                return 0, True, None
            if self._config.max_new_tokens > 0 and self._generated >= self._config.max_new_tokens:
                self._done = True
                self._state = _GenerationState.DONE
                return 0, True, None
            try:
                if self._state is _GenerationState.PREFILL:
                    self._state = _GenerationState.DECODE
                    logits = self._forward_prefill()
                else:
                    if self._last_token is None:
                        self._done = True
                        self._state = _GenerationState.DONE
                        return 0, True, None
                    logits = self._model.forward([self._last_token], self._session)
            except Exception as err:
                return 0, True, GenerationError(str(err))
            if not logits:
                return 0, True, GenerationError("empty logits")
            self._suppress_tokens(logits)
            try:
                tok = sample(logits, self._config.sampling, None)
            except Exception as err:
                return 0, True, GenerationError(str(err))
            self._generated += 1
            self._last_token = tok
            if self._finish_after_token(tok):
                self._done = True
                self._state = _GenerationState.DONE
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
        self._tokens: list[Token] = []

    def seed(self, prompt: list[Token]) -> None:
        with self._mu:
            if not self._tokens:
                self._tokens = list(prompt)

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
