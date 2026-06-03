"""Core model types mirroring oxidize-golang/core/model/model.go."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

Token = int
Logits = list[float]


class Session:
    def __init__(self) -> None:
        self._mu = threading.Lock()
        self._consumed = 0

    def consumed_tokens(self) -> int:
        with self._mu:
            return self._consumed

    def record_tokens(self, n: int) -> None:
        with self._mu:
            self._consumed += n

    def rewind_to(self, n: int) -> None:
        with self._mu:
            n = max(0, n)
            if n < self._consumed:
                self._consumed = n


class Error(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(f"model: {message}")


def new_errorf(fmt: str, *args: object) -> Error:
    return Error(fmt % args)


EmptyInputError = Error("empty input")


class ContextExceededError(Exception):
    def __init__(self, context_size: int, requested_total_tokens: int) -> None:
        self.context_size = context_size
        self.requested_total_tokens = requested_total_tokens
        super().__init__(
            f"model: context exceeded: {requested_total_tokens} > {context_size}"
        )


class Model(Protocol):
    def forward(self, tokens: list[Token], session: Session) -> Logits: ...

    def vocab_size(self) -> int: ...

    def context_size(self) -> int: ...

    def layer_count(self) -> int: ...


@dataclass
class Boxed:
    m: Model

    def forward(self, tokens: list[Token], session: Session) -> Logits:
        return self.m.forward(tokens, session)

    def vocab_size(self) -> int:
        return self.m.vocab_size()

    def context_size(self) -> int:
        return self.m.context_size()

    def layer_count(self) -> int:
        return self.m.layer_count()


def forward_many(m: Model, batch: list[list[Token]], session: Session) -> list[Logits]:
    out: list[Logits] = []
    for tokens in batch:
        out.append(m.forward(tokens, session))
    return out


class Architecture(StrEnum):
    LLAMA = "llama"
    MISTRAL = "mistral"
    MIXTRAL = "mixtral"
    DEEPSEEK = "deepseek"
    QWEN = "qwen"
    GEMMA = "gemma"
    PHI = "phi"
    FALCON = "falcon"
    GPT2 = "gpt2"
    GPTJ = "gptj"
    GPT_NEOX = "gpt_neox"
    MINIMAX = "minimax"


DEFAULT_ARCHITECTURE = Architecture.LLAMA


def uses_alibi(arch: Architecture) -> bool:
    return arch in (Architecture.GPTJ, Architecture.GPT_NEOX)


def uses_sliding_window(arch: Architecture) -> bool:
    return arch in (Architecture.MISTRAL, Architecture.MIXTRAL)


def uses_moe(arch: Architecture) -> bool:
    return arch in (Architecture.MIXTRAL, Architecture.DEEPSEEK)


def uses_parallel_attn_ffn(arch: Architecture) -> bool:
    return arch == Architecture.FALCON


def uses_mla(arch: Architecture) -> bool:
    return arch == Architecture.DEEPSEEK
