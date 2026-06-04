"""Legacy Llama model stubs mirroring oxidize-golang/core/model/llama.go."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from oxidize_python.core.model.model import Logits, Session, Token


class LlamaArchitecture(StrEnum):
    LLAMA2 = "llama2"
    LLAMA3 = "llama3"
    MISTRAL = "mistral"
    MIXTRAL = "mixtral"
    QWEN = "qwen"
    GEMMA = "gemma"
    PHI = "phi"
    FALCON = "falcon"
    GPT2 = "gpt2"
    GPTJ = "gptj"
    GPT_NEOX = "gpt_neox"


@dataclass
class LlamaConfig:
    architecture: LlamaArchitecture
    vocab_size: int
    context_size: int
    layer_count: int


class LlamaModel:
    def __init__(self, config: LlamaConfig) -> None:
        self.config = config

    def forward(self, _tokens: list[Token], _session: Session) -> Logits:
        return [0.0] * self.config.vocab_size

    def vocab_size(self) -> int:
        return self.config.vocab_size

    def context_size(self) -> int:
        return self.config.context_size

    def layer_count(self) -> int:
        return self.config.layer_count


def new_llama2() -> LlamaModel:
    return LlamaModel(LlamaConfig(LlamaArchitecture.LLAMA2, 32000, 2048, 32))


def new_llama3() -> LlamaModel:
    return LlamaModel(LlamaConfig(LlamaArchitecture.LLAMA3, 128256, 8192, 32))


def new_mistral() -> LlamaModel:
    return LlamaModel(LlamaConfig(LlamaArchitecture.MISTRAL, 32000, 32768, 32))


def new_mixtral() -> LlamaModel:
    return LlamaModel(LlamaConfig(LlamaArchitecture.MIXTRAL, 32000, 32768, 32))


def new_qwen() -> LlamaModel:
    return LlamaModel(LlamaConfig(LlamaArchitecture.QWEN, 151936, 32768, 32))


def new_gemma() -> LlamaModel:
    return LlamaModel(LlamaConfig(LlamaArchitecture.GEMMA, 256000, 8192, 18))


def new_phi() -> LlamaModel:
    return LlamaModel(LlamaConfig(LlamaArchitecture.PHI, 51200, 2048, 32))


def new_falcon() -> LlamaModel:
    return LlamaModel(LlamaConfig(LlamaArchitecture.FALCON, 65024, 2048, 32))


def new_gpt2() -> LlamaModel:
    return LlamaModel(LlamaConfig(LlamaArchitecture.GPT2, 50257, 1024, 12))


def new_gptj() -> LlamaModel:
    return LlamaModel(LlamaConfig(LlamaArchitecture.GPTJ, 50400, 2048, 28))


def new_gpt_neox() -> LlamaModel:
    return LlamaModel(LlamaConfig(LlamaArchitecture.GPT_NEOX, 50432, 2048, 32))
