"""DFlash draft model types mirroring oxidize-golang/core/model/dflash.go."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from oxidize_python.core.model.llama_decoder import LlamaDecoderStack
from oxidize_python.core.model.model import Logits, Session, Token


class DFlashFeature(StrEnum):
    DRAFT_FORWARD = "draft-forward"
    TARGET_FUSION = "target-fusion"
    HEURISTIC_BLOCK = "heuristic-block"
    WEIGHT_SHARING = "weight-sharing"


def implemented_dflash_features() -> list[DFlashFeature]:
    return list(DFlashFeature)


@dataclass
class DFlashConfig:
    hidden_size: int = 4096
    num_hidden_layers: int = 32
    intermediate_size: int = 11008
    num_attention_heads: int = 32
    num_key_value_heads: int = 32
    vocab_size: int = 32000
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    block_size: int = 4


@dataclass
class DFlashDraftModel:
    config: DFlashConfig
    stack: LlamaDecoderStack | None = None

    def forward(self, tokens: list[Token], session: Session) -> Logits:
        if not tokens:
            return []
        if self.stack is None or not self.stack.loaded():
            return [0.0] * max(1, self.config.vocab_size)
        from oxidize_python.core.model.llama_decoder_forward import (
            forward_batch,
            forward_token,
            logits,
        )

        if len(tokens) > 1:
            hidden = forward_batch(self.stack, [int(t) for t in tokens])
        else:
            hidden = forward_token(self.stack, int(tokens[0]))
        session.record_tokens(len(tokens))
        return logits(self.stack, hidden)

    def vocab_size(self) -> int:
        return self.config.vocab_size

    def context_size(self) -> int:
        return 2048

    def layer_count(self) -> int:
        return self.config.num_hidden_layers
