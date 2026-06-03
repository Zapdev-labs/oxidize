"""Inference model mirroring oxidize-golang/core/model/inference.go."""

from __future__ import annotations

import threading
from dataclasses import dataclass

from oxidize_python.core.ggufcore.gguf import MappedFile
from oxidize_python.core.kv_cache import Cache, EvictionStrategy
from oxidize_python.core.kv_cache import Config as KvConfig
from oxidize_python.core.model.inference_config import InferenceConfig
from oxidize_python.core.model.llama_decoder import LlamaDecoderStack
from oxidize_python.core.model.llama_decoder_forward import forward_batch, forward_token, logits
from oxidize_python.core.model.model import (
    ContextExceededError,
    EmptyInputError,
    Logits,
    Session,
    Token,
)


@dataclass
class WeightStorage:
    file: MappedFile | None = None


class Workspace:
    def __init__(self, capacity: int) -> None:
        self._mu = threading.Lock()
        self._scratch: list[float] = [0.0] * capacity

    def get(self, n: int) -> list[float]:
        with self._mu:
            if len(self._scratch) < n:
                self._scratch = [0.0] * n
            return self._scratch[:n]

    def capacity(self) -> int:
        return len(self._scratch)


class InferenceModel:
    def __init__(
        self,
        config: InferenceConfig,
        storage: WeightStorage,
        stack: LlamaDecoderStack | None,
    ) -> None:
        self.config = config
        self.workspace = Workspace(config.hidden_size * 4)
        self.storage = storage
        self.stack = stack
        kv_cfg = KvConfig(
            layer_count=config.layer_count,
            context_size=config.context_size,
            head_count=config.num_key_value_heads,
            head_dim=config.kv_head_dim(),
            dtype="f32",
            quantization=config.kv_quantization,
            eviction=EvictionStrategy.SLIDING_WINDOW,
        )
        self.kv_cache = Cache.new(kv_cfg)

    def forward(self, tokens: list[Token], session: Session) -> Logits:
        if not tokens:
            raise EmptyInputError
        requested = session.consumed_tokens() + len(tokens)
        if self.config.context_size > 0 and requested > self.config.context_size:
            raise ContextExceededError(self.config.context_size, requested)
        if self.stack is None or not self.stack.loaded():
            return [0.0] * self.config.vocab_size

        start_pos = session.consumed_tokens()
        if self.stack.position_offset != start_pos:
            self.stack.reset_cache()
            self.stack.position_offset = start_pos

        if len(tokens) > 1:
            batch = [int(t) for t in tokens]
            hidden = forward_batch(self.stack, batch)
        else:
            hidden = forward_token(self.stack, int(tokens[0]))
        out = logits(self.stack, hidden)
        session.record_tokens(len(tokens))
        return out

    def vocab_size(self) -> int:
        return self.config.vocab_size

    def context_size(self) -> int:
        return self.config.context_size

    def layer_count(self) -> int:
        return self.config.layer_count

    def rewind_to(self, _session: Session, n: int) -> None:
        n = max(0, n)
        if self.stack is not None:
            self.stack.reset_cache()
            self.stack.position_offset = n
        if self.kv_cache is not None:
            self.kv_cache = Cache.new(self.kv_cache.config)

    def __str__(self) -> str:
        return (
            f"InferenceModel{{arch={self.config.architecture} vocab={self.config.vocab_size} "
            f"ctx={self.config.context_size} layers={self.config.layer_count} "
            f"hidden={self.config.hidden_size}}}"
        )
