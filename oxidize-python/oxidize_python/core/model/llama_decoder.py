"""Llama decoder stack types mirroring oxidize-golang/core/model/llama_decoder.go."""

from __future__ import annotations

from dataclasses import dataclass, field

from oxidize_python.core.model.dflash_weights import F32Weight


@dataclass
class LlamaDecoderConfig:
    hidden_size: int
    layer_count: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    key_value_head_dim: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    # Gemma-family fields.
    sliding_window: int = 0
    rope_theta_swa: float = 0.0
    sliding_window_pattern: int = 0
    embedding_scale: float = 1.0
    gelu_ffn: bool = False
    sandwich_norm: bool = False

    def head_dim(self) -> int:
        if self.num_attention_heads == 0:
            return 0
        return self.hidden_size // self.num_attention_heads

    def kv_head_dim(self) -> int:
        if self.key_value_head_dim > 0:
            return self.key_value_head_dim
        return self.head_dim()

    def layer_is_global(self, idx: int) -> bool:
        """Whether layer idx uses global (full) attention rather than SWA.

        No window -> global. Window but no interleave pattern (Mistral) ->
        local. Gemma: every Nth layer (1-indexed) is global.
        """
        if self.sliding_window == 0:
            return True
        if self.sliding_window_pattern == 0:
            return False
        return (idx + 1) % self.sliding_window_pattern == 0

    def layer_rope_theta(self, idx: int) -> float:
        if self.rope_theta_swa > 0 and not self.layer_is_global(idx):
            return self.rope_theta_swa
        return self.rope_theta

    def layer_sliding_window(self, idx: int) -> int:
        if self.sliding_window > 0 and not self.layer_is_global(idx):
            return self.sliding_window
        return 0


@dataclass
class DecoderAttentionLayer:
    q_proj: F32Weight = field(default_factory=F32Weight.empty)
    k_proj: F32Weight = field(default_factory=F32Weight.empty)
    v_proj: F32Weight = field(default_factory=F32Weight.empty)
    o_proj: F32Weight = field(default_factory=F32Weight.empty)
    q_norm_weight: list[float] = field(default_factory=list)
    k_norm_weight: list[float] = field(default_factory=list)


@dataclass
class DecoderLayer:
    input_layernorm: list[float] = field(default_factory=list)
    # For Gemma (sandwich norm) post_attention_layernorm is the true
    # post-attention norm, pre_ffn_layernorm is the pre-MLP norm, and
    # post_ffn_layernorm is applied to the FFN output before its residual.
    post_attention_layernorm: list[float] = field(default_factory=list)
    pre_ffn_layernorm: list[float] = field(default_factory=list)
    post_ffn_layernorm: list[float] = field(default_factory=list)
    attention: DecoderAttentionLayer = field(default_factory=DecoderAttentionLayer)
    mlp_gate: F32Weight = field(default_factory=F32Weight.empty)
    mlp_up: F32Weight = field(default_factory=F32Weight.empty)
    mlp_down: F32Weight = field(default_factory=F32Weight.empty)


@dataclass
class DecoderKvLayerCache:
    keys: list[float] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    seq_len: int = 0

    def clear(self) -> None:
        self.keys = []
        self.values = []
        self.seq_len = 0

    def reserve_tokens(self, additional: int, kv_len: int) -> None:
        add = additional * kv_len
        if len(self.keys) + add > len(self.keys):
            self.keys.extend([0.0] * max(0, add))
        if len(self.values) + add > len(self.values):
            self.values.extend([0.0] * max(0, add))


@dataclass
class LlamaDecoderStack:
    config: LlamaDecoderConfig
    layers: list[DecoderLayer] = field(default_factory=list)
    tok_embeddings: F32Weight = field(default_factory=F32Weight.empty)
    output: F32Weight = field(default_factory=F32Weight.empty)
    norm: list[float] = field(default_factory=list)
    kv_cache: list[DecoderKvLayerCache] = field(default_factory=list)
    position_offset: int = 0

    @staticmethod
    def new(config: LlamaDecoderConfig) -> LlamaDecoderStack:
        return LlamaDecoderStack(
            config=config,
            kv_cache=[DecoderKvLayerCache() for _ in range(config.layer_count)],
        )

    def loaded(self) -> bool:
        return self.tok_embeddings.is_loaded() and self.output.is_loaded()

    def reset_cache(self) -> None:
        if len(self.kv_cache) != self.config.layer_count:
            self.kv_cache = [DecoderKvLayerCache() for _ in range(self.config.layer_count)]
        else:
            for c in self.kv_cache:
                c.clear()
        self.position_offset = 0

    def reserve_cache_tokens(self, tokens: int) -> None:
        kv_len = self.config.num_key_value_heads * self.config.kv_head_dim()
        for c in self.kv_cache:
            c.reserve_tokens(tokens, kv_len)

    def scale_embedding(self, x: list[float]) -> None:
        """Multiply embedding by config.embedding_scale (Gemma uses sqrt(hidden))."""
        scale = self.config.embedding_scale
        if scale in (0.0, 1.0):
            return
        for i in range(len(x)):
            x[i] *= scale

    def fill_token_embedding(self, token: int, output: list[float]) -> None:
        if not self.tok_embeddings.is_loaded():
            return
        vocab = self.tok_embeddings._output_dim()
        if vocab < 1:
            vocab = 1
        idx = token
        if idx >= vocab:
            idx = vocab - 1
        emb_w = self.tok_embeddings._input_dim()
        if emb_w == len(output):
            self.tok_embeddings.row(idx, output)
            return
        emb = [0.0] * emb_w
        self.tok_embeddings.row(idx, emb)
        n = min(len(output), len(emb))
        output[:n] = emb[:n]
