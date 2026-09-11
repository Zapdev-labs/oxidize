from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from scripts.dflash.config import DFlashTrainConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(var + self.eps) * self.weight


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return x * cos + rotate_half(x) * sin


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, theta: float) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, position_ids: torch.Tensor, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq = self.inv_freq.to(device=position_ids.device, dtype=torch.float32)
        freqs = torch.einsum("bi,j->bij", position_ids.float(), inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(dtype=dtype).unsqueeze(1)
        sin = emb.sin().to(dtype=dtype).unsqueeze(1)
        return cos, sin


class DFlashAttention(nn.Module):
    def __init__(self, cfg: DFlashTrainConfig) -> None:
        super().__init__()
        self.cfg = cfg
        q_out = cfg.num_attention_heads * cfg.head_dim
        kv_out = cfg.num_key_value_heads * cfg.head_dim
        self.q_proj = nn.Linear(cfg.hidden_size, q_out, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, kv_out, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, kv_out, bias=False)
        self.o_proj = nn.Linear(q_out, cfg.hidden_size, bias=False)
        self.q_norm = RMSNorm(cfg.head_dim, cfg.rms_norm_eps)
        self.k_norm = RMSNorm(cfg.head_dim, cfg.rms_norm_eps)
        self.n_rep = cfg.num_attention_heads // cfg.num_key_value_heads
        self.scale = cfg.head_dim ** -0.5

    def _shape_q(self, x: torch.Tensor, b: int, t: int) -> torch.Tensor:
        return x.view(b, t, self.cfg.num_attention_heads, self.cfg.head_dim).transpose(1, 2)

    def _shape_kv(self, x: torch.Tensor, b: int, t: int) -> torch.Tensor:
        return x.view(b, t, self.cfg.num_key_value_heads, self.cfg.head_dim).transpose(1, 2)

    def forward(
        self,
        noise: torch.Tensor,
        context: torch.Tensor,
        noise_pos: torch.Tensor,
        context_pos: torch.Tensor,
        rotary: RotaryEmbedding,
        attn_bias: torch.Tensor | None,
    ) -> torch.Tensor:
        bsz, blk, _ = noise.shape
        ctx_len = context.shape[1]
        q = self._shape_q(self.q_proj(noise), bsz, blk)
        q = self.q_norm(q)
        k_noise = self._shape_kv(self.k_proj(noise), bsz, blk)
        v_noise = self._shape_kv(self.v_proj(noise), bsz, blk)
        k_ctx = self._shape_kv(self.k_proj(context), bsz, ctx_len)
        v_ctx = self._shape_kv(self.v_proj(context), bsz, ctx_len)
        k_noise = self.k_norm(k_noise)
        k_ctx = self.k_norm(k_ctx)
        q_cos, q_sin = rotary(noise_pos, q.dtype)
        c_cos, c_sin = rotary(context_pos, k_ctx.dtype)
        q = apply_rope(q, q_cos, q_sin)
        k_noise = apply_rope(k_noise, q_cos, q_sin)
        k_ctx = apply_rope(k_ctx, c_cos, c_sin)
        k = torch.cat((k_ctx, k_noise), dim=2)
        v = torch.cat((v_ctx, v_noise), dim=2)
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)
        attn = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_bias, dropout_p=0.0, is_causal=False, scale=self.scale
        )
        attn = attn.transpose(1, 2).contiguous().view(bsz, blk, -1)
        return self.o_proj(attn)


class DFlashMLP(nn.Module):
    def __init__(self, cfg: DFlashTrainConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DFlashDecoderLayer(nn.Module):
    def __init__(self, cfg: DFlashTrainConfig) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.self_attn = DFlashAttention(cfg)
        self.mlp = DFlashMLP(cfg)

    def forward(
        self,
        hidden: torch.Tensor,
        context: torch.Tensor,
        noise_pos: torch.Tensor,
        context_pos: torch.Tensor,
        rotary: RotaryEmbedding,
        attn_bias: torch.Tensor | None,
    ) -> torch.Tensor:
        residual = hidden
        hidden = self.input_layernorm(hidden)
        hidden = residual + self.self_attn(
            hidden, context, noise_pos, context_pos, rotary, attn_bias
        )
        residual = hidden
        hidden = self.post_attention_layernorm(hidden)
        return residual + self.mlp(hidden)


class DFlashDraftModel(nn.Module):
    """Block-diffusion draft with persistent KV injection of fused target features."""

    def __init__(self, cfg: DFlashTrainConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.fc = nn.Linear(cfg.n_feat, cfg.hidden_size, bias=False)
        self.hidden_norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.layers = nn.ModuleList(DFlashDecoderLayer(cfg) for _ in range(cfg.num_hidden_layers))
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.rotary = RotaryEmbedding(cfg.head_dim, cfg.rope_theta)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)

    def fuse_target(self, target_hidden: torch.Tensor) -> torch.Tensor:
        return self.hidden_norm(self.fc(target_hidden))

    def forward(
        self,
        noise: torch.Tensor,
        target_hidden: torch.Tensor,
        noise_pos: torch.Tensor,
        context_pos: torch.Tensor,
        attn_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        context = self.fuse_target(target_hidden)
        hidden = noise
        for layer in self.layers:
            hidden = layer(hidden, context, noise_pos, context_pos, self.rotary, attn_bias)
        return self.norm(hidden)


def block_attention_bias(
    context_keep: torch.Tensor,
    block_size: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Allow each query to attend to kept context tokens and the whole draft block."""
    anchors, ctx_len = context_keep.shape
    keep = context_keep.to(dtype=torch.bool)
    ctx_mask = keep[:, None, None, :].expand(anchors, 1, block_size, ctx_len)
    block_mask = torch.ones(
        anchors, 1, block_size, block_size, dtype=torch.bool, device=context_keep.device
    )
    allow = torch.cat((ctx_mask, block_mask), dim=-1)
    bias = torch.zeros_like(allow, dtype=dtype)
    return bias.masked_fill(~allow, float("-inf"))


def sample_anchors(seq_len: int, block_size: int, max_anchors: int, generator: torch.Generator) -> torch.Tensor:
    # An anchor at position a predicts labels seq[a + 1 : a + block_size], so the
    # last valid anchor is seq_len - block_size (inclusive).
    last = seq_len - block_size
    if last < 1:
        return torch.zeros(0, dtype=torch.long)
    pool = torch.arange(1, last + 1)
    n = min(max_anchors, pool.numel())
    perm = torch.randperm(pool.numel(), generator=generator)[:n]
    return pool[perm].contiguous()
