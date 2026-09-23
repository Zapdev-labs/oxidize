"""vLLM implementation of K2HorizonForCausalLM (K2-Horizon-MoVA-36B-A4B).

Architecture (see modeling_k2_horizon.py in the HF repo):
  * 48 decoder layers; layers 0-2 are dense (standard GQA attention + SwiGLU MLP),
    layers 3-47 use MoVA attention + a sigmoid-routed MoE (100 experts, top-8,
    1 shared expert, bias used only for expert selection, x2.5 routed scale).
  * MoVA attention: q/k are plain projections, but V is a routed mixture of 64
    "value experts": v = sum_{e in top4} w_e * silu(W_e x), with the same
    sigmoid + selection-bias + renormalize + x2.5 routing as the MoE.
  * Every attention layer multiplies its output by softplus_{beta=ln2}(gate_proj(x)).
  * RMSNorms are grouped: hidden 2560 is normalised as 2 independent groups of 1280.

The value experts are stored as one ColumnParallelLinear whose output rows are
ordered [kv_head, expert, head_dim]. Column sharding over TP=8 therefore gives
each chip exactly its own KV head for all 64 experts, so the routed value
mixture stays chip-local and lines up with the attention head sharding.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from itertools import islice

import torch
import torch.nn.functional as F
from torch import nn

from vllm.config import VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe import FusedMoEFactory
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import SupportsPP
from vllm.model_executor.models.utils import (
    extract_layer_index,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)
from vllm.sequence import IntermediateTensors

_LN2 = math.log(2.0)


def _is_sparse_layer(config, layer_idx: int) -> bool:
    mlp_only = config.mlp_only_layers or []
    return (layer_idx not in mlp_only) and config.num_experts > 0 and (
        (layer_idx + 1) % config.decoder_sparse_step == 0)


def _softplus_ln2(x: torch.Tensor) -> torch.Tensor:
    # torch.nn.functional.softplus(x, beta=ln2, threshold=20)
    bx = x.float() * _LN2
    y = torch.where(bx > 20.0, x.float(), torch.log1p(torch.exp(bx)) / _LN2)
    return y.to(x.dtype)


def _route(logits: torch.Tensor, bias: torch.Tensor, top_k: int,
           scale: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Sigmoid scores; bias only picks experts; weights renormalised then scaled."""
    scores = torch.sigmoid(logits.float())
    idx = torch.topk(scores + bias.float(), top_k, dim=-1).indices
    w = torch.gather(scores, -1, idx)
    w = w / w.sum(dim=-1, keepdim=True)
    return w * scale, idx


class K2GroupedRMSNorm(nn.Module):
    """RMSNorm over `n_groups` equal slices of the last dim; vLLM residual API."""

    def __init__(self, hidden_size: int, n_groups: int, eps: float):
        super().__init__()
        self.n_groups = n_groups
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        h = x.float()
        h = h.reshape(*h.shape[:-1], self.n_groups, -1)
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.eps)
        h = h.reshape(*h.shape[:-2], -1)
        return (self.weight.float() * h).to(dtype)

    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None):
        if residual is None:
            return self._norm(x)
        x = x + residual
        return self._norm(x), x


class K2MLP(nn.Module):

    def __init__(self, hidden_size: int, intermediate_size: int,
                 reduce_results: bool = True, prefix: str = ""):
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size, [intermediate_size] * 2, bias=False,
            prefix=f"{prefix}.gate_up_proj")
        self.down_proj = RowParallelLinear(
            intermediate_size, hidden_size, bias=False,
            reduce_results=reduce_results, prefix=f"{prefix}.down_proj")
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        out, _ = self.down_proj(self.act_fn(gate_up))
        return out


class K2RouterGate(nn.Module):
    """Router weight + selection-only bias (checkpoint names: .weight / .bias)."""

    def __init__(self, hidden_size: int, num_experts: int, prefix: str = ""):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_experts, hidden_size),
                                   requires_grad=False)
        self.e_score_correction_bias = nn.Parameter(
            torch.zeros(num_experts, dtype=torch.float32), requires_grad=False)
        self.weight.weight_loader = default_weight_loader
        self.e_score_correction_bias.weight_loader = default_weight_loader
        self.out_dtype = None

    def forward(self, x):
        return F.linear(x, self.weight), None


class K2SparseMoeBlock(nn.Module):

    def __init__(self, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        self.routed_scaling_factor = float(config.router_scaling_factor or 1.0)
        self.gate = K2RouterGate(config.hidden_size, config.num_experts,
                                 prefix=f"{prefix}.gate")
        self.shared_experts = None
        if config.num_shared_experts > 0:
            self.shared_experts = K2MLP(
                config.hidden_size,
                config.moe_intermediate_size * config.num_shared_experts,
                reduce_results=True,
                prefix=f"{prefix}.shared_experts")
        # Routing is DeepSeek-V3 style minus groups: sigmoid scores, the bias only
        # picks experts, picked scores are renormalised. The x2.5 scale is applied
        # here (not in the router) so it is honoured by every MoE backend.
        self.experts = FusedMoEFactory(
            gate=self.gate,
            num_experts=config.num_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=bool(config.norm_topk_prob),
            scoring_func=config.router_score_func,
            e_score_correction_bias=self.gate.e_score_correction_bias,
            routed_scaling_factor=1.0,
            prefix=f"{prefix}.experts",
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        out = self.experts(hidden_states=hidden_states,
                           router_logits=hidden_states)
        out = out * self.routed_scaling_factor
        if self.shared_experts is not None:
            out = out + self.shared_experts(hidden_states)
        return out


class K2Attention(nn.Module):
    """Dense or MoVA attention, both with the softplus output gate."""

    def __init__(self, vllm_config: VllmConfig, mova: bool, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        tp = get_tensor_model_parallel_world_size()
        self.head_dim = config.head_dim
        self.total_heads = config.num_attention_heads
        self.total_kv_heads = config.num_key_value_heads
        assert self.total_heads % tp == 0 and self.total_kv_heads % tp == 0, (
            "K2 plugin assumes TP divides the 8 KV heads")
        self.num_heads = self.total_heads // tp
        self.num_kv_heads = self.total_kv_heads // tp
        self.mova = mova
        hs = config.hidden_size
        assert config.attention_gate_func == "softplus"
        assert not config.query_key_norm
        assert (config.rope_head_dim or self.head_dim) == self.head_dim

        self.q_proj = ColumnParallelLinear(hs, self.total_heads * self.head_dim,
                                           bias=False, prefix=f"{prefix}.q_proj")
        self.k_proj = ColumnParallelLinear(hs, self.total_kv_heads * self.head_dim,
                                           bias=False, prefix=f"{prefix}.k_proj")
        self.gate_proj = ColumnParallelLinear(hs, self.total_heads * self.head_dim,
                                              bias=False, prefix=f"{prefix}.gate_proj")
        if mova:
            self.n_vexp = config.mova_num_experts
            self.v_top_k = config.mova_num_experts_per_tok
            self.v_scale = float(config.router_scaling_factor or 1.0)
            self.v_router = K2RouterGate(hs, self.n_vexp, prefix=f"{prefix}.v_router")
            # rows ordered [kv_head, expert, head_dim] -> column shards = kv heads
            self.v_experts = ColumnParallelLinear(
                hs, self.total_kv_heads * self.n_vexp * self.head_dim,
                bias=False, prefix=f"{prefix}.v_experts")
        else:
            self.v_proj = ColumnParallelLinear(hs, self.total_kv_heads * self.head_dim,
                                               bias=False, prefix=f"{prefix}.v_proj")
        self.o_proj = RowParallelLinear(self.total_heads * self.head_dim, hs,
                                        bias=False, prefix=f"{prefix}.o_proj")
        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=config.max_position_embeddings,
            is_neox_style=True,
            rope_parameters=config.rope_parameters,
        )
        self.attn = Attention(self.num_heads, self.head_dim, self.head_dim**-0.5,
                              num_kv_heads=self.num_kv_heads,
                              cache_config=vllm_config.cache_config,
                              prefix=f"{prefix}.attn")

    def _mova_values(self, x: torch.Tensor) -> torch.Tensor:
        t = x.shape[0]
        logits = F.linear(x, self.v_router.weight)
        w, idx = _route(logits, self.v_router.e_score_correction_bias,
                        self.v_top_k, self.v_scale)
        dense_w = torch.zeros(t, self.n_vexp, dtype=torch.float32,
                              device=x.device).scatter(1, idx, w)
        y, _ = self.v_experts(x)                       # [T, kvh*E*D] (local kvh)
        y = F.silu(y.view(t, -1, self.n_vexp, self.head_dim))
        v = torch.einsum("thed,te->thd", y, dense_w.to(y.dtype))
        return v.reshape(t, -1)

    def forward(self, positions: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        q, _ = self.q_proj(x)
        k, _ = self.k_proj(x)
        if self.mova:
            v = self._mova_values(x)
        else:
            v, _ = self.v_proj(x)
        q, k = self.rotary_emb(positions, q, k)
        o = self.attn(q, k, v)
        g, _ = self.gate_proj(x)
        o = o * _softplus_ln2(g)
        out, _ = self.o_proj(o)
        return out


class K2DecoderLayer(nn.Module):

    def __init__(self, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        layer_idx = extract_layer_index(prefix)
        sparse = _is_sparse_layer(config, layer_idx)
        self.self_attn = K2Attention(vllm_config,
                                     mova=sparse and config.mova_num_experts > 0,
                                     prefix=f"{prefix}.self_attn")
        if sparse:
            self.mlp = K2SparseMoeBlock(vllm_config, prefix=f"{prefix}.mlp")
        else:
            self.mlp = K2MLP(config.hidden_size, config.intermediate_size,
                             prefix=f"{prefix}.mlp")
        g = config.layernorm_num_groups
        self.input_layernorm = K2GroupedRMSNorm(config.hidden_size, g, config.rms_norm_eps)
        self.post_attention_layernorm = K2GroupedRMSNorm(config.hidden_size, g,
                                                         config.rms_norm_eps)

    def forward(self, positions, hidden_states, residual):
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class K2HorizonModel(nn.Module):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size,
                                                   prefix=f"{prefix}.embed_tokens")
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: K2DecoderLayer(vllm_config=vllm_config, prefix=prefix),
            prefix=f"{prefix}.layers",
        )
        self.norm = K2GroupedRMSNorm(config.hidden_size, config.layernorm_num_groups,
                                     config.rms_norm_eps)
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(self, input_ids, positions, intermediate_tensors=None,
                inputs_embeds=None):
        if get_pp_group().is_first_rank:
            hidden_states = (inputs_embeds if inputs_embeds is not None
                             else self.embed_input_ids(input_ids))
            residual = None
        else:
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]
        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states, residual = layer(positions, hidden_states, residual)
        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states,
                                        "residual": residual})
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class K2HorizonForCausalLM(nn.Module, SupportsPP):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        self.config = config
        self.model = K2HorizonModel(vllm_config=vllm_config,
                                    prefix=maybe_prefix(prefix, "model"))
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size,
                                      prefix=maybe_prefix(prefix, "lm_head"))
        if config.tie_word_embeddings:
            self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(self, input_ids, positions, intermediate_tensors=None,
                inputs_embeds=None):
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    # ------------------------------------------------------------------ weights
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        cfg = self.config
        params = dict(self.named_parameters())
        modules = dict(self.named_modules())
        loaded: set[str] = set()
        # per-layer buffers for the 64 value experts: {layer_prefix: {e: tensor}}
        vexp_buf: dict[str, dict[int, torch.Tensor]] = {}
        moe_owners: set[str] = set()
        n_vexp = cfg.mova_num_experts
        kvh, hd = cfg.num_key_value_heads, cfg.head_dim

        def load(pname: str, w: torch.Tensor, *shard):
            if is_pp_missing_parameter(pname, self):
                return
            p = params[pname]
            loader = getattr(p, "weight_loader", default_weight_loader)
            loader(p, w, *shard)
            loaded.add(pname)

        for name, w in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if ".mlp.experts." in name:
                owner = name[:name.index(".mlp.experts.") + len(".mlp.experts")]
                if is_pp_missing_parameter(owner, self):
                    continue
                for _ in modules[owner].load_weights([(name[len(owner) + 1:], w)]):
                    pass
                moe_owners.add(owner)
                continue
            if ".self_attn.v_experts." in name:
                owner, rest = name.split(".self_attn.v_experts.")
                e = int(rest.split(".")[0])
                buf = vexp_buf.setdefault(owner, {})
                buf[e] = w
                if len(buf) == n_vexp:
                    stacked = torch.stack([buf[i] for i in range(n_vexp)])  # [E, kvh*D, H]
                    stacked = stacked.view(n_vexp, kvh, hd, -1).permute(1, 0, 2, 3)
                    load(f"{owner}.self_attn.v_experts.weight",
                         stacked.reshape(kvh * n_vexp * hd, -1).contiguous())
                    del vexp_buf[owner]
                continue
            if name.endswith(".mlp.gate.bias") or name.endswith(".self_attn.v_router.bias"):
                load(name[:-len("bias")] + "e_score_correction_bias", w.float())
                continue
            if ".mlp.gate_proj." in name or ".shared_experts.gate_proj." in name:
                load(name.replace("gate_proj", "gate_up_proj"), w, 0)
                continue
            if ".mlp.up_proj." in name or ".shared_experts.up_proj." in name:
                load(name.replace("up_proj", "gate_up_proj"), w, 1)
                continue
            if name not in params:
                raise KeyError(f"unexpected checkpoint tensor {name}")
            load(name, w)

        assert not vexp_buf, f"incomplete value-expert sets: {list(vexp_buf)}"
        loaded.update(n for n in params
                      if any(n.startswith(o + ".") for o in moe_owners))
        return loaded
