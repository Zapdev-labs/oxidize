from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from scripts.dflash.config import DFlashTrainConfig
from scripts.dflash.model import DFlashDraftModel


def _np(t: torch.Tensor) -> np.ndarray:
    return t.detach().to(torch.float32).cpu().contiguous().numpy()


def export_dflash_gguf(draft: DFlashDraftModel, cfg: DFlashTrainConfig, path: Path) -> Path:
    from gguf import GGUFWriter

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = GGUFWriter(path=str(path), arch="dflash-draft")
    writer.add_name("Qwen3.8-27B-ABLITERATED-DFlash")
    writer.add_uint32("dflash-draft.hidden_size", cfg.hidden_size)
    writer.add_uint32("dflash-draft.embedding_length", cfg.hidden_size)
    writer.add_uint32("dflash-draft.num_hidden_layers", cfg.num_hidden_layers)
    writer.add_uint32("dflash-draft.block_count", cfg.num_hidden_layers)
    writer.add_uint32("dflash-draft.context_length", 4096)
    writer.add_uint32("dflash-draft.num_attention_heads", cfg.num_attention_heads)
    writer.add_uint32("dflash-draft.attention.head_count", cfg.num_attention_heads)
    writer.add_uint32("dflash-draft.num_key_value_heads", cfg.num_key_value_heads)
    writer.add_uint32("dflash-draft.attention.head_count_kv", cfg.num_key_value_heads)
    writer.add_uint32("dflash-draft.attention.key_length", cfg.head_dim)
    writer.add_uint32("dflash-draft.intermediate_size", cfg.intermediate_size)
    writer.add_uint32("dflash-draft.feed_forward_length", cfg.intermediate_size)
    writer.add_float32("dflash-draft.rms_norm_eps", cfg.rms_norm_eps)
    writer.add_float32("dflash-draft.attention.layer_norm_rms_epsilon", cfg.rms_norm_eps)
    writer.add_float32("dflash-draft.rope_theta", cfg.rope_theta)
    writer.add_float32("dflash-draft.rope.freq_base", cfg.rope_theta)
    writer.add_uint32("dflash-draft.vocab_size", cfg.vocab_size)
    writer.add_uint32("dflash-draft.block_size", cfg.block_size)
    writer.add_uint32("dflash-draft.dflash.block_size", cfg.block_size)
    writer.add_uint32("dflash-draft.num_target_layers", cfg.num_target_layers)
    writer.add_uint32("dflash-draft.dflash.n_target_features", cfg.n_feat)
    writer.add_uint32("dflash-draft.mask_token_id", cfg.mask_token_id)
    writer.add_uint32("dflash-draft.dflash.mask_token_id", cfg.mask_token_id)
    writer.add_array("dflash-draft.target_layer_ids", cfg.target_layer_ids)
    writer.add_array("dflash-draft.dflash.target_layer_ids", cfg.target_layer_ids)
    writer.add_tensor("dflash_fc.weight", _np(draft.fc.weight))
    writer.add_tensor("dflash_hidden_norm.weight", _np(draft.hidden_norm.weight))
    writer.add_tensor("output_norm.weight", _np(draft.norm.weight))
    for i, layer in enumerate(draft.layers):
        writer.add_tensor(f"blk.{i}.attn_norm.weight", _np(layer.input_layernorm.weight))
        writer.add_tensor(f"blk.{i}.post_attention_norm.weight", _np(layer.post_attention_layernorm.weight))
        writer.add_tensor(f"blk.{i}.attn_q_norm.weight", _np(layer.self_attn.q_norm.weight))
        writer.add_tensor(f"blk.{i}.attn_k_norm.weight", _np(layer.self_attn.k_norm.weight))
        writer.add_tensor(f"blk.{i}.attn_q.weight", _np(layer.self_attn.q_proj.weight))
        writer.add_tensor(f"blk.{i}.attn_k.weight", _np(layer.self_attn.k_proj.weight))
        writer.add_tensor(f"blk.{i}.attn_v.weight", _np(layer.self_attn.v_proj.weight))
        writer.add_tensor(f"blk.{i}.attn_output.weight", _np(layer.self_attn.o_proj.weight))
        writer.add_tensor(f"blk.{i}.ffn_gate.weight", _np(layer.mlp.gate_proj.weight))
        writer.add_tensor(f"blk.{i}.ffn_up.weight", _np(layer.mlp.up_proj.weight))
        writer.add_tensor(f"blk.{i}.ffn_down.weight", _np(layer.mlp.down_proj.weight))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return path
