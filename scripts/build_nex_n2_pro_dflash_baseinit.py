#!/usr/bin/env python3
"""Build a DFlash baseinit GGUF for Nex-N2-Pro speculative decoding smoke tests."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from gguf import GGUFWriter
from safetensors.torch import load_file

BASE = Path("/home/ai/models/Nex-N2-Pro")
OUT = Path("/home/ai/gguf-out/Nex-N2-Pro-DFlash-baseinit-F32.gguf")
LAYER_FILE = BASE / "model-00007-of-00122.safetensors"
TARGET_LAYERS = [3, 15, 27, 39, 51, 59]
HIDDEN = 4096
INTER = 1024
N_LAYERS = 6
N_HEADS = 32
N_KV = 2
HEAD_DIM = 256
VOCAB = 248320
BLOCK = 8
MASK = 248318


def bf16_to_f32(t: torch.Tensor) -> np.ndarray:
    return t.detach().to(torch.float32).cpu().numpy()


def zeros(shape: tuple[int, ...]) -> np.ndarray:
    return np.zeros(shape, dtype=np.float32)


def main() -> None:
    cfg = json.loads((BASE / "config.json").read_text())
    text_cfg = cfg.get("text_config", cfg)
    print("Nex-N2-Pro text_config hidden_size", text_cfg.get("hidden_size"), flush=True)

    print(f"Loading Nex-N2-Pro tensors from {LAYER_FILE}", flush=True)
    st = load_file(str(LAYER_FILE), device="cpu")
    p = "model.language_model.layers.3."
    attn_norm = bf16_to_f32(st[p + "input_layernorm.weight"])
    post_key = p + "post_attention_layernorm.weight"
    post_norm = (
        bf16_to_f32(st[post_key])
        if post_key in st
        else attn_norm.copy()
    )
    ffn_gate = bf16_to_f32(st[p + "mlp.shared_expert.gate_proj.weight"])
    ffn_up = bf16_to_f32(st[p + "mlp.shared_expert.up_proj.weight"])
    ffn_down = bf16_to_f32(st[p + "mlp.shared_expert.down_proj.weight"])
    q_raw = bf16_to_f32(st[p + "self_attn.q_proj.weight"])
  # Qwen3.5 full-attn layers use gated Q: q_proj rows are 2x the attended query width.
    q_attn_rows = N_HEADS * HEAD_DIM
    if q_raw.shape[0] == 2 * q_attn_rows:
        q = q_raw[:q_attn_rows, :]
    else:
        q = q_raw
    k = bf16_to_f32(st[p + "self_attn.k_proj.weight"])
    v = bf16_to_f32(st[p + "self_attn.v_proj.weight"])
    o = bf16_to_f32(st[p + "self_attn.o_proj.weight"])
    q_norm = bf16_to_f32(st[p + "self_attn.q_norm.weight"])
    k_norm = bf16_to_f32(st[p + "self_attn.k_norm.weight"])

    print("Building DFlash target-hidden fusion weight", flush=True)
    fc = zeros((HIDDEN, HIDDEN * len(TARGET_LAYERS)))
    scale = np.float32(1.0 / len(TARGET_LAYERS))
    for i in range(len(TARGET_LAYERS)):
        s = i * HIDDEN
        fc[:, s : s + HIDDEN][np.arange(HIDDEN), np.arange(HIDDEN)] = scale
    hidden_norm = np.ones((HIDDEN,), dtype=np.float32)
    out_norm = post_norm.copy()

    print(f"Writing {OUT}", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    writer = GGUFWriter(path=str(OUT), arch="dflash-draft")
    writer.add_name("Nex-N2-Pro-DFlash-baseinit")
    writer.add_uint32("dflash-draft.hidden_size", HIDDEN)
    writer.add_uint32("dflash-draft.num_hidden_layers", N_LAYERS)
    writer.add_uint32("dflash-draft.num_attention_heads", N_HEADS)
    writer.add_uint32("dflash-draft.num_key_value_heads", N_KV)
    writer.add_uint32("dflash-draft.intermediate_size", INTER)
    writer.add_float32("dflash-draft.rms_norm_eps", 1e-6)
    writer.add_float32("dflash-draft.rope_theta", float(text_cfg.get("rope_theta", 10000000.0)))
    writer.add_uint32("dflash-draft.vocab_size", VOCAB)
    writer.add_uint32("dflash-draft.block_size", BLOCK)
    writer.add_uint32("dflash-draft.num_target_layers", len(TARGET_LAYERS))
    writer.add_uint32("dflash-draft.mask_token_id", MASK)
    writer.add_array("dflash-draft.target_layer_ids", TARGET_LAYERS)
    writer.add_tensor("dflash_fc.weight", fc)
    writer.add_tensor("dflash_hidden_norm.weight", hidden_norm)
    for i in range(N_LAYERS):
        print(f"queue layer {i}", flush=True)
        writer.add_tensor(f"blk.{i}.attn_norm.weight", attn_norm)
        writer.add_tensor(f"blk.{i}.post_attention_norm.weight", post_norm)
        writer.add_tensor(f"blk.{i}.attn_q_norm.weight", q_norm)
        writer.add_tensor(f"blk.{i}.attn_k_norm.weight", k_norm)
        writer.add_tensor(f"blk.{i}.attn_q.weight", q)
        writer.add_tensor(f"blk.{i}.attn_k.weight", k)
        writer.add_tensor(f"blk.{i}.attn_v.weight", v)
        writer.add_tensor(f"blk.{i}.attn_output.weight", o)
        writer.add_tensor(f"blk.{i}.ffn_gate.weight", ffn_gate)
        writer.add_tensor(f"blk.{i}.ffn_up.weight", ffn_up)
        writer.add_tensor(f"blk.{i}.ffn_down.weight", ffn_down)
    writer.add_tensor("output_norm.weight", out_norm)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print("DONE", OUT, OUT.stat().st_size, flush=True)


if __name__ == "__main__":
    main()
