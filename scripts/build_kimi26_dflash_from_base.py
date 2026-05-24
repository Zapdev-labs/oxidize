#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file
from gguf import GGUFWriter

BASE = Path("/home/ai/kimi-k2.6-base")
OUT = Path("/home/ai/Kimi-K2.6-DFlash-baseinit-F32.gguf")
LAYER_FILE = BASE / "model-00001-of-000064.safetensors"
TARGET_LAYERS = [1, 12, 24, 35, 47, 58]
HIDDEN = 7168
INTER = 18432
N_LAYERS = 6
N_HEADS = 64
N_KV = 8
HEAD_DIM = 128
VOCAB = 163840
BLOCK = 8
MASK = 163838

def bf16_to_f32(t):
    return t.detach().to(torch.float32).cpu().numpy()

def zeros(shape):
    return np.zeros(shape, dtype=np.float32)

print("Loading Kimi K2.6 base layer-0 dense tensors", flush=True)
st = load_file(str(LAYER_FILE), device="cpu")
p = "language_model.model.layers.0."
attn_norm = bf16_to_f32(st[p + "input_layernorm.weight"])
post_norm = bf16_to_f32(st[p + "post_attention_layernorm.weight"])
ffn_gate = bf16_to_f32(st[p + "mlp.gate_proj.weight"])
ffn_up = bf16_to_f32(st[p + "mlp.up_proj.weight"])
ffn_down = bf16_to_f32(st[p + "mlp.down_proj.weight"])
# Kimi K2.6 target uses MLA attention, while current Oxidize DFlash expects Qwen-style q/k/v/o.
# Seed those projections with shape-correct small zero matrices and reuse Kimi dense MLP/RMSNorm weights.
q = zeros((N_HEADS * HEAD_DIM, HIDDEN))
k = zeros((N_KV * HEAD_DIM, HIDDEN))
v = zeros((N_KV * HEAD_DIM, HIDDEN))
o = zeros((HIDDEN, N_HEADS * HEAD_DIM))
q_norm = np.ones((HEAD_DIM,), dtype=np.float32)
k_norm = np.ones((HEAD_DIM,), dtype=np.float32)

print("Building DFlash target-hidden fusion weight", flush=True)
fc = zeros((HIDDEN, HIDDEN * len(TARGET_LAYERS)))
scale = np.float32(1.0 / len(TARGET_LAYERS))
for i in range(len(TARGET_LAYERS)):
    s = i * HIDDEN
    fc[:, s:s+HIDDEN][np.arange(HIDDEN), np.arange(HIDDEN)] = scale
hidden_norm = np.ones((HIDDEN,), dtype=np.float32)
out_norm = post_norm.copy()

print(f"Writing {OUT}", flush=True)
writer = GGUFWriter(path=str(OUT), arch="dflash-draft")
writer.add_name("Kimi-K2.6-DFlash-baseinit")
writer.add_uint32("dflash-draft.hidden_size", HIDDEN)
writer.add_uint32("dflash-draft.num_hidden_layers", N_LAYERS)
writer.add_uint32("dflash-draft.num_attention_heads", N_HEADS)
writer.add_uint32("dflash-draft.num_key_value_heads", N_KV)
writer.add_uint32("dflash-draft.intermediate_size", INTER)
writer.add_float32("dflash-draft.rms_norm_eps", 1e-5)
writer.add_float32("dflash-draft.rope_theta", 1000000.0)
writer.add_uint32("dflash-draft.vocab_size", VOCAB)
writer.add_uint32("dflash-draft.block_size", BLOCK)
writer.add_uint32("dflash-draft.num_target_layers", 61)
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
