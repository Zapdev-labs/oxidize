# GLM-5.2 (glm-dsa) implementation spec for oxidize-cpp

Source: GGUF metadata read from `~/models/glm-5.2/target/UD-IQ1_M/...00001...gguf` + llama.cpp reference.

## Arch params (glm-dsa.* prefix)
- general.architecture = `glm-dsa`; size_label 256x22B; file_type 31 (IQ1_M mix); split.count 6; vocab 154880; tokenizer gpt2/BPE pre=glm4; eos 154820, bos 154822, eot 154827.
- block_count 79; nextn_predict_layers 1 → **78 transformer layers** (layer_count = 79-1). leading_dense_block_count 3 → layers 0-2 dense FFN, layers 3-77 MoE.
- embedding_length 6144; feed_forward_length 12288 (dense FFN inter).
- attention.head_count 64; head_count_kv 1; key_length 576; value_length 512; key_length_mla 256; value_length_mla 256.
- attention.q_lora_rank 2048; kv_lora_rank 512; layer_norm_rms_epsilon 1e-5.
- rope.freq_base 8e6; rope.dimension_count 64 (partial RoPE on 64 of 256 key dims).
- expert_count 256; expert_used_count 8; expert_group_count 1; expert_group_used_count 1; expert_gating_func 2 (SIGMOID); expert_feed_forward_length 2048; expert_shared_count 1; expert_weights_scale 2.5; expert_weights_norm true.
- DSA indexer (attention.indexer.*): head_count 32, key_length 128, top_k 2048 — **SKIP in Phase 1** (full attention over all cached positions; indexer only prunes for long-context perf).

## Per-block tensors (block 3, first MoE block)
```
attn_q_a.weight        Q5_K  [6144,2048]        ; attn_q_a_norm.weight  F32 [2048]
attn_q_b.weight        Q8_0  [2048,16384]       ; (16384 = 64 heads * 256 mla_key)
attn_kv_a_mqa.weight   Q8_0  [6144,576]         ; attn_kv_a_norm.weight F32 [512]
attn_k_b.weight        Q8_0  [192,512,64]       ; attn_v_b.weight       Q8_0 [512,256,64]
attn_output.weight     Q5_K  [16384,6144]
attn_norm.weight F32 [6144] ; ffn_norm.weight F32 [6144]
ffn_gate_inp.weight    F32   [6144,256]         ; router [cols=h, rows=n_experts] (matches gemv rows=256,cols=6144)
ffn_gate_exps.weight   IQ1_M [6144,2048,256] ; ffn_up_exps.weight IQ1_M [6144,2048,256]
ffn_down_exps.weight   IQ3_XXS [2048,6144,256]
ffn_gate_shexp.weight  Q5_K [6144,2048] ; ffn_up_shexp.weight Q5_K [6144,2048] ; ffn_down_shexp.weight Q6_K [2048,6144]
exp_probs_b.bias       F32 [256]  (sigmoid per-expert additive bias)
indexer.* (SKIP)
```
Distinct quant types across 1809 tensors: F32, Q8_0, Q5_K, Q6_K, **IQ1_M, IQ2_XXS, IQ3_XXS, IQ4_XS**, Q4_K, Q2_K, Q3_K. Loader MUST handle the four IQ types (others already supported). token_embd/output are Q4_K.

## Attention = MLA (NOT GQA). No attn_q/attn_k/attn_v tensors exist.
Decode step:
```
q_a   = W_qa * x                              [h->2048]
q_a   = rms_norm(q_a, attn_q_a_norm)          [2048]   (eps 1e-5)
q     = W_qb * q_a                            [2048->16384]  -> 64 heads x 256
apply partial RoPE(64) on first 64 of each 256-dim q head (theta 8e6)
kv_a  = W_kva * x                             [h->576]  -> [0:512]=kv_latent, [512:576]=k_rope(64)
kv_latent = rms_norm(kv_latent, attn_kv_a_norm) [512]
apply RoPE(64) to k_rope
store compressed latent (576) in KV cache (kv_heads=1, head_dim=576 -> kv_token_size 576 auto-correct)
per cached pos: k = W_kb*latent (+k_rope), v = W_vb*latent ; MHA(q, k, v) scale=1/sqrt(256)
out   = W_o * attn_out                        [16384->6144]
```
**k_b[192,512,64] / v_b[512,256,64] exact index interpretation must be confirmed against llama.cpp glm-dsa code** (the research phase delivers this). MLA "absorb" math == DeepSeek-V2.

## MoE gaps to close
1. **Shared expert (CRITICAL)**: load ffn_{gate,up,down}_shexp into LlamaLayer; run unconditionally at end of moe_ffn(), accumulate into ffn_out (weight 1.0). Currently dropped → wrong output.
2. expert_weights_norm: existing moe_ffn already normalizes top-k weights → OK for GLM.
3. sigmoid gating + exp_probs_b: already implemented (gating_func==2). OK.
4. router dim order [6144,256] matches existing gemv(rows=256,cols=6144). OK.
5. expert_weights_scale 2.5: apply as multiplier to routed-expert summed output (verify vs llama.cpp).

## Minimal code changes (dependency order)
1. config.hpp: Architecture::GlmDsa; InferenceConfig += q_lora_rank, kv_lora_rank, mla_key_dim, mla_val_dim, num_shared_experts, expert_weights_norm, expert_weights_scale.
2. gguf.cpp: architecture_from_name "glm-dsa"->GlmDsa; uses_mla |= GlmDsa; read new fields (arch_u32). Keep key_value_head_dim=576 for cache (do NOT override to 256).
3. model_llama.hpp: LlamaLayer += MLA fields (attn_q_a,attn_q_a_norm,attn_q_b,attn_kv_a_mqa,attn_kv_a_norm,attn_k_b,attn_v_b; is_mla()) + shexp fields (has_shared_expert()).
4. model_llama.cpp ctor: opt_w/opt_vec the new tensors; guard: `if (attn_q.empty() && !is_mla()) throw`.
5. model_llama.cpp run_layers: MLA branch (above) when is_mla().
6. model_llama.cpp moe_ffn: shared-expert accumulation + expert_weights_scale.
7. quant.cpp: IQ1_M/IQ2_XXS/IQ3_XXS/IQ4_XS dequant (+ grids) wired into dequantize_row + is_supported_quant. (Separate research delivers exact algorithm.)

## Verification ladder
1. Model LOADS without throw (all 6 shards, mmap).
2. First forward produces finite logits (no NaN), argmax token is a plausible id.
3. Greedy a few tokens; sanity vs llama.cpp glm-dsa on same prompt/tokens (if llama.cpp builds GLM).
NOTE: full coherent generation is a multi-day effort; make maximal VERIFIED progress and report honestly what loads/runs vs what is stubbed.
