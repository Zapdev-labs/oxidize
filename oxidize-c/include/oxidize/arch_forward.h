/*
 * arch_forward.h — architecture-specific forward passes (GPT-2, GPT-NeoX,
 * Falcon).
 *
 * The Llama/Mistral dense path lives in llama.c (`oc_llama_forward`).
 * These functions provide forward passes for architectures that differ
 * structurally from Llama:
 *
 *   - GPT-2:     LayerNorm + GeLU FFN, learned positional embeddings, no
 *                RoPE, parallel attention/FFN per layer.
 *   - GPT-NeoX:  LayerNorm + single-projection GeLU FFN, rotary positional
 *                embeddings.
 *   - Falcon:    LayerNorm + single-projection GeLU FFN, rotary positional
 *                embeddings, parallel attention/FFN per layer, multi-query
 *                attention (single KV head, n_head_kv == 1).
 *
 * Each function processes a single token, advances `sess->pos`, and writes
 * logits to `logits_out` (length model->cfg.vocab_size). The caller is
 * responsible for resetting/rewinding the session between sequences and for
 * sampling from `logits_out`.
 *
 * These implementations are correct but simplified (scalar reference
 * paths, single-pass online softmax attention). They reuse the workspace
 * buffers already allocated in `OcLlamaSession` by `oc_llama_session_init`,
 * so they are drop-in replacements for `oc_llama_forward` for the
 * architectures they cover.
 */
#ifndef OXIDIZE_ARCH_FORWARD_H
#define OXIDIZE_ARCH_FORWARD_H

#include <stdint.h>

#include "oxidize/error.h"
#include "oxidize/llama.h"

#ifdef __cplusplus
extern "C" {
#endif

/* GPT-2 forward pass for a single token.
 *
 * Pipeline per layer (parallel layout, GPT-2 convention):
 *   1. LayerNorm(x) → ln_1
 *   2. attn = o_proj(c_attn(ln_1)); x += attn
 *   3. LayerNorm(x) → ln_2
 *   4. mlp = mlp_c_fc(mlp_proj(act(mlp_c_fc(ln_2)))); x += mlp
 * with GeLU activation and learned positional embeddings added to the
 * token embedding before layer 0.
 *
 * `logits_out` may be NULL to skip the lm_head projection (prefill). */
OcError oc_arch_forward_gpt2(OcLlamaSession *sess, uint32_t token,
                              float *logits_out);

/* GPT-NeoX forward pass for a single token.
 *
 * Pipeline per layer (parallel attention/FFN, NeoX convention):
 *   1. LayerNorm(x) → ln (input_layernorm)
 *   2. attn = o_proj(query_key_value(ln)) + x; (RoPE applied to Q/K)
 *   3. LayerNorm(x) → ln (post_attention_layernorm)
 *   4. mlp = dense_4h_to_h(GELU(dense_h_to_4h(ln))); x += mlp
 * Rotary embeddings are applied to Q and K (split-halves NeoX layout). */
OcError oc_arch_forward_gpt_neox(OcLlamaSession *sess, uint32_t token,
                                  float *logits_out);

/* Falcon forward pass for a single token.
 *
 * Pipeline per layer (Falcon parallel layout):
 *   1. LayerNorm(x) → ln (input_layernorm)
 *   2. attn = o_proj(query_key_value(ln)); x += attn
 *   3. mlp = dense_4h_to_h(GELU(dense_h_to_4h(ln))); x += mlp
 * with RoPE applied to Q/K, multi-query attention (single KV head), and
 * the MLP fed from the shared layer-normalized input (parallel block). */
OcError oc_arch_forward_falcon(OcLlamaSession *sess, uint32_t token,
                                float *logits_out);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_ARCH_FORWARD_H */
