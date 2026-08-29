/*
 * arch_forward.c — architecture-specific forward passes (GPT-2, GPT-NeoX,
 * Falcon).
 *
 * These three forward passes differ from the Llama/Mistral path in
 * `llama.c` (`oc_llama_forward`) in norm type, activation, and layer
 * ordering. The Llama path uses RMSNorm → SwiGLU FFN with sequential
 * attention+FFN; these architectures use LayerNorm (mean+variance) and a
 * parallel attention/FFN layout per layer.
 *
 * Reused infrastructure (no duplication):
 *   - embed_token / matvec / attention_head helpers (defined in llama.c)
 *   - oc_rms_norm_f32, oc_apply_rope_f32, oc_swiglu_inplace_f32 (activation.c)
 *   - oc_gelu_exact_f32, oc_gelu_approx_f32 (activation.c)
 *   - oc_matvec_f32 / oc_matvec_quantized (matvec.c)
 *   - OcLlamaSession workspace (already sized for n_embd / n_ff / n_head)
 *
 * Architectural conventions (mirrors Rust `oxidize-core::inference`):
 *
 *   GPT-2:
 *     - LayerNorm (not RMSNorm). Two LayerNorm weights per layer
 *       (attn_norm = ln_1, ffn_norm = ln_2) stored as f32* in OcLlamaLayer.
 *     - GeLU FFN (exact erf-based GeLU activation; not SwiGLU). The FFN has
 *       a single up-projection (n_embd → n_ff) with a GeLU activation, then
 *       a down-projection (n_ff → n_embd). This is the canonical GPT-2
 *       FFN (no gated SwiGLU). For GGUF models converted with the standard
 *       converter, ffn_gate / ffn_up are both populated with the same
 *       n_ff-sized intermediate; we treat the up projection as the gate and
 *       the down projection as the projection.
 *       When the GGUF only carries the canonical single-hidden-layer FFN,
 *       ffn_gate.rows == n_ff and ffn_up.rows == n_ff. We apply GeLU to the
 *       gate output and multiply by up, then project with down. When only
 *       one of gate/up exists (common in GPT-2 GGUFs), we treat the FFN as
 *       gate = GeLU(W_up · x); down = W_down · gate (standard 2-layer MLP).
 *       This stub uses the ffn_up weight as the up-projection and
 *       ffn_down as the down-projection (the Llama-canonical names).
 *     - Learned positional embeddings added to the token embedding.
 *       These are stored in a separate tensor; the GGUF name is
 *       `position_embeddings.weight` (or `wpe`). When absent, we fall back
 *       to no positional embedding (the model must be loaded with one for
 *       correct output).
 *     - Parallel attention/FFN per layer: LayerNorm → attn → residual;
 *       LayerNorm → mlp → residual. (GPT-2 actually applies them
 *       sequentially; "parallel" here refers to the fused QKV c_attn
 *       projection, not the layer layout. The implementation uses the
 *       standard sequential GPT-2 block.)
 *
 *   GPT-NeoX:
 *     - LayerNorm + SwiGLU FFN, rotary positional embeddings.
 *     - Parallel attention/FFN per layer (input_layernorm feeds both attn
 *       and mlp; post_attention_layernorm feeds the residual path). The
 *       NeoX parallel layout is:
 *         ln = LayerNorm(x, input_layernorm)
 *         attn_out = o_proj(c_attn(ln))
 *         x = x + attn_out
 *         ln2 = LayerNorm(x, post_attention_layernorm)
 *         mlp_out = mlp_down(SwiGLU(mlp_gate(ln2), mlp_up(ln2)))
 *         x = x + mlp_out
 *       We implement the standard sequential variant here for clarity.
 *
 *   Falcon:
 *     - LayerNorm + SwiGLU FFN, rotary positional embeddings.
 *     - Parallel attention/FFN per layer:
 *         ln = LayerNorm(x, input_layernorm)
 *         attn_out = o_proj(c_attn(ln))
 *         x = x + attn_out
 *         mlp_out = mlp_down(SwiGLU(mlp_up(x), mlp_gate(x)))
 *         x = x + mlp_out
 *       (The Falcon parallel block uses the same LayerNorm for both attn
 *       and mlp input; the FFN operates on the post-residual x.)
 *     - Multi-query attention: n_head_kv == 1 (single shared KV head).
 *       All Q heads attend to the same K/V.
 *
 * LayerNorm implementation:
 *   out[i] = (x[i] - mean) / sqrt(var + eps) * weight[i] + bias[i]
 * where var = mean((x[i]-mean)^2). The eps for GPT-2/NeoX/Falcon defaults to
 * 1e-5 (the Llama rms_norm_eps field is reused for LayerNorm eps; the GGUF
 * converter writes the architecture's epsilon into this field).
 *
 * The forward pass is scalar (no SIMD): one Q head at a time, online softmax
 * over the cached K/V. This matches the parity-first philosophy of the C
 * port (correctness before speed). SIMD kernels can be layered on later.
 */
#include "oxidize/arch_forward.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "oxidize/activation.h"
#include "oxidize/llama.h"

#include "oxidize/log.h"
#include "oxidize/matvec.h"
#include "oxidize/model.h"
#include "oxidize/quant.h"
#include "oxidize/tensor_ops.h"

#include "llama_session_ops.h"

/* ─── LayerNorm ──────────────────────────────────────────────────────────
 *
 * out[i] = (x[i] - mean) / sqrt(var + eps) * weight[i] + bias[i]
 *
 * LayerNorm computes the mean and variance over the hidden dimension
 * (axis=-1), unlike RMSNorm which only uses the mean of squares. The
 * `weight` array has the same length as the hidden dimension (n_embd) and
 * is the learned scale (gamma). `bias` is the learned beta (loaded from
 * the GGUF's *.bias norm tensors); NULL means zero bias.
 *
 * The computation uses f32 accumulation throughout (matching the Rust scalar
 * reference). `eps` defaults to rms_norm_eps from the config (which the GGUF
 * converter populates with the architecture's layer_norm_epsilon). */
static void arch_layer_norm(const float *x, const float *weight,
                            const float *bias, float *out, size_t n, float eps)
{
    /* Compute mean. */
    float mean = 0.0f;
    for (size_t i = 0; i < n; i++) mean += x[i];
    mean /= (float)n;

    /* Compute variance (population variance: divide by n, not n-1). */
    float var = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float d = x[i] - mean;
        var += d * d;
    }
    var /= (float)n;

    float inv_std = 1.0f / sqrtf(var + eps);

    /* Normalize, scale, shift. */
    for (size_t i = 0; i < n; i++) {
        out[i] = (x[i] - mean) * inv_std * weight[i]
               + (bias ? bias[i] : 0.0f);
    }
}

/* ─── GeLU activation (GPT-2 uses the tanh approximation) ────────────────
 *
 * GPT-2's original implementation uses the tanh approximation of GeLU:
 *   gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
 *
 * GPT-NeoX and Falcon also use a single-projection GeLU MLP when no gate
 * tensor is present (the standard checkpoints). We use the approximation variant
 * (oc_gelu_approx_f32) to match the original GPT-2 reference implementation
 * (HuggingFace transformers GPT2MLP uses gelu_new which is the tanh
 * approximation). */
static void arch_gelu_inplace_f32(float *buf, size_t n)
{
    for (size_t i = 0; i < n; i++) {
        buf[i] = oc_gelu_approx_f32(buf[i]);
    }
}

/* ─── Online-softmax attention (shared by all three architectures) ───────
 *
 * Computes attention for a single query head against all cached K/V up to
 * `sess->pos`. Uses the online (streaming) softmax algorithm (Milakov &
 * Gimelshein 2018) so we never need to materialize the full score vector.
 *
 * For multi-query attention (Falcon) or GQA, the `kv_head` index selects
 * which KV head's cache to attend to. For GPT-2 (n_head_kv == n_head) each
 * Q head has its own KV head. For Falcon (n_head_kv == 1) all Q heads
 * share the single KV head (kv_head = 0 for all heads).
 *
 * Parameters:
 *   s        — session (provides KV cache, pos, config)
 *   head     — query head index (0..n_head-1)
 *   layer    — layer index (selects the KV cache slice)
 *   q_vec    — query vector for this head (length head_dim)
 *   out_vec  — output vector (length head_dim), written in place */


/* ─── GPT-2 layer forward ────────────────────────────────────────────────
 *
 * GPT-2 block (sequential, matching HF transformers GPT2Block):
 *   1. ln_1 = LayerNorm(x, attn_norm)
 *   2. qkv = c_attn(ln_1)         [fused QKV projection, n_embd → 3*n_embd]
 *   3. split qkv into Q, K, V     [each n_head * head_dim]
 *   4. attention → attn_out       [no RoPE; learned positional embeddings
 *                                  already added before layer 0]
 *   5. x = x + o_proj(attn_out)   [residual]
 *   6. ln_2 = LayerNorm(x, ffn_norm)
 *   7. h = GeLU(mlp_c_fc(ln_2))   [up-projection, n_embd → n_ff]
 *   8. x = x + mlp_proj(h)        [down-projection, n_ff → n_embd, residual]
 *
 * Weight mapping (canonical → GPT-2 HF):
 *   attn_q / attn_k / attn_v → transformer.h.N.attn.c_attn.weight
 *     (GPT-2 uses a single fused c_attn weight of shape [3*n_embd, n_embd].
 *      When loaded via the standard GGUF converter, the converter splits
 *      c_attn into three separate tensors: attn_q, attn_k, attn_v, each
 *      [n_embd, n_embd]. We use them as separate projections.)
 *   attn_output → transformer.h.N.attn.c_proj.weight
 *   ffn_up      → transformer.h.N.mlp.c_fc.weight      [n_ff, n_embd]
 *   ffn_down    → transformer.h.N.mlp.c_proj.weight     [n_embd, n_ff]
 *   attn_norm   → transformer.h.N.ln_1.weight           [n_embd]
 *   ffn_norm    → transformer.h.N.ln_2.weight           [n_embd]
 *
 * The GPT-2 FFN is a standard 2-layer MLP: h = GeLU(W_up · x); out = W_down · h.
 * There is no gated activation (no ffn_gate). We reuse the ffn_up slot for
 * the up-projection and ffn_down for the down-projection. ffn_gate is not
 * used in the GPT-2 path. */
static void arch_gpt2_layer(OcLlamaSession *s, uint32_t layer)
{
    const OcLlamaConfig *c = &s->model->cfg;
    OcLlamaLayer *L = &s->model->layers[layer];
    size_t hd = c->head_dim;
    size_t n_embd = c->n_embd;

    /* 1. Pre-attention LayerNorm (ln_1). */
    arch_layer_norm(s->x, L->attn_norm, L->attn_norm_bias, s->normed,
                    n_embd, c->rms_norm_eps);

    /* 2. Q/K/V projections (separate weights in the GGUF). */
    oc_llama_matvec(&L->attn_q, s->normed, s->q, s->dequant_temp);
    oc_llama_matvec(&L->attn_k, s->normed, s->k, s->dequant_temp);
    oc_llama_matvec(&L->attn_v, s->normed, s->v, s->dequant_temp);

    /* 3. No RoPE for GPT-2 (learned positional embeddings added before
     *    layer 0; see oc_arch_forward_gpt2). */

    /* 4. KV cache write at position `pos`. */
    size_t kv_off = ((size_t)layer * c->n_ctx + (size_t)s->pos)
                  * s->kv_row_floats;
    memcpy(s->kv_k + kv_off, s->k, s->kv_row_floats * sizeof(float));
    memcpy(s->kv_v + kv_off, s->v, s->kv_row_floats * sizeof(float));

    /* 5. Attention per head → s->attn_out. GPT-2 uses causal attention
     *    (each position attends to itself and all previous positions).
     *    The online-softmax attention_head already handles this because
     *    seq_len = pos+1 (only cached positions up to the current one). */
    for (uint32_t h = 0; h < c->n_head; h++) {
        oc_llama_attention_head(s, h, layer, s->q + h * hd,
                            s->attn_out + h * hd);
    }

    /* 6. Output projection + residual add. */
    oc_llama_matvec(&L->attn_output, s->attn_out, s->normed, s->dequant_temp);
    for (size_t i = 0; i < n_embd; i++) s->x[i] += s->normed[i];

    /* 7. Pre-FFN LayerNorm (ln_2). */
    arch_layer_norm(s->x, L->ffn_norm, L->ffn_norm_bias, s->normed,
                    n_embd, c->rms_norm_eps);

    /* 8. FFN: up-projection → GeLU → down-projection + residual.
     *    GPT-2 uses the tanh-approx GeLU. */
    oc_llama_matvec(&L->ffn_up, s->normed, s->ffn_gate, s->dequant_temp);
    arch_gelu_inplace_f32(s->ffn_gate, c->n_ff);
    oc_llama_matvec(&L->ffn_down, s->ffn_gate, s->normed, s->dequant_temp);
    for (size_t i = 0; i < n_embd; i++) s->x[i] += s->normed[i];
}

static void arch_gptj_layer(OcLlamaSession *s, uint32_t layer)
{
    const OcLlamaConfig *c = &s->model->cfg;
    OcLlamaLayer *L = &s->model->layers[layer];
    size_t hd = c->head_dim;
    size_t n_embd = c->n_embd;
    float rope_base = c->rope_theta > 0.0f ? c->rope_theta : 10000.0f;

    arch_layer_norm(s->x, L->attn_norm, L->attn_norm_bias, s->normed,
                    n_embd, c->rms_norm_eps);

    oc_llama_matvec(&L->attn_q, s->normed, s->q, s->dequant_temp);
    oc_llama_matvec(&L->attn_k, s->normed, s->k, s->dequant_temp);
    oc_llama_matvec(&L->attn_v, s->normed, s->v, s->dequant_temp);

    for (uint32_t h = 0; h < c->n_head; h++) {
        oc_tensor_rope_gptj_f32(s->q + h * hd, hd, s->pos, rope_base);
    }
    for (uint32_t h = 0; h < c->n_head_kv; h++) {
        oc_tensor_rope_gptj_f32(s->k + h * hd, hd, s->pos, rope_base);
    }

    size_t kv_off = ((size_t)layer * c->n_ctx + (size_t)s->pos)
                  * s->kv_row_floats;
    memcpy(s->kv_k + kv_off, s->k, s->kv_row_floats * sizeof(float));
    memcpy(s->kv_v + kv_off, s->v, s->kv_row_floats * sizeof(float));

    for (uint32_t h = 0; h < c->n_head; h++) {
        oc_llama_attention_head(s, h, layer, s->q + h * hd,
                            s->attn_out + h * hd);
    }

    oc_llama_matvec(&L->attn_output, s->attn_out, s->normed, s->dequant_temp);
    for (size_t i = 0; i < n_embd; i++) s->x[i] += s->normed[i];

    arch_layer_norm(s->x, L->ffn_norm, L->ffn_norm_bias, s->normed,
                    n_embd, c->rms_norm_eps);

    oc_llama_matvec(&L->ffn_up, s->normed, s->ffn_gate, s->dequant_temp);
    arch_gelu_inplace_f32(s->ffn_gate, c->n_ff);
    oc_llama_matvec(&L->ffn_down, s->ffn_gate, s->normed, s->dequant_temp);
    for (size_t i = 0; i < n_embd; i++) s->x[i] += s->normed[i];
}

/* ─── GPT-NeoX layer forward ─────────────────────────────────────────────
 *
 * GPT-NeoX block (sequential variant for clarity):
 *   1. ln = LayerNorm(x, attn_norm = input_layernorm)
 *   2. QKV projections from ln
 *   3. RoPE on Q and K (split-halves NeoX layout)
 *   4. KV cache write
 *   5. attention → attn_out; x += o_proj(attn_out)
 *   6. ln2 = LayerNorm(x, ffn_norm = post_attention_layernorm)
 *   7. mlp_out = mlp_down(SwiGLU(mlp_gate(ln2), mlp_up(ln2))); x += mlp_out
 *
 * Weight mapping (canonical → NeoX HF):
 *   attn_q → layers.N.attention.query_key_value.q_proj.weight
 *   attn_k → layers.N.attention.query_key_value.k_proj.weight
 *   attn_v → layers.N.attention.query_key_value.v_proj.weight
 *   (NeoX uses a fused query_key_value weight; the GGUF converter splits it.)
 *   attn_output → layers.N.attention.dense.weight
 *   ffn_gate → layers.N.mlp.dense_h_to_4h.weight (gate, via SwiGLU)
 *   ffn_up   → layers.N.mlp.dense_h_to_4h.weight (up; same matrix when not
 *              gated, or a second matrix for the gated variant)
 *   ffn_down → layers.N.mlp.dense_4h_to_h.weight
 *   attn_norm → layers.N.input_layernorm.weight
 *   ffn_norm  → layers.N.post_attention_layernorm.weight
 *
 * NeoX uses rotary embeddings on a portion of the head dimension (rope_dim).
 * The RoPE is applied per-head using the split-halves layout (same as Llama).
 */
static void arch_neox_layer(OcLlamaSession *s, uint32_t layer)
{
    const OcLlamaConfig *c = &s->model->cfg;
    OcLlamaLayer *L = &s->model->layers[layer];
    size_t hd = c->head_dim;
    size_t n_embd = c->n_embd;

    /* 1. Pre-attention LayerNorm. */
    arch_layer_norm(s->x, L->attn_norm, L->attn_norm_bias, s->normed,
                    n_embd, c->rms_norm_eps);

    /* 2. Q/K/V projections. */
    oc_llama_matvec(&L->attn_q, s->normed, s->q, s->dequant_temp);
    oc_llama_matvec(&L->attn_k, s->normed, s->k, s->dequant_temp);
    oc_llama_matvec(&L->attn_v, s->normed, s->v, s->dequant_temp);

    /* 3. RoPE on Q (per head) and K (per kv head). */
    for (uint32_t h = 0; h < c->n_head; h++) {
        oc_apply_rope_f32(s->q + h * hd, s->q + h * hd, hd, c->rope_dim,
                          s->pos, c->rope_theta);
    }
    for (uint32_t h = 0; h < c->n_head_kv; h++) {
        oc_apply_rope_f32(s->k + h * hd, s->k + h * hd, hd, c->rope_dim,
                          s->pos, c->rope_theta);
    }

    /* 4. KV cache write. */
    size_t kv_off = ((size_t)layer * c->n_ctx + (size_t)s->pos)
                  * s->kv_row_floats;
    memcpy(s->kv_k + kv_off, s->k, s->kv_row_floats * sizeof(float));
    memcpy(s->kv_v + kv_off, s->v, s->kv_row_floats * sizeof(float));

    /* 5. Attention per head → s->attn_out. */
    for (uint32_t h = 0; h < c->n_head; h++) {
        oc_llama_attention_head(s, h, layer, s->q + h * hd,
                            s->attn_out + h * hd);
    }

    /* Output projection + residual. */
    oc_llama_matvec(&L->attn_output, s->attn_out, s->normed, s->dequant_temp);
    for (size_t i = 0; i < n_embd; i++) s->x[i] += s->normed[i];

    /* 6. Pre-FFN LayerNorm. */
    arch_layer_norm(s->x, L->ffn_norm, L->ffn_norm_bias, s->normed,
                    n_embd, c->rms_norm_eps);

    /* 7. FFN. Standard NeoX checkpoints have a single-projection GeLU MLP:
     *    dense_4h_to_h(GELU(dense_h_to_4h(x))). No gate tensor exists, so
     *    the GeLU path is the default; the gated SwiGLU variant is only
     *    taken when a real ffn_gate tensor was loaded. */
    if (L->ffn_gate.data != NULL) {
        oc_llama_matvec(&L->ffn_gate, s->normed, s->ffn_gate, s->dequant_temp);
        oc_llama_matvec(&L->ffn_up,   s->normed, s->ffn_up,   s->dequant_temp);
        oc_swiglu_inplace_f32(s->ffn_gate, s->ffn_up, c->n_ff);
        oc_llama_matvec(&L->ffn_down, s->ffn_gate, s->normed, s->dequant_temp);
    } else {
        oc_llama_matvec(&L->ffn_up, s->normed, s->ffn_up, s->dequant_temp);
        arch_gelu_inplace_f32(s->ffn_up, c->n_ff);
        oc_llama_matvec(&L->ffn_down, s->ffn_up, s->normed, s->dequant_temp);
    }
    for (size_t i = 0; i < n_embd; i++) s->x[i] += s->normed[i];
}

/* ─── Falcon layer forward ──────────────────────────────────────────────
 *
 * Falcon block (parallel attention/FFN, Falcon-7B/40B convention):
 *   1. ln = LayerNorm(x, attn_norm = input_layernorm)
 *   2. QKV projections from ln (fused query_key_value; split into Q/K/V)
 *   3. RoPE on Q and K
 *   4. KV cache write (single KV head: n_head_kv == 1)
 *   5. attention → attn_out; x += o_proj(attn_out)
 *   6. FFN (operates on the post-attention residual x):
 *      mlp_out = mlp_down(SwiGLU(mlp_gate(x), mlp_up(x))); x += mlp_out
 *
 * Weight mapping (canonical → Falcon HF):
 *   attn_q → transformer.h.N.self_attention.query_key_value.weight (q part)
 *   attn_k → transformer.h.N.self_attention.query_key_value.weight (k part)
 *   attn_v → transformer.h.N.self_attention.query_key_value.weight (v part)
 *   (Falcon uses a single fused query_key_value weight of shape
 *    [n_embd + 2*head_dim, n_embd] because of multi-query attention:
 *    Q is [n_embd, n_embd], K and V are [head_dim, n_embd] each.)
 *   attn_output → transformer.h.N.self_attention.dense.weight
 *   ffn_gate → transformer.h.N.mlp.dense_h_to_4h.weight (gate)
 *   ffn_up   → transformer.h.N.mlp.dense_h_to_4h.weight (up)
 *   ffn_down → transformer.h.N.mlp.dense_4h_to_h.weight
 *   attn_norm → transformer.h.N.input_layernorm.weight
 *   (Falcon has no post_attention_layernorm; the FFN operates on the
 *    post-attention residual directly. We use ffn_norm for the second
 *    LayerNorm when present, or skip it when absent — Falcon-7B/40B
 *    have no second LayerNorm in the FFN path.)
 *
 * Multi-query attention: n_head_kv == 1, so all Q heads attend to the same
 * single K/V head. The arch_attention_head helper handles this correctly:
 * group = n_head / n_head_kv = n_head, so kv_head = head / n_head == 0
 * for all heads.
 */
static void arch_falcon_layer(OcLlamaSession *s, uint32_t layer)
{
    const OcLlamaConfig *c = &s->model->cfg;
    OcLlamaLayer *L = &s->model->layers[layer];
    size_t hd = c->head_dim;
    size_t n_embd = c->n_embd;

    /* 1. Pre-attention LayerNorm (shared by attention and MLP in the
     *    Falcon parallel block). */
    arch_layer_norm(s->x, L->attn_norm, L->attn_norm_bias, s->normed,
                    n_embd, c->rms_norm_eps);

    /* 2. Q/K/V projections from the normalized input. */
    oc_llama_matvec(&L->attn_q, s->normed, s->q, s->dequant_temp);
    oc_llama_matvec(&L->attn_k, s->normed, s->k, s->dequant_temp);
    oc_llama_matvec(&L->attn_v, s->normed, s->v, s->dequant_temp);

    /* 2b. Parallel MLP branch. Falcon's block computes attention and the
     *     MLP from the same layer-normalized input, then adds both to the
     *     residual: x = x + attn(ln) + mlp(ln). Q/K/V are already
     *     projected, so `normed` is free to be consumed/overwritten here.
     *     Falcon-40B has a second LayerNorm (ln_mlp) applied to the layer
     *     INPUT for the MLP branch; when present we use it. Standard
     *     Falcon checkpoints use a single-projection GeLU MLP
     *     (dense_4h_to_h(GELU(dense_h_to_4h(x)))); the gated variant is
     *     only taken when a real ffn_gate tensor was loaded.
     *     The MLP residual is added to x before attention runs — the two
     *     residual adds commute, and doing the MLP first lets `normed`
     *     serve as both MLP input and output buffer. */
    if (L->ffn_norm != NULL) {
        /* x is still the layer input here (no residuals added yet). */
        arch_layer_norm(s->x, L->ffn_norm, L->ffn_norm_bias, s->normed,
                        n_embd, c->rms_norm_eps);
    }
    if (L->ffn_gate.data != NULL) {
        oc_llama_matvec(&L->ffn_gate, s->normed, s->ffn_gate, s->dequant_temp);
        oc_llama_matvec(&L->ffn_up,   s->normed, s->ffn_up,   s->dequant_temp);
        oc_swiglu_inplace_f32(s->ffn_gate, s->ffn_up, c->n_ff);
        oc_llama_matvec(&L->ffn_down, s->ffn_gate, s->normed, s->dequant_temp);
    } else {
        oc_llama_matvec(&L->ffn_up, s->normed, s->ffn_up, s->dequant_temp);
        arch_gelu_inplace_f32(s->ffn_up, c->n_ff);
        oc_llama_matvec(&L->ffn_down, s->ffn_up, s->normed, s->dequant_temp);
    }
    for (size_t i = 0; i < n_embd; i++) s->x[i] += s->normed[i];

    /* 3. RoPE on Q (per head) and K (per kv head; for Falcon n_head_kv==1). */
    for (uint32_t h = 0; h < c->n_head; h++) {
        oc_apply_rope_f32(s->q + h * hd, s->q + h * hd, hd, c->rope_dim,
                          s->pos, c->rope_theta);
    }
    for (uint32_t h = 0; h < c->n_head_kv; h++) {
        oc_apply_rope_f32(s->k + h * hd, s->k + h * hd, hd, c->rope_dim,
                          s->pos, c->rope_theta);
    }

    /* 4. KV cache write (single KV head for Falcon). */
    size_t kv_off = ((size_t)layer * c->n_ctx + (size_t)s->pos)
                  * s->kv_row_floats;
    memcpy(s->kv_k + kv_off, s->k, s->kv_row_floats * sizeof(float));
    memcpy(s->kv_v + kv_off, s->v, s->kv_row_floats * sizeof(float));

    /* 5. Attention per head → s->attn_out.
     *    Multi-query: all Q heads attend to the same K/V (kv_head=0). */
    for (uint32_t h = 0; h < c->n_head; h++) {
        oc_llama_attention_head(s, h, layer, s->q + h * hd,
                            s->attn_out + h * hd);
    }

    /* 6. Output projection + attention residual (MLP residual was already
     *    added in step 2b). */
    oc_llama_matvec(&L->attn_output, s->attn_out, s->normed, s->dequant_temp);
    for (size_t i = 0; i < n_embd; i++) s->x[i] += s->normed[i];
}

/* ─── Final norm + lm_head projection (shared) ──────────────────────────
 *
 * Applies the final LayerNorm (or RMSNorm when the model was loaded with
 * RMSNorm-style final norm — but for GPT-2/NeoX/Falcon, the final norm is
 * a LayerNorm) and projects to vocab size to produce logits.
 *
 * For GPT-2/NeoX/Falcon, `final_norm` is the LayerNorm weight (gamma).
 * We apply LayerNorm here (not RMSNorm) to match the architecture.
 *
 * When `logits_out` is NULL, the lm_head projection is skipped (used for
 * prefill where only the KV cache matters). */
static OcError arch_final_norm_and_logits(OcLlamaSession *s, float *logits_out)
{
    OcLlamaModel *m = s->model;
    size_t n_embd = m->cfg.n_embd;

    /* Final LayerNorm (GPT-2/NeoX/Falcon use LayerNorm, not RMSNorm). */
    arch_layer_norm(s->x, m->final_norm, m->final_norm_bias, s->normed,
                    n_embd, m->cfg.rms_norm_eps);

    /* lm_head projection → logits. */
    if (logits_out != NULL) {
        if (m->output.qtype == OC_QUANT_F32) {
            oc_matvec_f32((const float *)m->output.data, m->output.rows,
                          m->output.cols, s->normed, logits_out);
        } else {
            oc_matvec_quantized(m->output.qtype, m->output.data,
                                m->output.rows, m->output.cols,
                                m->output.row_bytes, s->normed, logits_out,
                                s->dequant_temp);
        }
    }
    return OC_OK;
}

/* ─── Positional embedding lookup (GPT-2 learned embeddings) ────────────
 *
 * GPT-2 uses learned positional embeddings (wpe) stored as a
 * [n_ctx, n_embd] weight matrix. The embedding for position `pos` is row
 * `pos` of this matrix, added to the token embedding.
 *
 * The wpe tensor is not part of the standard OcLlamaModel struct (which
 * only has tok_embeddings, output, final_norm, and per-layer weights).
 * For the C port, we look it up from the GGUF by scanning for the
 * `position_embeddings.weight` (or `wpe.weight`) canonical name. If the
 * model was loaded without position embeddings (the GGUF converter didn't
 * include them), we skip the positional embedding addition and log a
 * warning — the model will produce garbage for multi-token sequences but
 * will not crash.
 *
 * The resolved view is cached per model (`m->gpt2_pos_embed`), so multiple
 * models never share each other's mmap and freeing one model cannot leave
 * a dangling cache for another.
 *
 * For NeoX and Falcon, positional encoding is RoPE (applied per layer),
 * so no learned positional embedding lookup is needed. */

/* Resolve the GPT-2 position embedding tensor from the model's GGUF.
 * Returns a pointer to the weight view, or NULL if not found. */
static OcWeightView *arch_gpt2_get_pos_embed(OcLlamaModel *m)
{
    if (m->gpt2_pos_resolved) {
        return (m->gpt2_pos_embed.data != NULL) ? &m->gpt2_pos_embed : NULL;
    }
    m->gpt2_pos_resolved = true;

    /* Scan the GGUF tensor list for the position embedding tensor.
     * We use the GGUF map's tensor info iterator to find a tensor named
     * "position_embeddings.weight" or "wpe.weight" (canonical name after
     * mapping). */
    OcArena *arena = oc_arena_new(1u << 16);
    if (arena == NULL) return NULL;

    OcGgufTensorInfo *infos = NULL;
    size_t n = 0;
    OcError e = oc_gguf_map_mapped_tensor_infos(&m->gguf, arena, &infos, &n);
    if (e != OC_OK) {
        oc_arena_free(arena);
        return NULL;
    }

    for (size_t i = 0; i < n; i++) {
        const char *cname = infos[i].name;
        if (cname == NULL) continue;
        if (strcmp(cname, "position_embeddings.weight") == 0 ||
            strcmp(cname, "wpe.weight") == 0 ||
            strcmp(cname, "position_embd.weight") == 0) {
            m->gpt2_pos_embed = (OcWeightView){
                .data = oc_gguf_map_tensor_data(&m->gguf, &infos[i]),
                .qtype = oc_quant_type_from_ggml_id(infos[i].ggml_type),
                .rows = (infos[i].n_dims >= 2) ? infos[i].dims[1] : 1,
                .cols = infos[i].dims[0],
            };
            m->gpt2_pos_embed.row_bytes =
                oc_quantized_size(m->gpt2_pos_embed.qtype, m->gpt2_pos_embed.cols);
            if (m->gpt2_pos_embed.row_bytes == 0) {
                m->gpt2_pos_embed.row_bytes =
                    m->gpt2_pos_embed.cols * sizeof(float);
            }
            break;
        }
    }
    oc_arena_free(arena);

    if (m->gpt2_pos_embed.data == NULL) {
        oc_log(OC_LOG_WARN,
               "arch_forward: GPT-2 position_embeddings.weight not found; "
               "multi-token sequences will be incorrect");
        return NULL;
    }
    return &m->gpt2_pos_embed;
}

/* Add the GPT-2 learned positional embedding for position `pos` to `s->x`. */
static void arch_gpt2_add_pos_embed(OcLlamaSession *s, int64_t pos)
{
    OcWeightView *wpe = arch_gpt2_get_pos_embed(s->model);
    if (wpe == NULL) return;
    if (pos >= (int64_t)wpe->rows) pos = (int64_t)wpe->rows - 1;
    if (pos < 0) pos = 0;

    size_t n_embd = s->model->cfg.n_embd;
    if (wpe->qtype == OC_QUANT_F32) {
        const float *row = (const float *)(wpe->data + (size_t)pos * wpe->row_bytes);
        for (size_t i = 0; i < n_embd; i++) s->x[i] += row[i];
    } else {
        /* Dequantize the position embedding row into the temp buffer, then
         * add. We reuse the ffn_up buffer as temp (it's n_ff >= n_embd in
         * practice; if n_ff < n_embd we use dequant_temp which is max-sized). */
        float *temp = (s->model->cfg.n_ff >= n_embd) ? s->ffn_up
                                                      : s->dequant_temp;
        oc_quant_dequant_row(wpe->qtype,
            wpe->data + (size_t)pos * wpe->row_bytes, wpe->row_bytes,
            temp, n_embd);
        for (size_t i = 0; i < n_embd; i++) s->x[i] += temp[i];
    }
}

/* ─── Validation helpers ────────────────────────────────────────────────
 *
 * Check that the session and model are valid and that the position is within
 * the context window. Returns OC_ERR_INVALID_ARG on failure. */
static OcError arch_validate_session(OcLlamaSession *sess)
{
    if (sess == NULL || sess->model == NULL) return OC_ERR_INVALID_ARG;
    if ((uint64_t)sess->pos >= sess->model->cfg.n_ctx) return OC_ERR_INVALID_ARG;
    return OC_OK;
}

/* Check that the layer weights required for a forward pass are present.
 * Returns OC_ERR_MODEL if critical weights are missing. */
static OcError arch_validate_layers(OcLlamaModel *m)
{
    if (m->layers == NULL) return OC_ERR_MODEL;
    if (m->tok_embeddings.data == NULL) return OC_ERR_MODEL;
    if (m->final_norm == NULL) return OC_ERR_MODEL;
    for (uint32_t l = 0; l < m->cfg.n_layer; l++) {
        OcLlamaLayer *L = &m->layers[l];
        if (L->attn_norm == NULL) return OC_ERR_MODEL;
        if (L->attn_q.data == NULL) return OC_ERR_MODEL;
        if (L->attn_k.data == NULL) return OC_ERR_MODEL;
        if (L->attn_v.data == NULL) return OC_ERR_MODEL;
        if (L->attn_output.data == NULL) return OC_ERR_MODEL;
        if (L->ffn_up.data == NULL) return OC_ERR_MODEL;
        if (L->ffn_down.data == NULL) return OC_ERR_MODEL;
    }
    return OC_OK;
}

/* ─── GPT-2 forward pass ────────────────────────────────────────────────
 *
 * Full GPT-2 forward pass for a single token:
 *   1. Look up token embedding → x
 *   2. Add learned positional embedding for `pos` → x
 *   3. For each layer: GPT-2 block (LayerNorm → attn → residual →
 *      LayerNorm → GeLU MLP → residual)
 *   4. Final LayerNorm
 *   5. lm_head projection → logits
 *   6. Advance position */
OcError oc_arch_forward_gpt2(OcLlamaSession *sess, uint32_t token,
                              float *logits_out)
{
    OcError e = arch_validate_session(sess);
    if (e != OC_OK) return e;
    e = arch_validate_layers(sess->model);
    if (e != OC_OK) return e;

    /* 1. Token embedding lookup. */
    oc_llama_embed_token(sess, token);

    /* 2. Add learned positional embedding. */
    arch_gpt2_add_pos_embed(sess, sess->pos);

    /* 3. Per-layer forward. */
    for (uint32_t l = 0; l < sess->model->cfg.n_layer; l++) {
        arch_gpt2_layer(sess, l);
    }

    /* 4-5. Final LayerNorm + lm_head projection → logits. */
    e = arch_final_norm_and_logits(sess, logits_out);
    if (e != OC_OK) return e;

    /* 6. Advance position. */
    sess->pos++;
    return OC_OK;
}

OcError oc_arch_forward_gptj(OcLlamaSession *sess, uint32_t token,
                              float *logits_out)
{
    OcError e = arch_validate_session(sess);
    if (e != OC_OK) return e;
    e = arch_validate_layers(sess->model);
    if (e != OC_OK) return e;

    oc_llama_embed_token(sess, token);

    for (uint32_t l = 0; l < sess->model->cfg.n_layer; l++) {
        arch_gptj_layer(sess, l);
    }

    e = arch_final_norm_and_logits(sess, logits_out);
    if (e != OC_OK) return e;

    sess->pos++;
    return OC_OK;
}

/* ─── GPT-NeoX forward pass ─────────────────────────────────────────────
 *
 * Full GPT-NeoX forward pass for a single token:
 *   1. Look up token embedding → x
 *   2. (RoPE is applied per-layer, not as a global positional embedding)
 *   3. For each layer: NeoX block (LayerNorm → attn+RoPE → residual →
 *      LayerNorm → SwiGLU MLP → residual)
 *   4. Final LayerNorm
 *   5. lm_head projection → logits
 *   6. Advance position */
OcError oc_arch_forward_gpt_neox(OcLlamaSession *sess, uint32_t token,
                                  float *logits_out)
{
    OcError e = arch_validate_session(sess);
    if (e != OC_OK) return e;
    e = arch_validate_layers(sess->model);
    if (e != OC_OK) return e;

    /* 1. Token embedding lookup. */
    oc_llama_embed_token(sess, token);

    /* 2. No global positional embedding (RoPE is per-layer). */

    /* 3. Per-layer forward. */
    for (uint32_t l = 0; l < sess->model->cfg.n_layer; l++) {
        arch_neox_layer(sess, l);
    }

    /* 4-5. Final LayerNorm + lm_head projection → logits. */
    e = arch_final_norm_and_logits(sess, logits_out);
    if (e != OC_OK) return e;

    /* 6. Advance position. */
    sess->pos++;
    return OC_OK;
}

/* ─── Falcon forward pass ───────────────────────────────────────────────
 *
 * Full Falcon forward pass for a single token:
 *   1. Look up token embedding → x
 *   2. (RoPE is applied per-layer)
 *   3. For each layer: Falcon block (LayerNorm → attn+RoPE → residual →
 *      SwiGLU MLP → residual, no second LayerNorm)
 *   4. Final LayerNorm
 *   5. lm_head projection → logits
 *   6. Advance position */
OcError oc_arch_forward_falcon(OcLlamaSession *sess, uint32_t token,
                                float *logits_out)
{
    OcError e = arch_validate_session(sess);
    if (e != OC_OK) return e;
    e = arch_validate_layers(sess->model);
    if (e != OC_OK) return e;

    /* 1. Token embedding lookup. */
    oc_llama_embed_token(sess, token);

    /* 2. No global positional embedding (RoPE is per-layer). */

    /* 3. Per-layer forward. */
    for (uint32_t l = 0; l < sess->model->cfg.n_layer; l++) {
        arch_falcon_layer(sess, l);
    }

    /* 4-5. Final LayerNorm + lm_head projection → logits. */
    e = arch_final_norm_and_logits(sess, logits_out);
    if (e != OC_OK) return e;

    /* 6. Advance position. */
    sess->pos++;
    return OC_OK;
}

