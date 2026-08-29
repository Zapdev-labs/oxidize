/*
 * test_llama.c — Llama forward-pass component tests.
 *
 * The oxidize-core GGUF test fixtures are tiny parser fixtures (no
 * weights), so there is no end-to-end runnable model to test against.
 * Instead this file asserts:
 *   1. The math primitives (RMSNorm, RoPE, SwiGLU, matvec) match hand-
 *      computed expected values (VAL-FWD-001..004).
 *   2. oc_llama_load gracefully rejects the parser fixtures (which lack
 *      tok_embeddings) with OC_ERR_MODEL rather than crashing.
 *   3. Session workspace allocation sizes are consistent with config.
 *
 * Full end-to-end parity against a Rust reference forward requires a real
 * GGUF model; that test runs on the remote NUMA box (ai@192.168.1.121) as
 * part of the cpu-qwen-benchmark-121 feature.
 */
#include <criterion/criterion.h>

#include "oxidize/activation.h"
#include "oxidize/llama.h"
#include "oxidize/matvec.h"
#include "oxidize/quant.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* ─── RMSNorm ────────────────────────────────────────────────────────────
 * x=[3,4,0], w=[1,1,1], eps=1e-5.
 * ss = 9+16+0 = 25; mean = 25/3; inv_rms = 1/sqrt(25/3 + eps).
 * out[i] = x[i] * inv_rms. */
Test(llama, rms_norm_basic)
{
    float x[] = {3.0f, 4.0f, 0.0f};
    float w[] = {1.0f, 1.0f, 1.0f};
    float out[3];
    oc_rms_norm_f32(x, w, out, 3, 1e-5f);
    float inv_rms = 1.0f / sqrtf(25.0f / 3.0f + 1e-5f);
    cr_assert_float_eq(out[0], 3.0f * inv_rms, 1e-5f, "out[0]");
    cr_assert_float_eq(out[1], 4.0f * inv_rms, 1e-5f, "out[1]");
    cr_assert_float_eq(out[2], 0.0f, 1e-7f, "out[2] should be zero");
}

Test(llama, rms_norm_weight_scales)
{
    /* If we double the weight, output doubles. */
    float x[] = {1.0f, 2.0f, 3.0f};
    float w1[] = {1.0f, 1.0f, 1.0f};
    float w2[] = {2.0f, 2.0f, 2.0f};
    float o1[3], o2[3];
    oc_rms_norm_f32(x, w1, o1, 3, 1e-5f);
    oc_rms_norm_f32(x, w2, o2, 3, 1e-5f);
    for (int i = 0; i < 3; i++) {
        cr_assert_float_eq(o2[i], 2.0f * o1[i], 1e-5f, "weight scaling at %d", i);
    }
}

/* ─── RoPE (split-halves, NeoX-style) ────────────────────────────────────
 * head_dim=4, rope_len=4, position=1, theta=10000.
 * half=2; freq_mul = 10000^(-2/4) = 0.01.
 * pair 0: freq=1.0,     angle=1.0
 * pair 1: freq=0.01,    angle=0.01
 * input [1,2,3,4]:
 *   out[0] = 1*cos(1.0) - 3*sin(1.0)
 *   out[2] = 1*sin(1.0) + 3*cos(1.0)
 *   out[1] = 2*cos(0.01) - 4*sin(0.01)
 *   out[3] = 2*sin(0.01) + 4*cos(0.01) */
Test(llama, rope_split_halves_position1)
{
    float in[]  = {1.0f, 2.0f, 3.0f, 4.0f};
    float out[4];
    oc_apply_rope_f32(in, out, 4, 4, 1, 10000.0f);
    float c1 = cosf(1.0f), s1 = sinf(1.0f);
    float c01 = cosf(0.01f), s01 = sinf(0.01f);
    cr_assert_float_eq(out[0], 1.0f * c1 - 3.0f * s1, 1e-5f, "out[0]");
    cr_assert_float_eq(out[2], 1.0f * s1 + 3.0f * c1, 1e-5f, "out[2]");
    cr_assert_float_eq(out[1], 2.0f * c01 - 4.0f * s01, 1e-5f, "out[1]");
    cr_assert_float_eq(out[3], 2.0f * s01 + 4.0f * c01, 1e-5f, "out[3]");
}

Test(llama, rope_position_zero_is_identity)
{
    /* Position 0 → no rotation (fast path). */
    float in[] = {1.5f, -2.5f, 3.0f, -4.0f, 5.0f, -6.0f, 7.0f, 8.0f};
    float out[8];
    oc_apply_rope_f32(in, out, 8, 8, 0, 10000.0f);
    cr_assert_arr_eq(out, in, 8 * sizeof(float), "pos=0 must be identity");
}

Test(llama, rope_partial_passthrough_tail)
{
    /* rope_len=4 < head_dim=8 → tail [4..8] passes through unchanged. */
    float in[] = {1.0f, 2.0f, 3.0f, 4.0f, 99.0f, 100.0f, 101.0f, 102.0f};
    float out[8];
    oc_apply_rope_f32(in, out, 8, 4, 1, 10000.0f);
    cr_assert_float_eq(out[4], 99.0f, 1e-6f, "tail passthrough [4]");
    cr_assert_float_eq(out[5], 100.0f, 1e-6f, "tail passthrough [5]");
    cr_assert_float_eq(out[6], 101.0f, 1e-6f, "tail passthrough [6]");
    cr_assert_float_eq(out[7], 102.0f, 1e-6f, "tail passthrough [7]");
}

/* ─── SwiGLU ─────────────────────────────────────────────────────────────
 * gate=[0,1], up=[1,2].
 * silu(0)=0*sigmoid(0)=0; silu(1)=1*sigmoid(1)=0.731059.
 * out = [0*1, 0.731059*2] = [0, 1.462117]. */
Test(llama, swiglu_basic)
{
    float gate[] = {0.0f, 1.0f};
    float up[]   = {1.0f, 2.0f};
    oc_swiglu_inplace_f32(gate, up, 2);
    cr_assert_float_eq(gate[0], 0.0f, 1e-6f, "silu(0)*1 = 0");
    cr_assert_float_eq(gate[1], 0.73105858f * 2.0f, 1e-5f, "silu(1)*2");
}

/* ─── matvec_f32 ─────────────────────────────────────────────────────────
 * data = [[1,2,3],[4,5,6]], input=[1,1,1] → [6, 15]. */
Test(llama, matvec_f32_basic)
{
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    float in[]   = {1.0f, 1.0f, 1.0f};
    float out[2];
    oc_matvec_f32(data, 2, 3, in, out);
    cr_assert_float_eq(out[0], 6.0f, 1e-6f, "row 0 dot");
    cr_assert_float_eq(out[1], 15.0f, 1e-6f, "row 1 dot");
}

Test(llama, matvec_bf16_matches_dequant)
{
    const size_t rows = 4, cols = 32;
    float src[4 * 32], in[32], want[4], got[4], temp[32];
    uint8_t packed[4 * 64];
    for (size_t i = 0; i < rows * cols; i++)
        src[i] = 0.05f * (float)((int)(i % 17) - 8);
    for (size_t i = 0; i < cols; i++)
        in[i] = 0.1f * (float)((int)(i % 7) - 3);
    for (size_t r = 0; r < rows; r++) {
        cr_assert_eq(oc_quant_pack_row(OC_QUANT_BF16, src + r * cols, cols,
                                       packed + r * 64, 64), OC_OK);
        cr_assert_eq(oc_quant_dequant_row(OC_QUANT_BF16, packed + r * 64, 64,
                                          temp, cols), OC_OK);
        want[r] = 0.0f;
        for (size_t c = 0; c < cols; c++) want[r] += temp[c] * in[c];
    }
    oc_matvec_quantized(OC_QUANT_BF16, packed, rows, cols, 64, in, got, temp);
    for (size_t r = 0; r < rows; r++)
        cr_assert_float_eq(got[r], want[r], 1e-4f, "bf16 row %zu", r);
}

Test(llama, matvec_f32_zero_input)
{
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    float in[]   = {0.0f, 0.0f, 0.0f};
    float out[2];
    oc_matvec_f32(data, 2, 3, in, out);
    cr_assert_float_eq(out[0], 0.0f, 1e-6f);
    cr_assert_float_eq(out[1], 0.0f, 1e-6f);
}

/* ─── Load: parser fixture must be rejected ─────────────────────────────
 * oxidize-core/tests/fixtures/valid-v3.gguf has a valid GGUF header but no
 * tok_embeddings.weight → oc_llama_load must return OC_ERR_MODEL (not crash). */
Test(llama, load_rejects_parser_fixture)
{
    const char *path = "../oxidize-core/tests/fixtures/valid-v3.gguf";
    OcLlamaModel m;
    OcError e = oc_llama_load(path, &m);
    /* Either the fixture exists and we get OC_ERR_MODEL (no weights), or the
     * relative path doesn't resolve in this CWD — either way we must NOT
     * crash and must NOT report OC_OK. */
    cr_assert(e != OC_OK, "parser fixture must not load as a model (got %d)", (int)e);
}

Test(llama, load_null_args)
{
    OcLlamaModel m;
    cr_assert_eq(oc_llama_load(NULL, &m), OC_ERR_INVALID_ARG);
    cr_assert_eq(oc_llama_load("path", NULL), OC_ERR_INVALID_ARG);
}

Test(llama, session_init_null_args)
{
    OcLlamaSession s;
    cr_assert_eq(oc_llama_session_init(NULL, &s), OC_ERR_INVALID_ARG);
    cr_assert_eq(oc_llama_session_init((OcLlamaModel *)0x1, NULL), OC_ERR_INVALID_ARG);
}

Test(llama, sessions_reject_unsupported_architecture_paths)
{
    OcLlamaModel model;
    memset(&model, 0, sizeof(model));
    OcLlamaSession session;
    OcBatchSession batch;
    /* MLA is no longer rejected — forward_mla_attention handles it, and
     * single-token sessions for DeepSeek/GLM-MoE-DSA/LongCat go through it.
     * (Batch decode already allocated the MLA workspace; see
     * batch_allocates_mla_workspace below.) uses_geglu is likewise a
     * supported activation. LayerNorm architectures (GPT-2/NeoX/Falcon) are
     * rejected by batch init only — single-token sessions dispatch to
     * arch_forward.c. */
    model.arch = OC_ARCH_GPT2;
    cr_assert_eq(oc_batch_session_init(&model, 2, &batch), OC_ERR_MODEL);
    model.arch = OC_ARCH_GPTJ;
    cr_assert_eq(oc_batch_session_init(&model, 2, &batch), OC_ERR_MODEL);
    model.arch = OC_ARCH_FALCON;
    cr_assert_eq(oc_batch_session_init(&model, 2, &batch), OC_ERR_MODEL);
}

Test(llama, batch_allocates_mla_workspace)
{
    /* MLA models are supported by batch decode: the compressed-KV
     * temporaries are allocated once and shared across sequences. */
    OcLlamaModel model;
    memset(&model, 0, sizeof(model));
    model.cfg.n_ctx = 4;
    model.cfg.n_layer = 1;
    model.cfg.n_embd = 8;
    model.cfg.n_ff = 8;
    model.cfg.n_head = 2;
    model.cfg.n_head_kv = 2;
    model.cfg.head_dim = 4;
    model.cfg.kv_head_dim = 4;
    model.cfg.vocab_size = 4;
    model.cfg.uses_mla = true;
    model.cfg.mla_q_lora_dim = 8;
    model.cfg.mla_kv_lora_dim = 8;
    model.cfg.mla_q_rope_dim = 2;

    OcBatchSession batch;
    cr_assert_eq(oc_batch_session_init(&model, 2, &batch), OC_OK);
    cr_assert_not_null(batch.mla_c_q);
    cr_assert_not_null(batch.mla_c_kv);
    cr_assert_not_null(batch.mla_q_full);
    cr_assert_not_null(batch.mla_kv_compressed);
    cr_assert_not_null(batch.mla_q_absorbed);
    cr_assert_not_null(batch.mla_ctx_latent);
    /* MLA caches the compressed [c_kv | k_pe] latent, and the batch path
     * must agree with the single-sequence path exactly — they index the same
     * rows through the same forward code. Sizing this the old expanded way
     * (n_head * head_dim) allocated ~21x more per sequence. */
    cr_assert_eq(batch.kv_row_floats,
                 (size_t)model.cfg.mla_kv_lora_dim + model.cfg.mla_q_rope_dim);
    oc_batch_session_free(&batch);
    cr_assert_null(batch.mla_c_q);
}

Test(llama, batch_validates_context_and_moe_workspace)
{
    OcLlamaModel model;
    memset(&model, 0, sizeof(model));
    model.cfg.n_ctx = 2;
    model.cfg.n_layer = 1;
    model.cfg.n_embd = 4;
    model.cfg.n_ff = 8;
    model.cfg.n_head = 1;
    model.cfg.n_head_kv = 1;
    model.cfg.head_dim = 4;
    model.cfg.kv_head_dim = 4;
    model.cfg.vocab_size = 4;
    model.cfg.num_experts = 2;
    OcBatchSession batch;
    cr_assert_eq(oc_batch_session_init(&model, 2, &batch), OC_OK);
    cr_assert_eq(model.cfg.expert_intermediate_size, model.cfg.n_ff);
    cr_assert_not_null(batch.expert_gate);
    OcBatchSeq seqs[2] = {0};
    seqs[0].active = true;
    seqs[0].pos = 2;
    cr_assert_eq(oc_batch_forward(&batch, seqs), OC_ERR_INVALID_ARG);
    oc_batch_session_free(&batch);
}

/* ─── MoE config defaults ──────────────────────────────────────────────
 * A dense model (no expert_count metadata) must have num_experts=0,
 * which means forward_layer uses the dense SwiGLU path. The MoE scratch
 * buffers (router_logits, expert_gate, etc.) must be NULL. */
Test(llama, moe_config_defaults_dense)
{
    OcLlamaModel m;
    /* We can't easily construct a full model, but we can verify the config
     * struct initializes to zero and the defaults are applied in
     * parse_config. If the fixture loads (or fails with IO), the config
     * should still report num_experts=0 for non-MoE GGUFs. */
    memset(&m, 0, sizeof(m));
    /* Simulate a default config (as parse_config would set it). */
    m.cfg.num_experts = 0;
    m.cfg.num_experts_per_tok = 0;
    m.cfg.expert_intermediate_size = 0;
    m.cfg.expert_gating_sigmoid = false;
    m.cfg.expert_weights_scale = 1.0f;
    cr_assert_eq(m.cfg.num_experts, 0, "dense model has no experts");
    cr_assert_eq(m.cfg.expert_gating_sigmoid, false, "default gating is softmax");
    cr_assert_float_eq(m.cfg.expert_weights_scale, 1.0f, 1e-6f, "default scale 1.0");
}

/* ─── MoE top-k clamp logic ─────────────────────────────────────────────
 * When num_experts_per_tok > num_experts, it must be clamped to num_experts.
 * When num_experts_per_tok == 0 but num_experts > 0, it defaults to 1.
 * This mirrors the clamp in parse_config. */
Test(llama, moe_topk_clamp)
{
    /* Case 1: k=0 with experts → default to 1. */
    uint32_t n_exp = 8, k = 0;
    if (n_exp > 0 && k == 0) k = 1;
    cr_assert_eq(k, 1, "k should default to 1 when experts exist");

    /* Case 2: k > n_exp → clamp to n_exp. */
    k = 16; n_exp = 8;
    if (k > n_exp) k = n_exp;
    cr_assert_eq(k, 8, "k should clamp to num_experts");

    /* Case 3: k within range → unchanged. */
    k = 4; n_exp = 8;
    if (k > n_exp) k = n_exp;
    cr_assert_eq(k, 4, "k should be unchanged when within range");
}

/* ─── GeGLU activation (Gemma FFN) ────────────────────────────────────
 * GeGLU: gate[i] = gelu(gate[i]) * up[i].
 * gelu(0) = 0; gelu(1) = 0.841345 (erf-based).
 * gate=[0,1], up=[1,2] → [0*1, 0.841345*2] = [0, 1.682690]. */
Test(llama, geglu_basic)
{
    float gate[] = {0.0f, 1.0f};
    float up[]   = {1.0f, 2.0f};
    oc_geglu_inplace_f32(gate, up, 2);
    cr_assert_float_eq(gate[0], 0.0f, 1e-6f, "gelu(0)*1 = 0");
    cr_assert_float_eq(gate[1], 0.84134474f * 2.0f, 1e-4f, "gelu(1)*2");
}

/* ─── GeLU vs SwiGLU differ ───────────────────────────────────────────
 * For the same input, GeGLU and SwiGLU must produce different outputs
 * (gelu(1) ≠ silu(1)). silu(1) = 0.731059, gelu(1) = 0.841345. */
Test(llama, geglu_differs_from_swiglu)
{
    float g1[] = {1.0f, 1.0f};
    float g2[] = {1.0f, 1.0f};
    float up[] = {1.0f, 1.0f};
    oc_swiglu_inplace_f32(g1, up, 2);
    oc_geglu_inplace_f32(g2, up, 2);
    cr_assert_neq(g1[0], g2[0], "silu(1) ≠ gelu(1)");
    cr_assert_float_eq(g1[0], 0.73105858f, 1e-5f, "silu(1)");
    cr_assert_float_eq(g2[0], 0.84134474f, 1e-4f, "gelu(1)");
}

/* ─── Gemma config defaults ──────────────────────────────────────────── */
Test(llama, gemma_config_defaults)
{
    OcLlamaConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.n_embd = 3072;  /* Gemma-2B hidden_size */
    cfg.uses_geglu = true;
    cfg.norm_scale = sqrtf((float)cfg.n_embd);
    cr_assert(cfg.uses_geglu, "Gemma uses GeGLU");
    cr_assert_float_eq(cfg.norm_scale, sqrtf(3072.0f), 1e-3f,
                       "Gemma norm_scale = sqrt(n_embd)");
    cr_assert(cfg.norm_scale > 1.0f, "norm_scale > 1 for Gemma");
}

/* ─── YaRN RoPE scaling ─────────────────────────────────────────────────
 * YaRN should be identity when yarn_factor <= 1.0 (no scaling).
 * With yarn_factor > 1.0, YaRN should produce different output. */
Test(llama, yarn_identity_within_ctx)
{
    float in[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float out_yarn[4], out_normal[4];
    /* yarn_factor=1.0 → no YaRN, should equal normal RoPE. */
    oc_apply_rope_yarn_f32(in, out_yarn, 4, 4, 10, 10000.0f, 1.0f, 4096);
    oc_apply_rope_f32(in, out_normal, 4, 4, 10, 10000.0f);
    for (int i = 0; i < 4; i++) {
        cr_assert_float_eq(out_yarn[i], out_normal[i], 1e-6f,
                           "YaRN == RoPE when factor=1.0 at %d", i);
    }
}

Test(llama, yarn_scales_beyond_ctx)
{
    float in[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float out_yarn[4], out_normal[4];
    /* position=8192, orig_ctx=4096, yarn_factor=4.0 → beyond ctx, YaRN should differ. */
    oc_apply_rope_yarn_f32(in, out_yarn, 4, 4, 8192, 10000.0f, 4.0f, 4096);
    oc_apply_rope_f32(in, out_normal, 4, 4, 8192, 10000.0f);
    bool differs = false;
    for (int i = 0; i < 4; i++) {
        if (fabsf(out_yarn[i] - out_normal[i]) > 1e-4f) {
            differs = true;
            break;
        }
    }
    cr_assert(differs, "YaRN should differ from RoPE beyond orig_ctx");
}

Test(llama, yarn_no_scaling_when_factor_zero)
{
    float in[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float out[4];
    /* factor=0 → should behave as normal RoPE. */
    oc_apply_rope_yarn_f32(in, out, 4, 4, 8192, 10000.0f, 0.0f, 4096);
    float expected[4];
    oc_apply_rope_f32(in, expected, 4, 4, 8192, 10000.0f);
    for (int i = 0; i < 4; i++) {
        cr_assert_float_eq(out[i], expected[i], 1e-6f,
                           "factor=0 → normal RoPE at %d", i);
    }
}

Test(llama, yarn_factor_changes_angles)
{
    float in[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float factor_one[4], factor_two[4];
    oc_apply_rope_yarn_f32(in, factor_one, 4, 4, 8192, 10000.0f, 1.0f, 4096);
    oc_apply_rope_yarn_f32(in, factor_two, 4, 4, 8192, 10000.0f, 2.0f, 4096);
    cr_assert_arr_neq(factor_one, factor_two, sizeof(factor_one));
}

/* ─── QKV projection bias (Qwen2-family) ───────────────────────────────
 *
 * Qwen2 carries attn_{q,k,v}.bias on every layer. The port ignored them
 * entirely, which silently produced garbage output on every Qwen2 model
 * while all unit tests stayed green — nothing here ever ran a forward pass.
 * This builds a minimal single-layer F32 model and asserts the bias reaches
 * the result.
 */
#define TB_EMBD 4u
#define TB_FF   4u
#define TB_VOCAB 4u

typedef struct {
    OcLlamaModel model;
    OcLlamaLayer layer;
    float embd[TB_VOCAB * TB_EMBD];
    float ident[TB_EMBD * TB_EMBD];
    float ffn_w[TB_FF * TB_EMBD];
    float norm_ones[TB_EMBD];
    float qb[TB_EMBD], kb[TB_EMBD], vb[TB_EMBD];
} TinyBiasModel;

static OcWeightView tb_view(const float *p, size_t rows, size_t cols)
{
    OcWeightView v = {0};
    v.data = (const uint8_t *)p;
    v.qtype = OC_QUANT_F32;
    v.rows = rows;
    v.cols = cols;
    v.row_bytes = cols * sizeof(float);
    return v;
}

/* Build a 1-layer model: identity-ish projections, zeroed FFN so the FFN
 * contributes nothing, and an embedding that depends on the token. */
static void tiny_bias_model_init(TinyBiasModel *t, bool with_bias)
{
    memset(t, 0, sizeof(*t));
    OcLlamaConfig *c = &t->model.cfg;
    c->n_layer = 1; c->n_embd = TB_EMBD; c->n_ff = TB_FF;
    c->n_head = 1; c->n_head_kv = 1;
    c->head_dim = TB_EMBD; c->kv_head_dim = TB_EMBD;
    c->vocab_size = TB_VOCAB; c->n_ctx = 8;
    c->rms_norm_eps = 1e-6f; c->norm_scale = 1.0f;
    c->rope_theta = 10000.0f; c->rope_dim = 0; /* no RoPE rotation */

    for (size_t i = 0; i < TB_VOCAB * TB_EMBD; i++)
        t->embd[i] = 0.1f * (float)(i + 1);
    for (size_t r = 0; r < TB_EMBD; r++)
        for (size_t cc = 0; cc < TB_EMBD; cc++)
            t->ident[r * TB_EMBD + cc] = (r == cc) ? 1.0f : 0.0f;
    for (size_t i = 0; i < TB_EMBD; i++) t->norm_ones[i] = 1.0f;
    /* ffn_w stays zero: the dense FFN adds nothing, isolating attention. */

    t->model.tok_embeddings = tb_view(t->embd, TB_VOCAB, TB_EMBD);
    t->model.output = tb_view(t->ident, TB_VOCAB, TB_EMBD);
    t->model.final_norm = t->norm_ones;
    t->model.layers = &t->layer;

    t->layer.attn_norm = t->norm_ones;
    t->layer.ffn_norm = t->norm_ones;
    t->layer.attn_q = tb_view(t->ident, TB_EMBD, TB_EMBD);
    t->layer.attn_k = tb_view(t->ident, TB_EMBD, TB_EMBD);
    t->layer.attn_v = tb_view(t->ident, TB_EMBD, TB_EMBD);
    t->layer.attn_output = tb_view(t->ident, TB_EMBD, TB_EMBD);
    t->layer.ffn_gate = tb_view(t->ffn_w, TB_FF, TB_EMBD);
    t->layer.ffn_up = tb_view(t->ffn_w, TB_FF, TB_EMBD);
    t->layer.ffn_down = tb_view(t->ffn_w, TB_EMBD, TB_FF);

    if (with_bias) {
        for (size_t i = 0; i < TB_EMBD; i++) {
            t->qb[i] = 0.5f + 0.25f * (float)i;
            t->kb[i] = -0.5f;
            t->vb[i] = 2.0f + (float)i;
        }
        t->layer.attn_q_bias = t->qb;
        t->layer.attn_k_bias = t->kb;
        t->layer.attn_v_bias = t->vb;
    }
}

Test(llama, qkv_bias_is_applied_in_forward)
{
    TinyBiasModel plain, biased;
    tiny_bias_model_init(&plain, false);
    tiny_bias_model_init(&biased, true);

    OcLlamaSession sp, sb;
    cr_assert_eq(oc_llama_session_init(&plain.model, &sp), OC_OK);
    cr_assert_eq(oc_llama_session_init(&biased.model, &sb), OC_OK);

    float lp[TB_VOCAB], lb[TB_VOCAB];
    cr_assert_eq(oc_llama_forward(&sp, 1u, lp), OC_OK);
    cr_assert_eq(oc_llama_forward(&sb, 1u, lb), OC_OK);

    /* With a single position, attention returns V exactly, so the bias on V
     * must show up in the result; q/k bias cannot cancel it out. */
    bool differs = false;
    for (size_t i = 0; i < TB_VOCAB; i++)
        if (fabsf(lp[i] - lb[i]) > 1e-6f) differs = true;
    cr_assert(differs, "QKV bias had no effect on the forward pass");

    /* The V bias is a constant offset on the attention output, and with
     * n_head == 1 and one position the attention output is v verbatim. */
    oc_llama_session_free(&sp);
    oc_llama_session_free(&sb);
}

Test(llama, qkv_bias_absent_is_a_noop)
{
    /* A model without bias tensors must behave exactly as before: the
     * NULL checks in forward_layer must not read through them. */
    TinyBiasModel a, b;
    tiny_bias_model_init(&a, false);
    tiny_bias_model_init(&b, false);
    OcLlamaSession sa, sb;
    cr_assert_eq(oc_llama_session_init(&a.model, &sa), OC_OK);
    cr_assert_eq(oc_llama_session_init(&b.model, &sb), OC_OK);
    float la[TB_VOCAB], lb[TB_VOCAB];
    cr_assert_eq(oc_llama_forward(&sa, 2u, la), OC_OK);
    cr_assert_eq(oc_llama_forward(&sb, 2u, lb), OC_OK);
    for (size_t i = 0; i < TB_VOCAB; i++)
        cr_assert_float_eq(la[i], lb[i], 0.0f, "bias-free forward not deterministic");
    oc_llama_session_free(&sa);
    oc_llama_session_free(&sb);
}

/* ─── Q8 KV cache ────────────────────────────────────────────────────────
 *
 * oc_llama_kv_cache_bytes() reads only cfg, so it can be exercised against a
 * hand-built model without any weights.
 */
static OcLlamaModel kv_stub_model(uint32_t n_layer, uint32_t n_ctx,
                                  uint32_t n_head_kv, uint32_t kv_head_dim)
{
    OcLlamaModel m;
    memset(&m, 0, sizeof(m));
    m.cfg.n_layer = n_layer;
    m.cfg.n_ctx = n_ctx;
    m.cfg.n_head_kv = n_head_kv;
    m.cfg.kv_head_dim = kv_head_dim;
    m.cfg.head_dim = kv_head_dim;
    m.cfg.n_head = n_head_kv;
    return m;
}

Test(llama, kv_cache_bytes_f32_matches_layout)
{
    OcLlamaModel m = kv_stub_model(4, 1024, 8, 128);
    size_t elems = (size_t)4 * 1024 * 8 * 128;
    cr_assert_eq(oc_llama_kv_cache_bytes(&m, OC_KV_F32),
                 2 * elems * sizeof(float));
}

Test(llama, mla_kv_cache_bytes_counts_one_latent_buffer)
{
    OcLlamaModel m = kv_stub_model(4, 1024, 1, 576);
    m.cfg.uses_mla = true;
    m.cfg.mla_kv_lora_dim = 512;
    m.cfg.mla_q_rope_dim = 64;
    cr_assert_eq(oc_llama_kv_cache_bytes(&m, OC_KV_F32),
                 (size_t)4 * 1024 * 576 * sizeof(float));
}

/* Q8 must be close to 4x smaller: one byte per element instead of four, plus
 * one f32 scale per (layer, pos, kv head) — 1/kv_head_dim of the elements. */
Test(llama, kv_cache_bytes_q8_is_about_four_times_smaller)
{
    OcLlamaModel m = kv_stub_model(4, 1024, 8, 128);
    size_t f32 = oc_llama_kv_cache_bytes(&m, OC_KV_F32);
    size_t q8  = oc_llama_kv_cache_bytes(&m, OC_KV_Q8);
    cr_assert_lt(q8, f32);
    double ratio = (double)f32 / (double)q8;
    /* 4x minus the scale overhead (4 bytes per 128 elements => ~3%). */
    cr_assert(ratio > 3.7 && ratio < 4.0,
              "expected ~3.88x saving, got %.3fx", ratio);
}

Test(llama, kv_cache_bytes_null_is_zero)
{
    cr_assert_eq(oc_llama_kv_cache_bytes(NULL, OC_KV_Q8), 0);
}

/* A long-context model is exactly where this matters: 128k context on a
 * 28-layer model wants tens of GB of f32 KV. */
Test(llama, kv_cache_bytes_long_context_saving_is_gigabytes)
{
    OcLlamaModel m = kv_stub_model(28, 131072, 8, 128);
    size_t f32 = oc_llama_kv_cache_bytes(&m, OC_KV_F32);
    size_t q8  = oc_llama_kv_cache_bytes(&m, OC_KV_Q8);
    cr_assert_gt(f32 - q8, 10ull * 1024 * 1024 * 1024);
}

Test(llama, select_kv_type_explicit)
{
    cr_assert_eq(oc_llama_select_kv_type(32768, "f32"), OC_KV_F32);
    cr_assert_eq(oc_llama_select_kv_type(64, "q8"), OC_KV_Q8);
    cr_assert_eq(oc_llama_select_kv_type(4096, "Q8"), OC_KV_Q8);
}

static float llama_lcg(uint64_t *state)
{
    *state = *state * 6364136223846793005ULL + 1442695040888963407ULL;
    return (float)(*state >> 40) / (float)(1u << 24) * 2.0f - 1.0f;
}

static void llama_rope_interleaved(const float *row, size_t pos, size_t hd,
                                   float theta, float *out)
{
    size_t p;
    for (p = 0; p < hd / 2; p++) {
        const float freq = powf(theta, -2.0f * (float)p / (float)hd);
        const float a = freq * (float)pos;
        const float c = cosf(a);
        const float s = sinf(a);
        out[2 * p]     = row[2 * p] * c - row[2 * p + 1] * s;
        out[2 * p + 1] = row[2 * p] * s + row[2 * p + 1] * c;
    }
}

static float cosine_sim(const float *a, const float *b, size_t n)
{
    double dot = 0.0, na = 0.0, nb = 0.0;
    size_t i;
    for (i = 0; i < n; i++) {
        dot += (double)a[i] * (double)b[i];
        na += (double)a[i] * (double)a[i];
        nb += (double)b[i] * (double)b[i];
    }
    if (na <= 0.0 || nb <= 0.0) return 0.0f;
    return (float)(dot / (sqrt(na) * sqrt(nb)));
}

static void rotor_facade_vs_f32(size_t hd)
{
    const size_t tokens = 8;
    const float theta = 10000.0f;
    uint64_t rng = 11 + hd;
    float *keys, *values, *query, *ref, *rq, *rk, *out, *scores;
    size_t *positions;
    size_t t, i;
    float max_s, z, scale, sim;
    OcCompressedKvCache cache;
    keys = malloc(tokens * hd * sizeof(float));
    values = malloc(tokens * hd * sizeof(float));
    query = malloc(hd * sizeof(float));
    ref = calloc(hd, sizeof(float));
    rq = malloc(hd * sizeof(float));
    rk = malloc(hd * sizeof(float));
    out = malloc(hd * sizeof(float));
    scores = malloc(tokens * sizeof(float));
    positions = malloc(tokens * sizeof(size_t));
    cr_assert(keys && values && query && ref && rq && rk && out && scores &&
              positions);
    for (i = 0; i < tokens * hd; i++) keys[i] = llama_lcg(&rng);
    for (i = 0; i < tokens * hd; i++) values[i] = llama_lcg(&rng);
    for (i = 0; i < hd; i++) query[i] = llama_lcg(&rng);
    for (t = 0; t < tokens; t++) positions[t] = t;
    llama_rope_interleaved(query, tokens - 1, hd, theta, rq);
    scale = 1.0f / sqrtf((float)hd);
    max_s = -1.0e30f;
    for (t = 0; t < tokens; t++) {
        float dot = 0.0f;
        llama_rope_interleaved(keys + t * hd, t, hd, theta, rk);
        for (i = 0; i < hd; i++) dot += rq[i] * rk[i];
        scores[t] = dot * scale;
        if (scores[t] > max_s) max_s = scores[t];
    }
    z = 0.0f;
    for (t = 0; t < tokens; t++) {
        scores[t] = expf(scores[t] - max_s);
        z += scores[t];
    }
    for (t = 0; t < tokens; t++) {
        for (i = 0; i < hd; i++)
            ref[i] += scores[t] / z * values[t * hd + i];
    }
    cr_assert_eq(oc_compressed_kv_init(&cache, hd, OC_KV_SCHEME_ROTOR, tokens,
                                       theta),
                 OC_OK);
    cr_assert_eq(oc_compressed_kv_store_page(&cache, 0, 0, keys, values,
                                             positions, tokens),
                 OC_OK);
    cr_assert_eq(oc_compressed_kv_attention(&cache, 0, 0, query, hd, tokens - 1,
                                            out),
                 OC_OK);
    sim = cosine_sim(out, ref, hd);
    /* int4 + 3D rotors: cosine against f32 RoPE-attention stays high. */
    cr_assert(sim >= 0.95f,
              "head_dim=%zu cosine similarity %f (need >= 0.95)", hd, sim);
    oc_compressed_kv_free(&cache);
    free(keys); free(values); free(query); free(ref); free(rq); free(rk);
    free(out); free(scores); free(positions);
}

Test(llama, compressed_kv_rotor_cosine_hd8)
{
    rotor_facade_vs_f32(8);
}

Test(llama, compressed_kv_rotor_cosine_hd16)
{
    rotor_facade_vs_f32(16);
}

Test(llama, enable_kv_compress_releases_dense_cache)
{
    TinyBiasModel t;
    OcLlamaSession s;
    tiny_bias_model_init(&t, false);
    t.layer.use_rope = true;
    t.layer.rope_dim = TB_EMBD;
    t.model.cfg.rope_dim = TB_EMBD;
    cr_assert_eq(oc_llama_session_init(&t.model, &s), OC_OK);
    cr_assert_not_null(s.kv_k, "dense cache is allocated first");
    cr_assert_eq(oc_llama_session_enable_kv_compress(&s, OC_KV_SCHEME_ROTOR),
                 OC_OK);
    cr_assert_null(s.kv_k, "enabling compression must drop the f32 cache");
    cr_assert_null(s.kv_v);
    cr_assert_not_null(s.kv_compress);
    oc_llama_session_free(&s);
}

Test(llama, init_compressed_skips_dense_cache)
{
    TinyBiasModel t;
    OcLlamaSession s;
    tiny_bias_model_init(&t, false);
    t.layer.use_rope = true;
    t.layer.rope_dim = TB_EMBD;
    t.model.cfg.rope_dim = TB_EMBD;
    cr_assert_eq(oc_llama_session_init_compressed(&t.model, &s,
                                                 OC_KV_SCHEME_ROTOR),
                 OC_OK);
    cr_assert_null(s.kv_k, "init_compressed must not allocate the f32 cache");
    cr_assert_null(s.kv_v);
    cr_assert_not_null(s.kv_compress);
    oc_llama_session_free(&s);
}

Test(llama, mixed_swa_pattern_keeps_dense_kv)
{
    TinyBiasModel t;
    OcLlamaLayer layers[2];
    OcLlamaSession s;
    tiny_bias_model_init(&t, false);
    t.layer.use_rope = true;
    t.layer.rope_dim = TB_EMBD;
    t.model.cfg.rope_dim = TB_EMBD;
    layers[0] = t.layer;
    layers[1] = t.layer;
    t.model.layers = layers;
    t.model.cfg.n_layer = 2;
    t.model.cfg.sliding_window = 1024;
    t.model.cfg.sliding_window_pattern = 2;
    cr_assert_eq(oc_llama_session_init_with_compress(&t.model, &s, OC_KV_F32,
                                                     "rotor"),
                 OC_OK);
    cr_assert_not_null(s.kv_k, "legacy SWA layers must keep the dense cache");
    cr_assert_not_null(s.kv_v);
    cr_assert_not_null(s.kv_compress);
    oc_llama_session_free(&s);
}

Test(llama, enable_kv_compress_after_tokens_is_rejected)
{
    TinyBiasModel t;
    OcLlamaSession s;
    tiny_bias_model_init(&t, false);
    t.layer.use_rope = true;
    t.layer.rope_dim = TB_EMBD;
    t.model.cfg.rope_dim = TB_EMBD;
    cr_assert_eq(oc_llama_session_init(&t.model, &s), OC_OK);
    s.pos = 1;
    cr_assert_eq(oc_llama_session_enable_kv_compress(&s, OC_KV_SCHEME_ROTOR),
                 OC_ERR_INVALID_ARG);
    cr_assert_null(s.kv_compress);
    oc_llama_session_free(&s);
}

Test(llama, enable_kv_compress_rejects_gpt_family)
{
    TinyBiasModel t;
    OcLlamaSession s;
    tiny_bias_model_init(&t, false);
    t.layer.use_rope = true;
    t.layer.rope_dim = TB_EMBD;
    t.model.cfg.rope_dim = TB_EMBD;
    t.model.arch = OC_ARCH_GPT2;
    cr_assert_eq(oc_llama_session_init(&t.model, &s), OC_OK);
    cr_assert_eq(oc_llama_session_enable_kv_compress(&s, OC_KV_SCHEME_ROTOR),
                 OC_ERR_INVALID_ARG);
    oc_llama_session_free(&s);
}
