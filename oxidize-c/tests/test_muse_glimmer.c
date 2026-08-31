/* test_muse_glimmer.c — Muse Glimmer config parse + forward-path wiring. */
#include <criterion/criterion.h>

#include "oxidize/gguf_writer.h"
#include "oxidize/llama.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Geometry of the miniature model written by build_muse_gguf(). Mirrors the
 * real model's shape relationships: GQA 4:1, head_dim not n_embd/n_head. */
#define MG_LAYERS   8u    /* two full [L,L,L,G] periods                  */
#define MG_EMBD     64u
#define MG_HEADS    4u
#define MG_KV_HEADS 1u
#define MG_HEAD_DIM 16u
#define MG_FF       32u
#define MG_VOCAB    19u
#define MG_WINDOW   12u
#define MG_PERIOD   4u
#define MG_SOFTCAP  20.0f
#define MG_LSCALE   0.1961161345243454f

static void add_f32_tensor(OcGgufWriter *w, const char *name,
                           uint64_t d0, uint64_t d1)
{
    uint64_t dims[2] = { d0, d1 };
    uint32_t n_dims = (d1 > 0) ? 2u : 1u;
    size_t n = (size_t)d0 * (size_t)(d1 > 0 ? d1 : 1);
    float *buf = calloc(n, sizeof(float));
    cr_assert_not_null(buf, "calloc tensor %s", name);
    /* Non-zero, deterministic and distinct per tensor so a mis-bound view
     * shows up as a value mismatch rather than as silent zeros. */
    for (size_t i = 0; i < n; i++) buf[i] = (float)((i % 7) + 1) * 0.25f;
    OcError e = oc_gguf_writer_add_tensor(w, name, n_dims, dims, 0 /* F32 */,
                                          buf, (uint64_t)(n * sizeof(float)));
    cr_assert_eq(e, OC_OK, "add_tensor(%s) failed: %d", name, (int)e);
    free(buf);
}

static void build_muse_gguf(const char *path)
{
    OcGgufWriter w;
    cr_assert_eq(oc_gguf_writer_init(path, "muse-glimmer", &w), OC_OK,
                 "writer init failed");

    oc_gguf_writer_add_uint32(&w, "muse-glimmer.block_count", MG_LAYERS);
    oc_gguf_writer_add_uint32(&w, "muse-glimmer.embedding_length", MG_EMBD);
    oc_gguf_writer_add_uint32(&w, "muse-glimmer.feed_forward_length", MG_FF);
    oc_gguf_writer_add_uint32(&w, "muse-glimmer.attention.head_count",
                              MG_HEADS);
    oc_gguf_writer_add_uint32(&w, "muse-glimmer.attention.head_count_kv",
                              MG_KV_HEADS);
    oc_gguf_writer_add_uint32(&w, "muse-glimmer.attention.key_length",
                              MG_HEAD_DIM);
    oc_gguf_writer_add_uint32(&w, "muse-glimmer.attention.value_length",
                              MG_HEAD_DIM);
    oc_gguf_writer_add_float32(
        &w, "muse-glimmer.attention.layer_norm_rms_epsilon", 1e-5f);
    oc_gguf_writer_add_uint32(&w, "muse-glimmer.attention.sliding_window",
                              MG_WINDOW);
    oc_gguf_writer_add_uint32(
        &w, "muse-glimmer.attention.sliding_window_pattern", MG_PERIOD);
    oc_gguf_writer_add_float32(&w, "muse-glimmer.rope.freq_base", 500000.0f);
    oc_gguf_writer_add_float32(&w, "muse-glimmer.final_logit_softcapping",
                               MG_SOFTCAP);
    oc_gguf_writer_add_float32(&w, "muse-glimmer.logit_scale", MG_LSCALE);
    oc_gguf_writer_add_uint32(&w, "muse-glimmer.context_length", 128);

    add_f32_tensor(&w, "token_embd.weight", MG_EMBD, MG_VOCAB);
    add_f32_tensor(&w, "output.weight", MG_EMBD, MG_VOCAB);
    add_f32_tensor(&w, "output_norm.weight", MG_EMBD, 0);

    const uint64_t q_rows = (uint64_t)MG_HEADS * MG_HEAD_DIM;
    const uint64_t kv_rows = (uint64_t)MG_KV_HEADS * MG_HEAD_DIM;
    for (unsigned l = 0; l < MG_LAYERS; l++) {
        char nm[80];
        snprintf(nm, sizeof nm, "blk.%u.attn_norm.weight", l);
        add_f32_tensor(&w, nm, MG_EMBD, 0);
        snprintf(nm, sizeof nm, "blk.%u.post_attention_norm.weight", l);
        add_f32_tensor(&w, nm, MG_EMBD, 0);
        snprintf(nm, sizeof nm, "blk.%u.ffn_norm.weight", l);
        add_f32_tensor(&w, nm, MG_EMBD, 0);
        snprintf(nm, sizeof nm, "blk.%u.post_ffw_norm.weight", l);
        add_f32_tensor(&w, nm, MG_EMBD, 0);
        snprintf(nm, sizeof nm, "blk.%u.attn_q.weight", l);
        add_f32_tensor(&w, nm, MG_EMBD, q_rows);
        snprintf(nm, sizeof nm, "blk.%u.attn_k.weight", l);
        add_f32_tensor(&w, nm, MG_EMBD, kv_rows);
        snprintf(nm, sizeof nm, "blk.%u.attn_v.weight", l);
        add_f32_tensor(&w, nm, MG_EMBD, kv_rows);
        /* The attention output gate: n_embd → n_head*head_dim. */
        snprintf(nm, sizeof nm, "blk.%u.attn_gate.weight", l);
        add_f32_tensor(&w, nm, MG_EMBD, q_rows);
        snprintf(nm, sizeof nm, "blk.%u.attn_output.weight", l);
        add_f32_tensor(&w, nm, q_rows, MG_EMBD);
        snprintf(nm, sizeof nm, "blk.%u.attn_q_norm.weight", l);
        add_f32_tensor(&w, nm, MG_HEAD_DIM, 0);
        snprintf(nm, sizeof nm, "blk.%u.attn_k_norm.weight", l);
        add_f32_tensor(&w, nm, MG_HEAD_DIM, 0);
        snprintf(nm, sizeof nm, "blk.%u.ffn_gate.weight", l);
        add_f32_tensor(&w, nm, MG_EMBD, MG_FF);
        snprintf(nm, sizeof nm, "blk.%u.ffn_up.weight", l);
        add_f32_tensor(&w, nm, MG_EMBD, MG_FF);
        snprintf(nm, sizeof nm, "blk.%u.ffn_down.weight", l);
        add_f32_tensor(&w, nm, MG_FF, MG_EMBD);
    }

    cr_assert_eq(oc_gguf_writer_finalize(&w), OC_OK, "finalize failed");
    oc_gguf_writer_free(&w);
}

/* Criterion runs each Test in its own process, in parallel, so each test
 * needs its own fixture path — a shared one lets one test unlink a file
 * another is still mmap'ing. */
#define FIXTURE(name) "/tmp/oxidize-c-muse-" name ".gguf"

Test(muse_glimmer, config_flags_and_geometry)
{
    const char *p = FIXTURE("cfg");
    build_muse_gguf(p);

    OcLlamaModel m;
    memset(&m, 0, sizeof m);
    cr_assert_eq(oc_llama_load(p, &m), OC_OK, "oc_llama_load failed");

    cr_assert_eq(m.arch, OC_ARCH_MUSE_GLIMMER, "arch");
    cr_assert_eq(m.cfg.n_layer, MG_LAYERS, "n_layer");
    cr_assert_eq(m.cfg.n_embd, MG_EMBD, "n_embd");
    /* head_dim comes from attention.key_length, NOT n_embd / n_head (which
     * would be 16 here by coincidence of the fixture — assert the source by
     * checking the GQA counts too). */
    cr_assert_eq(m.cfg.head_dim, MG_HEAD_DIM, "head_dim");
    cr_assert_eq(m.cfg.n_head_kv, MG_KV_HEADS, "n_head_kv");

    cr_assert(m.cfg.embd_rms_norm, "embedding must be RMS-normalized");
    cr_assert(m.cfg.attn_out_gate, "attention output gate must be enabled");
    cr_assert(m.cfg.rope_swa_only, "RoPE must be restricted to SWA layers");
    cr_assert_float_eq(m.cfg.post_norm_eps, 1e-8f, 1e-12f,
                       "sandwich norms use their own epsilon");
    cr_assert_float_eq(m.cfg.logit_softcap, MG_SOFTCAP, 1e-6f, "softcap");
    cr_assert_float_eq(m.cfg.logit_scale, MG_LSCALE, 1e-9f, "logit_scale");
    cr_assert_eq(m.cfg.sliding_window, MG_WINDOW, "window");
    /* No MoE, no MLA, no Gemma 4 dual geometry. */
    cr_assert_eq(m.cfg.num_experts, 0u, "dense FFN");
    cr_assert(!m.cfg.uses_mla, "not MLA");
    cr_assert(!m.cfg.uses_gemma4, "not Gemma 4");
    cr_assert(!m.cfg.uses_geglu, "SwiGLU, not GeGLU");

    oc_llama_free(&m);
    remove(p);
}

Test(muse_glimmer, three_local_layers_then_one_global)
{
    const char *p = FIXTURE("swa");
    build_muse_gguf(p);

    OcLlamaModel m;
    memset(&m, 0, sizeof m);
    cr_assert_eq(oc_llama_load(p, &m), OC_OK, "load");

    cr_assert_not_null(m.cfg.layer_is_swa, "per-layer pattern must be built");
    for (uint32_t l = 0; l < MG_LAYERS; l++) {
        /* llama.cpp set_swa_pattern: sliding iff l % period < period - 1. */
        const bool want_sliding = (l % MG_PERIOD) < (MG_PERIOD - 1u);
        cr_assert_eq(m.cfg.layer_is_swa[l] != 0, want_sliding,
            "layer %u should be %s", l, want_sliding ? "sliding" : "global");
        cr_assert_eq(m.layers[l].sliding_window,
                     want_sliding ? MG_WINDOW : 0u,
            "layer %u window", l);
        /* The RoPE/NoPE split follows the same pattern — global layers carry
         * no positional information at all. */
        cr_assert_eq(m.layers[l].use_rope, want_sliding,
            "layer %u should %s RoPE", l, want_sliding ? "apply" : "skip");
    }

    oc_llama_free(&m);
    remove(p);
}

Test(muse_glimmer, per_layer_tensors_bind)
{
    const char *p = FIXTURE("tensors");
    build_muse_gguf(p);

    OcLlamaModel m;
    memset(&m, 0, sizeof m);
    cr_assert_eq(oc_llama_load(p, &m), OC_OK, "load");

    for (uint32_t l = 0; l < MG_LAYERS; l++) {
        const OcLlamaLayer *L = &m.layers[l];
        cr_assert_not_null(L->attn_norm, "layer %u attn_norm", l);
        cr_assert_not_null(L->ffn_norm, "layer %u ffn_norm", l);
        cr_assert_not_null(L->post_attention_norm,
                           "layer %u post_attention_norm", l);
        cr_assert_not_null(L->post_ffw_norm, "layer %u post_ffw_norm", l);
        cr_assert_not_null(L->attn_q_norm, "layer %u attn_q_norm", l);
        cr_assert_not_null(L->attn_k_norm, "layer %u attn_k_norm", l);
        cr_assert_not_null(L->attn_q.data, "layer %u attn_q", l);
        cr_assert_not_null(L->attn_output.data, "layer %u attn_output", l);
        cr_assert_not_null(L->ffn_down.data, "layer %u ffn_down", l);
        /* The gate is what makes this arch different from a Gemma-shaped
         * dense model; an unbound view would silently skip the gating. */
        cr_assert_not_null(L->attn_gate.data, "layer %u attn_gate", l);
        cr_assert_eq(L->attn_gate.rows, (size_t)MG_HEADS * MG_HEAD_DIM,
            "layer %u attn_gate rows", l);
        cr_assert_eq(L->attn_gate.cols, (size_t)MG_EMBD,
            "layer %u attn_gate cols", l);
    }

    oc_llama_free(&m);
    remove(p);
}

Test(muse_glimmer, forward_produces_softcapped_logits)
{
    const char *p = FIXTURE("fwd");
    build_muse_gguf(p);

    OcLlamaModel m;
    memset(&m, 0, sizeof m);
    cr_assert_eq(oc_llama_load(p, &m), OC_OK, "load");

    OcLlamaSession s;
    memset(&s, 0, sizeof s);
    cr_assert_eq(oc_llama_session_init(&m, &s), OC_OK, "session init");

    float *logits = calloc(m.cfg.vocab_size, sizeof(float));
    cr_assert_not_null(logits, "calloc logits");

    /* Run past the first global layer's window so both attention kinds are
     * exercised, and past MG_WINDOW so the sliding layers actually drop
     * tokens off the front. */
    for (uint32_t i = 0; i < MG_WINDOW + 3u; i++) {
        cr_assert_eq(oc_llama_forward(&s, i % MG_VOCAB, logits), OC_OK,
                     "forward step %u", i);
    }

    for (uint32_t v = 0; v < m.cfg.vocab_size; v++) {
        cr_assert(isfinite(logits[v]), "logit %u is not finite", v);
        cr_assert(fabsf(logits[v]) < MG_SOFTCAP,
            "logit %u = %f must be softcapped inside (-%f, %f)",
            v, (double)logits[v], (double)MG_SOFTCAP, (double)MG_SOFTCAP);
    }

    free(logits);
    oc_llama_session_free(&s);
    oc_llama_free(&m);
    remove(p);
}

/* The batched prefill path must agree with the per-token path. */
Test(muse_glimmer, prefill_matches_per_token_forward)
{
    const char *p = FIXTURE("prefill");
    build_muse_gguf(p);

    OcLlamaModel m;
    memset(&m, 0, sizeof m);
    cr_assert_eq(oc_llama_load(p, &m), OC_OK, "load");

    enum { N_TOK = 17 };
    uint32_t tokens[N_TOK];
    for (int i = 0; i < N_TOK; i++) tokens[i] = (uint32_t)(i * 3 % MG_VOCAB);

    float *a = calloc(m.cfg.vocab_size, sizeof(float));
    float *b = calloc(m.cfg.vocab_size, sizeof(float));
    cr_assert_not_null(a, "calloc a");
    cr_assert_not_null(b, "calloc b");

    OcLlamaSession s1;
    memset(&s1, 0, sizeof s1);
    cr_assert_eq(oc_llama_session_init(&m, &s1), OC_OK, "session 1");
    for (int i = 0; i < N_TOK; i++) {
        cr_assert_eq(oc_llama_forward(&s1, tokens[i],
                                      (i == N_TOK - 1) ? a : NULL),
                     OC_OK, "forward %d", i);
    }
    oc_llama_session_free(&s1);

    OcLlamaSession s2;
    memset(&s2, 0, sizeof s2);
    cr_assert_eq(oc_llama_session_init(&m, &s2), OC_OK, "session 2");
    cr_assert_eq(oc_llama_prefill(&s2, tokens, N_TOK, 8, b), OC_OK, "prefill");
    oc_llama_session_free(&s2);

    for (uint32_t v = 0; v < m.cfg.vocab_size; v++) {
        cr_assert_float_eq(b[v], a[v], 1e-3f,
            "prefill logit %u = %f, per-token = %f", v,
            (double)b[v], (double)a[v]);
    }

    free(a);
    free(b);
    oc_llama_free(&m);
    remove(p);
}
