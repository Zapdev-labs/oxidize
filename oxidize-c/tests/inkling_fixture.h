/* inkling_fixture.h — a tiny synthetic Inkling-2 checkpoint for tests.
 *
 * Writes config.json, tokenizer.json, model-00001-of-00001.safetensors and
 * mtp.safetensors with deterministic pseudo-random weights into a temp
 * directory: 6 layers, a 2-layer loop core run 3 times (8 visits), 64-wide
 * hidden state, 4 experts (top 2) with NVFP4 experts on most layers and
 * MXFP8 experts on the last one, a 4-token sliding window so the SWA ring
 * wraps, and a 2-layer DSpark drafter. */
#ifndef OC_TEST_INKLING_FIXTURE_H
#define OC_TEST_INKLING_FIXTURE_H

#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif
#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define FX_E      64
#define FX_L      6
#define FX_H      2
#define FX_KVH    1
#define FX_HD     64
#define FX_NREL   16
#define FX_EXT    8
#define FX_SWA    4
#define FX_NE     4
#define FX_K      2
#define FX_FF     64
#define FX_V      250
#define FX_VPAD   256
#define FX_RANK   8
#define FX_D_L    2
#define FX_D_FF   128
#define FX_MR     256

typedef struct {
    char   name[96];
    char   dtype[16];
    size_t shape[4];
    int    nd;
    uint8_t *data;
    size_t bytes;
} FxT;

typedef struct {
    FxT   *t;
    size_t n, cap;
} FxSt;

static uint32_t fx_seed = 12345u;
static float fx_rand(void)
{
    fx_seed = fx_seed * 1664525u + 1013904223u;
    return (float)((int32_t)(fx_seed >> 8) - (1 << 23)) / (float)(1 << 23);
}

static uint16_t fx_bf16(float f)
{
    uint32_t u;
    memcpy(&u, &f, 4);
    return (uint16_t)((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}

static FxT *fx_add(FxSt *st, const char *name, const char *dtype, int nd, size_t d0, size_t d1, size_t d2)
{
    if (st->n == st->cap) {
        st->cap = st->cap ? st->cap * 2 : 128;
        st->t = realloc(st->t, st->cap * sizeof *st->t);
    }
    FxT *t = &st->t[st->n++];
    memset(t, 0, sizeof *t);
    snprintf(t->name, sizeof t->name, "%s", name);
    snprintf(t->dtype, sizeof t->dtype, "%s", dtype);
    t->nd = nd;
    t->shape[0] = d0; t->shape[1] = d1; t->shape[2] = d2;
    size_t n = d0 * (nd > 1 ? d1 : 1) * (nd > 2 ? d2 : 1);
    size_t es = strcmp(dtype, "F32") == 0 ? 4 : strcmp(dtype, "BF16") == 0 ? 2 : 1;
    t->bytes = n * es;
    t->data = calloc(t->bytes ? t->bytes : 1, 1);
    return t;
}

/* BF16 (or F32) tensor filled with scale * rand (+ offset). */
static void fx_dense(FxSt *st, const char *name, const char *dtype, int nd, size_t d0, size_t d1, size_t d2,
                     float scale, float offset)
{
    FxT *t = fx_add(st, name, dtype, nd, d0, d1, d2);
    size_t n = d0 * (nd > 1 ? d1 : 1) * (nd > 2 ? d2 : 1);
    for (size_t i = 0; i < n; i++) {
        float v = offset + scale * fx_rand();
        if (strcmp(dtype, "F32") == 0) memcpy(t->data + 4 * i, &v, 4);
        else { uint16_t h = fx_bf16(v); memcpy(t->data + 2 * i, &h, 2); }
    }
}

static void fx_cat(char *h, size_t cap, size_t *len, const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    const int n = vsnprintf(h + *len, cap - *len, fmt, ap);
    va_end(ap);
    if (n < 0 || (size_t)n + 8 >= cap - *len) abort();
    *len += (size_t)n;
}

static void fx_write_st(const char *path, const FxSt *st)
{
    size_t cap = 1 << 20, len = 0;
    char *h = malloc(cap);
    fx_cat(h, cap, &len, "{");
    size_t off = 0;
    for (size_t i = 0; i < st->n; i++) {
        const FxT *t = &st->t[i];
        fx_cat(h, cap, &len, "%s\"%s\":{\"dtype\":\"%s\",\"shape\":[", i ? "," : "", t->name, t->dtype);
        for (int k = 0; k < t->nd; k++) fx_cat(h, cap, &len, "%s%zu", k ? "," : "", t->shape[k]);
        fx_cat(h, cap, &len, "],\"data_offsets\":[%zu,%zu]}", off, off + t->bytes);
        off += t->bytes;
    }
    fx_cat(h, cap, &len, "}");
    while (len % 8) h[len++] = ' ';
    FILE *f = fopen(path, "wb");
    uint64_t hl = len;
    fwrite(&hl, 8, 1, f);
    fwrite(h, 1, len, f);
    for (size_t i = 0; i < st->n; i++) fwrite(st->t[i].data, 1, st->t[i].bytes, f);
    fclose(f);
    free(h);
}

static void fx_free(FxSt *st)
{
    for (size_t i = 0; i < st->n; i++) free(st->t[i].data);
    free(st->t);
    memset(st, 0, sizeof *st);
}

static bool fx_is_swa(int i) { return i % 2 == 0; }

/* NVFP4 experts: packed nibbles [NE, rows, cols/2], E4M3 scales [NE, rows, cols/16]. */
static void fx_experts_nvfp4(FxSt *st, const char *base, size_t rows, size_t cols)
{
    char n[160];
    snprintf(n, sizeof n, "%s", base);
    FxT *w = fx_add(st, n, "U8", 3, FX_NE, rows, cols / 2);
    for (size_t i = 0; i < w->bytes; i++) w->data[i] = (uint8_t)(fx_rand() * 127.0f + 128.0f);
    snprintf(n, sizeof n, "%s_scale", base);
    FxT *s = fx_add(st, n, "F8_E4M3", 3, FX_NE, rows, cols / 16);
    for (size_t i = 0; i < s->bytes; i++) s->data[i] = (uint8_t)(0x30 + (int)(fx_rand() * 6.0f));
    snprintf(n, sizeof n, "%s_amax", base);
    FxT *a = fx_add(st, n, "F32", 1, FX_NE, 0, 0);
    for (size_t e = 0; e < FX_NE; e++) { float v = 0.5f + 0.1f * (float)e; memcpy(a->data + 4 * e, &v, 4); }
    snprintf(n, sizeof n, "%s_outer_scale_denominator", base);
    FxT *d = fx_add(st, n, "F32", 1, 1, 0, 0);
    float dv = 6.0f;
    memcpy(d->data, &dv, 4);
}

/* MXFP8 experts: E4M3 weights [NE, rows, cols], UE8M0 scales [NE, rows, cols/32]. */
static void fx_experts_mxfp8(FxSt *st, const char *base, size_t rows, size_t cols)
{
    char n[160];
    snprintf(n, sizeof n, "%s", base);
    FxT *w = fx_add(st, n, "F8_E4M3", 3, FX_NE, rows, cols);
    for (size_t i = 0; i < w->bytes; i++) {
        uint8_t b = (uint8_t)(fx_rand() * 127.0f + 128.0f);
        if ((b & 0x7F) == 0x7F) b ^= 1;
        w->data[i] = b;
    }
    snprintf(n, sizeof n, "%s_scale", base);
    FxT *s = fx_add(st, n, "U8", 3, FX_NE, rows, cols / 32);
    for (size_t i = 0; i < s->bytes; i++) s->data[i] = (uint8_t)(120 + (int)(fx_rand() * 2.0f));
}

/* Build the checkpoint in a fresh temp dir; returns its path (static buffer). */
static const char *fx_build(void)
{
    static char dir[256];
    snprintf(dir, sizeof dir, "/tmp/oc_inkling_fx_XXXXXX");
    if (!mkdtemp(dir)) return NULL;
    char p[512];
    fx_seed = 12345u;

    /* config.json */
    snprintf(p, sizeof p, "%s/config.json", dir);
    FILE *f = fopen(p, "w");
    fprintf(f,
        "{\"model_type\":\"inkling2_loop\",\"eos_token_id\":249,\"text_config\":{"
        "\"hidden_size\":%d,\"num_hidden_layers\":%d,\"vocab_size\":%d,\"unpadded_vocab_size\":%d,"
        "\"num_attention_heads\":%d,\"num_key_value_heads\":%d,\"head_dim\":%d,\"d_rel\":%d,\"rel_extent\":%d,"
        "\"rms_norm_eps\":1e-06,\"qk_l2_clip_threshold\":22.6,\"rel_l2_clip_threshold\":8.0,"
        "\"log_scaling_n_floor\":4,\"log_scaling_alpha\":0.1,\"local_layer_ids\":[0,2,4],\"sconv_kernel_size\":4,"
        "\"final_logit_softcapping\":30.0,\"sliding_window_size\":%d,\"n_layers_per_attn_res_block\":2,"
        "\"n_routed_experts\":%d,\"num_experts_per_tok\":%d,\"intermediate_size\":%d,\"mlp_gate_softcap\":4.0,"
        "\"mlp_up_softcap\":25.0,\"route_scale\":2.0,"
        "\"loop_config\":{\"core_start\":2,\"core_end\":4,\"num_passes\":3,\"gate_rank\":%d,\"is_varloop\":false}},"
        "\"mtp_config\":{\"hidden_size\":%d,\"num_hidden_layers\":%d,\"num_attention_heads\":%d,"
        "\"num_key_value_heads\":%d,\"head_dim\":%d,\"intermediate_size\":%d,\"rms_norm_eps\":1e-06,"
        "\"rope_theta\":1000000.0,\"local_layer_ids\":[0],\"sliding_window_size\":%d,\"block_size\":4,"
        "\"markov_rank\":%d,\"markov_head_type\":\"rnn_hidden\",\"target_layer_ids\":[1,3,7,10]}}\n",
        FX_E, FX_L, FX_VPAD, FX_V, FX_H, FX_KVH, FX_HD, FX_NREL, FX_EXT, FX_SWA, FX_NE, FX_K, FX_FF, FX_RANK,
        FX_E, FX_D_L, FX_H, FX_KVH, FX_HD, FX_D_FF, FX_SWA, FX_MR);
    fclose(f);

    /* tokenizer.json: single-letter byte tokens a.. plus padding names */
    snprintf(p, sizeof p, "%s/tokenizer.json", dir);
    f = fopen(p, "w");
    fprintf(f, "{\"model\":{\"type\":\"BPE\",\"vocab\":{");
    for (int i = 0; i < FX_V - 1; i++) fprintf(f, "%s\"tok%d\":%d", i ? "," : "", i, i);
    fprintf(f, "},\"merges\":[]},\"added_tokens\":[{\"id\":%d,\"content\":\"<|end|>\"}]}\n", FX_V - 1);
    fclose(f);

    /* target */
    FxSt st = { 0 };
    char n[160];
    fx_dense(&st, "model.llm.embed.weight", "BF16", 2, FX_VPAD, FX_E, 0, 0.5f, 0.0f);
    fx_dense(&st, "model.llm.embed_norm.weight", "BF16", 1, FX_E, 0, 0, 0.1f, 1.0f);
    fx_dense(&st, "model.llm.unembed.weight", "BF16", 2, FX_VPAD, FX_E, 0, 0.2f, 0.0f);
    fx_dense(&st, "model.llm.norm.weight", "BF16", 1, FX_E, 0, 0, 0.1f, 1.0f);
    fx_dense(&st, "model.llm.attn_res.output_query", "F32", 1, FX_E, 0, 0, 0.3f, 0.0f);
    fx_dense(&st, "model.llm.middle_cycle_cross_pass_queries", "F32", 2, 4, FX_E, 0, 0.3f, 0.0f);
    fx_dense(&st, "model.llm.middle_cycle_cross_pass_kv_queries", "F32", 2, 4, FX_E, 0, 0.3f, 0.0f);
    for (int i = 0; i < FX_L; i++) {
#define NM(s) (snprintf(n, sizeof n, "model.llm.layers.%d." s, i), n)
        const size_t QD = FX_H * FX_HD, KV = FX_KVH * FX_HD;
        fx_dense(&st, NM("attn.wq_du.weight"), "BF16", 2, QD, FX_E, 0, 0.2f, 0.0f);
        fx_dense(&st, NM("attn.wk_dv.weight"), "BF16", 2, KV, FX_E, 0, 0.2f, 0.0f);
        fx_dense(&st, NM("attn.wv_dv.weight"), "BF16", 2, KV, FX_E, 0, 0.2f, 0.0f);
        fx_dense(&st, NM("attn.wo_ud.weight"), "BF16", 2, FX_E, QD, 0, 0.2f, 0.0f);
        fx_dense(&st, NM("attn.gate_attn.weight"), "BF16", 2, QD, FX_E, 0, 0.2f, 0.0f);
        fx_dense(&st, NM("attn.wr_du.weight"), "BF16", 2, FX_H * FX_NREL, FX_E, 0, 0.2f, 0.0f);
        fx_dense(&st, NM("attn.attn_sink_proj.weight"), "BF16", 2, FX_H, FX_E, 0, 0.2f, 0.0f);
        fx_dense(&st, NM("attn.q_l2_clip.scale"), "BF16", 2, FX_H, FX_HD, 0, 0.1f, 1.0f);
        fx_dense(&st, NM("attn.rel_l2_clip.scale"), "BF16", 2, FX_H, FX_NREL, 0, 0.1f, 1.0f);
        fx_dense(&st, NM("attn.rel_logits_proj.proj"), "BF16", 2, FX_NREL, FX_EXT, 0, 0.3f, 0.0f);
        fx_dense(&st, NM("attn_sconv.weight"), "BF16", 3, FX_E, 1, 4, 0.3f, 0.0f);
        fx_dense(&st, NM("mlp_sconv.weight"), "BF16", 3, FX_E, 1, 4, 0.3f, 0.0f);
        if (fx_is_swa(i)) {
            fx_dense(&st, NM("attn.k_sconv.weight"), "BF16", 3, KV, 1, 4, 0.3f, 0.0f);
            fx_dense(&st, NM("attn.v_sconv.weight"), "BF16", 3, KV, 1, 4, 0.3f, 0.0f);
            fx_dense(&st, NM("attn_norm.weight"), "BF16", 1, FX_E, 0, 0, 0.1f, 1.0f);
        } else {
            fx_dense(&st, NM("attn.k_sconv.static_w"), "BF16", 2, 4, KV, 0, 0.3f, 0.0f);
            fx_dense(&st, NM("attn.v_sconv.static_w"), "BF16", 2, 4, KV, 0, 0.3f, 0.0f);
            fx_dense(&st, NM("attn.k_sconv.weight_proj"), "BF16", 2, FX_KVH * 4, FX_E, 0, 0.2f, 0.0f);
            fx_dense(&st, NM("attn.v_sconv.weight_proj"), "BF16", 2, FX_KVH * 4, FX_E, 0, 0.2f, 0.0f);
            fx_dense(&st, NM("q_input_norm.weight"), "BF16", 1, FX_E, 0, 0, 0.1f, 1.0f);
            fx_dense(&st, NM("k_input_norm.weight"), "BF16", 1, FX_E, 0, 0, 0.1f, 1.0f);
            fx_dense(&st, NM("v_input_norm.weight"), "BF16", 1, FX_E, 0, 0, 0.1f, 1.0f);
        }
        fx_dense(&st, NM("mlp_norm.weight"), "BF16", 1, FX_E, 0, 0, 0.1f, 1.0f);
        fx_dense(&st, NM("mlp.gate.weight"), "BF16", 2, FX_NE, FX_E, 0, 0.5f, 0.0f);
        fx_dense(&st, NM("mlp.gate.bias"), "F32", 1, FX_NE, 0, 0, 0.1f, 0.0f);
        if (i == FX_L - 1) {
            fx_experts_mxfp8(&st, NM("mlp.experts.w13_weight"), 2 * FX_FF, FX_E);
            fx_experts_mxfp8(&st, NM("mlp.experts.w2_weight"), FX_E, FX_FF);
        } else {
            fx_experts_nvfp4(&st, NM("mlp.experts.w13_weight"), 2 * FX_FF, FX_E);
            fx_experts_nvfp4(&st, NM("mlp.experts.w2_weight"), FX_E, FX_FF);
        }
        if (i >= 2 && i < 4) {
            static const char *ctl[4] = { "attn", "k", "v", "mlp" };
            for (int c = 0; c < 4; c++) {
                snprintf(n, sizeof n, "model.llm.layers.%d.controllers.%s.down_weight", i, ctl[c]);
                fx_dense(&st, n, "BF16", 2, FX_RANK, 2 * FX_E, 0, 0.2f, 0.0f);
                snprintf(n, sizeof n, "model.llm.layers.%d.controllers.%s.gain_weight", i, ctl[c]);
                fx_dense(&st, n, "BF16", 2, FX_E, FX_RANK, 0, 0.2f, 0.0f);
                snprintf(n, sizeof n, "model.llm.layers.%d.controllers.%s.forget_weight", i, ctl[c]);
                fx_dense(&st, n, "BF16", 2, FX_E, FX_RANK, 0, 0.2f, 0.0f);
            }
        }
        snprintf(n, sizeof n, "model.llm.attn_res.mlp_queries.%d", i);
        fx_dense(&st, n, "F32", 1, FX_E, 0, 0, 0.3f, 0.0f);
        if (i > 0) {
            snprintf(n, sizeof n, "model.llm.attn_res.attn_queries.%d", i - 1);
            if (fx_is_swa(i)) fx_dense(&st, n, "F32", 1, FX_E, 0, 0, 0.3f, 0.0f);
            else fx_dense(&st, n, "F32", 2, 3, FX_E, 0, 0.3f, 0.0f);
        }
#undef NM
    }
    snprintf(p, sizeof p, "%s/model-00001-of-00001.safetensors", dir);
    fx_write_st(p, &st);
    fx_free(&st);

    /* drafter */
    for (int i = 0; i < FX_D_L; i++) {
#define NM(s) (snprintf(n, sizeof n, "layers.%d." s, i), n)
        const size_t QD = FX_H * FX_HD, KV = FX_KVH * FX_HD;
        fx_dense(&st, NM("input_layernorm.weight"), "BF16", 1, FX_E, 0, 0, 0.1f, 1.0f);
        fx_dense(&st, NM("post_attention_layernorm.weight"), "BF16", 1, FX_E, 0, 0, 0.1f, 1.0f);
        fx_dense(&st, NM("dflare_norm.weight"), "BF16", 1, FX_E, 0, 0, 0.1f, 1.0f);
        fx_dense(&st, NM("self_attn.q_proj.weight"), "BF16", 2, QD, FX_E, 0, 0.2f, 0.0f);
        fx_dense(&st, NM("self_attn.k_proj.weight"), "BF16", 2, KV, FX_E, 0, 0.2f, 0.0f);
        fx_dense(&st, NM("self_attn.v_proj.weight"), "BF16", 2, KV, FX_E, 0, 0.2f, 0.0f);
        fx_dense(&st, NM("self_attn.o_proj.weight"), "BF16", 2, FX_E, QD, 0, 0.2f, 0.0f);
        fx_dense(&st, NM("self_attn.k_proj_ctx.weight"), "BF16", 2, KV, FX_E, 0, 0.2f, 0.0f);
        fx_dense(&st, NM("self_attn.v_proj_ctx.weight"), "BF16", 2, KV, FX_E, 0, 0.2f, 0.0f);
        fx_dense(&st, NM("self_attn.q_norm.weight"), "BF16", 1, FX_HD, 0, 0, 0.1f, 1.0f);
        fx_dense(&st, NM("self_attn.k_norm.weight"), "BF16", 1, FX_HD, 0, 0, 0.1f, 1.0f);
        fx_dense(&st, NM("self_attn.k_norm_ctx.weight"), "BF16", 1, FX_HD, 0, 0, 0.1f, 1.0f);
        fx_dense(&st, NM("mlp.gate_proj.weight"), "BF16", 2, FX_D_FF, FX_E, 0, 0.2f, 0.0f);
        fx_dense(&st, NM("mlp.up_proj.weight"), "BF16", 2, FX_D_FF, FX_E, 0, 0.2f, 0.0f);
        fx_dense(&st, NM("mlp.down_proj.weight"), "BF16", 2, FX_E, FX_D_FF, 0, 0.2f, 0.0f);
#undef NM
    }
    fx_dense(&st, "fuse_weight", "BF16", 2, FX_D_L + 1, 4, 0, 0.5f, 0.0f);
    fx_dense(&st, "norm.weight", "BF16", 1, FX_E, 0, 0, 0.1f, 1.0f);
    fx_dense(&st, "mask_embed", "BF16", 1, FX_E, 0, 0, 0.5f, 0.0f);
    fx_dense(&st, "markov_head.dflare_norm.weight", "BF16", 1, FX_E, 0, 0, 0.1f, 1.0f);
    fx_dense(&st, "markov_head.hidden_out.weight", "BF16", 2, FX_E, FX_MR, 0, 0.1f, 0.0f);
    fx_dense(&st, "markov_head.hidden_out.bias", "BF16", 1, FX_E, 0, 0, 0.1f, 0.0f);
    fx_dense(&st, "markov_head.markov_w1.weight", "BF16", 2, FX_VPAD, FX_MR, 0, 0.3f, 0.0f);
    fx_dense(&st, "markov_head.rnn_proj.weight", "BF16", 2, 3 * FX_MR, 2 * FX_MR + 2 * FX_E, 0, 0.05f, 0.0f);
    fx_dense(&st, "markov_head.rnn_proj.bias", "BF16", 1, 3 * FX_MR, 0, 0, 0.1f, 0.0f);
    snprintf(p, sizeof p, "%s/mtp.safetensors", dir);
    fx_write_st(p, &st);
    fx_free(&st);
    return dir;
}

static void fx_cleanup(const char *dir)
{
    char cmd[600];
    snprintf(cmd, sizeof cmd, "rm -rf '%s'", dir);
    if (system(cmd) != 0) { /* best effort */ }
}

#endif
