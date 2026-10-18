/* Native qwen4exp EXL3 checkpoints (CYBER-FROST class).
 *
 * Builds a tiny but structurally complete qwen4_exp checkpoint — 4 layers
 * (3 gated delta-net + 1 full attention), 2 routed experts per layer,
 * hyper-connection blocks, and optionally the MTP draft head — as raw
 * SafeTensors, loads it through oc_llama_load_exl3_native() and runs
 * decode + prefill through the real forward paths. */
#define _POSIX_C_SOURCE 200809L
#include <criterion/criterion.h>
#include "oxidize/exl3_overlay.h"
#include "oxidize/exl3.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define D 128u          /* n_embd                  */
#define NL 4u           /* layers, layer 3 full    */
#define NE 2u           /* routed experts          */
#define IS 128u         /* expert intermediate     */
#define HC 2u           /* hc_mult, hc_dim = 256   */
#define HCD (HC * D)
#define HCLR 8u
#define QROWS 256u      /* fused [q|gate], 2 * 2 * 64 */
#define KVROWS 128u
#define HDIM 64u
#define SINNER 256u     /* ssm_inner_size          */
#define SCONV 768u      /* 2 * 256 + 256           */
#define VOCAB 128u

/* f32 → f16 (hardcoded encodings of a small exact-value set) */
static uint16_t hsel(size_t i)
{
    static const uint16_t t[4] = {0x3c00, 0x3800, 0x4000, 0x3400}; /* 1 .5 2 .25 */
    return t[i % 4];
}
static void put_f16(unsigned char *p, uint16_t h) { p[0] = h & 255; p[1] = h >> 8; }
static uint16_t bf16_from_f32(float v)
{
    uint32_t b;
    memcpy(&b, &v, 4);
    return (uint16_t)(b >> 16);
}

/* ── SafeTensors writer: payloads accumulate in memory, then one write ── */
typedef struct {
    char *hdr; size_t hn, hcap;
    unsigned char *pay; size_t pn, pcap;
    size_t ntensors;
} Stw;

static void stw_init(Stw *w)
{
    memset(w, 0, sizeof(*w));
    w->hcap = 1u << 16; w->pcap = 1u << 20;
    w->hdr = malloc(w->hcap);
    w->pay = malloc(w->pcap);
    cr_assert_not_null(w->hdr);
    cr_assert_not_null(w->pay);
    w->hdr[w->hn++] = '{';
}

static void stw_grow(Stw *w, size_t extra)
{
    if (w->hn + extra > w->hcap) {
        w->hcap = (w->hn + extra) * 2;
        w->hdr = realloc(w->hdr, w->hcap);
        cr_assert_not_null(w->hdr);
    }
    if (w->pn + extra > w->pcap) {
        w->pcap = (w->pn + extra) * 2;
        w->pay = realloc(w->pay, w->pcap);
        cr_assert_not_null(w->pay);
    }
}

/* dims are the ON-DISK shape; payload appended verbatim. */
static void stw_add(Stw *w, const char *name, const char *dtype,
                    const size_t *dims, size_t nd, const void *data,
                    size_t bytes)
{
    stw_grow(w, bytes + 256);
    char dims_json[128];
    dims_json[0] = 0;
    for (size_t i = 0; i < nd; i++)
        snprintf(dims_json + strlen(dims_json),
                 sizeof(dims_json) - strlen(dims_json), "%s%zu",
                 i ? "," : "", dims[i]);
    w->hn += (size_t)snprintf(w->hdr + w->hn, w->hcap - w->hn,
        "%s\"%s\":{\"dtype\":\"%s\",\"shape\":[%s],\"data_offsets\":[%zu,%zu]}",
        w->ntensors ? "," : "", name, dtype, dims_json, w->pn, w->pn + bytes);
    memcpy(w->pay + w->pn, data, bytes);
    w->pn += bytes;
    w->ntensors++;
}

static void stw_write(const Stw *w, const char *path)
{
    FILE *f = fopen(path, "wb");
    cr_assert_not_null(f);
    stw_grow((Stw *)w, 16);
    char *hdr = malloc(w->hn + 2);
    cr_assert_not_null(hdr);
    memcpy(hdr, w->hdr, w->hn);
    hdr[w->hn] = '}';
    size_t n = w->hn + 1;
    uint8_t le[8];
    for (size_t i = 0; i < 8; i++) le[i] = (uint8_t)((uint64_t)n >> (i * 8));
    cr_assert_eq(fwrite(le, 1, 8, f), 8);
    cr_assert_eq(fwrite(hdr, 1, n, f), n);
    cr_assert_eq(fwrite(w->pay, 1, w->pn, f), w->pn);
    free(hdr);
    fclose(f);
}

static void stw_free(Stw *w) { free(w->hdr); free(w->pay); memset(w, 0, sizeof(*w)); }

/* ── tensor builders ──────────────────────────────────────────────────── */

/* 4-bit trellis tensor [in/16, out/16, 64] I16 with deterministic bytes. */
static void stw_trellis(Stw *w, const char *key, size_t in, size_t out)
{
    size_t kt = in / 16, nt = out / 16, bytes = kt * nt * 64 * 2;
    unsigned char *p = malloc(bytes);
    cr_assert_not_null(p);
    for (size_t i = 0; i < bytes; i++) p[i] = (unsigned char)(i * 7 + 3);
    size_t dims[3] = {kt, nt, 64};
    char name[160];
    snprintf(name, sizeof(name), "%s.trellis", key);
    stw_add(w, name, "I16", dims, 3, p, bytes);
    free(p);
    /* suh/svh F16 scales */
    unsigned char sc[4096];
    size_t dims1[1];
    dims1[0] = in;
    for (size_t i = 0; i < in; i++) put_f16(sc + i * 2, hsel(i));
    snprintf(name, sizeof(name), "%s.suh", key);
    stw_add(w, name, "F16", dims1, 1, sc, in * 2);
    dims1[0] = out;
    for (size_t i = 0; i < out; i++) put_f16(sc + i * 2, hsel(i + 1));
    snprintf(name, sizeof(name), "%s.svh", key);
    stw_add(w, name, "F16", dims1, 1, sc, out * 2);
    snprintf(name, sizeof(name), "%s.mul1", key);
    uint32_t one = 1;
    stw_add(w, name, "I32", NULL, 0, &one, 4);
}

static void stw_f16(Stw *w, const char *name, size_t rows, size_t cols)
{
    size_t n = rows * cols, dims[2] = {rows, cols};
    unsigned char *p = malloc(n * 2);
    cr_assert_not_null(p);
    for (size_t i = 0; i < n; i++) put_f16(p + i * 2, hsel(i));
    stw_add(w, name, "F16", dims, 2, p, n * 2);
    free(p);
}

static void stw_f16_vec(Stw *w, const char *name, size_t n)
{
    size_t dims[1] = {n};
    unsigned char *p = malloc(n * 2);
    cr_assert_not_null(p);
    for (size_t i = 0; i < n; i++) put_f16(p + i * 2, hsel(i));
    stw_add(w, name, "F16", dims, 1, p, n * 2);
    free(p);
}

static void stw_bf16_vec(Stw *w, const char *name, size_t n, float base)
{
    size_t dims[1] = {n};
    unsigned char *p = malloc(n * 2);
    cr_assert_not_null(p);
    for (size_t i = 0; i < n; i++) {
        uint16_t h = bf16_from_f32(base + 0.25f * (float)(i % 3));
        put_f16(p + i * 2, h);
    }
    stw_add(w, name, "BF16", dims, 1, p, n * 2);
    free(p);
}

/* ── checkpoint assembly ──────────────────────────────────────────────── */

static void write_config(const char *dir, bool with_mtp)
{
    char path[256];
    snprintf(path, sizeof(path), "%s/config.json", dir);    FILE *f = fopen(path, "w");
    cr_assert_not_null(f);
    fprintf(f, "{\"model_type\":\"qwen4_exp\",\"text_config\":{"
        "\"model_type\":\"qwen4_exp_text\",\"vocab_size\":%u,"
        "\"hidden_size\":%u,\"num_hidden_layers\":%u,"
        "\"num_attention_heads\":2,\"num_key_value_heads\":2,\"head_dim\":%u,"
        "\"max_position_embeddings\":64,\"rms_norm_eps\":0.00001,"
        "\"hidden_act\":\"silu\",\"num_experts\":%u,\"num_experts_per_tok\":1,"
        "\"moe_intermediate_size\":%u,\"shared_expert_intermediate_size\":%u,"
        "\"full_attention_interval\":4,"
        "\"layer_types\":[\"linear_attention\",\"linear_attention\","
        "\"linear_attention\",\"full_attention\"],"
        "\"linear_conv_kernel_dim\":4,\"linear_key_head_dim\":128,"
        "\"linear_num_key_heads\":2,\"linear_num_value_heads\":2,"
        "\"linear_value_head_dim\":128,"
        "\"hc_count\":%u,\"hc_lowrank\":%u,"
        "\"indexer_n_heads\":1,\"indexer_head_dim\":32,"
        "\"indexer_budget\":8,\"indexer_compress_ratio\":2,"
        "\"mtp_num_hidden_layers\":%u,"
        "\"rope_parameters\":{\"rope_theta\":10000000,"
        "\"partial_rotary_factor\":0.25}},\"tie_word_embeddings\":false}",
        VOCAB, D, NL, HDIM, NE, IS, IS, HC, HCLR, with_mtp ? 1u : 0u);
    fclose(f);
}

static void write_tokenizer(const char *dir)
{
    char path[256];
    snprintf(path, sizeof(path), "%s/tokenizer.json", dir);
    FILE *f = fopen(path, "w");
    cr_assert_not_null(f);
    fprintf(f, "{\"model\":{\"type\":\"BPE\",\"vocab\":{");
    for (size_t i = 0; i < VOCAB; i++) {
        if (i < 2) fprintf(f, "%s\"%c\":%zu", i ? "," : "", i ? 'b' : 'a', i);
        else fprintf(f, ",\"t%zu\":%zu", i, i);
    }
    fprintf(f, "},\"merges\":[]},\"pre_tokenizer\":{\"type\":\"ByteLevel\","
        "\"use_regex\":false,\"add_prefix_space\":false},"
        "\"decoder\":{\"type\":\"ByteLevel\"}}");
    fclose(f);
}

/* hc side tensors for one layer under `stem` (…attn_hyper_connection). */
static void stw_hc(Stw *w, const char *stem)
{
    char name[160];
    snprintf(name, sizeof(name), "%s.hc_norm.weight", stem);
    stw_f16_vec(w, name, HCD);
    snprintf(name, sizeof(name), "%s.input_mix_weight_down.weight", stem);
    stw_f16(w, name, HCLR, HCD);
    snprintf(name, sizeof(name), "%s.input_mix_weight_up.weight", stem);
    stw_f16(w, name, HCD, HCLR);
    snprintf(name, sizeof(name), "%s.block_inject_weight.weight", stem);
    stw_f16(w, name, HC, HCD);
}

static void stw_moe(Stw *w, const char *stem)
{
    char key[160];
    snprintf(key, sizeof(key), "%s.gate", stem);
    char name[176];
    snprintf(name, sizeof(name), "%s.weight", key);
    stw_f16(w, name, NE, D);
    for (size_t ex = 0; ex < NE; ex++) {
        snprintf(key, sizeof(key), "%s.experts.%zu.gate_proj", stem, ex);
        stw_trellis(w, key, D, IS);
        key[strlen(key) - strlen("gate_proj")] = 0;
        strcat(key, "up_proj");
        stw_trellis(w, key, D, IS);
        key[strlen(key) - strlen("up_proj")] = 0;
        strcat(key, "down_proj");
        stw_trellis(w, key, IS, D);
    }
    snprintf(key, sizeof(key), "%s.shared_expert.gate_proj", stem);
    stw_trellis(w, key, D, IS);
    snprintf(key, sizeof(key), "%s.shared_expert.up_proj", stem);
    stw_trellis(w, key, D, IS);
    snprintf(key, sizeof(key), "%s.shared_expert.down_proj", stem);
    stw_trellis(w, key, IS, D);
    snprintf(key, sizeof(key), "%s.shared_expert_gate", stem);
    snprintf(name, sizeof(name), "%s.weight", key);
    stw_f16(w, name, 1, D);
}

static void write_checkpoint(const char *dir, bool with_mtp)
{
    char path[256];
    snprintf(path, sizeof(path), "%s/model.safetensors", dir);
    Stw w;
    stw_init(&w);
    char key[176], name[192];

    snprintf(name, sizeof(name), "model.language_model.embed_tokens.weight");
    {
        size_t dims[2] = {VOCAB, D};
        unsigned char *p = malloc(VOCAB * D * 2);
        cr_assert_not_null(p);
        for (size_t i = 0; i < VOCAB * D; i++)
            put_f16(p + i * 2, bf16_from_f32(0.01f * (float)(i % 7)));
        stw_add(&w, name, "BF16", dims, 2, p, VOCAB * D * 2);
        free(p);
    }
    stw_trellis(&w, "lm_head", D, VOCAB);
    snprintf(name, sizeof(name),
             "model.language_model.hyper_connection_mixer.hc_norm.weight");
    stw_f16_vec(&w, name, HCD);
    snprintf(name, sizeof(name),
             "model.language_model.hyper_connection_mixer.input_mix_weight_down.weight");
    stw_f16(&w, name, HCLR, HCD);
    snprintf(name, sizeof(name),
             "model.language_model.hyper_connection_mixer.input_mix_weight_up.weight");
    stw_f16(&w, name, HCD, HCLR);

    for (size_t l = 0; l < NL; l++) {
        char stem[128];
        snprintf(stem, sizeof(stem), "model.language_model.layers.%zu", l);
        snprintf(name, sizeof(name), "%s.attn_hyper_connection", stem);
        stw_hc(&w, name);
        snprintf(name, sizeof(name), "%s.mlp_hyper_connection", stem);
        stw_hc(&w, name);
        snprintf(name, sizeof(name), "%s.mlp", stem);
        stw_moe(&w, name);

        if (l < 3) {
            snprintf(key, sizeof(key), "%s.linear_attn.in_proj_qkv", stem);
            stw_trellis(&w, key, D, SCONV);
            snprintf(key, sizeof(key), "%s.linear_attn.in_proj_z", stem);
            stw_trellis(&w, key, D, SINNER);
            snprintf(key, sizeof(key), "%s.linear_attn.out_proj", stem);
            stw_trellis(&w, key, SINNER, D);
            snprintf(name, sizeof(name), "%s.linear_attn.A_log", stem);
            stw_bf16_vec(&w, name, 2, -1.0f);
            snprintf(name, sizeof(name), "%s.linear_attn.dt_bias", stem);
            stw_bf16_vec(&w, name, 2, 0.5f);
            /* conv kernel [conv_dim, 1, 4] BF16 */
            size_t dims[3] = {SCONV, 1, 4};
            unsigned char *p = malloc(SCONV * 4 * 2);
            cr_assert_not_null(p);
            for (size_t i = 0; i < SCONV * 4; i++)
                put_f16(p + i * 2, bf16_from_f32(0.125f * (float)(i % 3)));
            snprintf(name, sizeof(name), "%s.linear_attn.conv1d.weight", stem);
            stw_add(&w, name, "BF16", dims, 3, p, SCONV * 4 * 2);
            free(p);
            snprintf(name, sizeof(name), "%s.linear_attn.norm.weight", stem);
            stw_bf16_vec(&w, name, 128, 1.0f);
            snprintf(name, sizeof(name), "%s.linear_attn.in_proj_a.weight", stem);
            stw_f16(&w, name, 2, D);
            snprintf(name, sizeof(name), "%s.linear_attn.in_proj_b.weight", stem);
            stw_f16(&w, name, 2, D);
        } else {
            snprintf(key, sizeof(key), "%s.self_attn.q_proj", stem);
            stw_trellis(&w, key, D, QROWS);
            snprintf(key, sizeof(key), "%s.self_attn.k_proj", stem);
            stw_trellis(&w, key, D, KVROWS);
            snprintf(key, sizeof(key), "%s.self_attn.v_proj", stem);
            stw_trellis(&w, key, D, KVROWS);
            snprintf(key, sizeof(key), "%s.self_attn.o_proj", stem);
            stw_trellis(&w, key, HDIM * 2u, D); /* q rows only, no gate */
            snprintf(name, sizeof(name), "%s.self_attn.q_norm.weight", stem);
            stw_bf16_vec(&w, name, HDIM, 0.0f);
            snprintf(name, sizeof(name), "%s.self_attn.k_norm.weight", stem);
            stw_bf16_vec(&w, name, HDIM, 0.0f);
        }
    }

    if (with_mtp) {
        snprintf(name, sizeof(name), "mtp.layers.0.self_attn.q_proj");
        stw_trellis(&w, name, D, QROWS);
        snprintf(name, sizeof(name), "mtp.layers.0.self_attn.k_proj");
        stw_trellis(&w, name, D, KVROWS);
        snprintf(name, sizeof(name), "mtp.layers.0.self_attn.v_proj");
        stw_trellis(&w, name, D, KVROWS);
        snprintf(name, sizeof(name), "mtp.layers.0.self_attn.o_proj");
        stw_trellis(&w, name, HDIM * 2u, D);
        snprintf(name, sizeof(name), "mtp.layers.0.self_attn.q_norm.weight");
        stw_bf16_vec(&w, name, HDIM, 0.0f);
        snprintf(name, sizeof(name), "mtp.layers.0.self_attn.k_norm.weight");
        stw_bf16_vec(&w, name, HDIM, 0.0f);
        snprintf(name, sizeof(name), "mtp.layers.0.attn_hyper_connection");
        stw_hc(&w, name);
        snprintf(name, sizeof(name), "mtp.layers.0.mlp_hyper_connection");
        stw_hc(&w, name);
        snprintf(name, sizeof(name), "mtp.layers.0.mlp");
        stw_moe(&w, name);
        stw_trellis(&w, "mtp.fc_embedding", D, D);
        stw_trellis(&w, "mtp.fc_hidden", D, D);
        snprintf(name, sizeof(name), "mtp.pre_fc_norm_embedding.weight");
        stw_bf16_vec(&w, name, D, 1.0f);
        snprintf(name, sizeof(name), "mtp.pre_fc_norm_hidden.weight");
        stw_f16_vec(&w, name, HCD);
        snprintf(name, sizeof(name), "mtp.hyper_connection_mixer.hc_norm.weight");
        stw_f16_vec(&w, name, HCD);
        snprintf(name, sizeof(name),
                 "mtp.hyper_connection_mixer.input_mix_weight_down.weight");
        stw_f16(&w, name, HCLR, HCD);
        snprintf(name, sizeof(name),
                 "mtp.hyper_connection_mixer.input_mix_weight_up.weight");
        stw_f16(&w, name, HCD, HCLR);
    }

    stw_write(&w, path);
    stw_free(&w);
    write_config(dir, with_mtp);
    write_tokenizer(dir);
}

static void remove_dir(const char *dir)
{
    char path[256];
    const char *files[] = {"model.safetensors", "config.json", "tokenizer.json"};
    for (size_t i = 0; i < 3; i++) {
        snprintf(path, sizeof(path), "%s/%s", dir, files[i]);
        unlink(path);
    }
    rmdir(dir);
}

Test(exl3_qwen4exp, load_and_forward)
{
    char dir[64];
    strcpy(dir, "/tmp/oxidize-q4e-XXXXXX");
    cr_assert_not_null(mkdtemp(dir));
    write_checkpoint(dir, false);

    OcLlamaModel m;
    OcError rc = oc_llama_load_exl3_native(dir, &m);
    cr_assert_eq(rc, OC_OK, "load rc=%d", (int)rc);
    cr_assert(m.cfg.is_qwen4exp);
    cr_assert_eq(m.cfg.n_layer, NL);
    cr_assert_eq(m.cfg.num_experts, NE);
    cr_assert_eq(m.cfg.num_experts_per_tok, 1u);
    cr_assert_eq(m.cfg.hc_mult, HC);
    cr_assert_eq(m.cfg.ssm_inner_size, SINNER);
    cr_assert_eq(m.cfg.rope_dim, HDIM / 4u);
    cr_assert_eq(m.cfg.n_recurrent_layers, 3u);
    cr_assert_eq(m.cfg.n_full_attention_layers, 1u);
    cr_assert_eq(m.cfg.n_ff, IS);
    cr_assert_eq(m.cfg.expert_intermediate_size, IS);
    cr_assert_eq(m.cfg.rope_theta, 10000000.0f);
    cr_assert_eq(m.cfg.recurrent_layers[0], 1u);
    cr_assert_eq(m.cfg.recurrent_layers[3], 0u);
    cr_assert_not(m.cfg.ple_enabled);
    cr_assert_not_null(m.layers[0].exl3_gate);
    cr_assert_not_null(m.layers[3].attn_q.exl3);
    cr_assert_not_null(m.layers[3].attn_q_norm);
    cr_assert_not_null(m.layers[0].attn_qkv.exl3);
    cr_assert_not_null(m.output.exl3);
    cr_assert_not_null(m.layers[0].hc_attn_norm);
    cr_assert_not_null(m.layers[0].ssm_a_f32);
    cr_assert_null(m.mtp.layer.attn_q.data);
    cr_assert_not(m.mtp.present);

    /* A_log must arrive as A = -exp(A_log) in tiled order; the values here
     * are base -1.0/-0.75/-0.5 so A is in (0,1). */
    for (size_t i = 0; i < 2; i++)
        cr_assert(m.layers[0].ssm_a_f32[i] < 0.0f &&
                  m.layers[0].ssm_a_f32[i] > -2.0f);

    /* Decode and prefill must agree. */
    OcLlamaSession a, b;
    cr_assert_eq(oc_llama_session_init_kv(&m, &a, OC_KV_F32), OC_OK);
    cr_assert_eq(oc_llama_session_init_kv(&m, &b, OC_KV_F32), OC_OK);
    uint32_t tokens[] = {1, 3, 5};
    float x[VOCAB], y[VOCAB];
    for (size_t i = 0; i < 3; i++)
        cr_assert_eq(oc_llama_forward(&a, tokens[i], x), OC_OK);
    cr_assert_eq(oc_llama_prefill(&b, tokens, 3, 3, y), OC_OK);
    for (size_t i = 0; i < VOCAB; i++) {
        cr_assert(isfinite(x[i]) && isfinite(y[i]));
        cr_assert_float_eq(x[i], y[i], 1e-4f + fabsf(y[i]) * 1e-4f,
                           "logit %zu", i);
    }
    oc_llama_session_free(&a);
    oc_llama_session_free(&b);
    oc_llama_free(&m);
    remove_dir(dir);
}

Test(exl3_qwen4exp, mtp_head)
{
    char dir[64];
    strcpy(dir, "/tmp/oxidize-q4e-XXXXXX");
    cr_assert_not_null(mkdtemp(dir));
    write_checkpoint(dir, true);

    OcLlamaModel m;
    OcError rc = oc_llama_load_exl3_native(dir, &m);
    cr_assert_eq(rc, OC_OK, "mtp load rc=%d", (int)rc);
    cr_assert(m.mtp.present);
    cr_assert_eq(m.cfg.nextn_predict_layers, 1u);
    cr_assert_not_null(m.mtp.layer.attn_q.exl3);
    cr_assert_not_null(m.mtp.layer.exl3_gate);
    cr_assert_not_null(m.mtp.eh_proj.data);
    cr_assert_eq(m.mtp.eh_proj.cols, 2u * D);
    cr_assert_eq(m.mtp.eh_proj.rows, D);
    /* eh_proj must be the column interleave [W_emb | W_h] of the two
     * dequantized fc matrices — not the two D×D matrices written
     * back-to-back. Rebuild the source matrix (same deterministic
     * contents stw_trellis wrote: packed = i*7+3, mul1 4-bit) and
     * compare every fused entry against an independent dequant. */
    {
        const size_t pbytes = (D / 16) * (D / 16) * 64 * 2;
        uint8_t *packed = malloc(pbytes);
        float *suh = malloc(D * sizeof(float));
        float *svh = malloc(D * sizeof(float));
        float *emb = malloc(D * D * sizeof(float));
        float *hid = malloc(D * D * sizeof(float));
        cr_assert(packed && suh && svh && emb && hid);
        for (size_t i = 0; i < pbytes; i++)
            packed[i] = (uint8_t)(i * 7 + 3);
        static const float scales[4] = {1.0f, 0.5f, 2.0f, 0.25f};
        for (size_t i = 0; i < D; i++) {
            suh[i] = scales[i % 4];
            svh[i] = scales[(i + 1) % 4];
        }
        OcExl3Matrix fe = {
            .input_dim = D, .output_dim = D,
            .packed = packed, .packed_size = pbytes,
            .bits_twice = 8, .codebook = 2,
            .suh = suh, .svh = svh,
        };
        cr_assert_eq(oc_exl3_dequant_f32(&fe, emb), OC_OK);
        cr_assert_eq(oc_exl3_dequant_f32(&fe, hid), OC_OK);
        const float *eh = (const float *)m.mtp.eh_proj.data;
        for (size_t r = 0; r < D; r++)
            for (size_t c = 0; c < 2u * D; c++) {
                const float want =
                    c < D ? emb[r * D + c] : hid[r * D + c - D];
                cr_assert_float_eq(eh[r * 2u * D + c], want, 1e-5f,
                                   "eh_proj[%zu][%zu]", r, c);
            }
        free(packed); free(suh); free(svh); free(emb); free(hid);
    }
    cr_assert_not_null(m.mtp.enorm);
    cr_assert_not_null(m.mtp.hnorm);
    cr_assert_not_null(m.mtp.hc_head_norm);

    OcLlamaSession s;
    cr_assert_eq(oc_llama_session_init_kv(&m, &s, OC_KV_F32), OC_OK);
    float x[VOCAB];
    cr_assert_eq(oc_llama_forward(&s, 1, x), OC_OK);
    for (size_t i = 0; i < VOCAB; i++) cr_assert(isfinite(x[i]));
    oc_llama_session_free(&s);
    oc_llama_free(&m);
    remove_dir(dir);
}
