/* test_longcat.c — LongCat-2.0 config parse + ScMoE tensor dispatch.
 *
 * LongCat packs TWO attention+FFN sub-blocks into each GGUF `blk.N`, so a
 * 38-block file is a 76-layer model. Sub-block tensors carry a `_0`/`_1`
 * marker on the stem (`blk.3.attn_norm_1.weight`); the router, router bias
 * and expert pool appear once per block and belong to the even sub-layer.
 *
 * These fixtures are geometrically faithful but tiny: real LongCat-2.0 is
 * 8192-wide with 768+128 experts, which would be a 1.6T-param file. The
 * shapes here are scaled down; only the NAMING and LAYER-REMAP behaviour is
 * under test.
 *
 * Reference for the contract: LongCat-2.0 config.json + a header parse of
 * LongCat-2.0-BF16.gguf.
 */
#include <criterion/criterion.h>
#include "oxidize/gguf_writer.h"
#include "oxidize/llama.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Geometry of the miniature model written by build_longcat_gguf(). */
#define LC_BLOCKS      3u        /* GGUF blocks   -> 6 internal layers */
#define LC_LAYERS      (LC_BLOCKS * 2u)
#define LC_EMBD        64u
#define LC_HEADS       4u
#define LC_KEY_LEN     24u       /* nope 16 + rope 8                   */
#define LC_ROPE_DIM    8u
#define LC_VALUE_LEN   16u
#define LC_Q_LORA      12u
#define LC_KV_LORA     8u
#define LC_FF          32u       /* dense ffn_hidden_size              */
#define LC_EXPERT_FF   6u
#define LC_ROUTED      5u
#define LC_ZERO        2u
#define LC_SLOTS       (LC_ROUTED + LC_ZERO)
#define LC_TOPK        3u
#define LC_VOCAB       17u
#define LC_NGRAM       4u        /* (neighbor_num 3 - 1) * split_num 2 */

static void add_f32_tensor(OcGgufWriter *w, const char *name,
                           uint64_t d0, uint64_t d1)
{
    uint64_t dims[2] = { d0, d1 };
    uint32_t n_dims = (d1 > 0) ? 2u : 1u;
    size_t n = (size_t)d0 * (size_t)(d1 > 0 ? d1 : 1);
    float *buf = calloc(n, sizeof(float));
    cr_assert_not_null(buf, "calloc tensor %s", name);
    /* Non-zero, deterministic, and distinct per tensor so a mis-bound view
     * shows up as a value mismatch rather than silently reading zeros. */
    for (size_t i = 0; i < n; i++) buf[i] = (float)((i % 7) + 1) * 0.25f;
    OcError e = oc_gguf_writer_add_tensor(w, name, n_dims, dims, 0 /* F32 */,
                                          buf, (uint64_t)(n * sizeof(float)));
    cr_assert_eq(e, OC_OK, "add_tensor(%s) failed: %d", name, (int)e);
    free(buf);
}

/* Write a miniature but structurally complete LongCat GGUF. */
static void build_longcat_gguf(const char *path)
{
    OcGgufWriter w;
    OcError e = oc_gguf_writer_init(path, "longcat", &w);
    cr_assert_eq(e, OC_OK, "writer init failed: %d", (int)e);

    oc_gguf_writer_add_uint32(&w, "longcat.block_count", LC_BLOCKS);
    oc_gguf_writer_add_uint32(&w, "longcat.embedding_length", LC_EMBD);
    oc_gguf_writer_add_uint32(&w, "longcat.feed_forward_length", LC_FF);
    oc_gguf_writer_add_uint32(&w, "longcat.expert_feed_forward_length",
                              LC_EXPERT_FF);
    oc_gguf_writer_add_uint32(&w, "longcat.expert_count", LC_ROUTED);
    oc_gguf_writer_add_uint32(&w, "longcat.zero_expert_count", LC_ZERO);
    oc_gguf_writer_add_uint32(&w, "longcat.expert_used_count", LC_TOPK);
    oc_gguf_writer_add_float32(&w, "longcat.expert_weights_scale", 9.0f);
    oc_gguf_writer_add_uint32(&w, "longcat.attention.head_count", LC_HEADS);
    oc_gguf_writer_add_uint32(&w, "longcat.attention.head_count_kv", 1);
    oc_gguf_writer_add_uint32(&w, "longcat.attention.key_length", LC_KEY_LEN);
    oc_gguf_writer_add_uint32(&w, "longcat.attention.value_length", LC_VALUE_LEN);
    oc_gguf_writer_add_uint32(&w, "longcat.attention.q_lora_rank", LC_Q_LORA);
    oc_gguf_writer_add_uint32(&w, "longcat.attention.kv_lora_rank", LC_KV_LORA);
    oc_gguf_writer_add_float32(&w, "longcat.attention.layer_norm_rms_epsilon",
                               1e-5f);
    oc_gguf_writer_add_uint32(&w, "longcat.rope.dimension_count", LC_ROPE_DIM);
    oc_gguf_writer_add_float32(&w, "longcat.rope.freq_base", 1000000.0f);
    oc_gguf_writer_add_string(&w, "longcat.rope.scaling.type", "yarn");
    oc_gguf_writer_add_float32(&w, "longcat.rope.scaling.factor", 120.0f);
    oc_gguf_writer_add_uint32(&w,
        "longcat.rope.scaling.original_context_length", 8192);
    oc_gguf_writer_add_uint32(&w, "longcat.context_length", 256);
    oc_gguf_writer_add_uint32(&w, "longcat.vocab_size", LC_VOCAB);
    /* (neighbor_num - 1) * split_num = (3-1)*2 = 4 tables. */
    oc_gguf_writer_add_uint32(&w, "longcat.ngram.neighbor_num", 3);
    oc_gguf_writer_add_uint32(&w, "longcat.ngram.split_num", 2);

    /* Top level. */
    add_f32_tensor(&w, "token_embd.weight", LC_EMBD, LC_VOCAB);
    add_f32_tensor(&w, "output.weight", LC_EMBD, LC_VOCAB);
    add_f32_tensor(&w, "output_norm.weight", LC_EMBD, 0);
    for (unsigned i = 0; i < LC_NGRAM; i++) {
        char nm[64];
        snprintf(nm, sizeof nm, "ngram_embd_%u.weight", i);
        add_f32_tensor(&w, nm, LC_KV_LORA, 32u + i);
        snprintf(nm, sizeof nm, "ngram_proj_%u.weight", i);
        add_f32_tensor(&w, nm, LC_KV_LORA, LC_EMBD);
    }

    for (unsigned b = 0; b < LC_BLOCKS; b++) {
        char nm[80];
        for (unsigned s = 0; s < 2; s++) {
            snprintf(nm, sizeof nm, "blk.%u.attn_norm_%u.weight", b, s);
            add_f32_tensor(&w, nm, LC_EMBD, 0);
            snprintf(nm, sizeof nm, "blk.%u.ffn_norm_%u.weight", b, s);
            add_f32_tensor(&w, nm, LC_EMBD, 0);
            snprintf(nm, sizeof nm, "blk.%u.attn_q_a_%u.weight", b, s);
            add_f32_tensor(&w, nm, LC_EMBD, LC_Q_LORA);
            snprintf(nm, sizeof nm, "blk.%u.attn_q_a_norm_%u.weight", b, s);
            add_f32_tensor(&w, nm, LC_Q_LORA, 0);
            snprintf(nm, sizeof nm, "blk.%u.attn_q_b_%u.weight", b, s);
            add_f32_tensor(&w, nm, LC_Q_LORA, LC_HEADS * LC_KEY_LEN);
            snprintf(nm, sizeof nm, "blk.%u.attn_kv_a_mqa_%u.weight", b, s);
            add_f32_tensor(&w, nm, LC_EMBD, LC_KV_LORA + LC_ROPE_DIM);
            snprintf(nm, sizeof nm, "blk.%u.attn_kv_a_norm_%u.weight", b, s);
            add_f32_tensor(&w, nm, LC_KV_LORA, 0);
            snprintf(nm, sizeof nm, "blk.%u.attn_k_b_%u.weight", b, s);
            add_f32_tensor(&w, nm, LC_KV_LORA, LC_HEADS * 16u);
            snprintf(nm, sizeof nm, "blk.%u.attn_v_b_%u.weight", b, s);
            add_f32_tensor(&w, nm, LC_KV_LORA, LC_HEADS * LC_VALUE_LEN);
            snprintf(nm, sizeof nm, "blk.%u.attn_output_%u.weight", b, s);
            add_f32_tensor(&w, nm, LC_EMBD, LC_EMBD);
            snprintf(nm, sizeof nm, "blk.%u.ffn_gate_%u.weight", b, s);
            add_f32_tensor(&w, nm, LC_EMBD, LC_FF);
            snprintf(nm, sizeof nm, "blk.%u.ffn_up_%u.weight", b, s);
            add_f32_tensor(&w, nm, LC_EMBD, LC_FF);
            snprintf(nm, sizeof nm, "blk.%u.ffn_down_%u.weight", b, s);
            add_f32_tensor(&w, nm, LC_FF, LC_EMBD);
        }
        /* Once per block — belongs to the even sub-layer. */
        snprintf(nm, sizeof nm, "blk.%u.ffn_gate_inp.weight", b);
        add_f32_tensor(&w, nm, LC_EMBD, LC_SLOTS);
        snprintf(nm, sizeof nm, "blk.%u.exp_probs_b.bias", b);
        add_f32_tensor(&w, nm, LC_SLOTS, 0);
        snprintf(nm, sizeof nm, "blk.%u.ffn_gate_exps.weight", b);
        add_f32_tensor(&w, nm, LC_EMBD, LC_EXPERT_FF * LC_ROUTED);
        snprintf(nm, sizeof nm, "blk.%u.ffn_up_exps.weight", b);
        add_f32_tensor(&w, nm, LC_EMBD, LC_EXPERT_FF * LC_ROUTED);
        snprintf(nm, sizeof nm, "blk.%u.ffn_down_exps.weight", b);
        add_f32_tensor(&w, nm, LC_EXPERT_FF, LC_EMBD * LC_ROUTED);
    }

    cr_assert_eq(oc_gguf_writer_finalize(&w), OC_OK, "finalize failed");
    oc_gguf_writer_free(&w);
}

/* Criterion runs each Test in its own process, in parallel. A shared fixture
 * path would let one test unlink the file another is still mmap'ing, so each
 * caller supplies its own name. */
#define FIXTURE(name) "/tmp/oxidize-c-longcat-" name ".gguf"

Test(longcat, config_doubles_block_count_into_layers)
{
    const char *p = FIXTURE("cfg");
    build_longcat_gguf(p);

    OcLlamaModel m;
    memset(&m, 0, sizeof m);
    OcError e = oc_llama_load(p, &m);
    cr_assert_eq(e, OC_OK, "oc_llama_load failed: %d", (int)e);

    cr_assert(m.cfg.is_longcat, "is_longcat must be set for arch=longcat");
    /* The headline invariant: 3 GGUF blocks are 6 internal layers. */
    cr_assert_eq(m.cfg.n_layer, LC_LAYERS,
        "n_layer should be 2 * block_count = %u, got %u",
        LC_LAYERS, m.cfg.n_layer);
    cr_assert_eq(m.arch, OC_ARCH_LONGCAT, "arch should be OC_ARCH_LONGCAT");

    oc_llama_free(&m);
    remove(p);
}

Test(longcat, config_mla_and_moe_geometry)
{
    const char *p = FIXTURE("geom");
    build_longcat_gguf(p);

    OcLlamaModel m;
    memset(&m, 0, sizeof m);
    cr_assert_eq(oc_llama_load(p, &m), OC_OK, "load");

    cr_assert(m.cfg.uses_mla, "LongCat uses MLA");
    cr_assert_eq(m.cfg.mla_q_lora_dim, LC_Q_LORA, "q_lora_rank");
    cr_assert_eq(m.cfg.mla_kv_lora_dim, LC_KV_LORA, "kv_lora_rank");
    cr_assert_eq(m.cfg.mla_q_rope_dim, LC_ROPE_DIM, "rope dim");
    /* key_length - rope = nope. */
    cr_assert_eq(m.cfg.mla_kv_nope_head_dim, LC_KEY_LEN - LC_ROPE_DIM,
        "nope head dim should be key_length - rope, got %u",
        m.cfg.mla_kv_nope_head_dim);
    /* value_length is read explicitly and is NOT assumed equal to nope. */
    cr_assert_eq(m.cfg.mla_v_head_dim, LC_VALUE_LEN,
        "v_head_dim should come from attention.value_length, got %u",
        m.cfg.mla_v_head_dim);
    cr_assert_eq(m.cfg.head_dim, LC_KEY_LEN, "head_dim = key_length");
    cr_assert_eq(m.cfg.n_head_kv, 1u, "MLA caches a single latent");

    cr_assert_eq(m.cfg.zero_expert_count, LC_ZERO, "zero_expert_count");
    cr_assert_eq(m.cfg.ngram_n_grams, LC_NGRAM,
        "(neighbor_num-1)*split_num should be %u, got %u",
        LC_NGRAM, m.cfg.ngram_n_grams);
    cr_assert_eq(m.cfg.ngram_split_num, 2u, "split_num");

    oc_llama_free(&m);
    remove(p);
}

Test(longcat, sub_block_tensors_bind_to_split_layers)
{
    const char *p = FIXTURE("split");
    build_longcat_gguf(p);

    OcLlamaModel m;
    memset(&m, 0, sizeof m);
    cr_assert_eq(oc_llama_load(p, &m), OC_OK, "load");

    /* Every one of the 6 internal layers must have its own per-sub-block
     * weights — this is what fails if the `_0`/`_1` marker is ignored. */
    for (uint32_t l = 0; l < LC_LAYERS; l++) {
        cr_assert_not_null(m.layers[l].attn_norm,
            "layer %u attn_norm unbound", l);
        cr_assert_not_null(m.layers[l].ffn_norm,
            "layer %u ffn_norm unbound", l);
        cr_assert_not_null(m.layers[l].mla_q_a_norm,
            "layer %u mla_q_a_norm unbound", l);
        cr_assert_not_null(m.layers[l].mla_kv_a_norm,
            "layer %u mla_kv_a_norm unbound", l);
        cr_assert_not_null(m.layers[l].mla_q_a.data,
            "layer %u mla_q_a unbound", l);
        cr_assert_not_null(m.layers[l].mla_q_b.data,
            "layer %u mla_q_b unbound", l);
        cr_assert_not_null(m.layers[l].mla_kv_a_mqa.data,
            "layer %u mla_kv_a_mqa unbound", l);
        cr_assert_not_null(m.layers[l].mla_k_b.data,
            "layer %u mla_k_b unbound", l);
        cr_assert_not_null(m.layers[l].mla_v_b.data,
            "layer %u mla_v_b unbound", l);
        cr_assert_not_null(m.layers[l].attn_output.data,
            "layer %u attn_output unbound", l);
        cr_assert_not_null(m.layers[l].ffn_down.data,
            "layer %u ffn_down unbound", l);
    }

    oc_llama_free(&m);
    remove(p);
}

Test(longcat, per_block_tensors_bind_to_even_sub_layer)
{
    const char *p = FIXTURE("even");
    build_longcat_gguf(p);

    OcLlamaModel m;
    memset(&m, 0, sizeof m);
    cr_assert_eq(oc_llama_load(p, &m), OC_OK, "load");

    /* Router, router bias and the expert pool exist once per GGUF block and
     * attach to the EVEN sub-layer; the odd sub-layer shares them and must
     * therefore be left unbound rather than silently aliasing. */
    for (uint32_t b = 0; b < LC_BLOCKS; b++) {
        uint32_t even = b * 2u, odd = b * 2u + 1u;

        cr_assert_not_null(m.layers[even].ffn_gate_inp.data,
            "block %u router should bind to sub-layer %u", b, even);
        cr_assert_not_null(m.layers[even].exp_probs_b,
            "block %u exp_probs_b should bind to sub-layer %u", b, even);
        cr_assert_not_null(m.layers[even].ffn_gate_exps.data,
            "block %u gate_exps should bind to sub-layer %u", b, even);
        cr_assert_not_null(m.layers[even].ffn_down_exps.data,
            "block %u down_exps should bind to sub-layer %u", b, even);

        cr_assert_null(m.layers[odd].ffn_gate_inp.data,
            "sub-layer %u must not have its own router", odd);
        cr_assert_null(m.layers[odd].exp_probs_b,
            "sub-layer %u must not have its own router bias", odd);
    }

    /* The router covers routed + zero slots. */
    cr_assert_eq(m.layers[0].ffn_gate_inp.rows, (size_t)LC_SLOTS,
        "router should have %u rows (routed + zero), got %zu",
        LC_SLOTS, m.layers[0].ffn_gate_inp.rows);

    oc_llama_free(&m);
    remove(p);
}

Test(longcat, ngram_tables_bind_with_per_table_row_counts)
{
    const char *p = FIXTURE("ngram");
    build_longcat_gguf(p);

    OcLlamaModel m;
    memset(&m, 0, sizeof m);
    cr_assert_eq(oc_llama_load(p, &m), OC_OK, "load");

    for (unsigned i = 0; i < LC_NGRAM; i++) {
        cr_assert_not_null(m.ngram_embd[i].data,
            "ngram_embd_%u unbound", i);
        cr_assert_not_null(m.ngram_proj[i].data,
            "ngram_proj_%u unbound", i);
        /* Row counts differ per table and must come from the tensor's own
         * shape — real LongCat uses 16476898 + 2i, which no formula in the
         * loader should try to reproduce. */
        cr_assert_eq(m.ngram_embd[i].rows, (size_t)(32u + i),
            "ngram_embd_%u rows should be %u, got %zu",
            i, 32u + i, m.ngram_embd[i].rows);
    }
    /* Tables past the configured count stay zeroed. */
    cr_assert_null(m.ngram_embd[LC_NGRAM].data,
        "table %u should be unbound", LC_NGRAM);

    oc_llama_free(&m);
    remove(p);
}
