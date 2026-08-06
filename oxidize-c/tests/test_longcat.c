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
#include "oxidize/inf_model.h"
#include "oxidize/weight_storage.h"
#include "oxidize/quant.h"
#include "oxidize/weight_ops.h"
#include "oxidize/inference.h"
#include <math.h>

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

/* ─── MLA V projection ───────────────────────────────────────────────────
 *
 * v_b is stored exactly like k_b: HF's kv_b_proj [n_heads*(nope+v), kv_lora]
 * is split so head h contributes rows [h*(nope+v)+nope, h*(nope+v)+nope+v),
 * leaving head h's V block as `v_head_dim` contiguous rows of `kv_lora`
 * columns. That is head-major, which is what oc_gemv_weight_head expects.
 *
 * Two things used to be wrong here, and both silently produced garbage
 * rather than an error:
 *   - the GEMV was hand-rolled with a [kv_lora, v_dim, n_heads] index, which
 *     is neither the file's layout nor self-consistent; and
 *   - it read through oc_weight_storage_f32_data(), which returns NULL for
 *     mmap-backed quantized storage. BF16 counts as quantized, so V was
 *     identically zero for every real model.
 */

#define VB_HEADS   3u
#define VB_VDIM    4u
#define VB_LORA    8u

/* Reference: head-major (head, v, l) -> ((head*v_dim) + v) * kv_lora + l. */
static float vb_ref(size_t head, size_t v, size_t l)
{
    return (float)((head * VB_VDIM + v) * VB_LORA + l) * 0.03125f - 1.0f;
}

Test(longcat, mla_v_projection_is_head_major)
{
    size_t total = (size_t)VB_HEADS * VB_VDIM * VB_LORA;
    float *w = malloc(total * sizeof(float));
    cr_assert_not_null(w, "alloc");
    for (size_t hd = 0; hd < VB_HEADS; hd++)
        for (size_t v = 0; v < VB_VDIM; v++)
            for (size_t l = 0; l < VB_LORA; l++)
                w[(hd * VB_VDIM + v) * VB_LORA + l] = vb_ref(hd, v, l);

    OcWeightStorage ws;
    oc_weight_storage_init(&ws);
    cr_assert_eq(oc_weight_storage_f32(&ws, w, total), OC_OK, "f32 storage");

    float c_kv[VB_LORA];
    for (size_t l = 0; l < VB_LORA; l++) c_kv[l] = (float)(l + 1) * 0.5f;

    for (uint32_t hd = 0; hd < VB_HEADS; hd++) {
        float out[VB_VDIM];
        memset(out, 0, sizeof out);
        OcError e = oc_gemv_weight_head(&ws, VB_VDIM, VB_LORA, hd, VB_HEADS,
                                        c_kv, out);
        cr_assert_eq(e, OC_OK, "gemv head %u", hd);
        for (size_t v = 0; v < VB_VDIM; v++) {
            float want = 0.0f;
            for (size_t l = 0; l < VB_LORA; l++) want += vb_ref(hd, v, l) * c_kv[l];
            cr_assert_float_eq(out[v], want, 1e-4f,
                "head %u v %zu: got %.6f want %.6f", hd, v, out[v], want);
        }
    }
    oc_weight_storage_free(&ws);
}

Test(longcat, mla_v_projection_nonzero_for_quantized_storage)
{
    /* The regression that made V vanish: quantized storage returned NULL
     * from the f32 accessor and the projection was skipped, leaving the
     * output buffer at its memset zero. Any real model hits this path,
     * because BF16 is a quantized type here. */
    size_t total = (size_t)VB_HEADS * VB_VDIM * VB_LORA;
    float *ref = malloc(total * sizeof(float));
    cr_assert_not_null(ref, "alloc");
    for (size_t i = 0; i < total; i++) ref[i] = (float)((i % 11) + 1) * 0.125f;

    /* Store the same values as BF16 (top 16 bits of the f32 pattern). */
    uint8_t *bf16 = malloc(total * 2);
    cr_assert_not_null(bf16, "alloc bf16");
    for (size_t i = 0; i < total; i++) {
        uint32_t bits;
        memcpy(&bits, &ref[i], 4);
        uint16_t hi = (uint16_t)(bits >> 16);
        memcpy(bf16 + i * 2, &hi, 2);
    }

    OcWeightStorage ws;
    oc_weight_storage_init(&ws);
    cr_assert_eq(oc_weight_storage_quantized(&ws, OC_QUANT_BF16, bf16,
                                             total * 2), OC_OK, "bf16 storage");

    float c_kv[VB_LORA];
    for (size_t l = 0; l < VB_LORA; l++) c_kv[l] = 1.0f;

    bool any_nonzero = false;
    for (uint32_t hd = 0; hd < VB_HEADS; hd++) {
        float out[VB_VDIM];
        memset(out, 0, sizeof out);
        OcError e = oc_gemv_weight_head(&ws, VB_VDIM, VB_LORA, hd, VB_HEADS,
                                        c_kv, out);
        cr_assert_eq(e, OC_OK, "gemv head %u on quantized storage", hd);
        for (size_t v = 0; v < VB_VDIM; v++) {
            float want = 0.0f;
            for (size_t l = 0; l < VB_LORA; l++)
                want += ref[(hd * VB_VDIM + v) * VB_LORA + l];
            /* BF16 keeps 8 mantissa bits, so allow ~1% per summed term. */
            cr_assert_float_eq(out[v], want, 0.05f * (float)VB_LORA,
                "head %u v %zu: got %.6f want %.6f", hd, v, out[v], want);
            if (out[v] != 0.0f) any_nonzero = true;
        }
    }
    cr_assert(any_nonzero,
        "V projection must not be identically zero for quantized storage");

    free(ref);
    oc_weight_storage_free(&ws);
}

/* ─── Zero-expert (identity) MoE routing ─────────────────────────────────
 *
 * LongCat appends `zero_expert_count` identity experts after the routed
 * ones. They hold no weights and return their input unchanged. Three
 * properties distinguish this from ordinary top-k MoE, and all three are
 * silent if wrong:
 *   - the router spans routed + zero slots, so top-k can pick "do nothing";
 *   - exp_probs_b steers SELECTION only, never the applied gate;
 *   - there is NO renormalization over top-k, because routed mass summing
 *     to less than 1 is the mechanism by which a token skips work.
 */

#define ZE_H       2u
#define ZE_IFF     2u
#define ZE_ROUTED  1u
#define ZE_ZERO    1u
#define ZE_SLOTS   (ZE_ROUTED + ZE_ZERO)

typedef struct {
    OcWeightStorage gate_inp, gate_exps, up_exps, down_exps;
} ZeExperts;

/* Router that produces logits [l0, l1] from a fixed input of all ones. */
static void ze_build(ZeExperts *z, float l0, float l1)
{
    float *router = malloc(ZE_SLOTS * ZE_H * sizeof(float));
    /* input is {1,1}, so each row must sum to the wanted logit. */
    router[0] = l0 * 0.5f; router[1] = l0 * 0.5f;
    router[2] = l1 * 0.5f; router[3] = l1 * 0.5f;
    oc_weight_storage_init(&z->gate_inp);
    cr_assert_eq(oc_weight_storage_f32(&z->gate_inp, router, ZE_SLOTS * ZE_H),
                 OC_OK, "router");

    /* One routed expert. gate=identity, up=all ones, down=identity. */
    size_t n = (size_t)ZE_ROUTED * ZE_IFF * ZE_H;
    float *g = calloc(n, sizeof(float));
    float *u = calloc(n, sizeof(float));
    float *d = calloc(n, sizeof(float));
    for (size_t i = 0; i < ZE_IFF; i++) { g[i * ZE_H + i] = 1.0f; d[i * ZE_H + i] = 1.0f; }
    for (size_t i = 0; i < n; i++) u[i] = 1.0f;
    oc_weight_storage_init(&z->gate_exps);
    oc_weight_storage_init(&z->up_exps);
    oc_weight_storage_init(&z->down_exps);
    cr_assert_eq(oc_weight_storage_f32(&z->gate_exps, g, n), OC_OK, "gate");
    cr_assert_eq(oc_weight_storage_f32(&z->up_exps,   u, n), OC_OK, "up");
    cr_assert_eq(oc_weight_storage_f32(&z->down_exps, d, n), OC_OK, "down");
}

static void ze_free(ZeExperts *z)
{
    oc_weight_storage_free(&z->gate_inp);
    oc_weight_storage_free(&z->gate_exps);
    oc_weight_storage_free(&z->up_exps);
    oc_weight_storage_free(&z->down_exps);
}

static void ze_cfg(OcInferenceConfig *cfg, float scale)
{
    oc_inference_config_init(cfg);
    cfg->hidden_size = ZE_H;
    cfg->intermediate_size = ZE_IFF;
    cfg->expert_intermediate_size = ZE_IFF;
    cfg->num_experts = ZE_ROUTED;
    cfg->zero_expert_count = ZE_ZERO;
    cfg->num_experts_per_tok = 1;
    cfg->expert_weights_scale = scale;
    cfg->expert_gating_sigmoid = false;
}

/* Run the MoE with the given router logits and bias; return ffn_out. */
static void ze_run(float l0, float l1, const float *bias, float scale,
                   float *out)
{
    ZeExperts z;
    ze_build(&z, l0, l1);
    OcInferenceConfig cfg;
    ze_cfg(&cfg, scale);

    float normed[ZE_H] = { 1.0f, 1.0f };
    float gate_s[ZE_IFF], up_s[ZE_IFF], exp_out[ZE_H];
    float logits[ZE_SLOTS];
    OcExpertScore scores[ZE_SLOTS];

    OcError e = oc_moe_ffn_forward(&z.gate_inp, &z.gate_exps, &z.up_exps,
                                    &z.down_exps, bias, &cfg, normed, out,
                                    gate_s, up_s, exp_out, logits, scores);
    cr_assert_eq(e, OC_OK, "moe forward failed: %d", (int)e);
    ze_free(&z);
}

Test(longcat, zero_expert_contributes_identity)
{
    /* Slot 1 is the zero expert. Give it the larger logit so it wins top-1.
     * softmax([0, 4]) -> p1 = 1/(1+e^-4) = 0.98201379.
     * Expected output = p1 * scale * normed = 0.98201379 * 9 * 1. */
    float out[ZE_H] = {0};
    ze_run(0.0f, 4.0f, NULL, 9.0f, out);

    float p1 = 1.0f / (1.0f + expf(-4.0f));
    float want = p1 * 9.0f * 1.0f;
    for (size_t i = 0; i < ZE_H; i++)
        cr_assert_float_eq(out[i], want, 1e-4f,
            "zero expert should pass input through scaled: got %.6f want %.6f",
            out[i], want);
}

Test(longcat, zero_expert_output_is_not_renormalized)
{
    /* If top-k were renormalized to sum 1, the single selected expert would
     * always get weight 1.0 and the output would be exactly `scale`,
     * regardless of how confident the router was. That would erase the
     * "skip work" mechanism entirely. Two different logit gaps must give
     * two different magnitudes. */
    float lo[ZE_H] = {0}, hi[ZE_H] = {0};
    ze_run(0.0f, 1.0f, NULL, 1.0f, lo);
    ze_run(0.0f, 6.0f, NULL, 1.0f, hi);

    cr_assert(hi[0] > lo[0] + 0.1f,
        "a more confident router must route more mass: %.6f vs %.6f",
        hi[0], lo[0]);
    cr_assert(lo[0] < 0.95f,
        "weight %.6f looks renormalized to 1.0", lo[0]);
}

Test(longcat, exp_probs_b_biases_selection_not_weight)
{
    /* Routed expert (slot 0) has the higher probability, but a large bias on
     * slot 1 flips WHICH expert wins. The applied gate must still be slot
     * 1's UNBIASED probability -- folding the bias into the weight would
     * inflate the contribution far past 1.0. */
    float bias[ZE_SLOTS] = { 0.0f, 10.0f };
    float out[ZE_H] = {0};
    ze_run(4.0f, 0.0f, bias, 1.0f, out);

    /* softmax([4,0]) -> p1 = 1/(1+e^4) = 0.01798621. The zero expert wins
     * selection on bias, and contributes identity * p1. */
    float p1 = 1.0f / (1.0f + expf(4.0f));
    cr_assert_float_eq(out[0], p1, 1e-4f,
        "gate should be the unbiased probability %.6f, got %.6f", p1, out[0]);
    cr_assert(out[0] < 1.0f,
        "bias leaked into the weight: %.6f", out[0]);
}

Test(longcat, routed_expert_still_runs_when_it_wins)
{
    /* Sanity: with the routed expert winning, the zero-expert path must not
     * swallow it. For x = {1,1}: gate is the identity so gate[i] = 1 and
     * silu(1) = 0.7310586; up is an all-ones [2,2] matrix, so it sums both
     * input lanes to 2.0; down is the identity. Per lane that is
     * silu(1) * 2, gated by the router probability. */
    float out[ZE_H] = {0};
    ze_run(4.0f, 0.0f, NULL, 1.0f, out);

    float p0 = 1.0f / (1.0f + expf(-4.0f));
    float silu1 = 1.0f / (1.0f + expf(-1.0f));
    float want = p0 * silu1 * (float)ZE_H;
    for (size_t i = 0; i < ZE_H; i++)
        cr_assert_float_eq(out[i], want, 1e-4f,
            "routed expert output: got %.6f want %.6f", out[i], want);
}
