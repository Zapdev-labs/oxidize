#include <stdarg.h>
#include <criterion/criterion.h>
#include "oxidize/eagle4.h"
#include "oxidize/gguf.h"
#include "oxidize/safetensors.h"

#include <stdio.h>
#include <string.h>
#include <unistd.h>

static void tiny(OcEagle4Config *cfg)
{
    oc_eagle4_config_init(cfg);
    cfg->hidden_size = 4;
    cfg->target_hidden_size = 4;
    cfg->n_extract_layers = 1;
    cfg->extract_layers[0] = 0;
    cfg->num_attention_heads = 1;
    cfg->num_key_value_heads = 1;
    cfg->head_dim = 4;
    cfg->intermediate_size = 4;
    cfg->vocab_size = 4;
    cfg->draft_vocab_size = 4;
    cfg->rank = 4;
    cfg->candidate_k = 1;
    cfg->sliding_window = 2;
    cfg->num_hidden_layers = 1;
    cfg->n_experts = 0;
}

static void ident(float *w, size_t n)
{
    size_t i;
    memset(w, 0, n * n * sizeof(float));
    for (i = 0; i < n; i++) w[i * n + i] = 1.0f;
}

Test(eagle4, released_shape_and_policy)
{
    OcEagle4Config cfg;
    OcEagle4Harness h;
    OcEagle4Policy tree, chain, narrow, wide_ctx;
    oc_eagle4_config_init(&cfg);
    oc_eagle4_config_set_released(&cfg);
    cr_assert_eq(cfg.sliding_window, 4096);
    cr_assert_eq(cfg.rank, 128);
    cr_assert_eq(cfg.candidate_k, 128);

    oc_eagle4_harness_init(&h);
    oc_eagle4_policy_for_batch(&h, 1, 128, &tree);
    oc_eagle4_policy_for_batch(&h, 64, 128, &chain);
    oc_eagle4_policy_for_batch(&h, 384, 128, &narrow);
    oc_eagle4_policy_for_batch(&h, 16, 40000, &wide_ctx);
    cr_assert_eq(tree.mode, OC_EAGLE4_MODE_TREE);
    cr_assert_eq(tree.max_branch, 4);
    cr_assert_eq(tree.verify_cap, 16);
    cr_assert_eq(chain.mode, OC_EAGLE4_MODE_CHAIN);
    cr_assert_eq(chain.verify_cap, 4);
    cr_assert_eq(chain.max_branch, 1);
    cr_assert_eq(narrow.verify_cap, 3);
    cr_assert_eq(wide_ctx.verify_cap, 8);
    cr_assert_eq(wide_ctx.max_branch, 1);
}

Test(eagle4, select_level_branches_early_and_stops)
{
    OcEagle4Harness h;
    OcEagle4Policy policy;
    OcEagle4Cand cands[4];
    uint32_t chosen[4];
    uint32_t n, i;
    oc_eagle4_harness_init(&h);
    h.min_gain_per_us = 0.0f;
    oc_eagle4_policy_for_batch(&h, 1, 32, &policy);
    for (i = 0; i < 4; i++) {
        cands[i].parent = -1;
        cands[i].cond_prob = 0.5f / (float)(i + 1);
        cands[i].path_prob = cands[i].cond_prob;
    }
    n = oc_eagle4_select_level(&h, &policy, 1, 1, 0, cands, 4, chosen, 4);
    cr_assert_eq(n, 4);

    h.min_gain_per_us = 1.0e9f;
    n = oc_eagle4_select_level(&h, &policy, 1, 2, 2, cands, 4, chosen, 4);
    cr_assert_eq(n, 0);
    n = oc_eagle4_select_level(&h, &policy, 1, 1, 0, cands, 4, chosen, 4);
    cr_assert_eq(n, 1);

    oc_eagle4_policy_for_batch(&h, 64, 32, &policy);
    h.min_gain_per_us = 0.0f;
    n = oc_eagle4_select_level(&h, &policy, 64, 1, 0, cands, 4, chosen, 4);
    cr_assert_eq(n, 1);
    cr_assert_eq(policy.verify_cap, 4);
}

Test(eagle4, mask_and_greedy_accept)
{
    OcEagle4Tree tree;
    uint8_t mask[9];
    uint32_t next[3];
    uint32_t out[4];
    uint32_t n;
    memset(&tree, 0, sizeof(tree));
    tree.n_nodes = 3;
    tree.nodes[0].token = 1;
    tree.nodes[0].parent = -1;
    tree.nodes[1].token = 7;
    tree.nodes[1].parent = -1;
    tree.nodes[2].token = 2;
    tree.nodes[2].parent = 0;
    oc_eagle4_tree_mask(&tree, mask);
    cr_assert_eq(mask[0], 1);
    cr_assert_eq(mask[1], 0);
    cr_assert_eq(mask[1 * 3 + 1], 1);
    cr_assert_eq(mask[2 * 3 + 0], 1);
    cr_assert_eq(mask[2 * 3 + 2], 1);
    cr_assert_eq(mask[2 * 3 + 1], 0);

    next[0] = 2;
    next[1] = 0;
    next[2] = 9;
    n = oc_eagle4_accept_greedy(&tree, 1, next, out, 4);
    cr_assert_eq(n, 2);
    cr_assert_eq(out[0], 1);
    cr_assert_eq(out[1], 2);
    n = oc_eagle4_accept_greedy(&tree, 3, next, out, 4);
    cr_assert_eq(n, 0);
}

Test(eagle4, speculative_accept_is_deterministic)
{
    float target[4] = {0, 1, 0, 0};
    float draft[4] = {1, 0, 0, 0};
    float one[4] = {0, 0, 0, 1};
    OcEagle4Step steps[2];
    uint32_t out[4];
    uint32_t rng = 1;
    uint32_t n;
    memset(steps, 0, sizeof(steps));
    steps[0].draft_token = 0;
    steps[0].p = 0.0f;
    steps[0].q = 1.0f;
    steps[0].target_dist = target;
    steps[0].draft_dist = draft;
    steps[0].vocab = 4;
    n = oc_eagle4_speculative_accept(steps, 1, &rng, out, 4);
    cr_assert_eq(n, 1);
    cr_assert_eq(out[0], 1);

    steps[0].p = 1.0f;
    steps[0].q = 1.0f;
    steps[0].draft_token = 4;
    steps[0].target_dist = one;
    steps[0].draft_dist = one;
    steps[1] = steps[0];
    steps[1].draft_token = 5;
    n = oc_eagle4_speculative_accept(steps, 2, &rng, out, 4);
    cr_assert_eq(n, 3);
    cr_assert_eq(out[0], 4);
    cr_assert_eq(out[1], 5);
    cr_assert_eq(out[2], 3);
}

Test(eagle4, low_rank_head_touches_k_rows_and_window)
{
    OcEagle4Config cfg;
    OcEagle4Draft m;
    float *logits;
    uint64_t before;
    uint32_t i;
    tiny(&cfg);
    cfg.candidate_k = 2;
    cfg.sliding_window = 2;
    cr_assert_eq(oc_eagle4_model_init(&m, &cfg), OC_OK);
    cr_assert_eq(oc_eagle4_alloc_weight(&m, &m.tok_embeddings, 4, 4), OC_OK);
    cr_assert_eq(oc_eagle4_alloc_weight(&m, &m.lm_head, 4, 4), OC_OK);
    cr_assert_eq(oc_eagle4_alloc_weight(&m, &m.lm_down, 4, 4), OC_OK);
    cr_assert_eq(oc_eagle4_alloc_weight(&m, &m.lm_score, 4, 4), OC_OK);
    ident(m.lm_head.data, 4);
    ident(m.lm_down.data, 4);
    ident(m.lm_score.data, 4);
    for (i = 0; i < 4; i++) m.tok_embeddings.data[i * 4 + i] = 1.0f;
    logits = (float *)calloc(4, sizeof(float));
    cr_assert_not_null(logits);
    oc_eagle4_reset_stats(&m);
    cr_assert_eq(oc_eagle4_forward(&m, 1, logits, 4), OC_OK);
    cr_assert_eq(m.stats.exact_rows, 2);
    cr_assert_eq(m.stats.score_rows, 4);
    cr_assert(logits[1] > logits[0]);

    for (i = 0; i < 3; i++) cr_assert_eq(oc_eagle4_forward(&m, 0, NULL, 0), OC_OK);
    before = m.stats.attn_keys;
    cr_assert_eq(oc_eagle4_forward(&m, 0, NULL, 0), OC_OK);
    cr_assert_eq(m.stats.attn_keys - before, 2);
    free(logits);
    oc_eagle4_model_free(&m);
}

Test(eagle4, d2t_scatters_exact_logit)
{
    OcEagle4Config cfg;
    OcEagle4Draft m;
    float logits[4];
    static uint32_t mapv[4] = {0, 3, 1, 2};
    uint32_t i;
    tiny(&cfg);
    cfg.candidate_k = 1;
    cr_assert_eq(oc_eagle4_model_init(&m, &cfg), OC_OK);
    cr_assert_eq(oc_eagle4_alloc_weight(&m, &m.tok_embeddings, 4, 4), OC_OK);
    cr_assert_eq(oc_eagle4_alloc_weight(&m, &m.lm_head, 4, 4), OC_OK);
    cr_assert_eq(oc_eagle4_alloc_weight(&m, &m.lm_down, 4, 4), OC_OK);
    cr_assert_eq(oc_eagle4_alloc_weight(&m, &m.lm_score, 4, 4), OC_OK);
    ident(m.lm_head.data, 4);
    ident(m.lm_down.data, 4);
    ident(m.lm_score.data, 4);
    for (i = 0; i < 4; i++) m.tok_embeddings.data[i * 4 + i] = 1.0f;
    m.d2t = mapv;
    m.n_d2t = 4;
    cr_assert_eq(oc_eagle4_forward(&m, 1, logits, 4), OC_OK);
    cr_assert(logits[3] > logits[0]);
    cr_assert(logits[1] < logits[3]);
    m.d2t = NULL;
    oc_eagle4_model_free(&m);
}

Test(eagle4, propose_chain_then_commit)
{
    OcEagle4Config cfg;
    OcEagle4Draft m;
    OcEagle4Harness h;
    OcEagle4Tree tree;
    uint32_t i;
    tiny(&cfg);
    cfg.candidate_k = 1;
    cfg.sliding_window = 16;
    cr_assert_eq(oc_eagle4_model_init(&m, &cfg), OC_OK);
    cr_assert_eq(oc_eagle4_alloc_weight(&m, &m.tok_embeddings, 4, 4), OC_OK);
    cr_assert_eq(oc_eagle4_alloc_weight(&m, &m.lm_head, 4, 4), OC_OK);
    cr_assert_eq(oc_eagle4_alloc_weight(&m, &m.lm_down, 4, 4), OC_OK);
    cr_assert_eq(oc_eagle4_alloc_weight(&m, &m.lm_score, 4, 4), OC_OK);
    ident(m.lm_head.data, 4);
    ident(m.lm_down.data, 4);
    ident(m.lm_score.data, 4);
    for (i = 0; i < 4; i++)
        m.tok_embeddings.data[i * 4 + ((i + 1) % 4)] = 1.0f;
    oc_eagle4_harness_init(&h);
    h.max_depth = 4;
    h.max_branch = 1;
    h.verify_cap = 4;
    h.min_gain_per_us = 0.0f;
    h.tree_batch_max = 4;
    cr_assert_eq(oc_eagle4_propose(&m, &h, 0, 1, 8, &tree), OC_OK);
    cr_assert_eq(tree.n_nodes, 4);
    cr_assert_eq(tree.nodes[0].token, 1);
    cr_assert_eq(tree.nodes[1].token, 2);
    cr_assert_eq(tree.nodes[2].token, 3);
    cr_assert_eq(tree.nodes[3].token, 0);
    cr_assert_eq(oc_eagle4_kv_len(&m), 1);
    cr_assert_eq(oc_eagle4_commit(&m, (uint32_t[]){1, 2, 3, 0}, 4), OC_OK);
    cr_assert_eq(oc_eagle4_kv_len(&m), 5);
    oc_eagle4_model_free(&m);
}

Test(eagle4, moe_expert_is_selected)
{
    OcEagle4Config cfg;
    OcEagle4Draft m;
    tiny(&cfg);
    cfg.hidden_size = 2;
    cfg.head_dim = 2;
    cfg.intermediate_size = 2;
    cfg.vocab_size = 2;
    cfg.draft_vocab_size = 2;
    cfg.n_experts = 2;
    cfg.top_k_experts = 1;
    cr_assert_eq(oc_eagle4_model_init(&m, &cfg), OC_OK);
    cr_assert_eq(oc_eagle4_alloc_weight(&m, &m.tok_embeddings, 2, 2), OC_OK);
    cr_assert_eq(oc_eagle4_alloc_f32(&m, &m.layers[0].router, 4), OC_OK);
    cr_assert_eq(oc_eagle4_alloc_weight(&m, &m.layers[0].expert_gate[0], 2, 2), OC_OK);
    cr_assert_eq(oc_eagle4_alloc_weight(&m, &m.layers[0].expert_up[0], 2, 2), OC_OK);
    cr_assert_eq(oc_eagle4_alloc_weight(&m, &m.layers[0].expert_down[0], 2, 2), OC_OK);
    m.tok_embeddings.data[0] = 1.0f;
    m.layers[0].router[0] = 10.0f;
    m.layers[0].expert_gate[0].data[0] = 4.0f;
    m.layers[0].expert_up[0].data[0] = 1.0f;
    m.layers[0].expert_down[0].data[0] = 1.0f;
    m.layers[0].expert_down[0].data[3] = 1.0f;
    cr_assert_eq(oc_eagle4_forward(&m, 0, NULL, 0), OC_OK);
    cr_assert(m.g_embeddings[0] > 4.0f);
    oc_eagle4_model_free(&m);
}

Test(eagle4, train_step_lowers_loss_and_refuses_gguf)
{
    OcEagle4Config cfg;
    OcEagle4Draft m;
    float features[4] = {0.2f, 0, 0, 0};
    float before = 0, after = 0;
    uint32_t pred = 99;
    int step;
    char path[64];
    tiny(&cfg);
    cr_assert_eq(oc_eagle4_model_init(&m, &cfg), OC_OK);
    cr_assert_eq(oc_eagle4_alloc_weight(&m, &m.tok_embeddings, 4, 4), OC_OK);
    cr_assert_eq(oc_eagle4_alloc_weight(&m, &m.lm_head, 4, 4), OC_OK);
    m.tok_embeddings.data[0] = 1.0f;
    cr_assert_eq(oc_eagle4_eval(&m, features, 4, 0, 2, &before, NULL), OC_OK);
    for (step = 0; step < 40; step++)
        cr_assert_eq(oc_eagle4_train_step(&m, features, 4, 0, 2, 0.4f, NULL), OC_OK);
    cr_assert_eq(oc_eagle4_eval(&m, features, 4, 0, 2, &after, &pred), OC_OK);
    cr_assert(after < before);
    cr_assert_eq(pred, 2);
    cr_assert(oc_eagle4_path_is_gguf("draft.gguf"));
    cr_assert(oc_eagle4_path_is_gguf("Draft.GGUF"));
    cr_assert(!oc_eagle4_path_is_gguf("draft.safetensors"));
    snprintf(path, sizeof(path), "/tmp/oc-eagle4-%d.gguf", (int)getpid());
    cr_assert_eq(oc_eagle4_train_safetensors(&m, path, 0.1f, 1, NULL), OC_ERR_INVALID_ARG);
    cr_assert_eq(oc_eagle4_load_safetensors(&m, path), OC_ERR_INVALID_ARG);
    cr_assert_eq(oc_eagle4_write_examples(path, features, 1, 4,
                                         (uint32_t[]){0}, (uint32_t[]){2}),
                 OC_ERR_INVALID_ARG);
    oc_eagle4_model_free(&m);
}

Test(eagle4, safetensors_roundtrip_trains_and_exports_gguf)
{
    OcEagle4Config cfg;
    OcEagle4Draft m, loaded;
    float features[4] = {1, 0, 0, 0};
    uint32_t tokens[1] = {0};
    uint32_t targets[1] = {1};
    float before, after;
    char st_path[64], ex_path[64], gg_path[64];
    OcGgufFile gg;
    const OcGgufTensorInfo *ti;
    int pid = (int)getpid();
    tiny(&cfg);
    cr_assert_eq(oc_eagle4_model_init(&m, &cfg), OC_OK);
    cr_assert_eq(oc_eagle4_alloc_weight(&m, &m.tok_embeddings, 4, 4), OC_OK);
    cr_assert_eq(oc_eagle4_alloc_weight(&m, &m.lm_head, 4, 4), OC_OK);
    m.tok_embeddings.data[0] = 0.5f;
    m.tok_embeddings.data[5] = 0.25f;
    snprintf(st_path, sizeof(st_path), "/tmp/oc-eagle4-%d.safetensors", pid);
    snprintf(ex_path, sizeof(ex_path), "/tmp/oc-eagle4-ex-%d.safetensors", pid);
    snprintf(gg_path, sizeof(gg_path), "/tmp/oc-eagle4-out-%d.gguf", pid);
    cr_assert_eq(oc_eagle4_save_safetensors(&m, st_path), OC_OK);
    memset(&loaded, 0, sizeof(loaded));
    cr_assert_eq(oc_eagle4_load_safetensors(&loaded, st_path), OC_OK);
    cr_assert_float_eq(loaded.tok_embeddings.data[0], 0.5f, 1e-6);
    cr_assert_float_eq(loaded.tok_embeddings.data[5], 0.25f, 1e-6);
    cr_assert_eq(loaded.config.hidden_size, 4);
    cr_assert_eq(oc_eagle4_write_examples(ex_path, features, 1, 4, tokens, targets), OC_OK);
    cr_assert_eq(oc_eagle4_eval(&loaded, features, 4, 0, 1, &before, NULL), OC_OK);
    cr_assert_eq(oc_eagle4_train_safetensors(&loaded, ex_path, 0.4f, 30, NULL), OC_OK);
    cr_assert_eq(oc_eagle4_eval(&loaded, features, 4, 0, 1, &after, NULL), OC_OK);
    cr_assert(after < before);
    cr_assert_eq(oc_eagle4_export_gguf(&loaded, gg_path), OC_OK);
    memset(&gg, 0, sizeof(gg));
    cr_assert_eq(oc_gguf_open(gg_path, &gg), OC_OK);
    ti = oc_gguf_tensor_get(&gg, "eagle4.embed");
    cr_assert_not_null(ti);
    cr_assert_eq(ti->ggml_type, 0);
    cr_assert_eq(memcmp(gg.backing_buf + ti->absolute_offset,
                        loaded.tok_embeddings.data, 16), 0);
    cr_assert_not_null(oc_gguf_metadata_get(&gg, "eagle4.format"));
    oc_gguf_free(&gg);
    oc_eagle4_model_free(&loaded);
    oc_eagle4_model_free(&m);
    unlink(st_path);
    unlink(ex_path);
    unlink(gg_path);
}

typedef struct StSpec {
    const char *name;
    const char *dtype;
    uint64_t shape[2];
    int ndims;
    size_t esz;
    int64_t len; /* < 0: derive from shape */
} StSpec;

/* Write a zero-filled SafeTensors file. A spec with len == -2 writes
 * data_offsets with end < start. */
/* Append formatted text, failing the test instead of overflowing json[]. */
static size_t st_append(char *buf, size_t cap, size_t used, const char *fmt, ...)
{
    va_list ap;
    int w;
    cr_assert_lt(used, cap);
    va_start(ap, fmt);
    w = vsnprintf(buf + used, cap - used, fmt, ap);
    va_end(ap);
    cr_assert(w >= 0 && (size_t)w < cap - used);
    return used + (size_t)w;
}

static void write_st(const char *path, const StSpec *s, size_t n)
{
    char json[4096];
    size_t used = 0, off = 0, i, total = 0;
    FILE *fp;
    uint8_t le[8];
    int k;
    used = st_append(json, sizeof(json), used, "{");
    for (i = 0; i < n; i++) {
        size_t len = s[i].len >= 0 ? (size_t)s[i].len
                                   : (size_t)(s[i].shape[0] * (s[i].ndims > 1 ? s[i].shape[1] : 1)) * s[i].esz;
        size_t a = off, b = off + len;
        if (s[i].len == -2) { a = 8; b = 0; len = 0; }
        if (s[i].ndims == 1)
            used = st_append(json, sizeof(json), used,
                "%s\"%s\":{\"dtype\":\"%s\",\"shape\":[%llu],\"data_offsets\":[%zu,%zu]}",
                i ? "," : "", s[i].name, s[i].dtype, (unsigned long long)s[i].shape[0], a, b);
        else
            used = st_append(json, sizeof(json), used,
                "%s\"%s\":{\"dtype\":\"%s\",\"shape\":[%llu,%llu],\"data_offsets\":[%zu,%zu]}",
                i ? "," : "", s[i].name, s[i].dtype, (unsigned long long)s[i].shape[0],
                (unsigned long long)s[i].shape[1], a, b);
        off += len;
    }
    used = st_append(json, sizeof(json), used, "}");
    while ((used & 7u) && used < sizeof(json)) json[used++] = ' ';
    total = off + 8;
    fp = fopen(path, "wb");
    cr_assert_not_null(fp);
    for (k = 0; k < 8; k++) le[k] = (uint8_t)(((uint64_t)used >> (8 * k)) & 0xff);
    fwrite(le, 1, 8, fp);
    fwrite(json, 1, used, fp);
    for (i = 0; i < total; i++) fputc(0, fp);
    fclose(fp);
}

Test(eagle4, safetensors_rejects_short_data_and_reversed_offsets)
{
    char path[64];
    OcSafetensorsFile st;
    const OcSafetensorsTensor *t = NULL;
    OcEagle4Draft m;
    StSpec small = { "a", "F32", {4, 4}, 2, 4, 8 };
    StSpec rev = { "a", "F32", {2, 0}, 1, 4, -2 };
    StSpec cfg = { "eagle4.config", "F32", {28, 0}, 1, 4, 16 };
    snprintf(path, sizeof(path), "/tmp/oc-eagle4-st-%d.safetensors", (int)getpid());

    write_st(path, &small, 1);
    memset(&st, 0, sizeof(st));
    cr_assert_eq(oc_safetensors_open(path, &st), OC_OK);
    cr_assert_eq(oc_safetensors_get_tensor(&st, "a", &t), OC_OK);
    cr_assert(!oc_st_tensor_bytes_ok(t, sizeof(float)));
    cr_assert(!oc_st_tensor_bytes_ok(t, 2)); /* 16 elems * 2 != 8 */
    oc_safetensors_close(&st);

    write_st(path, &cfg, 1);
    memset(&m, 0, sizeof(m));
    cr_assert_eq(oc_eagle4_load_safetensors(&m, path), OC_ERR_FORMAT);

    write_st(path, &rev, 1);
    memset(&st, 0, sizeof(st));
    cr_assert_eq(oc_safetensors_open(path, &st), OC_ERR_FORMAT);
    unlink(path);
}

Test(eagle4, load_rejects_mismatched_projection)
{
    OcEagle4Config cfg;
    OcEagle4Draft m, loaded;
    char path[64];
    tiny(&cfg);
    snprintf(path, sizeof(path), "/tmp/oc-eagle4-bad-%d.safetensors", (int)getpid());
    cr_assert_eq(oc_eagle4_model_init(&m, &cfg), OC_OK);
    cr_assert_eq(oc_eagle4_alloc_weight(&m, &m.tok_embeddings, 4, 4), OC_OK);
    cr_assert_eq(oc_eagle4_alloc_weight(&m, &m.layers[0].q_proj, 3, 4), OC_OK);
    cr_assert_eq(oc_eagle4_save_safetensors(&m, path), OC_OK);
    memset(&loaded, 0, sizeof(loaded));
    cr_assert_eq(oc_eagle4_load_safetensors(&loaded, path), OC_ERR_FORMAT);
    oc_eagle4_model_free(&m);
    unlink(path);
}

static void hf_specs(StSpec *s, uint64_t o_cols)
{
    StSpec base[] = {
        { "embed_tokens.weight", "BF16", {8, 4}, 2, 2, -1 },
        { "fc.weight", "BF16", {4, 4}, 2, 2, -1 },
        { "layers.0.self_attn.q_proj.weight", "BF16", {4, 8}, 2, 2, -1 },
        { "layers.0.self_attn.k_proj.weight", "BF16", {4, 8}, 2, 2, -1 },
        { "layers.0.self_attn.v_proj.weight", "BF16", {4, 8}, 2, 2, -1 },
        { "layers.0.self_attn.o_proj.weight", "BF16", {4, 4}, 2, 2, -1 },
        { "layers.0.mlp.gate_proj.weight", "BF16", {4, 4}, 2, 2, -1 },
        { "layers.0.mlp.up_proj.weight", "BF16", {4, 4}, 2, 2, -1 },
        { "layers.0.mlp.down_proj.weight", "BF16", {4, 4}, 2, 2, -1 },
        { "efficient_lm_head_down_proj.weight", "BF16", {4, 4}, 2, 2, -1 },
        { "efficient_lm_head.weight", "BF16", {8, 4}, 2, 2, -1 },
        { "lm_head.weight", "BF16", {8, 4}, 2, 2, -1 },
        { "norm.weight", "BF16", {4, 0}, 1, 2, -1 },
    };
    memcpy(s, base, sizeof(base));
    s[5].shape[1] = o_cols;
}

Test(eagle4, hf_rejects_bad_shape_and_slides_kv)
{
    StSpec s[13];
    OcEagle4HfBench b;
    char path[64];
    snprintf(path, sizeof(path), "/tmp/oc-eagle4-hf-%d.safetensors", (int)getpid());
    hf_specs(s, 4);
    write_st(path, s, 13);
    /* 4200 tokens pushes past the 4096+8 KV capacity and forces a slide. */
    cr_assert_eq(oc_eagle4_hf_bench(path, 1, 4200, &b), OC_OK);
    cr_assert_eq(b.hidden, 4);
    hf_specs(s, 6);
    write_st(path, s, 13);
    cr_assert_eq(oc_eagle4_hf_bench(path, 1, 1, &b), OC_ERR_FORMAT);
    s[12].shape[0] = 3;
    s[5].shape[1] = 4;
    write_st(path, s, 13);
    cr_assert_eq(oc_eagle4_hf_bench(path, 1, 1, &b), OC_ERR_FORMAT);
    unlink(path);
}

/* Every eagle4.layer.<L>.* tensor must bind to its own slot, not q_proj. */
Test(eagle4, load_binds_each_layer_tensor)
{
    OcEagle4Config cfg;
    OcEagle4Draft m, loaded;
    OcEagle4Layer *a, *b;
    char path[64];
    OcEagle4Weight *ws[10];
    float **vs[3];
    size_t i;
    tiny(&cfg);
    cfg.n_experts = 2;
    cfg.top_k_experts = 1;
    snprintf(path, sizeof(path), "/tmp/oc-eagle4-bind-%d.safetensors", (int)getpid());
    cr_assert_eq(oc_eagle4_model_init(&m, &cfg), OC_OK);
    a = &m.layers[0];
    ws[0] = &a->q_proj; ws[1] = &a->k_proj; ws[2] = &a->v_proj; ws[3] = &a->o_proj;
    ws[4] = &a->mlp_gate; ws[5] = &a->mlp_up; ws[6] = &a->mlp_down;
    ws[7] = &a->expert_gate[1]; ws[8] = &a->expert_up[1]; ws[9] = &a->expert_down[1];
    for (i = 0; i < 10; i++) {
        cr_assert_eq(oc_eagle4_alloc_weight(&m, ws[i], 4, 4), OC_OK);
        ws[i]->data[0] = (float)(i + 1);
    }
    vs[0] = &a->attn_norm; vs[1] = &a->ffn_norm; vs[2] = &a->router;
    for (i = 0; i < 3; i++) {
        cr_assert_eq(oc_eagle4_alloc_f32(&m, vs[i], i == 2 ? 8 : 4), OC_OK);
        (*vs[i])[0] = (float)(20 + i);
    }
    cr_assert_eq(oc_eagle4_save_safetensors(&m, path), OC_OK);
    memset(&loaded, 0, sizeof(loaded));
    cr_assert_eq(oc_eagle4_load_safetensors(&loaded, path), OC_OK);
    b = &loaded.layers[0];
    {
        OcEagle4Weight *lw[10] = { &b->q_proj, &b->k_proj, &b->v_proj, &b->o_proj,
                                   &b->mlp_gate, &b->mlp_up, &b->mlp_down,
                                   &b->expert_gate[1], &b->expert_up[1],
                                   &b->expert_down[1] };
        float *lv[3] = { b->attn_norm, b->ffn_norm, b->router };
        for (i = 0; i < 10; i++) {
            cr_assert_not_null(lw[i]->data, "weight %zu unbound", i);
            cr_assert_float_eq(lw[i]->data[0], (float)(i + 1), 1e-6, "weight %zu", i);
        }
        for (i = 0; i < 3; i++) {
            cr_assert_not_null(lv[i], "vec %zu unbound", i);
            cr_assert_float_eq(lv[i][0], (float)(20 + i), 1e-6, "vec %zu", i);
        }
        cr_assert_null(b->expert_gate[0].data);
    }
    oc_eagle4_model_free(&loaded);
    oc_eagle4_model_free(&m);
    unlink(path);
}

static void fill_det(float *p, size_t n, uint32_t seed)
{
    size_t i;
    for (i = 0; i < n; i++) {
        seed = seed * 1664525u + 1013904223u;
        p[i] = (float)((seed >> 8) & 1023) / 512.0f - 1.0f;
    }
}

static void kv_draft(OcEagle4Draft *m)
{
    OcEagle4Config cfg;
    OcEagle4Layer *ly;
    tiny(&cfg);
    cfg.candidate_k = 4;
    cfg.sliding_window = 16;
    cr_assert_eq(oc_eagle4_model_init(m, &cfg), OC_OK);
    ly = &m->layers[0];
    cr_assert_eq(oc_eagle4_alloc_weight(m, &m->tok_embeddings, 4, 4), OC_OK);
    cr_assert_eq(oc_eagle4_alloc_weight(m, &m->lm_head, 4, 4), OC_OK);
    cr_assert_eq(oc_eagle4_alloc_weight(m, &m->lm_down, 4, 4), OC_OK);
    cr_assert_eq(oc_eagle4_alloc_weight(m, &m->lm_score, 4, 4), OC_OK);
    cr_assert_eq(oc_eagle4_alloc_weight(m, &ly->q_proj, 4, 4), OC_OK);
    cr_assert_eq(oc_eagle4_alloc_weight(m, &ly->k_proj, 4, 4), OC_OK);
    cr_assert_eq(oc_eagle4_alloc_weight(m, &ly->v_proj, 4, 4), OC_OK);
    cr_assert_eq(oc_eagle4_alloc_weight(m, &ly->o_proj, 4, 4), OC_OK);
    fill_det(m->tok_embeddings.data, 16, 1);
    fill_det(m->lm_head.data, 16, 2);
    fill_det(m->lm_down.data, 16, 3);
    fill_det(m->lm_score.data, 16, 4);
    fill_det(ly->q_proj.data, 16, 5);
    fill_det(ly->k_proj.data, 16, 6);
    fill_det(ly->v_proj.data, 16, 7);
    fill_det(ly->o_proj.data, 16, 8);
}

/* A tree node's children must see that node's own KV row, not the row a
 * later sibling wrote at the same offset. Compare against a chain. */
Test(eagle4, tree_children_see_own_branch_kv)
{
    OcEagle4Draft m;
    OcEagle4Harness h;
    OcEagle4Tree chain, tree;
    uint32_t i, j, p;
    int checked = 0;
    kv_draft(&m);
    oc_eagle4_harness_init(&h);
    h.min_gain_per_us = 0.0f;
    h.tree_batch_max = 4;
    h.max_depth = 3;
    for (p = 0; p < 4; p++) {
        oc_eagle4_reset_cache(&m);
        h.max_branch = 1;
        h.verify_cap = 3;
        cr_assert_eq(oc_eagle4_propose(&m, &h, p, 1, 8, &chain), OC_OK);
        cr_assert_eq(chain.n_nodes, 3);

        oc_eagle4_reset_cache(&m);
        h.max_branch = 4;
        h.verify_cap = 64;
        cr_assert_eq(oc_eagle4_propose(&m, &h, p, 1, 8, &tree), OC_OK);
        cr_assert_eq(oc_eagle4_kv_len(&m), 1);
        cr_assert_eq(tree.nodes[0].token, chain.nodes[0].token);
        cr_assert(tree.n_nodes > 2 && tree.nodes[1].depth == 1, "need a sibling after node 0");
        /* Grandchildren of node 0 along the chain path were drafted from a
         * forward that attends to node 0's KV row after siblings wrote theirs. */
        for (i = 0; i < tree.n_nodes; i++) {
            if (tree.nodes[i].parent != 0 || tree.nodes[i].token != chain.nodes[1].token)
                continue;
            for (j = 0; j < tree.n_nodes; j++) {
                if (tree.nodes[j].parent == (int32_t)i &&
                    tree.nodes[j].token == chain.nodes[2].token) {
                    cr_assert_float_eq(tree.nodes[j].cond_prob, chain.nodes[2].cond_prob,
                                       1e-5, "prefix %u", p);
                    checked++;
                }
            }
        }
    }
    cr_assert(checked > 0);
    oc_eagle4_model_free(&m);
}

/* q.rows > 6*hidden used to under-size the HF scratch arena. */
Test(eagle4, hf_wide_query_fits_arena)
{
    StSpec s[] = {
        { "embed_tokens.weight", "BF16", {8, 2}, 2, 2, -1 },
        { "fc.weight", "BF16", {2, 2}, 2, 2, -1 },
        { "layers.0.self_attn.q_proj.weight", "BF16", {16, 4}, 2, 2, -1 },
        { "layers.0.self_attn.k_proj.weight", "BF16", {2, 4}, 2, 2, -1 },
        { "layers.0.self_attn.v_proj.weight", "BF16", {2, 4}, 2, 2, -1 },
        { "layers.0.self_attn.o_proj.weight", "BF16", {2, 16}, 2, 2, -1 },
        { "layers.0.mlp.gate_proj.weight", "BF16", {2, 2}, 2, 2, -1 },
        { "layers.0.mlp.up_proj.weight", "BF16", {2, 2}, 2, 2, -1 },
        { "layers.0.mlp.down_proj.weight", "BF16", {2, 2}, 2, 2, -1 },
        { "efficient_lm_head_down_proj.weight", "BF16", {2, 2}, 2, 2, -1 },
        { "efficient_lm_head.weight", "BF16", {8, 2}, 2, 2, -1 },
        { "lm_head.weight", "BF16", {8, 2}, 2, 2, -1 },
        { "norm.weight", "BF16", {2, 0}, 1, 2, -1 },
    };
    OcEagle4HfBench b;
    char path[64];
    snprintf(path, sizeof(path), "/tmp/oc-eagle4-hfq-%d.safetensors", (int)getpid());
    write_st(path, s, 13);
    cr_assert_eq(oc_eagle4_hf_bench(path, 1, 2, &b), OC_OK);
    cr_assert_eq(b.hidden, 2);
    unlink(path);
}

static void write_raw_st(const char *path, const char *json, size_t data_len)
{
    FILE *fp = fopen(path, "wb");
    size_t used = strlen(json), i;
    uint8_t le[8];
    int k;
    cr_assert_not_null(fp);
    for (k = 0; k < 8; k++) le[k] = (uint8_t)(((uint64_t)used >> (8 * k)) & 0xff);
    fwrite(le, 1, 8, fp);
    fwrite(json, 1, used, fp);
    for (i = 0; i < data_len; i++) fputc(0, fp);
    fclose(fp);
}

Test(eagle4, safetensors_rejects_bad_offset_count)
{
    char path[64];
    OcSafetensorsFile st;
    snprintf(path, sizeof(path), "/tmp/oc-eagle4-off-%d.safetensors", (int)getpid());
    write_raw_st(path, "{\"a\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[4]}}", 8);
    memset(&st, 0, sizeof(st));
    cr_assert_eq(oc_safetensors_open(path, &st), OC_ERR_FORMAT);
    write_raw_st(path, "{\"a\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4,8]}}", 8);
    memset(&st, 0, sizeof(st));
    cr_assert_eq(oc_safetensors_open(path, &st), OC_ERR_FORMAT);
    write_raw_st(path, "{\"a\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]}}", 8);
    memset(&st, 0, sizeof(st));
    cr_assert_eq(oc_safetensors_open(path, &st), OC_OK);
    oc_safetensors_close(&st);
    unlink(path);
}

/* Short example payloads and 2-byte token dtypes must be refused. */
Test(eagle4, train_rejects_short_examples)
{
    OcEagle4Config cfg;
    OcEagle4Draft m;
    char path[64];
    StSpec s[] = {
        { "features", "F32", {2, 4}, 2, 4, 8 },
        { "tokens", "U32", {2, 0}, 1, 4, -1 },
        { "targets", "U32", {2, 0}, 1, 4, -1 },
    };
    tiny(&cfg);
    snprintf(path, sizeof(path), "/tmp/oc-eagle4-ex-bad-%d.safetensors", (int)getpid());
    cr_assert_eq(oc_eagle4_model_init(&m, &cfg), OC_OK);
    cr_assert_eq(oc_eagle4_alloc_weight(&m, &m.lm_head, 4, 4), OC_OK);
    write_st(path, s, 3);
    cr_assert_eq(oc_eagle4_train_safetensors(&m, path, 0.1f, 1, NULL), OC_ERR_FORMAT);
    s[0].len = -1;
    s[1].dtype = "U16";
    s[1].esz = 2;
    write_st(path, s, 3);
    cr_assert_eq(oc_eagle4_train_safetensors(&m, path, 0.1f, 1, NULL), OC_ERR_FORMAT);
    oc_eagle4_model_free(&m);
    unlink(path);
}
