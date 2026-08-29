#include <criterion/criterion.h>
#include <string.h>

#include "oxidize/spec_tree.h"


Test(spec_tree, config_init_defaults)
{
    OcSpecTreeConfig cfg;
    cr_assert_eq(oc_spec_tree_config_init(&cfg), OC_OK);
    cr_assert_eq(cfg.max_depth, OC_SPEC_TREE_DEFAULT_MAX_DEPTH);
    cr_assert_eq(cfg.max_children, OC_SPEC_TREE_DEFAULT_MAX_CHILDREN);
    cr_assert_eq(cfg.max_nodes, OC_SPEC_TREE_DEFAULT_MAX_NODES);
}

Test(spec_tree, config_init_null)
{
    cr_assert_neq(oc_spec_tree_config_init(NULL), OC_OK);
}

Test(spec_tree, init_free)
{
    OcSpecTreeConfig cfg;
    oc_spec_tree_config_init(&cfg);
    cfg.max_nodes = 8;
    OcSpecTree t;
    cr_assert_eq(oc_spec_tree_init(&cfg, &t), OC_OK);
    cr_assert_eq(t.n_nodes, 0u);
    cr_assert_eq(t.root_idx, -1);
    cr_assert_eq(t.max_nodes, 8u);
    oc_spec_tree_free(&t);
    cr_assert_null(t.nodes);
}

Test(spec_tree, init_default_config)
{
    OcSpecTree t;
    cr_assert_eq(oc_spec_tree_init(NULL, &t), OC_OK);
    cr_assert_eq(t.config.max_depth, OC_SPEC_TREE_DEFAULT_MAX_DEPTH);
    oc_spec_tree_free(&t);
}

Test(spec_tree, init_bad_args)
{
    OcSpecTree t;
    cr_assert_neq(oc_spec_tree_init(NULL, NULL), OC_OK);
    OcSpecTreeConfig cfg;
    oc_spec_tree_config_init(&cfg);
    cfg.max_nodes = 0;
    cr_assert_neq(oc_spec_tree_init(&cfg, &t), OC_OK);
}

Test(spec_tree, free_null_safe)
{
    oc_spec_tree_free(NULL);
}


Test(spec_tree, add_root)
{
    OcSpecTree t;
    cr_assert_eq(oc_spec_tree_init(NULL, &t), OC_OK);
    uint32_t idx = 999u;
    cr_assert_eq(oc_spec_tree_add_root(&t, 42u, -1.5f, &idx), OC_OK);
    cr_assert_eq(idx, 0u);
    cr_assert_eq(t.root_idx, 0);
    cr_assert_eq(t.n_nodes, 1u);
    const OcSpecNode *n = oc_spec_tree_get_node(&t, 0);
    cr_assert_not_null(n);
    cr_assert_eq(n->token_id, 42u);
    cr_assert_float_eq(n->logprob, -1.5f, 1e-9f);
    cr_assert_eq(n->parent_idx, -1);
    cr_assert_eq(n->depth, 0u);
    oc_spec_tree_free(&t);
}

Test(spec_tree, add_root_twice_fails)
{
    OcSpecTree t;
    cr_assert_eq(oc_spec_tree_init(NULL, &t), OC_OK);
    cr_assert_eq(oc_spec_tree_add_root(&t, 1u, 0.0f, NULL), OC_OK);
    cr_assert_neq(oc_spec_tree_add_root(&t, 2u, 0.0f, NULL), OC_OK);
    oc_spec_tree_free(&t);
}

Test(spec_tree, add_child)
{
    OcSpecTree t;
    cr_assert_eq(oc_spec_tree_init(NULL, &t), OC_OK);
    cr_assert_eq(oc_spec_tree_add_root(&t, 1u, 0.0f, NULL), OC_OK);
    uint32_t idx = 0;
    cr_assert_eq(oc_spec_tree_add_child(&t, 0, 2u, -0.5f, &idx), OC_OK);
    cr_assert_eq(idx, 1u);
    cr_assert_eq(t.n_nodes, 2u);
    const OcSpecNode *c = oc_spec_tree_get_node(&t, 1);
    cr_assert_eq(c->parent_idx, 0);
    cr_assert_eq(c->depth, 1u);
    cr_assert_eq(c->token_id, 2u);
    const OcSpecNode *p = oc_spec_tree_get_node(&t, 0);
    cr_assert_eq(p->n_children, 1u);
    cr_assert_eq(p->children_idx[0], 1);
    oc_spec_tree_free(&t);
}

Test(spec_tree, add_child_bad_parent)
{
    OcSpecTree t;
    cr_assert_eq(oc_spec_tree_init(NULL, &t), OC_OK);
    cr_assert_neq(oc_spec_tree_add_child(&t, 0, 1u, 0.0f, NULL), OC_OK);
    cr_assert_eq(oc_spec_tree_add_root(&t, 1u, 0.0f, NULL), OC_OK);
    cr_assert_neq(oc_spec_tree_add_child(&t, 5, 2u, 0.0f, NULL), OC_OK);
    oc_spec_tree_free(&t);
}

Test(spec_tree, max_children_cap)
{
    OcSpecTreeConfig cfg;
    oc_spec_tree_config_init(&cfg);
    cfg.max_children = 2;
    cfg.max_nodes = 16;
    OcSpecTree t;
    cr_assert_eq(oc_spec_tree_init(&cfg, &t), OC_OK);
    cr_assert_eq(oc_spec_tree_add_root(&t, 1u, 0.0f, NULL), OC_OK);
    cr_assert_eq(oc_spec_tree_add_child(&t, 0, 2u, 0.0f, NULL), OC_OK);
    cr_assert_eq(oc_spec_tree_add_child(&t, 0, 3u, 0.0f, NULL), OC_OK);
    cr_assert_neq(oc_spec_tree_add_child(&t, 0, 4u, 0.0f, NULL), OC_OK);
    oc_spec_tree_free(&t);
}

Test(spec_tree, max_depth_cap)
{
    OcSpecTreeConfig cfg;
    oc_spec_tree_config_init(&cfg);
    cfg.max_depth = 1;
    cfg.max_nodes = 16;
    OcSpecTree t;
    cr_assert_eq(oc_spec_tree_init(&cfg, &t), OC_OK);
    cr_assert_eq(oc_spec_tree_add_root(&t, 1u, 0.0f, NULL), OC_OK);
    cr_assert_eq(oc_spec_tree_add_child(&t, 0, 2u, 0.0f, NULL), OC_OK);
    /* depth 2 > max_depth 1 -> fails */
    cr_assert_neq(oc_spec_tree_add_child(&t, 1, 3u, 0.0f, NULL), OC_OK);
    oc_spec_tree_free(&t);
}

Test(spec_tree, max_nodes_oom)
{
    OcSpecTreeConfig cfg;
    oc_spec_tree_config_init(&cfg);
    cfg.max_nodes = 2;
    OcSpecTree t;
    cr_assert_eq(oc_spec_tree_init(&cfg, &t), OC_OK);
    cr_assert_eq(oc_spec_tree_add_root(&t, 1u, 0.0f, NULL), OC_OK);
    cr_assert_eq(oc_spec_tree_add_child(&t, 0, 2u, 0.0f, NULL), OC_OK);
    cr_assert_neq(oc_spec_tree_add_child(&t, 0, 3u, 0.0f, NULL), OC_OK);
    oc_spec_tree_free(&t);
}

Test(spec_tree, get_node_out_of_range)
{
    OcSpecTree t;
    cr_assert_eq(oc_spec_tree_init(NULL, &t), OC_OK);
    cr_assert_null(oc_spec_tree_get_node(&t, 0));
    cr_assert_eq(oc_spec_tree_add_root(&t, 1u, 0.0f, NULL), OC_OK);
    cr_assert_not_null(oc_spec_tree_get_node(&t, 0));
    cr_assert_null(oc_spec_tree_get_node(&t, -1));
    cr_assert_null(oc_spec_tree_get_node(&t, 5));
    oc_spec_tree_free(&t);
}


Test(spec_tree, get_path)
{
    OcSpecTree t;
    cr_assert_eq(oc_spec_tree_init(NULL, &t), OC_OK);
    cr_assert_eq(oc_spec_tree_add_root(&t, 10u, 0.0f, NULL), OC_OK);   /* 0 */
    cr_assert_eq(oc_spec_tree_add_child(&t, 0, 20u, 0.0f, NULL), OC_OK); /* 1 */
    cr_assert_eq(oc_spec_tree_add_child(&t, 1, 30u, 0.0f, NULL), OC_OK); /* 2 */
    uint32_t toks[4] = {0};
    size_t len = 0;
    cr_assert_eq(oc_spec_tree_get_path(&t, 2, toks, 4, &len), OC_OK);
    cr_assert_eq(len, 3u);
    cr_assert_eq(toks[0], 10u);
    cr_assert_eq(toks[1], 20u);
    cr_assert_eq(toks[2], 30u);
    oc_spec_tree_free(&t);
}

Test(spec_tree, get_path_buffer_too_small)
{
    OcSpecTree t;
    cr_assert_eq(oc_spec_tree_init(NULL, &t), OC_OK);
    cr_assert_eq(oc_spec_tree_add_root(&t, 1u, 0.0f, NULL), OC_OK);
    cr_assert_eq(oc_spec_tree_add_child(&t, 0, 2u, 0.0f, NULL), OC_OK);
    uint32_t toks[1] = {0};
    size_t len = 0;
    cr_assert_neq(oc_spec_tree_get_path(&t, 1, toks, 1, &len), OC_OK);
    oc_spec_tree_free(&t);
}

Test(spec_tree, get_leaves)
{
    OcSpecTree t;
    cr_assert_eq(oc_spec_tree_init(NULL, &t), OC_OK);
    /* root -> [1, 2], 1 -> [3]; leaves: 2, 3 */
    cr_assert_eq(oc_spec_tree_add_root(&t, 0u, 0.0f, NULL), OC_OK);
    cr_assert_eq(oc_spec_tree_add_child(&t, 0, 1u, 0.0f, NULL), OC_OK);
    cr_assert_eq(oc_spec_tree_add_child(&t, 0, 2u, 0.0f, NULL), OC_OK);
    cr_assert_eq(oc_spec_tree_add_child(&t, 1, 3u, 0.0f, NULL), OC_OK);
    int32_t leaves[4] = {0};
    size_t cnt = 0;
    cr_assert_eq(oc_spec_tree_get_leaves(&t, leaves, 4, &cnt), OC_OK);
    cr_assert_eq(cnt, 2u);
    /* Node 2 and node 3 should be the leaves. */
    bool has2 = false, has3 = false;
    for (size_t i = 0; i < cnt; i++) {
        if (leaves[i] == 2) has2 = true;
        if (leaves[i] == 3) has3 = true;
    }
    cr_assert(has2 && has3);
    oc_spec_tree_free(&t);
}


Test(spec_tree, mark_accepted)
{
    OcSpecTree t;
    cr_assert_eq(oc_spec_tree_init(NULL, &t), OC_OK);
    cr_assert_eq(oc_spec_tree_add_root(&t, 0u, 0.0f, NULL), OC_OK);
    cr_assert_eq(oc_spec_tree_add_child(&t, 0, 1u, 0.0f, NULL), OC_OK);
    cr_assert_eq(oc_spec_tree_add_child(&t, 1, 2u, 0.0f, NULL), OC_OK);
    cr_assert_eq(oc_spec_tree_mark_accepted(&t, 2), OC_OK);
    cr_assert(oc_spec_tree_get_node(&t, 0)->accepted);
    cr_assert(oc_spec_tree_get_node(&t, 1)->accepted);
    cr_assert(oc_spec_tree_get_node(&t, 2)->accepted);
    oc_spec_tree_free(&t);
}

Test(spec_tree, mark_accepted_bad)
{
    OcSpecTree t;
    cr_assert_eq(oc_spec_tree_init(NULL, &t), OC_OK);
    cr_assert_eq(oc_spec_tree_add_root(&t, 0u, 0.0f, NULL), OC_OK);
    cr_assert_neq(oc_spec_tree_mark_accepted(&t, 5), OC_OK);
    cr_assert_neq(oc_spec_tree_mark_accepted(&t, -1), OC_OK);
    oc_spec_tree_free(&t);
}

Test(spec_tree, prune_keeps_accepted_chain)
{
    OcSpecTree t;
    cr_assert_eq(oc_spec_tree_init(NULL, &t), OC_OK);
    /* root=0 */
    cr_assert_eq(oc_spec_tree_add_root(&t, 0u, 0.0f, NULL), OC_OK);
    cr_assert_eq(oc_spec_tree_add_child(&t, 0, 1u, 0.0f, NULL), OC_OK);
    cr_assert_eq(oc_spec_tree_add_child(&t, 1, 2u, 0.0f, NULL), OC_OK);
    cr_assert_eq(oc_spec_tree_add_child(&t, 0, 3u, 0.0f, NULL), OC_OK);
    /* Accept node 2 -> 0,1,2 accepted. */
    cr_assert_eq(oc_spec_tree_mark_accepted(&t, 2), OC_OK);
    cr_assert_eq(t.n_nodes, 4u);
    cr_assert_eq(oc_spec_tree_prune(&t), OC_OK);
    /* After prune: nodes 0,1,2 remain; node 3 dropped. */
    cr_assert_eq(t.n_nodes, 3u);
    cr_assert_eq(t.root_idx, 0);
    /* Verify token ids preserved. */
    cr_assert_eq(oc_spec_tree_get_node(&t, 0)->token_id, 0u);
    cr_assert_eq(oc_spec_tree_get_node(&t, 1)->token_id, 1u);
    cr_assert_eq(oc_spec_tree_get_node(&t, 2)->token_id, 2u);
    /* Parent links rewritten. */
    cr_assert_eq(oc_spec_tree_get_node(&t, 1)->parent_idx, 0);
    cr_assert_eq(oc_spec_tree_get_node(&t, 2)->parent_idx, 1);
    /* Root has only 1 child now (3 was pruned). */
    cr_assert_eq(oc_spec_tree_get_node(&t, 0)->n_children, 1u);
    oc_spec_tree_free(&t);
}

Test(spec_tree, prune_empty_tree)
{
    OcSpecTree t;
    cr_assert_eq(oc_spec_tree_init(NULL, &t), OC_OK);
    cr_assert_eq(oc_spec_tree_prune(&t), OC_OK);
    cr_assert_eq(t.n_nodes, 0u);
    oc_spec_tree_free(&t);
}

Test(spec_tree, prune_keeps_root_when_none_accepted)
{
    /* If nothing is accepted, root survives as a sentinel. */
    OcSpecTree t;
    cr_assert_eq(oc_spec_tree_init(NULL, &t), OC_OK);
    cr_assert_eq(oc_spec_tree_add_root(&t, 0u, 0.0f, NULL), OC_OK);
    cr_assert_eq(oc_spec_tree_add_child(&t, 0, 1u, 0.0f, NULL), OC_OK);
    cr_assert_eq(oc_spec_tree_prune(&t), OC_OK);
    cr_assert_eq(t.n_nodes, 1u);
    cr_assert_eq(oc_spec_tree_get_node(&t, 0)->token_id, 0u);
    oc_spec_tree_free(&t);
}


Test(spec_tree, size_and_depth)
{
    OcSpecTree t;
    cr_assert_eq(oc_spec_tree_init(NULL, &t), OC_OK);
    cr_assert_eq(oc_spec_tree_size(&t), 0u);
    cr_assert_eq(oc_spec_tree_depth(&t), 0u);
    cr_assert_eq(oc_spec_tree_add_root(&t, 0u, 0.0f, NULL), OC_OK);
    cr_assert_eq(oc_spec_tree_add_child(&t, 0, 1u, 0.0f, NULL), OC_OK);
    cr_assert_eq(oc_spec_tree_add_child(&t, 1, 2u, 0.0f, NULL), OC_OK);
    cr_assert_eq(oc_spec_tree_size(&t), 3u);
    cr_assert_eq(oc_spec_tree_depth(&t), 2u);
    oc_spec_tree_free(&t);
}

Test(spec_tree, size_null_safe)
{
    cr_assert_eq(oc_spec_tree_size(NULL), 0u);
    cr_assert_eq(oc_spec_tree_depth(NULL), 0u);
}
