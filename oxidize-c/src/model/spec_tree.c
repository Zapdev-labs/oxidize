/*
 * spec_tree.c — Tree-based speculative decoding implementation.
 */
#include "oxidize/spec_tree.h"

#include <stdlib.h>
#include <string.h>

/* ─── Helpers ───────────────────────────────────────────────────────── */

static void oc_spec_node_init(OcSpecNode *node, uint32_t token_id,
                              float logprob, int32_t parent_idx,
                              uint32_t depth)
{
    memset(node, 0, sizeof(*node));
    node->token_id   = token_id;
    node->logprob    = logprob;
    node->parent_idx = parent_idx;
    node->depth      = depth;
    node->accepted   = false;
    for (uint32_t i = 0; i < OC_SPEC_TREE_MAX_CHILDREN; i++) {
        node->children_idx[i] = -1;
    }
    node->n_children = 0;
}

/* ─── API ────────────────────────────────────────────────────────────── */

OcError oc_spec_tree_config_init(OcSpecTreeConfig *cfg)
{
    if (!cfg) return OC_ERR_INVALID_ARG;
    cfg->max_depth    = OC_SPEC_TREE_DEFAULT_MAX_DEPTH;
    cfg->max_children = OC_SPEC_TREE_DEFAULT_MAX_CHILDREN;
    cfg->max_nodes    = OC_SPEC_TREE_DEFAULT_MAX_NODES;
    return OC_OK;
}

OcError oc_spec_tree_init(const OcSpecTreeConfig *config, OcSpecTree *out)
{
    if (!out) return OC_ERR_INVALID_ARG;

    OcSpecTreeConfig cfg;
    if (config) {
        cfg = *config;
    } else {
        oc_spec_tree_config_init(&cfg);
    }

    if (cfg.max_nodes == 0) return OC_ERR_INVALID_ARG;
    if (cfg.max_children > OC_SPEC_TREE_MAX_CHILDREN)
        cfg.max_children = OC_SPEC_TREE_MAX_CHILDREN;
    if (cfg.max_children == 0) cfg.max_children = 1;

    memset(out, 0, sizeof(*out));
    out->config    = cfg;
    out->max_nodes = cfg.max_nodes;
    out->nodes = malloc((size_t)cfg.max_nodes * sizeof(OcSpecNode));
    if (!out->nodes) return OC_ERR_OOM;

    out->n_nodes  = 0;
    out->root_idx = -1;
    return OC_OK;
}

void oc_spec_tree_free(OcSpecTree *tree)
{
    if (!tree) return;
    free(tree->nodes);
    memset(tree, 0, sizeof(*tree));
    tree->root_idx = -1;
}

OcError oc_spec_tree_add_root(OcSpecTree *tree, uint32_t token_id,
                              float logprob, uint32_t *out_idx)
{
    if (!tree) return OC_ERR_INVALID_ARG;
    if (tree->root_idx >= 0) return OC_ERR_INVALID_ARG;
    if (tree->n_nodes >= tree->max_nodes) return OC_ERR_OOM;

    uint32_t idx = tree->n_nodes;
    oc_spec_node_init(&tree->nodes[idx], token_id, logprob, -1, 0);
    tree->n_nodes++;
    tree->root_idx = (int32_t)idx;
    if (out_idx) *out_idx = idx;
    return OC_OK;
}

OcError oc_spec_tree_add_child(OcSpecTree *tree, int32_t parent_idx,
                               uint32_t token_id, float logprob,
                               uint32_t *out_idx)
{
    if (!tree) return OC_ERR_INVALID_ARG;
    if (parent_idx < 0 || (uint32_t)parent_idx >= tree->n_nodes)
        return OC_ERR_INVALID_ARG;
    if (tree->n_nodes >= tree->max_nodes) return OC_ERR_OOM;

    OcSpecNode *parent = &tree->nodes[parent_idx];
    if (parent->n_children >= tree->config.max_children)
        return OC_ERR_INVALID_ARG;

    uint32_t new_depth = parent->depth + 1;
    if (new_depth > tree->config.max_depth) return OC_ERR_INVALID_ARG;

    uint32_t idx = tree->n_nodes;
    oc_spec_node_init(&tree->nodes[idx], token_id, logprob,
                      parent_idx, new_depth);

    /* Register the child on the parent (we know there is room). */
    parent->children_idx[parent->n_children] = (int32_t)idx;
    parent->n_children++;

    tree->n_nodes++;
    if (out_idx) *out_idx = idx;
    return OC_OK;
}

const OcSpecNode *oc_spec_tree_get_node(const OcSpecTree *tree, int32_t idx)
{
    if (!tree || idx < 0 || (uint32_t)idx >= tree->n_nodes) return NULL;
    return &tree->nodes[idx];
}

OcError oc_spec_tree_get_path(const OcSpecTree *tree, int32_t node_idx,
                              uint32_t *out_tokens, size_t max_tokens,
                              size_t *out_len)
{
    if (!tree || !out_tokens || !out_len) return OC_ERR_INVALID_ARG;
    *out_len = 0;
    if (node_idx < 0 || (uint32_t)node_idx >= tree->n_nodes)
        return OC_ERR_INVALID_ARG;

    /* Walk up collecting tokens into a small local stack, then reverse. */
    uint32_t tmp[256];
    size_t   count = 0;
    int32_t  cur   = node_idx;
    while (cur >= 0 && count < sizeof(tmp) / sizeof(tmp[0])) {
        tmp[count++] = tree->nodes[cur].token_id;
        cur = tree->nodes[cur].parent_idx;
    }
    if (cur >= 0) {
        /* Path longer than the local stack: caller asked for too much. */
        return OC_ERR_INVALID_ARG;
    }
    if (count > max_tokens) return OC_ERR_INVALID_ARG;

    /* Reverse: root -> node. */
    for (size_t i = 0; i < count; i++) {
        out_tokens[i] = tmp[count - 1 - i];
    }
    *out_len = count;
    return OC_OK;
}

OcError oc_spec_tree_get_leaves(const OcSpecTree *tree,
                                int32_t *out_indices, size_t max_leaves,
                                size_t *out_count)
{
    if (!tree || !out_indices || !out_count) return OC_ERR_INVALID_ARG;
    *out_count = 0;
    size_t count = 0;
    for (uint32_t i = 0; i < tree->n_nodes; i++) {
        if (tree->nodes[i].n_children == 0) {
            if (count >= max_leaves) return OC_ERR_INVALID_ARG;
            out_indices[count++] = (int32_t)i;
        }
    }
    *out_count = count;
    return OC_OK;
}

OcError oc_spec_tree_mark_accepted(OcSpecTree *tree, int32_t node_idx)
{
    if (!tree) return OC_ERR_INVALID_ARG;
    if (node_idx < 0 || (uint32_t)node_idx >= tree->n_nodes)
        return OC_ERR_INVALID_ARG;

    int32_t cur = node_idx;
    while (cur >= 0) {
        tree->nodes[cur].accepted = true;
        cur = tree->nodes[cur].parent_idx;
    }
    return OC_OK;
}

OcError oc_spec_tree_prune(OcSpecTree *tree)
{
    if (!tree) return OC_ERR_INVALID_ARG;
    if (tree->n_nodes == 0) return OC_OK;

    /* A node survives if it (and all its ancestors) is accepted. In
     * tree-based speculative decoding, only fully-accepted paths are kept;
     * a rejected branch is dropped along with its descendants.
     * The root is preserved as a sentinel so the tree is never empty. */
    bool *keep = malloc(tree->n_nodes * sizeof(bool));
    if (!keep) return OC_ERR_OOM;

    for (uint32_t i = 0; i < tree->n_nodes; i++) {
        bool survived = true;
        int32_t cur = (int32_t)i;
        while (cur >= 0) {
            if (!tree->nodes[cur].accepted) { survived = false; break; }
            cur = tree->nodes[cur].parent_idx;
        }
        /* Root always survives so the tree is never empty after prune. */
        if (i == 0 && (uint32_t)tree->root_idx == 0) survived = true;
        keep[i] = survived;
    }

    /* Build the compacted node list and a remap from old idx -> new idx. */
    int32_t *remap = malloc(tree->n_nodes * sizeof(int32_t));
    if (!remap) { free(keep); return OC_ERR_OOM; }
    for (uint32_t i = 0; i < tree->n_nodes; i++) remap[i] = -1;

    OcSpecNode *new_nodes = malloc(tree->max_nodes * sizeof(OcSpecNode));
    if (!new_nodes) { free(keep); free(remap); return OC_ERR_OOM; }

    uint32_t new_count = 0;
    for (uint32_t i = 0; i < tree->n_nodes; i++) {
        if (!keep[i]) continue;
        remap[i] = (int32_t)new_count;
        new_nodes[new_count] = tree->nodes[i];
        new_count++;
    }

    /* Rewrite parent / children indices via the remap. */
    for (uint32_t i = 0; i < new_count; i++) {
        OcSpecNode *n = &new_nodes[i];
        n->parent_idx = (n->parent_idx >= 0) ? remap[n->parent_idx] : -1;
        uint32_t out_c = 0;
        for (uint32_t c = 0; c < n->n_children; c++) {
            int32_t child = n->children_idx[c];
            if (child >= 0 && remap[child] >= 0) {
                n->children_idx[out_c++] = remap[child];
            }
        }
        for (uint32_t c = out_c; c < OC_SPEC_TREE_MAX_CHILDREN; c++) {
            n->children_idx[c] = -1;
        }
        n->n_children = out_c;
    }

    free(tree->nodes);
    tree->nodes   = new_nodes;
    tree->n_nodes = new_count;
    tree->root_idx = (new_count > 0) ? 0 : -1;

    free(keep);
    free(remap);
    return OC_OK;
}

uint32_t oc_spec_tree_size(const OcSpecTree *tree)
{
    return tree ? tree->n_nodes : 0;
}

uint32_t oc_spec_tree_depth(const OcSpecTree *tree)
{
    if (!tree || tree->n_nodes == 0) return 0;
    uint32_t max_d = 0;
    for (uint32_t i = 0; i < tree->n_nodes; i++) {
        if (tree->nodes[i].depth > max_d) max_d = tree->nodes[i].depth;
    }
    return max_d;
}
