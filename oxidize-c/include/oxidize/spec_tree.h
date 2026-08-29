/* spec_tree.h — Tree-based speculative decoding support. */
#ifndef OXIDIZE_SPEC_TREE_H
#define OXIDIZE_SPEC_TREE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"

#ifdef __cplusplus
extern "C" {
#endif


#define OC_SPEC_TREE_MAX_CHILDREN 8
#define OC_SPEC_TREE_DEFAULT_MAX_DEPTH    4
#define OC_SPEC_TREE_DEFAULT_MAX_CHILDREN 8
#define OC_SPEC_TREE_DEFAULT_MAX_NODES    64


typedef struct OcSpecNode {
    uint32_t token_id;
    float    logprob;
    int32_t  parent_idx;                    /* -1 for root                 */
    int32_t  children_idx[OC_SPEC_TREE_MAX_CHILDREN];
    uint32_t n_children;
    uint32_t depth;
    bool     accepted;
} OcSpecNode;

typedef struct OcSpecTreeConfig {
    uint32_t max_depth;     /* default 4                                    */
    uint32_t max_children;  /* default 8 (capped at OC_SPEC_TREE_MAX_CHILDREN) */
    uint32_t max_nodes;     /* default 64                                   */
} OcSpecTreeConfig;

typedef struct OcSpecTree {
    OcSpecNode *nodes;
    uint32_t    max_nodes;
    uint32_t    n_nodes;
    int32_t     root_idx;   /* -1 when empty                                */
    OcSpecTreeConfig config;
} OcSpecTree;


/* Initialize config with defaults. */
OcError oc_spec_tree_config_init(OcSpecTreeConfig *cfg);

/* Allocate a tree. Returns an empty tree (no root). */
OcError oc_spec_tree_init(const OcSpecTreeConfig *config, OcSpecTree *out);

/* Free all owned storage and reset state. Safe on already-freed tree. */
void oc_spec_tree_free(OcSpecTree *tree);

/* Add the root node. Returns its index (0) via out_idx if non-NULL.
 * Fails if the tree already has a root. */
OcError oc_spec_tree_add_root(OcSpecTree *tree, uint32_t token_id,
                              float logprob, uint32_t *out_idx);

/* Add a child node under `parent_idx`. */
OcError oc_spec_tree_add_child(OcSpecTree *tree, int32_t parent_idx,
                               uint32_t token_id, float logprob,
                               uint32_t *out_idx);

/* Get a pointer to the node at `idx`, or NULL if out of range. */
const OcSpecNode *oc_spec_tree_get_node(const OcSpecTree *tree, int32_t idx);

/* Collect the path of token ids from root down to `node_idx` (inclusive).
 * Writes into `out_tokens` (capacity `max_tokens`). Sets `*out_len` to the
 * number of tokens written. */
OcError oc_spec_tree_get_path(const OcSpecTree *tree, int32_t node_idx,
                              uint32_t *out_tokens, size_t max_tokens,
                              size_t *out_len);

/* Collect indices of all leaf nodes (nodes with n_children == 0).
 * Writes into `out_indices` (capacity `max_leaves`). Sets `*out_count`. */
OcError oc_spec_tree_get_leaves(const OcSpecTree *tree,
                                int32_t *out_indices, size_t max_leaves,
                                size_t *out_count);

/* Mark `node_idx` and all of its ancestors as accepted. */
OcError oc_spec_tree_mark_accepted(OcSpecTree *tree, int32_t node_idx);

/* Prune rejected branches. Removes any node that is not accepted AND has no
 * accepted ancestor. Compacts the node array and rewrites parent/child
 * indices. Root is preserved if accepted or has any accepted descendant. */
OcError oc_spec_tree_prune(OcSpecTree *tree);

/* Number of currently-stored nodes. */
uint32_t oc_spec_tree_size(const OcSpecTree *tree);

/* Maximum depth among all nodes (0 if empty). */
uint32_t oc_spec_tree_depth(const OcSpecTree *tree);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_SPEC_TREE_H */
