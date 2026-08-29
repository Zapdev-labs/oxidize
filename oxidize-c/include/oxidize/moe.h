#ifndef OXIDIZE_MOE_H
#define OXIDIZE_MOE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    OC_MOE_ROUTE_TOP_K    = 0,  /* select top-k experts, renormalize       */
    OC_MOE_ROUTE_TOP_P    = 1,  /* nucleus-style cumulative-prob cutoff     */
    OC_MOE_ROUTE_SOFTMAX  = 2,  /* weighted sum over all experts            */
    OC_MOE_ROUTE__COUNT,
} OcMoeRoutingMethod;

typedef struct OcMoeConfig {
    uint32_t n_experts;            /* total number of experts               */
    uint32_t n_active_experts;     /* top-k experts per token (0 → 1)        */
    uint32_t expert_size;          /* expert FFN intermediate size            */
    uint32_t hidden_dim;           /* input/output hidden dimension           */
    OcMoeRoutingMethod routing_method;
    float    top_p;                /* cumulative probability cutoff (TOP_P)   */
    float    router_z_loss;        /* z-loss regularizer weight (0 = off)     */
    float    router_aux_loss;      /* aux load-balance loss weight (0 = off)   */
    bool     normalize_weights;    /* renormalize selected weights to sum 1   */
} OcMoeConfig;

typedef struct OcMoeRouter {
    OcMoeConfig config;

    /* Gate weight matrix: [n_experts × hidden_dim], row-major f32.
     * Owned (malloc'd), freed by oc_moe_router_free. */
    float *gate_weights;

    /* Expert weight matrices, each [expert_size × hidden_dim], row-major f32. */
    float *expert_weights;   /* [n_experts * expert_size * hidden_dim]      */
    float *expert_up;        /* [n_experts * expert_size * hidden_dim] or NULL */
    float *expert_down;      /* [n_experts * hidden_dim * expert_size] or NULL */

    /* Stats. */
    uint64_t total_tokens;
    uint64_t *expert_usage_counts;   /* length n_experts                    */
    double   routing_entropy_sum;   /* running sum of per-token entropies  */
} OcMoeRouter;

typedef struct OcMoeRouteResult {
    uint32_t  n_selected;            /* actual number of experts selected   */
    uint32_t  expert_indices[64];    /* selected expert indices (capped)     */
    float     expert_weights[64];     /* corresponding normalized weights     */
    float     gate_logits[64];        /* raw gate logits for selected experts */
    float     entropy;                /* Shannon entropy of the routing dist  */
} OcMoeRouteResult;

/* 64 is a generous upper bound; real models use 2-8 active experts. */
#define OC_MOE_MAX_EXPERTS_PER_TOKEN 64

typedef struct OcMoeStats {
    uint64_t  total_tokens;
    uint32_t  n_experts;
    uint32_t  n_active_experts;
    uint64_t *expert_usage_counts;   /* length n_experts (owned)             */
    double    routing_entropy;        /* mean routing entropy                  */
} OcMoeStats;


/* Initialize a router with the given config. */
OcError oc_moe_router_init(OcMoeRouter *r, const OcMoeConfig *config);

/* Set the gate weight matrix (copies `n_experts * hidden_dim` floats from
 * `weights` into the router's owned buffer). Pass NULL to zero-init. */
OcError oc_moe_router_set_gate(OcMoeRouter *r, const float *weights);

/* Set expert weight matrices. Each pointer may be NULL (expert_forward
 * will use the available matrices). Copies n_experts * rows * cols floats. */
OcError oc_moe_router_set_experts(OcMoeRouter *r,
                                  const float *gate_proj,
                                  const float *up_proj,
                                  const float *down_proj);

/* Free a router and all owned buffers. Safe on NULL. */
void oc_moe_router_free(OcMoeRouter *r);

/* Route a hidden state to top-k experts. */
OcError oc_moe_route(OcMoeRouter *r,
                     const float *hidden,
                     OcMoeRouteResult *out,
                     float *temp_logits);

/* Compute a single expert's output for the given input. */
OcError oc_moe_expert_forward(const OcMoeRouter *r,
                              uint32_t expert_idx,
                              const float *x,
                              float *out, size_t *out_len,
                              float *temp);

/* Combine expert outputs using routing weights. */
OcError oc_moe_combine(const OcMoeRouteResult *result,
                       const float *const *expert_outs,
                       size_t n_selected,
                       size_t out_len,
                       float *combined);

/* Get a snapshot of the router's stats. Copies into `stats` (which takes */
OcError oc_moe_get_stats(const OcMoeRouter *r, OcMoeStats *stats);

/* Free an OcMoeStats snapshot (frees expert_usage_counts). Safe on NULL. */
void oc_moe_stats_free(OcMoeStats *stats);

/* Format stats as JSON into `buf` (up to `cap-1` chars, NUL-terminated).
 * Returns the number of bytes written (excluding NUL). If `buf` is NULL or
 * cap==0, returns the length that would have been written. */
size_t oc_moe_stats_format(const OcMoeStats *stats, char *buf, size_t cap);

/* Human-readable name for a routing method. Never NULL. */
const char *oc_moe_routing_method_name(OcMoeRoutingMethod method);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_MOE_H */
