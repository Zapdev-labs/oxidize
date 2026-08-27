/*
 * moe.h — Mixture-of-Experts routing and expert forward pass.
 *
 * Port of the MoE routing logic used by Mixtral, DeepSeek-V2/V3, Qwen-MoE,
 * and other Mixture-of-Experts architectures. Each MoE layer contains:
 *
 *   1. A gate (router) weight matrix of shape [n_experts, hidden_dim]. The
 *      router computes gate logits = gate @ hidden, applies softmax, and
 *      selects the top-k experts. The selected weights are renormalized to
 *      sum to 1.
 *   2. A set of expert weight matrices (each a standard FFN: gate_proj,
 *      up_proj, down_proj with SwiGLU). Each expert is an independent MLP.
 *   3. A combine step that computes the weighted sum of selected expert
 *      outputs.
 *
 * Routing methods:
 *   - OC_MOE_ROUTE_TOP_K:    softmax over all experts, pick top-k, renormalize
 *   - OC_MOE_ROUTE_TOP_P:    nucleus-style: sort by probability, keep minimal
 *                             set whose cumulative prob >= p, renormalize
 *   - OC_MOE_ROUTE_SOFTMAX:  weighted sum over ALL experts (k = n_experts)
 *
 * Stats tracking: per-expert usage counts and routing entropy (Shannon) are
 * accumulated across calls to oc_moe_route for load-balance diagnostics.
 *
 * JSON format produced by oc_moe_stats_format:
 *   {
 *     "total_tokens": 1234,
 *     "n_experts": 8,
 *     "n_active_experts": 2,
 *     "expert_usage_counts": [100, 98, 95, ...],
 *     "routing_entropy": 2.0134
 *   }
 */
#ifndef OXIDIZE_MOE_H
#define OXIDIZE_MOE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ─── Routing method ───────────────────────────────────────────────────── */
typedef enum {
    OC_MOE_ROUTE_TOP_K    = 0,  /* select top-k experts, renormalize       */
    OC_MOE_ROUTE_TOP_P    = 1,  /* nucleus-style cumulative-prob cutoff     */
    OC_MOE_ROUTE_SOFTMAX  = 2,  /* weighted sum over all experts            */
    OC_MOE_ROUTE__COUNT,
} OcMoeRoutingMethod;

/* ─── Config ──────────────────────────────────────────────────────────── */
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

/* ─── Router state ─────────────────────────────────────────────────────── */
typedef struct OcMoeRouter {
    OcMoeConfig config;

    /* Gate weight matrix: [n_experts × hidden_dim], row-major f32.
     * Owned (malloc'd), freed by oc_moe_router_free. */
    float *gate_weights;

    /* Expert weight matrices, each [expert_size × hidden_dim], row-major f32.
     * For SwiGLU FFN we need gate + up + down; for a single-projection expert
     * (used by tests + simple routing demos) only `expert_weights` is used.
     * All owned (malloc'd), freed by oc_moe_router_free.
     *
     * Expert forward (SwiGLU):  out = down @ (silu(gate @ x) * (up @ x))
     *   gate_weights_e:  [expert_size × hidden_dim]
     *   up_weights_e:    [expert_size × hidden_dim]
     *   down_weights_e: [hidden_dim × expert_size]
     *
     * If up_weights/down_weights are NULL, expert_forward falls back to a
     * single-projection matvec: out = expert_weights @ x  (output length
     * = expert_size). This is used by tests + lightweight demos.
     */
    float *expert_weights;   /* [n_experts * expert_size * hidden_dim]      */
    float *expert_up;        /* [n_experts * expert_size * hidden_dim] or NULL */
    float *expert_down;      /* [n_experts * hidden_dim * expert_size] or NULL */

    /* Stats. */
    uint64_t total_tokens;
    uint64_t *expert_usage_counts;   /* length n_experts                    */
    double   routing_entropy_sum;   /* running sum of per-token entropies  */
} OcMoeRouter;

/* ─── Routing result ───────────────────────────────────────────────────── */
typedef struct OcMoeRouteResult {
    uint32_t  n_selected;            /* actual number of experts selected   */
    uint32_t  expert_indices[64];    /* selected expert indices (capped)     */
    float     expert_weights[64];     /* corresponding normalized weights     */
    float     gate_logits[64];        /* raw gate logits for selected experts */
    float     entropy;                /* Shannon entropy of the routing dist  */
} OcMoeRouteResult;

/* 64 is a generous upper bound; real models use 2-8 active experts. */
#define OC_MOE_MAX_EXPERTS_PER_TOKEN 64

/* ─── Stats ────────────────────────────────────────────────────────────── */
typedef struct OcMoeStats {
    uint64_t  total_tokens;
    uint32_t  n_experts;
    uint32_t  n_active_experts;
    uint64_t *expert_usage_counts;   /* length n_experts (owned)             */
    double    routing_entropy;        /* mean routing entropy                  */
} OcMoeStats;

/* ─── API ──────────────────────────────────────────────────────────────── */

/* Initialize a router with the given config. Allocates gate_weights and
 * expert weight buffers. `config.gate_weights` etc. must be set by the
 * caller via oc_moe_router_set_gate / oc_moe_router_set_expert after init.
 * Returns OC_ERR_INVALID_ARG if `r` is NULL or config is invalid.
 * Returns OC_ERR_OOM on allocation failure. */
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

/* Route a hidden state to top-k experts.
 *
 * Given `hidden` (length == config.hidden_dim), computes gate logits, applies
 * the configured routing method, selects experts, and normalizes weights.
 * The result is written to `out` (expert_indices, expert_weights, gate_logits,
 * entropy). Stats (usage counts, entropy sum) are updated.
 *
 * `temp_logits` is a caller-provided scratch buffer of length >= n_experts
 * (used to avoid internal malloc). May be NULL (router will malloc/free
 * internally).
 *
 * Returns OC_ERR_INVALID_ARG on NULL args or invalid config.
 * Returns OC_OK on success. */
OcError oc_moe_route(OcMoeRouter *r,
                     const float *hidden,
                     OcMoeRouteResult *out,
                     float *temp_logits);

/* Compute a single expert's output for the given input.
 *
 * Expert `expert_idx` forward pass on input `x` (length == hidden_dim):
 *   - If up_weights and down_weights are both set (SwiGLU):
 *       gate = silu(gate_proj @ x)   [length expert_size]
 *       up   = up_proj @ x            [length expert_size]
 *       out  = down_proj @ (gate * up) [length hidden_dim]
 *   - If only expert_weights is set (single projection):
 *       out  = expert_weights @ x    [length expert_size]
 *
 * `out` must be of length `*out_len` (SwiGLU: hidden_dim; single: expert_size).
 * On return, `*out_len` is set to the actual output length.
 * `temp` is a scratch buffer of length >= expert_size (for the SwiGLU gate/up
 * intermediates). May be NULL for the single-projection path.
 *
 * Returns OC_ERR_INVALID_ARG on NULL/invalid args.
 * Returns OC_ERR_MODEL if expert_idx >= n_experts.
 * Returns OC_OK on success. */
OcError oc_moe_expert_forward(const OcMoeRouter *r,
                              uint32_t expert_idx,
                              const float *x,
                              float *out, size_t *out_len,
                              float *temp);

/* Combine expert outputs using routing weights.
 *
 * Given an array of expert outputs (`expert_outs`, `n_selected` entries, each
 * of length `out_len`) and the routing weights from OcMoeRouteResult, computes
 * the weighted sum into `combined` (length `out_len`).
 *
 *   combined[j] = sum_i (result->expert_weights[i] * expert_outs[i][j])
 *
 * Returns OC_ERR_INVALID_ARG on NULL/invalid args.
 * Returns OC_OK on success. */
OcError oc_moe_combine(const OcMoeRouteResult *result,
                       const float *const *expert_outs,
                       size_t n_selected,
                       size_t out_len,
                       float *combined);

/* Get a snapshot of the router's stats. Copies into `stats` (which takes
 * ownership of a freshly-malloc'd `expert_usage_counts` array — caller must
 * free `stats->expert_usage_counts`). Returns OC_ERR_INVALID_ARG on NULL.
 * Returns OC_ERR_OOM on allocation failure. */
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
