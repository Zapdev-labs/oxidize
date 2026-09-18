/*
 * prerouter.h — Cross-token MoE routing prediction (Edge0).
 *
 * Each owner layer runs fc1 → erf-GELU → fc2 plus a linear residual on
 * concat(hidden, onehot(curr_topk), onehot(prev_topk)). The prediction for
 * layer N+1 at token t+1 is consumed as routing at decode so SSD reads can
 * overlap the current forward pass.
 */
#ifndef OXIDIZE_PREROUTER_H
#define OXIDIZE_PREROUTER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct OcExpertStreamPool OcExpertStreamPool;
typedef struct OcPrerouter OcPrerouter;

OcError oc_prerouter_new(uint32_t n_layers, uint32_t n_experts,
                         uint32_t hidden_dim, uint32_t top_k,
                         OcPrerouter **out);

/* Load Edge0 `prerouter_*.safetensors` (`layers.N.fc1/fc2/linear_init`). */
OcError oc_prerouter_load_safetensors(OcPrerouter *p, const char *path);

void oc_prerouter_set_replace_routing(OcPrerouter *p, bool replace);
void oc_prerouter_set_stream(OcPrerouter *p, OcExpertStreamPool *stream);
bool oc_prerouter_replace_routing(const OcPrerouter *p);
uint32_t oc_prerouter_n_heads(const OcPrerouter *p);
uint32_t oc_prerouter_top_k(const OcPrerouter *p);

bool oc_prerouter_has_prediction(const OcPrerouter *p, uint32_t layer);

/* Fill `sel[0..k)` and write selected weights into `weights_out[sel[i]]`. */
OcError oc_prerouter_consume(OcPrerouter *p, uint32_t layer,
                             uint32_t *sel, uint32_t k, float *weights_out);

/* After native or predicted routing: record one-hots, run the owner head,
 * prefetch the predicted experts for the next token. */
OcError oc_prerouter_commit(OcPrerouter *p, uint32_t layer,
                            const float *hidden,
                            const uint32_t *sel, uint32_t k);

void oc_prerouter_reset(OcPrerouter *p);
void oc_prerouter_free(OcPrerouter *p);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_PREROUTER_H */
