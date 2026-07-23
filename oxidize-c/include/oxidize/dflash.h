/*
 * dflash.h — DFlash speculative decoding for C port.
 *
 * Ports the DFlash algorithm from oxidize-core/src/model/dflash.rs.
 * DFlash uses a small draft model to generate candidate continuations
 * that are verified by the target model in a single forward pass.
 */
#ifndef OXIDIZE_DFLASH_H
#define OXIDIZE_DFLASH_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"

#ifdef __cplusplus
extern "C" {
#endif

#define OC_DFLASH_MAX_DRAFT 16
#define OC_DFLASH_MAX_CONTEXT 4096

typedef struct {
    uint32_t max_draft_tokens;
    uint32_t verification_window;
    float acceptance_threshold;
    bool adaptive;
} OcDFlashConfig;

typedef struct {
    OcDFlashConfig config;
    uint32_t draft_tokens[OC_DFLASH_MAX_DRAFT];
    float draft_logprobs[OC_DFLASH_MAX_DRAFT];
    uint32_t target_tokens[OC_DFLASH_MAX_DRAFT];
    float target_logprobs[OC_DFLASH_MAX_DRAFT];
    uint32_t n_draft;
    uint32_t n_accepted;
    uint64_t total_accepted;
    uint64_t total_proposed;
} OcDFlashState;

OcError oc_dflash_config_init(OcDFlashConfig *cfg);
OcError oc_dflash_state_init(OcDFlashState *state, const OcDFlashConfig *cfg);
OcError oc_dflash_set_draft(OcDFlashState *state, const uint32_t *tokens,
                           const float *logprobs, uint32_t n);
OcError oc_dflash_set_target(OcDFlashState *state, const uint32_t *tokens,
                            const float *logprobs, uint32_t n);
OcError oc_dflash_verify(OcDFlashState *state, uint32_t *out_accepted, uint32_t *out_n);
OcError oc_dflash_get_accepted(const OcDFlashState *state, const uint32_t **out_tokens, uint32_t *out_n);
float oc_dflash_acceptance_rate(const OcDFlashState *state);
uint32_t oc_dflash_avg_acceptance(const OcDFlashState *state);
void oc_dflash_state_free(OcDFlashState *state);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_DFLASH_H */
