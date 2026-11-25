/*
 * inkling_gpu.h — private interface between inkling.c and the Vulkan
 * forward in inkling_vk.c. Without OC_VULKAN every entry reports
 * OC_ERR_BACKEND.
 */
#ifndef OXIDIZE_INKLING_GPU_H
#define OXIDIZE_INKLING_GPU_H

#include "oxidize/inkling.h"

#define INK_GPU_MAX_BATCH  1024  /* tokens per GPU forward call    */
#define INK_GPU_MAX_LOGITS 32    /* rows of an all_logits GPU call */

typedef struct InkGpu InkGpu;

bool    ink_gpu_compiled(void);
OcError ink_gpu_model_load(OcInklingModel *m);
void    ink_gpu_model_free(OcInklingModel *m);

OcError ink_gpu_new(const OcInklingModel *m, uint32_t n_ctx, uint32_t n_taps, InkGpu **out);
void    ink_gpu_free(InkGpu *g);

/* One forward of n <= INK_GPU_MAX_BATCH tokens at positions pos0... keep > 0
 * first commits the conv history of the previous call's first keep tokens
 * (calls of more than 32 tokens can only be kept whole).
 * taps receives [n][n_taps][n_embd] for the listed visits. */
OcError ink_gpu_forward(InkGpu *g, const uint32_t *tokens, size_t n, uint32_t pos0, uint32_t keep,
                        float *logits, bool all_logits, const int32_t *tap_visits, uint32_t n_taps,
                        float *taps);

/* Commit the previous call's first keep inputs (as the next forward would)
 * and save rows 0..2 of every visit's conv history and the sliding-window
 * K/V rows before position pos in a slot (< OC_INK_SAVE_SLOTS); restore
 * copies them back. */
OcError ink_gpu_save(InkGpu *g, uint32_t keep, uint32_t slot, uint32_t pos);
OcError ink_gpu_restore(InkGpu *g, uint32_t slot, uint32_t pos);

/* ─── DSpark drafter on the GPU ─────────────────────────────────────────── */

typedef struct InkGpuDraft InkGpuDraft;

/* The session's GPU forward, or NULL when it runs on the CPU. */
InkGpu *ink_session_gpu(const OcInklingSession *s);

/* Drafter weights, K/V caches and scratch on the target session's GPU. */
OcError ink_gpu_draft_new(InkGpu *g, const OcInklingDraft *d, uint32_t n_ctx, InkGpuDraft **out);
void    ink_gpu_draft_free(InkGpuDraft *gd);

/* Context K/V of every drafter layer from the taps of the first n tokens of
 * the session's last forward call, at positions pos0... */
OcError ink_gpu_draft_inject(InkGpuDraft *gd, uint32_t n, uint32_t pos0);

/* Drafter block forward of nb <= 8 inputs h [nb][n_embd] at positions
 * pos0..., then the Markov walk from anchor with context state frnn: the
 * greedy token of each slot 1..nb-1 into tok and its softcapped top
 * probability into ptop. */
OcError ink_gpu_draft_forward(InkGpuDraft *gd, const float *h, uint32_t nb, uint32_t pos0, uint32_t anchor,
                              const float *frnn, uint32_t *tok, float *ptop);

#endif
