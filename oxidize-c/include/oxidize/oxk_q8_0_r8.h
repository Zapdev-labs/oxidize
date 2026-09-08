/*
 * Q8_0_R8 — ik_llama.cpp interleaved Q8_0 (ggml type 208).
 *
 * Eight consecutive Q8_0 rows share one 272-byte superblock (8×34). Byte
 * budget matches row-major Q8_0; only the layout changes.
 */
#ifndef OXIDIZE_OXK_Q8_0_R8_H
#define OXIDIZE_OXK_Q8_0_R8_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define OC_Q8_0_R8_PACK      8u
#define OC_Q8_0_R8_SB_BYTES  272u

void oc_q8_0_r8_repack(const uint8_t *q8_0, size_t nrows, size_t cols,
                       uint8_t *dst);

int oc_q8_0_r8_unpack_to_q8_0_inplace(uint8_t *data, size_t nrows, size_t cols);

#ifdef __cplusplus
}
#endif

#endif
