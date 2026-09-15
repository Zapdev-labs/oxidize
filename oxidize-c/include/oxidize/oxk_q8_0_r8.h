/*
 * Q8_0_R8 — ik_llama.cpp interleaved Q8_0 (ggml type 208).
 *
 * Eight consecutive Q8_0 rows share one 272-byte superblock (8x34). The byte
 * budget matches row-major Q8_0; only the layout changes.
 *
 * Superblock layout (one superblock covers rows 8g..8g+7, columns
 * 32*ib..32*ib+31; `k` is the row within the group, 0..7):
 *
 *   bytes [0, 16)    8 scales, 2 opaque bytes each (an f16, little-endian on
 *                    disk). Row k's scale lives at byte offset 2*k.
 *   bytes [16, 272)  256 int8 quants, addressed as qs[0..255] with
 *                    qs = superblock + 16. For l in 0..3 and i in 0..3:
 *                        qs[32*l + 4*k + i]       = row k, element (4*l + i)
 *                        qs[32*l + 4*k + i + 128] = row k, element (4*l+i+16)
 *
 * Superblocks are laid out block-major within a group of 8 rows: superblock
 * `ib` of group `g` starts at byte (g * nblock + ib) * 272, where
 * nblock = cols / 32.
 *
 * NOTE — this is an OFFLINE repack format only. `OC_QUANT_Q8_0_R8` has no
 * runtime decoder: `oc_quant_block_size()` reports a zero layout for it and
 * `oc_quant_is_runtime_supported()` returns false, so a GGUF that stores
 * weights as ggml type 208 is rejected at load time rather than silently
 * decoded through the row-major Q8_0 path (which would read the interleaved
 * bytes as if they were row-major and produce garbage). Use
 * `oc_q8_0_r8_unpack_to_q8_0_inplace()` to convert such a tensor back to
 * row-major Q8_0 before feeding it to the inference path.
 *
 * Buffer sizing: a repacked tensor occupies exactly the same number of bytes
 * as the row-major Q8_0 form, i.e. `nrows * oc_quantized_size(OC_QUANT_Q8_0,
 * cols)` bytes. (`oc_quantized_size(OC_QUANT_Q8_0_R8, ...)` deliberately
 * returns 0 — see the note above.)
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

/* Repack `nrows * (cols/32)` row-major Q8_0 blocks from `q8_0` into the
 * interleaved Q8_0_R8 layout in `dst`. Both buffers must hold at least
 * `nrows * (cols/32) * 34` bytes and must not overlap.
 *
 * Returns 0 on success, -1 if `q8_0` or `dst` is NULL, if `cols` is not a
 * multiple of 32, if `nrows` is not a multiple of 8, or if the derived byte
 * span would overflow `size_t`. `cols == 0` or `nrows == 0` is valid and
 * succeeds without writing anything. */
int oc_q8_0_r8_repack(const uint8_t *q8_0, size_t nrows, size_t cols,
                      uint8_t *dst);

/* Convert `data` from the interleaved Q8_0_R8 layout back to row-major Q8_0,
 * in place. `data` must hold at least `nrows * (cols/32) * 34` bytes.
 *
 * Returns 0 on success, -1 if `data` is NULL, if `cols` is not a multiple of
 * 32, if `nrows` is not a multiple of 8, if the derived byte span would
 * overflow `size_t`, or on scratch-allocation failure. `cols == 0` or
 * `nrows == 0` is valid and succeeds without doing any work. */
int oc_q8_0_r8_unpack_to_q8_0_inplace(uint8_t *data, size_t nrows, size_t cols);

#ifdef __cplusplus
}
#endif

#endif
