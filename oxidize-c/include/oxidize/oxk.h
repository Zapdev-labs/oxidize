/*
 * oxk.h — OXK (Oxidize Kernels) hand-tuned CPU GEMV kernels for the C port.
 *
 * Port of the Rust `oxidize-kernels` crate: bit-exact quantized GEMV row-dot
 * and matvec kernels with scalar / AVX2 / AVX-512 variants. The per-row math
 * is bit-identical to the scalar reference in src/compute/oxk.c (same integer
 * op sequence, same per-block f32 accumulation order) — the SIMD variants
 * only change throughput, never the result. Tests/tests/test_oxk.c asserts
 * exact equality between every variant pair on randomized inputs.
 *
 * Scope (Phase 1):
 *   - Q4_0 × Q8_0, Q4_1 × Q8_0, Q8_0 × Q8_0 dot products
 *   - Q4_K × Q8_K, Q5_K × Q8_K, Q6_K × Q8_K dot products (super-block layout)
 *   - Quantized-weight × f32-input matvec for Q4_0, Q4_K, Q8_0
 *
 * Compile model (mirrors simd.h): every kernel is compiled into every build.
 * The AVX2 kernels live in oxk_avx2.c and are guarded by
 * `__attribute__((target("avx2,fma,f16c")))`; the AVX-512 BW+DQ+VNNI kernels
 * live in oxk_avx512.c and are guarded by
 * `__attribute__((target("avx512bw,avx512dq,avx512vnni")))`. The dispatcher
 * (`oc_oxk_init` → `OcOxkContext`) selects the best variant per host at
 * first use. On hosts without AVX2 the scalar kernels handle every type.
 *
 * Bit-exactness invariants (VAL-OXK-001..006):
 *   - Q4_0/Q8_0: single f32 multiply per element → identical rounding.
 *   - Q4_1/Q4_K/Q5_K/Q6_K: separate vmulps + vaddps/vsubps (NO FMA) so the
 *     two FP roundings match the scalar `a*b ± c` exactly.
 *   - f16 → f32: vcvtph2ps is the canonical f16→f32 conversion (f16 is a
 *     strict subset of f32); bit-identical to the scalar bit-twiddle in
 *     `oc_oxk_f16_le_to_f32`.
 */
#ifndef OXIDIZE_OXK_H
#define OXIDIZE_OXK_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ─── Block-size constants (mirror quant.h; replicated here so OXK is
 * self-contained, matching oxidize-kernels/Cargo.toml having no deps). ──── */

#define OC_OXK_QK4_0      32u
#define OC_OXK_QK4_1      32u
#define OC_OXK_QK8_0      32u
#define OC_OXK_QK_K       256u

#define OC_OXK_BLOCK_Q4_0_SIZE   18u   /* f16 d + 16 packed 4-bit values      */
#define OC_OXK_BLOCK_Q4_1_SIZE   20u   /* f16 d + f16 m + 16 packed 4-bit     */
#define OC_OXK_BLOCK_Q8_0_SIZE   34u   /* f16 d + 32 int8 values              */
#define OC_OXK_BLOCK_Q4_K_SIZE  144u  /* f16 d + f16 dmin + 12 scales + 128 nibbles */
#define OC_OXK_BLOCK_Q5_K_SIZE  176u  /* f16 d + f16 dmin + 12 scales + 128 nibbles + 32qh */
#define OC_OXK_BLOCK_Q6_K_SIZE  210u  /* f16 d + 16 ql_scales + 192 packed (4+4-bit) */
#define OC_OXK_BLOCK_Q8_K_SIZE  292u  /* f32 d + 256 int8 + 16 i16 bsums     */

/* ─── Capability detection + kernel table ────────────────────────────────
 *
 * Detected once via `oc_oxk_init` (which calls `__builtin_cpu_supports` on
 * gcc/clang) and cached in a process-global `OcOxkContext`. The level is
 * monotonic: AVX-512 implies AVX2 implies SCALAR. Callers reach the kernels
 * through the dispatch function pointers in `OcOxkContext`, OR through the
 * direct `oc_oxk_dot_*` / `oc_oxk_matvec_*` entry points (which themselves
 * dispatch through the global context).
 */
typedef enum {
    OC_OXK_SCALAR  = 0,   /* no usable SIMD acceleration                */
    OC_OXK_AVX2    = 1,   /* AVX2 + FMA + F16C (Haswell+, Zen2+)        */
    OC_OXK_AVX512  = 2,   /* AVX-512 BW + DQ + VNNI (Cascade Lake+, Zen4) */
    OC_OXK_NEON    = 3,   /* AArch64 Advanced SIMD — see oxk_neon.h. Not
                           * ordered against the x86 tiers; compare for
                           * equality, never `>=`. */
} OcOxkLevel;

typedef struct OcOxkCaps {
    OcOxkLevel level;
    bool has_f16c;    /* f16 → f32 hardware conversion (needed by AVX2+)   */
    bool has_fma;    /* FMA3 (available on every AVX2 host in practice)   */
    bool has_vnni;   /* AVX-512 VNNI (VPDPBUSD)                           */
    bool has_neon;   /* AArch64 Advanced SIMD                             */
    const char *name; /* "scalar" | "avx2" | "avx512" | "neon"            */
} OcOxkCaps;

/* Function-pointer types for each kernel family. The dot products take a
 * flat byte buffer for the weight row (`row`), a `blocks_per_row` count, and
 * the matching Q8 activation buffer (`q8`); they return the f32 dot. The
 * matvec kernels iterate `n_rows` rows of `row_bytes` each. */
typedef float (*OcOxkDotQ4_0Q8_0_fn)(const uint8_t *row, size_t blocks,
                                     const uint8_t *q8);
typedef float (*OcOxkDotQ4_1Q8_0_fn)(const uint8_t *row, size_t blocks,
                                     const uint8_t *q8);
typedef float (*OcOxkDotQ4_KQ8_K_fn)(const uint8_t *row, size_t blocks,
                                    const uint8_t *q8);
typedef float (*OcOxkDotQ5_KQ8_K_fn)(const uint8_t *row, size_t blocks,
                                    const uint8_t *q8);
typedef float (*OcOxkDotQ6_KQ8_K_fn)(const uint8_t *row, size_t blocks,
                                    const uint8_t *q8);
typedef float (*OcOxkDotQ8_0Q8_0_fn)(const uint8_t *row, size_t blocks,
                                     const uint8_t *q8);

typedef void (*OcOxkMatvecQ4_0F32_fn)(const uint8_t *w, size_t n_rows,
                                     size_t row_bytes, const float *x,
                                     float *out);
typedef void (*OcOxkMatvecQ4_KF32_fn)(const uint8_t *w, size_t n_rows,
                                     size_t row_bytes, const float *x,
                                     float *out);
typedef void (*OcOxkMatvecQ8_0F32_fn)(const uint8_t *w, size_t n_rows,
                                     size_t row_bytes, const float *x,
                                     float *out);

typedef struct OcOxkContext {
    OcOxkCaps caps;
    /* Dot-product dispatch table — one slot per quant pair. */
    OcOxkDotQ4_0Q8_0_fn dot_q4_0_q8_0;
    OcOxkDotQ4_1Q8_0_fn dot_q4_1_q8_0;
    OcOxkDotQ4_KQ8_K_fn dot_q4_k_q8_k;
    OcOxkDotQ5_KQ8_K_fn dot_q5_k_q8_k;
    OcOxkDotQ6_KQ8_K_fn dot_q6_k_q8_k;
    OcOxkDotQ8_0Q8_0_fn dot_q8_0_q8_0;
    /* Matvec dispatch table — one slot per quant type. */
    OcOxkMatvecQ4_0F32_fn matvec_q4_0_f32;
    OcOxkMatvecQ4_KF32_fn matvec_q4_k_f32;
    OcOxkMatvecQ8_0F32_fn matvec_q8_0_f32;
} OcOxkContext;

/* Detect CPU features once and populate the global context's dispatch table
 * with the best kernel for the host. Thread-safe via C11 one-time init.
 * Returns a pointer to the (now-initialized) global context. */
const OcOxkContext *oc_oxk_init(void);

/* Returns the cached capability struct (triggers init on first call). */
const OcOxkCaps *oc_oxk_caps(void);

/* ─── Dispatched entry points ────────────────────────────────────────────
 *
 * These route through the global context's function pointers. They are the
 * preferred production API — callers do not need to know which ISA is active.
 *
 * `row`     : packed quant buffer for one weight row, `blocks_per_row` blocks
 *             laid out back-to-back (`row_bytes = blocks * block_size`).
 * `q8`      : packed Q8 activation buffer, `blocks_per_row` blocks.
 * Returns the f32 dot product of the dequantized weight row and the
 * dequantized Q8 activation row.
 */
float oc_oxk_dot_q4_0_q8_0(const uint8_t *row, size_t blocks_per_row,
                            const uint8_t *q8);
float oc_oxk_dot_q4_1_q8_0(const uint8_t *row, size_t blocks_per_row,
                            const uint8_t *q8);
float oc_oxk_dot_q4_k_q8_k(const uint8_t *row, size_t blocks_per_row,
                           const uint8_t *q8);
float oc_oxk_dot_q5_k_q8_k(const uint8_t *row, size_t blocks_per_row,
                           const uint8_t *q8);
float oc_oxk_dot_q6_k_q8_k(const uint8_t *row, size_t blocks_per_row,
                           const uint8_t *q8);
float oc_oxk_dot_q8_0_q8_0(const uint8_t *row, size_t blocks_per_row,
                            const uint8_t *q8);

/* Quantized weight × f32-input matvec. `w` is `n_rows` rows of `row_bytes`
 * each, laid out back-to-back; `x` is the f32 activation vector of length
 * `row_bytes / block_size * elements_per_block`; `out` receives `n_rows`
 * f32 results. */
void oc_oxk_matvec_q4_0_f32(const uint8_t *w, size_t n_rows, size_t row_bytes,
                            const float *x, float *out);
void oc_oxk_matvec_q4_k_f32(const uint8_t *w, size_t n_rows, size_t row_bytes,
                            const float *x, float *out);
void oc_oxk_matvec_q8_0_f32(const uint8_t *w, size_t n_rows, size_t row_bytes,
                            const float *x, float *out);

/* ─── Scalar reference (always compiled, always available) ──────────────
 *
 * Used by the dispatcher on scalar hosts and by parity tests as the
 * reference. The AVX2 / AVX-512 variants must match these bit-for-bit. */
float oc_oxk_dot_q4_0_q8_0_scalar(const uint8_t *row, size_t blocks_per_row,
                                  const uint8_t *q8);
float oc_oxk_dot_q4_1_q8_0_scalar(const uint8_t *row, size_t blocks_per_row,
                                  const uint8_t *q8);
float oc_oxk_dot_q4_k_q8_k_scalar(const uint8_t *row, size_t blocks_per_row,
                                  const uint8_t *q8);
float oc_oxk_dot_q5_k_q8_k_scalar(const uint8_t *row, size_t blocks_per_row,
                                  const uint8_t *q8);
float oc_oxk_dot_q6_k_q8_k_scalar(const uint8_t *row, size_t blocks_per_row,
                                  const uint8_t *q8);
float oc_oxk_dot_q8_0_q8_0_scalar(const uint8_t *row, size_t blocks_per_row,
                                  const uint8_t *q8);

void oc_oxk_matvec_q4_0_f32_scalar(const uint8_t *w, size_t n_rows,
                                   size_t row_bytes, const float *x, float *out);
void oc_oxk_matvec_q4_k_f32_scalar(const uint8_t *w, size_t n_rows,
                                   size_t row_bytes, const float *x, float *out);
void oc_oxk_matvec_q8_0_f32_scalar(const uint8_t *w, size_t n_rows,
                                   size_t row_bytes, const float *x, float *out);

/* ─── AVX2 direct entry points (for tests/benchmarks) ────────────────────
 *
 * Guarded by `__attribute__((target("avx2,fma,f16c")))` and present in
 * every build; only callable on AVX2 hosts. Tests must check
 * `oc_oxk_caps()->level >= OC_OXK_AVX2` before calling. */
float oc_oxk_dot_q4_0_q8_0_avx2(const uint8_t *row, size_t blocks_per_row,
                                const uint8_t *q8);
float oc_oxk_dot_q4_1_q8_0_avx2(const uint8_t *row, size_t blocks_per_row,
                                const uint8_t *q8);
float oc_oxk_dot_q4_k_q8_k_avx2(const uint8_t *row, size_t blocks_per_row,
                                const uint8_t *q8);
float oc_oxk_dot_q5_k_q8_k_avx2(const uint8_t *row, size_t blocks_per_row,
                                const uint8_t *q8);
float oc_oxk_dot_q6_k_q8_k_avx2(const uint8_t *row, size_t blocks_per_row,
                                const uint8_t *q8);
float oc_oxk_dot_q8_0_q8_0_avx2(const uint8_t *row, size_t blocks_per_row,
                                const uint8_t *q8);
void oc_oxk_matvec_q4_0_f32_avx2(const uint8_t *w, size_t n_rows,
                                 size_t row_bytes, const float *x, float *out);
void oc_oxk_matvec_q4_k_f32_avx2(const uint8_t *w, size_t n_rows,
                                 size_t row_bytes, const float *x, float *out);
void oc_oxk_matvec_q8_0_f32_avx2(const uint8_t *w, size_t n_rows,
                                 size_t row_bytes, const float *x, float *out);

/* ─── AVX-512 BW+DQ+VNNI direct entry points ────────────────────────────
 *
 * Guarded by `__attribute__((target("avx512bw,avx512dq,avx512vnni")))`.
 * VNNI kernels use `_mm512_dpbusd_epi32` for int8 × uint8 multiply-add. */
float oc_oxk_dot_q4_0_q8_0_avx512(const uint8_t *row, size_t blocks_per_row,
                                  const uint8_t *q8);
float oc_oxk_dot_q4_1_q8_0_avx512(const uint8_t *row, size_t blocks_per_row,
                                  const uint8_t *q8);
float oc_oxk_dot_q4_k_q8_k_avx512(const uint8_t *row, size_t blocks_per_row,
                                  const uint8_t *q8);
float oc_oxk_dot_q5_k_q8_k_avx512(const uint8_t *row, size_t blocks_per_row,
                                  const uint8_t *q8);
float oc_oxk_dot_q6_k_q8_k_avx512(const uint8_t *row, size_t blocks_per_row,
                                  const uint8_t *q8);
float oc_oxk_dot_q8_0_q8_0_avx512(const uint8_t *row, size_t blocks_per_row,
                                  const uint8_t *q8);
void oc_oxk_matvec_q4_0_f32_avx512(const uint8_t *w, size_t n_rows,
                                   size_t row_bytes, const float *x, float *out);
void oc_oxk_matvec_q4_k_f32_avx512(const uint8_t *w, size_t n_rows,
                                   size_t row_bytes, const float *x, float *out);
void oc_oxk_matvec_q8_0_f32_avx512(const uint8_t *w, size_t n_rows,
                                   size_t row_bytes, const float *x, float *out);

/* ─── Shared helpers (exposed for tests + reuse by SIMD files) ───────────
 *
 * Bit-exact f16 → f32 conversion (no libm). Mirrors the Rust
 * `f16_le_to_f32` and the scalar bit-twiddle in quantization.c. */
float oc_oxk_f16_le_to_f32(const uint8_t p[2]);

/* Decode (scale, min) for sub-group `j` (0..7) from a Q4_K/Q5_K 12-byte
 * scale field. Identical to llama.cpp's `get_scale_min_k4`. */
void oc_oxk_get_scale_min_k4(unsigned j, const uint8_t scales[12],
                             uint8_t *scale, uint8_t *min);

/* Read the i16 bsum at `index` (0..15) from a Q8_K block's bsums field. */
int16_t oc_oxk_read_q8_k_bsum(const uint8_t *bsums, size_t index);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_OXK_H */
