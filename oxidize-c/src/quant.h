/* Dequant + quantized-row dot kernels. Ported from oxidize-cpp/src/quant.cpp
 * and tensor_cpu.cpp (scalar reference paths), plus the AL5_XS custom type. */
#ifndef OC_QUANT_H
#define OC_QUANT_H

#include <stddef.h>
#include <stdint.h>

/* ggml type ids we handle (raw ids straight from GGUF). Layouts follow
 * ggml-quants.c / llama.cpp, NOT oxidize-core's Rust dequant: the Rust simple
 * quants (Q4_1/Q5_0/Q5_1) use an interleaved 2i/2i+1 nibble order and its Q3_K
 * main loop is buggy (wrong shift, qs never advanced). ggml — and the existing
 * oxidize-C Q4_0 — use the split j/j+16 order; that is what is implemented. */
enum {
  OC_F32 = 0,
  OC_F16 = 1,
  OC_Q4_0 = 2,
  OC_Q4_1 = 3,
  OC_Q5_0 = 6,
  OC_Q5_1 = 7,
  OC_Q8_0 = 8,
  OC_Q2_K = 10,
  OC_Q3_K = 11,
  OC_Q4_K = 12,
  OC_Q5_K = 13,
  OC_Q6_K = 14,
  /* IQ grid-codebook quants (ids straight from ggml.h; NOT the oxidize-core
   * Rust enum, which mis-numbers IQ2_S/IQ3_S and gives several wrong block
   * sizes — GGUF stores these ggml ids on disk). Scalar dequant only. */
  OC_IQ2_XXS = 16,
  OC_IQ2_XS = 17,
  OC_IQ3_XXS = 18,
  OC_IQ1_S = 19,
  OC_IQ3_S = 21,
  OC_IQ2_S = 22,
  OC_IQ4_XS = 23,
  OC_IQ1_M = 29,
  OC_BF16 = 30,
  /* AL custom family (240-243). Only AL5_XS (243) is decodable for now;
   * 240-242 parse (accepted by gguf.c) but have no kernels yet. */
  OC_AL5_XS = 243,
};

#define OC_QK 32       /* values per block for Q4_0/Q8_0/AL5_XS/Q4_1/Q5_0/Q5_1 */
#define OC_QK_K 256    /* values per block for K-quants and IQ4_XS */
#define OC_BLK_Q4_0 18 /* f16 d + 16 nibble bytes */
#define OC_BLK_Q4_1 20 /* f16 d + f16 m + 16 nibble bytes */
#define OC_BLK_Q5_0 22 /* f16 d + 4-byte qh + 16 nibble bytes */
#define OC_BLK_Q5_1 24 /* f16 d + f16 m + 4-byte qh + 16 nibble bytes */
#define OC_BLK_Q8_0 34 /* f16 d + 32 int8 */
#define OC_BLK_Q2_K 84  /* 16 scale bytes + 64 qs + f16 d + f16 dmin */
#define OC_BLK_Q3_K 110 /* 32 hmask + 64 qs + 12 packed scales + f16 d */
#define OC_BLK_Q4_K 144
#define OC_BLK_Q5_K 176
#define OC_BLK_Q6_K 210
#define OC_BLK_IQ4_XS 136 /* f16 d + u16 scales_h + 4 scales_l + 128 qs */
/* IQ grid quants, all QK_K=256 values/block (exact ggml block sizes). */
#define OC_BLK_IQ2_XXS 66  /* f16 d + 32 u16 qs */
#define OC_BLK_IQ2_XS 74   /* f16 d + 32 u16 qs + 8 scales */
#define OC_BLK_IQ2_S 82    /* f16 d + 64 qs + 8 qh + 8 scales */
#define OC_BLK_IQ3_XXS 98  /* f16 d + 64 qs + 32 scales/signs */
#define OC_BLK_IQ3_S 110   /* f16 d + 64 qs + 8 qh + 32 signs + 4 scales */
#define OC_BLK_IQ1_S 50    /* f16 d + 32 qs + 8 u16 qh */
#define OC_BLK_IQ1_M 56    /* 32 qs + 16 qh + 8 scales (scale packed in scales) */
#define OC_BLK_AL5_XS 14 /* f16 scale + 12 bytes of 3-bit codes (32 codes) */

/* ---- ISA selection ----
 * Every kernel family is compiled into the binary; which one oc_dot_row /
 * oc_dot_row_q8 / oc_q8_quantize actually call is a table of function pointers
 * resolved once, before main(), from cpuid. So a binary built anywhere runs
 * everywhere and still uses the widest ISA the host CPU (and its OS, via
 * XCR0) actually supports.
 *
 * OC_ISA_AVX512 additionally enables the VNNI int8 path, but only when the CPU
 * has AVX512VNNI — a Skylake-X gets the AVX-512 float kernels and the scalar
 * int8 path. oc_isa_active_name() spells out exactly what was bound.
 *
 * The OC_ISA env var (scalar|avx2|avx512|neon) forces a family at startup;
 * forcing one the CPU does not have clamps down to the best it does. neon is
 * only meaningful on aarch64, avx2/avx512 only on x86. Set once at startup or
 * from a single-threaded test; not synchronized. */
typedef enum {
  OC_ISA_SCALAR = 0,
  OC_ISA_AVX2 = 1,
  OC_ISA_AVX512 = 2,
  OC_ISA_NEON = 3, /* aarch64 ASIMD; a parallel tier, never live in the same
                    * binary as AVX2/AVX512 (arch is fixed at compile time) */
  OC_ISA_AUTO = 99,
} OcIsa;
void oc_force_isa(OcIsa isa);
OcIsa oc_isa_available(void); /* best family this CPU + this build support */
OcIsa oc_isa_active(void);    /* family currently bound in the table */
const char* oc_isa_active_name(void); /* e.g. "avx512+vnni", "avx2", "scalar" */
/* Test hook: identity of the kernel currently bound for `ggml_type`. Only
 * useful for asserting that forcing an ISA actually rebound the table. */
const void* oc_dot_impl(uint32_t ggml_type);

float oc_f16_to_f32(uint16_t bits);
uint16_t oc_f32_to_f16(float f);

/* AL5_XS value mapping. The exact mapping is being verified empirically and
 * may be adjusted — keep every use routed through this one function. */
float al5xs_lut(unsigned code, float scale);
/* AL5_XS bit unpacking: LSB-first, code i occupies bits [3i, 3i+3) of the
 * 12-byte little-endian bitstream. Isolated so the order can be flipped. */
void al5xs_unpack(const uint8_t qs[12], uint8_t codes[32]);
void al5xs_pack(const uint8_t codes[32], uint8_t qs[12]); /* for tests */
/* Encode 32 floats into one 14-byte AL5_XS block (MSE-refined scale). */
void oc_al5xs_encode_block(const float w[32], uint8_t out[14]);

/* Bytes per row of `cols` values for a ggml type; 0 if unsupported. */
size_t oc_row_bytes(uint32_t ggml_type, size_t cols);

/* Dequantize one row of n values. Returns 0, or -1 for unsupported type.
 * ISA-dispatched for the types oc_matmul leans on (it unpacks every weight in
 * the model once per batch, and cannot fuse that into the FMA the way the dot
 * kernels below do). Other types fall through to the scalar decoder. */
int oc_dequant_row(uint32_t ggml_type, const uint8_t* src, float* dst, size_t n);
const void* oc_dequant_impl(uint32_t ggml_type); /* test hook; NULL = scalar */

/* dot(dequant(row), x) over `cols` values. Row must be a whole row pointer. */
float oc_dot_row(uint32_t ggml_type, const uint8_t* row, const float* x, size_t cols);

/* ---- GEMM inner kernels ----
 * Rank-kb update of the token accumulators for R weight rows at once:
 *
 *   acc[i*n + t] += sum_k w[i*wstride + k] * xp[k*n + t]   i < R, t < n
 *
 * `w` holds R dequantized weight row-blocks (kb values each) and `xp` is the
 * activation panel packed k-major, so all n token accumulators stay in
 * registers for the whole kb loop and each weight block is touched once. That
 * reuse is the entire point of oc_matmul — it is what turns a memory-bound GEMV
 * into a compute-bound GEMM. They live here, not in tensor.c, because the ISA
 * dispatch table does.
 *
 * PREFER row4; row1 exists only for the 1-3 rows left at the end of a tile.
 * Four rows is not arbitrary. One row keeps 2 accumulators live, and with a
 * 4-cycle FMA latency and 2 FMA units a chain of 2 sustains 0.5 FMA/cycle of
 * the 2 the core can retire. Eight independent chains (4 rows x 2 vectors) is
 * what covers the latency; it also amortizes each panel load over 4 broadcasts
 * instead of 1. Measured on a Zen3+ at gemma-4-31B shapes: 1 row 275 GFLOP/s,
 * 2 rows 335, 4 rows — see `make gemm-bench`.
 *
 * PRECONDITION: n is a multiple of 16 (oc_matmul zero-pads the panel to it), so
 * the kernels have no token tail — the pad lanes multiply by zero and are
 * dropped when the accumulator is written out. */
void oc_gemm_row(float* acc, const float* w, const float* xp, size_t kb, size_t n);
void oc_gemm_row4(float* acc, const float* w, size_t wstride, const float* xp,
                  size_t kb, size_t n);
const void* oc_gemm_impl(void); /* test hook: which kernel is bound */

void oc_quantize_row_q4_0(const float* x, uint8_t* out, size_t n); /* for tests */

/* ---- int8 activation path (AVX512-VNNI) ----
 * Activations quantized once per matvec into 256-value blocks: int8 codes,
 * one f32 scale per block, and per-16 int32 sums (for K-quant min/offset
 * terms). oc_dot_row_q8 is a drop-in for oc_dot_row on supported types. */
typedef struct {
  const int8_t* q;     /* [n] */
  const float* d;      /* [n/256] block scales */
  const int32_t* bsum; /* [n/16] per-16 sums of q */
} OcQ8Act;
int oc_q8_dot_supported(uint32_t ggml_type); /* 1 if VNNI path exists */
void oc_q8_quantize(const float* x, size_t n, int8_t* q, float* d, int32_t* bsum);
float oc_dot_row_q8(uint32_t ggml_type, const uint8_t* row, const OcQ8Act* a,
                    size_t cols);

/* ---- rotoquant KV cache (TurboQuant-style rotated 4-bit) ----
 * Normalized in-place fast Walsh-Hadamard transform; self-inverse and
 * orthonormal, so dot(FHT(a), FHT(b)) == dot(a, b). n must be a power of 2. */
void oc_fht(float* v, size_t n);
/* Asymmetric 4-bit row codec: n values -> n/2 packed bytes (low nibble first)
 * + meta[0]=scale, meta[1]=min; x ~= min + scale*q. n must be even. */
void oc_kvq_encode(const float* x, size_t n, uint8_t* out, float* meta);
void oc_kvq_decode(const uint8_t* in, size_t n, const float* meta, float* x);

#endif
