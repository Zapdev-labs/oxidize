/*
 * cuda_mmq.cu — device-resident quantized matvec kernels.
 *
 * See include/oxidize/cuda_mmq.h for why this exists. In short: weights stay
 * in their packed GGUF form in device memory and are dequantized in registers
 * inside the dot product, instead of being expanded to f32 on the host.
 *
 * Each matvec kernel assigns one WARP per output row, MQ_WARPS rows per block.
 * Lanes stride over the row's 32-element groups, accumulate partial dot
 * products, and combine with warp shuffles — no shared memory, no barriers.
 * No thread ever materializes a whole super-block, which keeps register
 * pressure flat (the pre-existing k_q4k_matvec in cuda_kernels.cu declared
 * `float vals[256]` per thread — 1 KiB of local-memory spill each — and is
 * superseded by this file).
 *
 * Block layouts are bit-compatible ports of src/compute/quantization.c:
 *   Q4_K  144 B / 256 vals : d:f16, min:f16, 12 B scale table, 128 B nibbles
 *   Q6_K  210 B / 256 vals : 128 B ql, 64 B qh, 16 B int8 scales, d:f16
 *   Q8_0   34 B /  32 vals : d:f16, 32 int8
 */
#include "oxidize/cuda_mmq.h"
#include "oxidize/quant.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

/* Must match include/oxidize/quant.h. */
#define MQ_QK_K            256u
#define MQ_QK8_0            32u
#define MQ_BLOCK_Q4_K_SIZE  144u
#define MQ_BLOCK_Q5_K_SIZE  176u
#define MQ_BLOCK_Q6_K_SIZE  210u
#define MQ_BLOCK_Q8_0_SIZE   34u
#define MQ_BLOCK_Q4_0_SIZE   18u
#define MQ_BLOCK_Q5_0_SIZE   22u
#define MQ_BLOCK_AL5_XS_SIZE 14u
#define MQ_BLOCK_IQ4_XS_SIZE 136u

/* Device-side block strides. These may exceed the on-disk stride to buy
 * alignment: Q6_K's 210 bytes is only 2-aligned, which forbids 16-byte vector
 * loads, so on the device it is padded to 224 (= 14*16). Every ql/qh span
 * inside a block is then 16-aligned too. Cost is 6.7% extra VRAM on Q6_K
 * tensors, which buys vector loads on ~a quarter of a Q4_K_M model's weights.
 * Q4_K's 144 (= 9*16) and Q8_0's 34 need no padding — 144 is already aligned,
 * and Q8_0 keeps scalar loads. */
#define MQ_DEV_Q4_K_SIZE   144u
/* Q5_K's 176 = 11*16 is already 16-aligned; no padding needed. */
#define MQ_DEV_Q5_K_SIZE   176u
#define MQ_DEV_Q6_K_SIZE   224u
#define MQ_DEV_Q8_0_SIZE    34u
#define MQ_DEV_Q4_0_SIZE    18u
#define MQ_DEV_Q5_0_SIZE    22u
#define MQ_DEV_AL5_XS_SIZE  14u
#define MQ_DEV_IQ4_XS_SIZE 136u

#define MQ_BLOCK_THREADS   256u   /* threads per block on the get_row path */

/* ─── Device helpers ──────────────────────────────────────────────────────── */

/* f16 (little-endian pair) → f32.
 *
 * Semantically identical to quantization.c::f16_le_to_f32, but a single
 * hardware conversion instead of the scalar bit-twiddling that function needs
 * on the host. This runs once per 32-element group per lane on the hottest
 * path in the model, and the portable version's denormal branch is a
 * data-dependent `while` loop — the last thing you want inside a warp. */
__device__ __forceinline__ float mq_f16(uint8_t b0, uint8_t b1)
{
    const unsigned short bits =
        (unsigned short)((unsigned)b0 | ((unsigned)b1 << 8));
    return __half2float(__ushort_as_half(bits));
}

/* IQ4_NL nonlinear 4-bit codebook. Mirrors quant_tables.h::KVALUES_IQ4NL,
 * which is itself generated from ggml-common.h. Held in constant memory: every
 * lane of a warp indexes it with unrelated nibbles, but the table is 16 bytes
 * and stays resident in the constant cache, so the broadcast-vs-gather
 * distinction costs nothing at this size. */
__device__ __constant__ int8_t MQ_KVALUES_IQ4NL[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10,
    1, 13, 25, 38, 53, 69, 89, 113
};

/* Q4_K 6-bit scale/min unpack. Mirrors quantization.c::get_scale_min_k4. */
__device__ __forceinline__ void mq_scale_min_k4(uint32_t j,
                                                const uint8_t *scales,
                                                uint8_t *out_sc, uint8_t *out_m)
{
    if (j < 4u) {
        *out_sc = (uint8_t)(scales[j] & 63u);
        *out_m  = (uint8_t)(scales[j + 4] & 63u);
    } else {
        *out_sc = (uint8_t)((scales[j + 4] & 0x0Fu) | ((scales[j - 4] >> 6) << 4));
        *out_m  = (uint8_t)(((scales[j + 4] >> 4) & 0x0Fu) | ((scales[j] >> 6) << 4));
    }
}

/* ── Per-group dot products ───────────────────────────────────────────────
 *
 * The unit of parallel work is a 32-element GROUP, not a whole super-block.
 * This matters: a Q4_K row of a 1536-wide tensor is only 6 super-blocks, so
 * super-block granularity left 250 of 256 threads idle. At group granularity
 * the same row is 48 units, and consecutive threads read adjacent bytes of
 * `qs`, which also coalesces far better.
 *
 * `g` is the group index within the row; `x` points at x[g * 32].
 */

/* Q4_K: 8 groups per super-block. Group g maps to scale index (g % 8) — the
 * low nibbles of qs[q_base..+31] for even groups, the high nibbles for odd.
 *
 * The dequantized value is `d*sc*q - min*m`, so a group's contribution
 * factors into d*sc*sum(x*q) - min*m*sum(x): two accumulators rather than 32
 * reconstructed weights. */
__device__ __forceinline__ float mq_q4k_group_dot(const uint8_t *row,
                                                  uint32_t g, const float *x)
{
    const uint8_t *blk = row + (size_t)(g >> 3) * MQ_DEV_Q4_K_SIZE;
    const uint32_t gw = g & 7u;

    const float d  = mq_f16(blk[0], blk[1]);
    const float mn = mq_f16(blk[2], blk[3]);
    uint8_t sc, m;
    mq_scale_min_k4(gw, blk + 4, &sc, &m);

    const uint8_t *qs = blk + 16 + (size_t)(gw >> 1) * 32u;
    const bool hi = (gw & 1u) != 0u;

    /* 16-byte vector loads rather than 32 scalar byte loads. Alignment is
     * guaranteed: a Q4_K row is (cols/256)*144 bytes and 144 = 9*16, so every
     * block start is 16-aligned, as is blk+16+32k. Same for x: g*32 floats is
     * a 128-byte offset from a cudaMalloc'd (256-aligned) base. */
    const uint4 w0 = *reinterpret_cast<const uint4 *>(qs);
    const uint4 w1 = *reinterpret_cast<const uint4 *>(qs + 16);
    const uint32_t words[8] = { w0.x, w0.y, w0.z, w0.w,
                                w1.x, w1.y, w1.z, w1.w };

    float qx = 0.0f, sx = 0.0f;
    #pragma unroll
    for (uint32_t wi = 0u; wi < 8u; wi++) {
        const uint32_t word = words[wi];
        const float4 xv = *reinterpret_cast<const float4 *>(x + wi * 4u);
        const float xs[4] = { xv.x, xv.y, xv.z, xv.w };
        #pragma unroll
        for (uint32_t b = 0u; b < 4u; b++) {
            /* Byte l of the span is word l/4, byte l%4 (little-endian), which
             * pairs with x[wi*4 + b]. */
            const uint32_t byte = (word >> (b * 8u)) & 0xFFu;
            const uint32_t q = hi ? (byte >> 4) : (byte & 0x0Fu);
            qx += xs[b] * (float)q;
            sx += xs[b];
        }
    }
    return d * (float)sc * qx - mn * (float)m * sx;
}

/* Q5_K: 8 groups per super-block. Layout per quantization.c::dequant_q5_k:
 *   d:f16, min:f16, 12 B scale table, 32 B qh, 128 B nibbles.
 * Group g's nibbles are qs[(g/2)*32 ..+31] (low nibbles for even g, high for
 * odd), and its fifth bit for byte l is bit g of qh[l] — one bit per
 * 64-element group, the u1/u2 stepping masks of the dequant, NOT a flat
 * 256-bit field. Same `d*sc*sum(x*q) - min*m*sum(x)` factoring as Q4_K. */
__device__ __forceinline__ float mq_q5k_group_dot(const uint8_t *row,
                                                  uint32_t g, const float *x)
{
    const uint8_t *blk = row + (size_t)(g >> 3) * MQ_DEV_Q5_K_SIZE;
    const uint32_t gw = g & 7u;

    const float d  = mq_f16(blk[0], blk[1]);
    const float mn = mq_f16(blk[2], blk[3]);
    uint8_t sc, m;
    mq_scale_min_k4(gw, blk + 4, &sc, &m);

    const uint8_t *qh = blk + 16;
    const uint8_t *qs = blk + 48 + (size_t)(gw >> 1) * 32u;
    const bool hi = (gw & 1u) != 0u;

    /* 176 = 11*16 keeps blk 16-aligned; qh (blk+16) and qs (blk+48+32k) are
     * both 16-aligned spans. */
    const uint4 w0 = *reinterpret_cast<const uint4 *>(qs);
    const uint4 w1 = *reinterpret_cast<const uint4 *>(qs + 16);
    const uint4 h0 = *reinterpret_cast<const uint4 *>(qh);
    const uint4 h1 = *reinterpret_cast<const uint4 *>(qh + 16);
    const uint32_t words[8] = { w0.x, w0.y, w0.z, w0.w,
                                w1.x, w1.y, w1.z, w1.w };
    const uint32_t hbits[8] = { h0.x, h0.y, h0.z, h0.w,
                                h1.x, h1.y, h1.z, h1.w };

    float qx = 0.0f, sx = 0.0f;
    #pragma unroll
    for (uint32_t wi = 0u; wi < 8u; wi++) {
        const uint32_t word = words[wi];
        const uint32_t hw   = hbits[wi];
        const float4 xv = *reinterpret_cast<const float4 *>(x + wi * 4u);
        const float xs[4] = { xv.x, xv.y, xv.z, xv.w };
        #pragma unroll
        for (uint32_t b = 0u; b < 4u; b++) {
            const uint32_t byte  = (word >> (b * 8u)) & 0xFFu;
            const uint32_t hbyte = (hw   >> (b * 8u)) & 0xFFu;
            const uint32_t nib = hi ? (byte >> 4) : (byte & 0x0Fu);
            const uint32_t q = nib | (((hbyte >> gw) & 1u) << 4);
            qx += xs[b] * (float)q;
            sx += xs[b];
        }
    }
    return d * (float)sc * qx - mn * (float)m * sx;
}

/* Q6_K: 8 groups per super-block. Layout per quantization.c: ql[0..127],
 * qh[128..191], int8 scales[192..207], d at [208..209]. Group G splits as
 * half = G/4 (which 64-byte ql / 32-byte qh span) and j = G%4 (which of the
 * four nibble/high-bit combinations); the scale index also advances with
 * l/16 inside the group. */
__device__ __forceinline__ float mq_q6k_group_dot(const uint8_t *row,
                                                  uint32_t g, const float *x)
{
    const uint8_t *blk = row + (size_t)(g >> 3) * MQ_DEV_Q6_K_SIZE;
    const uint32_t G = g & 7u;
    const uint32_t half = G >> 2;
    const uint32_t j    = G & 3u;

    const float d = mq_f16(blk[208], blk[209]);
    const uint8_t *ql = blk + (size_t)half * 64u + ((j & 1u) ? 32u : 0u);
    const uint8_t *qh = blk + 128 + (size_t)half * 32u;
    const int8_t  *sc = (const int8_t *)(blk + 192) + (size_t)half * 8u
                      + (size_t)j * 2u;
    const uint32_t shift = j * 2u;

    /* Vector loads: the 224-byte device stride makes blk 16-aligned, and both
     * spans sit at 16-aligned offsets within it (half*64 + {0,32}; 128 +
     * half*32). */
    const uint4 l0 = *reinterpret_cast<const uint4 *>(ql);
    const uint4 l1 = *reinterpret_cast<const uint4 *>(ql + 16);
    const uint4 h0 = *reinterpret_cast<const uint4 *>(qh);
    const uint4 h1 = *reinterpret_cast<const uint4 *>(qh + 16);
    const uint32_t lw[8] = { l0.x, l0.y, l0.z, l0.w, l1.x, l1.y, l1.z, l1.w };
    const uint32_t hw[8] = { h0.x, h0.y, h0.z, h0.w, h1.x, h1.y, h1.z, h1.w };

    /* The scale index only changes at l == 16, so split the loop in two halves
     * and hoist it out of the inner body. */
    float acc = 0.0f;
    #pragma unroll
    for (uint32_t wi = 0u; wi < 8u; wi++) {
        const float4 xv = *reinterpret_cast<const float4 *>(x + wi * 4u);
        const float xs[4] = { xv.x, xv.y, xv.z, xv.w };
        const float s = (float)sc[(wi * 4u) >> 4];
        float part = 0.0f;
        #pragma unroll
        for (uint32_t b = 0u; b < 4u; b++) {
            const uint32_t lb = (lw[wi] >> (b * 8u)) & 0xFFu;
            const uint32_t hb = (hw[wi] >> (b * 8u)) & 0xFFu;
            const int32_t lo = (j < 2u) ? (int32_t)(lb & 0x0Fu)
                                        : (int32_t)(lb >> 4);
            const int32_t q = lo | ((int32_t)((hb >> shift) & 3u) << 4);
            part += (float)(q - 32) * xs[b];
        }
        acc += s * part;
    }
    return d * acc;
}

/* Q8_0: the block IS a 32-element group. */
__device__ __forceinline__ float mq_q8_0_group_dot(const uint8_t *row,
                                                   uint32_t g, const float *x)
{
    const uint8_t *blk = row + (size_t)g * MQ_DEV_Q8_0_SIZE;
    const float d = mq_f16(blk[0], blk[1]);
    const int8_t *q = (const int8_t *)(blk + 2);
    const float4 *x4 = reinterpret_cast<const float4 *>(x);
    float acc = 0.0f;
    #pragma unroll
    for (uint32_t i = 0u; i < 8u; i++) {
        const float4 xv = x4[i];
        const uint32_t b = i * 4u;
        acc += (float)q[b] * xv.x + (float)q[b + 1u] * xv.y
             + (float)q[b + 2u] * xv.z + (float)q[b + 3u] * xv.w;
    }
    return acc * d;
}

/* Q4_0 / AL5: 18-byte block is one 32-element group.
 * out[i] = ((nibble)-8)*d with low nibble → i, high nibble → i+16. */
__device__ __forceinline__ float mq_q4_0_group_dot(const uint8_t *row,
                                                   uint32_t g, const float *x)
{
    const uint8_t *blk = row + (size_t)g * MQ_DEV_Q4_0_SIZE;
    const float d = mq_f16(blk[0], blk[1]);
    const uint8_t *qs = blk + 2;
    const float4 *x0 = reinterpret_cast<const float4 *>(x);
    const float4 *x1 = reinterpret_cast<const float4 *>(x + 16);
    float acc = 0.0f;
    #pragma unroll
    for (uint32_t i = 0; i < 4u; i++) {
        const float4 a = x0[i];
        const float4 b = x1[i];
        const uint32_t base = i * 4u;
        const uint8_t p0 = qs[base], p1 = qs[base + 1u];
        const uint8_t p2 = qs[base + 2u], p3 = qs[base + 3u];
        acc += (float)((int32_t)(p0 & 0x0Fu) - 8) * a.x
             + (float)((int32_t)(p1 & 0x0Fu) - 8) * a.y
             + (float)((int32_t)(p2 & 0x0Fu) - 8) * a.z
             + (float)((int32_t)(p3 & 0x0Fu) - 8) * a.w;
        acc += (float)((int32_t)(p0 >> 4) - 8) * b.x
             + (float)((int32_t)(p1 >> 4) - 8) * b.y
             + (float)((int32_t)(p2 >> 4) - 8) * b.z
             + (float)((int32_t)(p3 >> 4) - 8) * b.w;
    }
    return acc * d;
}

/* Q5_0 / AL6: 22-byte block. High bits in a 32-bit field at blk+2. */
__device__ __forceinline__ float mq_q5_0_group_dot(const uint8_t *row,
                                                   uint32_t g, const float *x)
{
    const uint8_t *blk = row + (size_t)g * MQ_DEV_Q5_0_SIZE;
    const float d = mq_f16(blk[0], blk[1]);
    const uint32_t qh = (uint32_t)blk[2] | ((uint32_t)blk[3] << 8)
                      | ((uint32_t)blk[4] << 16) | ((uint32_t)blk[5] << 24);
    const uint8_t *qs = blk + 6;
    float acc = 0.0f;
    #pragma unroll
    for (uint32_t j = 0; j < 16u; j++) {
        const uint32_t x0 = (uint32_t)(qs[j] & 0x0Fu) | (((qh >> j) & 1u) << 4);
        const uint32_t x1 = (uint32_t)(qs[j] >> 4)
                          | (((qh >> (j + 16u)) & 1u) << 4);
        acc += (float)((int32_t)x0 - 16) * x[j];
        acc += (float)((int32_t)x1 - 16) * x[j + 16u];
    }
    return acc * d;
}

/* AL5_XS: 14-byte block, 32 × 3-bit levels, (l-4)*d. */
__device__ __forceinline__ float mq_al5_xs_group_dot(const uint8_t *row,
                                                     uint32_t g, const float *x)
{
    const uint8_t *blk = row + (size_t)g * MQ_DEV_AL5_XS_SIZE;
    const float d = mq_f16(blk[0], blk[1]);
    const uint8_t *packed = blk + 2;
    float acc = 0.0f;
    uint32_t bitpos = 0u;
    for (uint32_t i = 0; i < 32u; i++) {
        uint32_t v = 0u;
        #pragma unroll
        for (int b = 0; b < 3; b++) {
            const uint32_t byte_idx = bitpos / 8u;
            const uint32_t bit_idx = bitpos % 8u;
            if ((packed[byte_idx] >> bit_idx) & 1u) v |= (1u << b);
            bitpos += 1u;
        }
        acc += (float)((int32_t)v - 4) * x[i];
    }
    return acc * d;
}

/* IQ4_XS: 8 groups per super-block, and the 32-element group maps exactly onto
 * one ib32 sub-block — the natural unit here, unlike the K-quants where a
 * group is half a scale span. Layout per quantization.c::dequant_iq4_xs:
 *   d[0..2] f16, scales_h[2..4] u16 LE, scales_l[4..8], qs[8..136].
 *
 * Sub-block ib owns 16 bytes at qs + ib*16 and a 6-bit signed scale split
 * across scales_l (low nibble) and scales_h (high 2 bits). Within the group,
 * byte j's low nibble is value j and its high nibble is value j+16 — so the
 * two halves of x are read as separate 16-float spans rather than
 * interleaved. Values are codebook indices, not magnitudes, which is why this
 * cannot reuse the K-quant "sum(x*q), sum(x)" factoring: there is no additive
 * min term, but the dequantized value is a table lookup, so only the single
 * `dl` scale factors out. */
__device__ __forceinline__ float mq_iq4_xs_group_dot(const uint8_t *row,
                                                     uint32_t g, const float *x)
{
    const uint8_t *blk = row + (size_t)(g >> 3) * MQ_DEV_IQ4_XS_SIZE;
    const uint32_t ib = g & 7u;

    const float d = mq_f16(blk[0], blk[1]);
    const uint32_t scales_h = (uint32_t)blk[2] | ((uint32_t)blk[3] << 8);
    const uint8_t *scales_l = blk + 4;

    const int32_t ls_l = (int32_t)((scales_l[ib >> 1] >> (4u * (ib & 1u))) & 0x0Fu);
    const int32_t ls_h = (int32_t)((scales_h >> (2u * ib)) & 3u) << 4;
    const float dl = d * (float)((ls_l | ls_h) - 32);

    /* 8-byte loads: blk is 8-aligned (136 = 17*8) and qs sits at blk + 8, so
     * qs + ib*16 is 8-aligned. uint4 would need 16 and is not guaranteed. */
    const uint8_t *qs = blk + 8 + (size_t)ib * 16u;
    const uint2 p0 = *reinterpret_cast<const uint2 *>(qs);
    const uint2 p1 = *reinterpret_cast<const uint2 *>(qs + 8);
    const uint32_t words[4] = { p0.x, p0.y, p1.x, p1.y };

    float acc = 0.0f;
    #pragma unroll
    for (uint32_t wi = 0u; wi < 4u; wi++) {
        const uint32_t word = words[wi];
        /* x + g*32 is 128-byte aligned, so both spans are 16-aligned. */
        const float4 xlo = *reinterpret_cast<const float4 *>(x + wi * 4u);
        const float4 xhi = *reinterpret_cast<const float4 *>(x + 16u + wi * 4u);
        const float ls[4] = { xlo.x, xlo.y, xlo.z, xlo.w };
        const float hs[4] = { xhi.x, xhi.y, xhi.z, xhi.w };
        #pragma unroll
        for (uint32_t b = 0u; b < 4u; b++) {
            const uint32_t byte = (word >> (b * 8u)) & 0xFFu;
            acc += (float)MQ_KVALUES_IQ4NL[byte & 0x0Fu] * ls[b];
            acc += (float)MQ_KVALUES_IQ4NL[byte >> 4]    * hs[b];
        }
    }
    return dl * acc;
}

/* ─── Reduction ──────────────────────────────────────────────────────────── */

/* Sum across a warp with shuffles: no shared memory and no __syncthreads.
 * The previous block-wide tree reduction cost six barriers per row for as
 * little as 32 MACs of work per thread, which dominated narrow tensors. */
__device__ __forceinline__ float mq_warp_reduce(float v)
{
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_down_sync(0xffffffffu, v, off);
    return v;
}

/* ─── Matvec kernels (one warp per output row) ────────────────────────────── */

/* One WARP per output row, MQ_WARPS rows per block. Lanes stride over the
 * row's 32-element groups and combine with shuffles. This keeps every lane
 * busy on narrow tensors and removes the barrier traffic entirely. */
#define MQ_WARPS 8u

#define MQ_DEFINE_MATVEC(NAME, DOT)                                          \
__global__ void NAME(const uint8_t *w, const float *x, float *out,           \
                     size_t rows, uint32_t n_groups, size_t row_bytes)       \
{                                                                            \
    const uint32_t lane = threadIdx.x & 31u;                                 \
    const size_t row = (size_t)blockIdx.x * MQ_WARPS + (threadIdx.x >> 5);   \
    if (row >= rows) return;                                                 \
    const uint8_t *rw = w + row * row_bytes;                                 \
    float partial = 0.0f;                                                    \
    for (uint32_t g = lane; g < n_groups; g += 32u)                          \
        partial += DOT(rw, g, x + (size_t)g * 32u);                          \
    partial = mq_warp_reduce(partial);                                       \
    if (lane == 0u) out[row] = partial;                                      \
}

MQ_DEFINE_MATVEC(k_mmq_matvec_q4k,    mq_q4k_group_dot)
MQ_DEFINE_MATVEC(k_mmq_matvec_q5k,    mq_q5k_group_dot)
MQ_DEFINE_MATVEC(k_mmq_matvec_q6k,    mq_q6k_group_dot)
MQ_DEFINE_MATVEC(k_mmq_matvec_q8_0,   mq_q8_0_group_dot)
MQ_DEFINE_MATVEC(k_mmq_matvec_q4_0,   mq_q4_0_group_dot)
MQ_DEFINE_MATVEC(k_mmq_matvec_q5_0,   mq_q5_0_group_dot)
MQ_DEFINE_MATVEC(k_mmq_matvec_al5_xs, mq_al5_xs_group_dot)
MQ_DEFINE_MATVEC(k_mmq_matvec_iq4_xs, mq_iq4_xs_group_dot)

/* MoE variant: the expert to use is read from device memory, so routing never
 * round-trips to the host and the whole token stays one async submission.
 * Expert `e` occupies rows [e*rows, (e+1)*rows) of the stacked tensor. */
#define MQ_DEFINE_MATVEC_EXPERT(NAME, DOT)                                   \
__global__ void NAME(const uint8_t *w, const float *x, float *out,           \
                     size_t rows, uint32_t n_groups, size_t row_bytes,       \
                     const uint32_t *expert_sel, uint32_t slot)              \
{                                                                            \
    const uint32_t lane = threadIdx.x & 31u;                                 \
    const size_t r = (size_t)blockIdx.x * MQ_WARPS + (threadIdx.x >> 5);     \
    if (r >= rows) return;                                                   \
    const uint8_t *rw = w + ((size_t)expert_sel[slot] * rows + r) * row_bytes;\
    float partial = 0.0f;                                                    \
    for (uint32_t g = lane; g < n_groups; g += 32u)                          \
        partial += DOT(rw, g, x + (size_t)g * 32u);                          \
    partial = mq_warp_reduce(partial);                                       \
    if (lane == 0u) out[r] = partial;                                        \
}

MQ_DEFINE_MATVEC_EXPERT(k_mmq_matvec_exp_q4k,    mq_q4k_group_dot)
MQ_DEFINE_MATVEC_EXPERT(k_mmq_matvec_exp_q5k,    mq_q5k_group_dot)
MQ_DEFINE_MATVEC_EXPERT(k_mmq_matvec_exp_q6k,    mq_q6k_group_dot)
MQ_DEFINE_MATVEC_EXPERT(k_mmq_matvec_exp_q8_0,   mq_q8_0_group_dot)
MQ_DEFINE_MATVEC_EXPERT(k_mmq_matvec_exp_q4_0,   mq_q4_0_group_dot)
MQ_DEFINE_MATVEC_EXPERT(k_mmq_matvec_exp_q5_0,   mq_q5_0_group_dot)
MQ_DEFINE_MATVEC_EXPERT(k_mmq_matvec_exp_al5_xs, mq_al5_xs_group_dot)
MQ_DEFINE_MATVEC_EXPERT(k_mmq_matvec_exp_iq4_xs, mq_iq4_xs_group_dot)

/* ─── Single-row dequantize (embedding lookup) ────────────────────────────── */

#define MQ_DEFINE_GETROW(NAME, BLOCK_BYTES, VALS, EXPAND)                    \
__global__ void NAME(const uint8_t *w, uint32_t token, float *out,           \
                     size_t cols, size_t row_bytes)                          \
{                                                                            \
    const uint8_t *rw = w + (size_t)token * row_bytes;                       \
    const size_t n_blocks = cols / (VALS);                                   \
    for (size_t b = blockIdx.x * blockDim.x + threadIdx.x; b < n_blocks;     \
         b += (size_t)gridDim.x * blockDim.x)                                \
        EXPAND(rw + b * (BLOCK_BYTES), out + b * (VALS));                    \
}

/* Full expansion of one block — only used on the embedding path, where a
 * single row per token makes the register cost irrelevant. */
__device__ __forceinline__ void mq_q4k_expand(const uint8_t *blk, float *out)
{
    const float d  = mq_f16(blk[0], blk[1]);
    const float mn = mq_f16(blk[2], blk[3]);
    const uint8_t *scales = blk + 4;
    const uint8_t *qs     = blk + 16;
    uint32_t is = 0u, off = 0u;
    for (uint32_t gp = 0u; gp < 4u; gp++) {
        const uint32_t q_base = gp * 32u;
        uint8_t sc1, m1, sc2, m2;
        mq_scale_min_k4(is,      scales, &sc1, &m1);
        mq_scale_min_k4(is + 1u, scales, &sc2, &m2);
        const float d1 = d * (float)sc1, min1 = mn * (float)m1;
        const float d2 = d * (float)sc2, min2 = mn * (float)m2;
        for (uint32_t l = 0u; l < 32u; l++) {
            out[off + l]       = d1 * (float)(qs[q_base + l] & 0x0Fu) - min1;
            out[off + 32u + l] = d2 * (float)(qs[q_base + l] >> 4)    - min2;
        }
        off += 64u;
        is  += 2u;
    }
}

__device__ __forceinline__ void mq_q5k_expand(const uint8_t *blk, float *out)
{
    const float d  = mq_f16(blk[0], blk[1]);
    const float mn = mq_f16(blk[2], blk[3]);
    const uint8_t *scales = blk + 4;
    const uint8_t *qh     = blk + 16;
    const uint8_t *qs     = blk + 48;
    uint32_t is = 0u, off = 0u;
    for (uint32_t gp = 0u; gp < 4u; gp++) {
        const uint8_t *ql = qs + gp * 32u;
        uint8_t sc1, m1, sc2, m2;
        mq_scale_min_k4(is,      scales, &sc1, &m1);
        mq_scale_min_k4(is + 1u, scales, &sc2, &m2);
        const float d1 = d * (float)sc1, min1 = mn * (float)m1;
        const float d2 = d * (float)sc2, min2 = mn * (float)m2;
        for (uint32_t l = 0u; l < 32u; l++) {
            const uint32_t q1 = (uint32_t)(ql[l] & 0x0Fu)
                              | (((uint32_t)(qh[l] >> (2u * gp)) & 1u) << 4);
            const uint32_t q2 = (uint32_t)(ql[l] >> 4)
                              | (((uint32_t)(qh[l] >> (2u * gp + 1u)) & 1u) << 4);
            out[off + l]       = d1 * (float)q1 - min1;
            out[off + 32u + l] = d2 * (float)q2 - min2;
        }
        off += 64u;
        is  += 2u;
    }
}

__device__ __forceinline__ void mq_q6k_expand(const uint8_t *blk, float *out)
{
    const float d = mq_f16(blk[208], blk[209]);
    const uint8_t *ql = blk;
    const uint8_t *qh = blk + 128;
    const int8_t  *sc = (const int8_t *)(blk + 192);
    uint32_t o = 0u;
    for (uint32_t group = 0u; group < 2u; group++) {
        const uint32_t ql_off = group * 64u;
        const uint32_t qh_off = group * 32u;
        const uint32_t sc_off = group * 8u;
        for (uint32_t l = 0u; l < 32u; l++) {
            const uint32_t is_idx = l >> 4;
            const uint8_t h = qh[qh_off + l];
            const int32_t q1 = (int32_t)(ql[ql_off + l] & 0x0Fu)
                             | ((int32_t)(h & 3u) << 4);
            const int32_t q2 = (int32_t)(ql[ql_off + l + 32u] & 0x0Fu)
                             | ((int32_t)((h >> 2) & 3u) << 4);
            const int32_t q3 = (int32_t)(ql[ql_off + l] >> 4)
                             | ((int32_t)((h >> 4) & 3u) << 4);
            const int32_t q4 = (int32_t)(ql[ql_off + l + 32u] >> 4)
                             | ((int32_t)((h >> 6) & 3u) << 4);
            out[o + l]        = d * (float)sc[sc_off + is_idx]      * (float)(q1 - 32);
            out[o + 32u + l]  = d * (float)sc[sc_off + is_idx + 2u] * (float)(q2 - 32);
            out[o + 64u + l]  = d * (float)sc[sc_off + is_idx + 4u] * (float)(q3 - 32);
            out[o + 96u + l]  = d * (float)sc[sc_off + is_idx + 6u] * (float)(q4 - 32);
        }
        o += 128u;
    }
}

__device__ __forceinline__ void mq_q8_0_expand(const uint8_t *blk, float *out)
{
    const float d = mq_f16(blk[0], blk[1]);
    const int8_t *q = (const int8_t *)(blk + 2);
    for (uint32_t i = 0u; i < MQ_QK8_0; i++) out[i] = (float)q[i] * d;
}

__device__ __forceinline__ void mq_q4_0_expand(const uint8_t *blk, float *out)
{
    const float d = mq_f16(blk[0], blk[1]);
    const uint8_t *qs = blk + 2;
    for (uint32_t i = 0; i < 16u; i++) {
        const uint8_t p = qs[i];
        out[i]      = (float)((int32_t)(p & 0x0Fu) - 8) * d;
        out[i + 16] = (float)((int32_t)(p >> 4) - 8) * d;
    }
}

__device__ __forceinline__ void mq_q5_0_expand(const uint8_t *blk, float *out)
{
    const float d = mq_f16(blk[0], blk[1]);
    const uint32_t qh = (uint32_t)blk[2] | ((uint32_t)blk[3] << 8)
                      | ((uint32_t)blk[4] << 16) | ((uint32_t)blk[5] << 24);
    const uint8_t *qs = blk + 6;
    for (uint32_t j = 0; j < 16u; j++) {
        const uint32_t x0 = (uint32_t)(qs[j] & 0x0Fu) | (((qh >> j) & 1u) << 4);
        const uint32_t x1 = (uint32_t)(qs[j] >> 4)
                          | (((qh >> (j + 16u)) & 1u) << 4);
        out[j]      = (float)((int32_t)x0 - 16) * d;
        out[j + 16] = (float)((int32_t)x1 - 16) * d;
    }
}

__device__ __forceinline__ void mq_al5_xs_expand(const uint8_t *blk, float *out)
{
    const float d = mq_f16(blk[0], blk[1]);
    const uint8_t *packed = blk + 2;
    uint32_t bitpos = 0u;
    for (uint32_t i = 0; i < 32u; i++) {
        uint32_t v = 0u;
        for (int b = 0; b < 3; b++) {
            const uint32_t byte_idx = bitpos / 8u;
            const uint32_t bit_idx = bitpos % 8u;
            if ((packed[byte_idx] >> bit_idx) & 1u) v |= (1u << b);
            bitpos += 1u;
        }
        out[i] = (float)((int32_t)v - 4) * d;
    }
}

/* IQ4_XS full-block expansion. Same layout walk as mq_iq4_xs_group_dot, but
 * writing all 256 values instead of contracting them against x. */
__device__ __forceinline__ void mq_iq4_xs_expand(const uint8_t *blk, float *out)
{
    const float d = mq_f16(blk[0], blk[1]);
    const uint32_t scales_h = (uint32_t)blk[2] | ((uint32_t)blk[3] << 8);
    const uint8_t *scales_l = blk + 4;
    const uint8_t *qs = blk + 8;

    for (uint32_t ib = 0u; ib < 8u; ib++) {
        const int32_t ls_l =
            (int32_t)((scales_l[ib >> 1] >> (4u * (ib & 1u))) & 0x0Fu);
        const int32_t ls_h = (int32_t)((scales_h >> (2u * ib)) & 3u) << 4;
        const float dl = d * (float)((ls_l | ls_h) - 32);
        const uint8_t *q = qs + (size_t)ib * 16u;
        float *o = out + (size_t)ib * 32u;
        for (uint32_t j = 0u; j < 16u; j++) {
            o[j]       = dl * (float)MQ_KVALUES_IQ4NL[q[j] & 0x0Fu];
            o[j + 16u] = dl * (float)MQ_KVALUES_IQ4NL[q[j] >> 4];
        }
    }
}

MQ_DEFINE_GETROW(k_mmq_getrow_q4k,    MQ_DEV_Q4_K_SIZE,   MQ_QK_K,  mq_q4k_expand)
MQ_DEFINE_GETROW(k_mmq_getrow_q5k,    MQ_DEV_Q5_K_SIZE,   MQ_QK_K,  mq_q5k_expand)
MQ_DEFINE_GETROW(k_mmq_getrow_q6k,    MQ_DEV_Q6_K_SIZE,   MQ_QK_K,  mq_q6k_expand)
MQ_DEFINE_GETROW(k_mmq_getrow_q8_0,   MQ_DEV_Q8_0_SIZE,   MQ_QK8_0, mq_q8_0_expand)
MQ_DEFINE_GETROW(k_mmq_getrow_q4_0,   MQ_DEV_Q4_0_SIZE,   MQ_QK8_0, mq_q4_0_expand)
MQ_DEFINE_GETROW(k_mmq_getrow_q5_0,   MQ_DEV_Q5_0_SIZE,   MQ_QK8_0, mq_q5_0_expand)
MQ_DEFINE_GETROW(k_mmq_getrow_al5_xs, MQ_DEV_AL5_XS_SIZE, MQ_QK8_0, mq_al5_xs_expand)
MQ_DEFINE_GETROW(k_mmq_getrow_iq4_xs, MQ_DEV_IQ4_XS_SIZE, MQ_QK_K,  mq_iq4_xs_expand)

/* ─── Host API ───────────────────────────────────────────────────────────── */

/* On-disk block bytes / device block bytes / values-per-block, or all zero for
 * an unsupported type. */
static void mq_layout(uint32_t qtype, size_t *src_block, size_t *dev_block,
                      size_t *vals)
{
    switch ((OcGgufQuantizationType)qtype) {
    case OC_QUANT_Q4_K_S:
    case OC_QUANT_Q4_K_M:
        *src_block = MQ_BLOCK_Q4_K_SIZE; *dev_block = MQ_DEV_Q4_K_SIZE;
        *vals = MQ_QK_K;  return;
    case OC_QUANT_Q5_K_S:
    case OC_QUANT_Q5_K_M:
        *src_block = MQ_BLOCK_Q5_K_SIZE; *dev_block = MQ_DEV_Q5_K_SIZE;
        *vals = MQ_QK_K;  return;
    case OC_QUANT_Q6_K:
        *src_block = MQ_BLOCK_Q6_K_SIZE; *dev_block = MQ_DEV_Q6_K_SIZE;
        *vals = MQ_QK_K;  return;
    case OC_QUANT_Q8_0:
    case OC_QUANT_AL8:
        *src_block = MQ_BLOCK_Q8_0_SIZE; *dev_block = MQ_DEV_Q8_0_SIZE;
        *vals = MQ_QK8_0; return;
    case OC_QUANT_Q4_0:
    case OC_QUANT_AL5:
        *src_block = MQ_BLOCK_Q4_0_SIZE; *dev_block = MQ_DEV_Q4_0_SIZE;
        *vals = MQ_QK8_0; return;
    case OC_QUANT_Q5_0:
    case OC_QUANT_AL6:
        *src_block = MQ_BLOCK_Q5_0_SIZE; *dev_block = MQ_DEV_Q5_0_SIZE;
        *vals = MQ_QK8_0; return;
    case OC_QUANT_AL5_XS:
        *src_block = MQ_BLOCK_AL5_XS_SIZE; *dev_block = MQ_DEV_AL5_XS_SIZE;
        *vals = MQ_QK8_0; return;
    case OC_QUANT_IQ4_XS:
        *src_block = MQ_BLOCK_IQ4_XS_SIZE; *dev_block = MQ_DEV_IQ4_XS_SIZE;
        *vals = MQ_QK_K;  return;
    default:
        *src_block = 0; *dev_block = 0; *vals = 0; return;
    }
}

extern "C" bool oc_cuda_mmq_supported(uint32_t qtype, size_t cols)
{
    size_t sb = 0, db = 0, vals = 0;
    mq_layout(qtype, &sb, &db, &vals);
    if (sb == 0 || vals == 0) return false;
    /* A row must be a whole number of blocks: K-quants need cols % 256 == 0,
     * which rules out shapes like n_embd = 896. Those keep the f32 path. */
    return cols != 0 && (cols % vals) == 0;
}

extern "C" size_t oc_cuda_mmq_row_bytes(uint32_t qtype, size_t cols)
{
    if (!oc_cuda_mmq_supported(qtype, cols)) return 0;
    size_t sb = 0, db = 0, vals = 0;
    mq_layout(qtype, &sb, &db, &vals);
    return (cols / vals) * db;
}

extern "C" bool oc_cuda_mmq_block_layout(uint32_t qtype, size_t cols,
                                         size_t *src_block, size_t *dev_block,
                                         size_t *n_blocks)
{
    if (!oc_cuda_mmq_supported(qtype, cols)) return false;
    size_t sb = 0, db = 0, vals = 0;
    mq_layout(qtype, &sb, &db, &vals);
    if (src_block) *src_block = sb;
    if (dev_block) *dev_block = db;
    if (n_blocks)  *n_blocks  = cols / vals;
    return true;
}

extern "C" bool oc_cuda_mmq_matvec(uint32_t qtype, const void *d_weights,
                                   const float *d_x, float *d_out,
                                   size_t rows, size_t cols, void *stream)
{
    if (!d_weights || !d_x || !d_out || rows == 0) return false;
    const size_t row_bytes = oc_cuda_mmq_row_bytes(qtype, cols);
    if (row_bytes == 0) return false;

    const uint32_t n_groups = (uint32_t)(cols / 32u);
    const uint32_t block = MQ_WARPS * 32u;
    const size_t grid = (rows + MQ_WARPS - 1u) / MQ_WARPS;
    cudaStream_t s = (cudaStream_t)stream;
    const uint8_t *w = (const uint8_t *)d_weights;

    switch ((OcGgufQuantizationType)qtype) {
    case OC_QUANT_Q4_K_S:
    case OC_QUANT_Q4_K_M:
        k_mmq_matvec_q4k<<<grid, block, 0, s>>>(w, d_x, d_out, rows,
                                                n_groups, row_bytes);
        break;
    case OC_QUANT_Q5_K_S:
    case OC_QUANT_Q5_K_M:
        k_mmq_matvec_q5k<<<grid, block, 0, s>>>(w, d_x, d_out, rows,
                                                n_groups, row_bytes);
        break;
    case OC_QUANT_Q6_K:
        k_mmq_matvec_q6k<<<grid, block, 0, s>>>(w, d_x, d_out, rows,
                                                n_groups, row_bytes);
        break;
    case OC_QUANT_Q8_0:
    case OC_QUANT_AL8:
        k_mmq_matvec_q8_0<<<grid, block, 0, s>>>(w, d_x, d_out, rows,
                                                 n_groups, row_bytes);
        break;
    case OC_QUANT_Q4_0:
    case OC_QUANT_AL5:
        k_mmq_matvec_q4_0<<<grid, block, 0, s>>>(w, d_x, d_out, rows,
                                                 n_groups, row_bytes);
        break;
    case OC_QUANT_Q5_0:
    case OC_QUANT_AL6:
        k_mmq_matvec_q5_0<<<grid, block, 0, s>>>(w, d_x, d_out, rows,
                                                 n_groups, row_bytes);
        break;
    case OC_QUANT_AL5_XS:
        k_mmq_matvec_al5_xs<<<grid, block, 0, s>>>(w, d_x, d_out, rows,
                                                   n_groups, row_bytes);
        break;
    case OC_QUANT_IQ4_XS:
        k_mmq_matvec_iq4_xs<<<grid, block, 0, s>>>(w, d_x, d_out, rows,
                                                   n_groups, row_bytes);
        break;
    default:
        return false;
    }
    return cudaGetLastError() == cudaSuccess;
}

extern "C" bool oc_cuda_mmq_matvec_expert(uint32_t qtype, const void *d_weights,
                                          const float *d_x, float *d_out,
                                          size_t rows, size_t cols,
                                          const uint32_t *d_expert_sel,
                                          uint32_t slot, void *stream)
{
    if (!d_weights || !d_x || !d_out || !d_expert_sel || rows == 0) return false;
    const size_t row_bytes = oc_cuda_mmq_row_bytes(qtype, cols);
    if (row_bytes == 0) return false;

    const uint32_t n_groups = (uint32_t)(cols / 32u);
    const uint32_t block = MQ_WARPS * 32u;
    const size_t grid = (rows + MQ_WARPS - 1u) / MQ_WARPS;
    cudaStream_t s = (cudaStream_t)stream;
    const uint8_t *w = (const uint8_t *)d_weights;

    switch ((OcGgufQuantizationType)qtype) {
    case OC_QUANT_Q4_K_S:
    case OC_QUANT_Q4_K_M:
        k_mmq_matvec_exp_q4k<<<grid, block, 0, s>>>(w, d_x, d_out, rows,
            n_groups, row_bytes, d_expert_sel, slot);
        break;
    case OC_QUANT_Q5_K_S:
    case OC_QUANT_Q5_K_M:
        k_mmq_matvec_exp_q5k<<<grid, block, 0, s>>>(w, d_x, d_out, rows,
            n_groups, row_bytes, d_expert_sel, slot);
        break;
    case OC_QUANT_Q6_K:
        k_mmq_matvec_exp_q6k<<<grid, block, 0, s>>>(w, d_x, d_out, rows,
            n_groups, row_bytes, d_expert_sel, slot);
        break;
    case OC_QUANT_Q8_0:
    case OC_QUANT_AL8:
        k_mmq_matvec_exp_q8_0<<<grid, block, 0, s>>>(w, d_x, d_out, rows,
            n_groups, row_bytes, d_expert_sel, slot);
        break;
    case OC_QUANT_Q4_0:
    case OC_QUANT_AL5:
        k_mmq_matvec_exp_q4_0<<<grid, block, 0, s>>>(w, d_x, d_out, rows,
            n_groups, row_bytes, d_expert_sel, slot);
        break;
    case OC_QUANT_Q5_0:
    case OC_QUANT_AL6:
        k_mmq_matvec_exp_q5_0<<<grid, block, 0, s>>>(w, d_x, d_out, rows,
            n_groups, row_bytes, d_expert_sel, slot);
        break;
    case OC_QUANT_AL5_XS:
        k_mmq_matvec_exp_al5_xs<<<grid, block, 0, s>>>(w, d_x, d_out, rows,
            n_groups, row_bytes, d_expert_sel, slot);
        break;
    case OC_QUANT_IQ4_XS:
        k_mmq_matvec_exp_iq4_xs<<<grid, block, 0, s>>>(w, d_x, d_out, rows,
            n_groups, row_bytes, d_expert_sel, slot);
        break;
    default:
        return false;
    }
    return cudaGetLastError() == cudaSuccess;
}

extern "C" bool oc_cuda_mmq_get_row(uint32_t qtype, const void *d_weights,
                                    uint32_t token, float *d_out,
                                    size_t cols, void *stream)
{
    if (!d_weights || !d_out) return false;
    const size_t row_bytes = oc_cuda_mmq_row_bytes(qtype, cols);
    if (row_bytes == 0) return false;

    size_t sb = 0, db = 0, vals = 0;
    mq_layout(qtype, &sb, &db, &vals);
    const size_t n_blocks = cols / vals;
    const uint32_t block = MQ_BLOCK_THREADS;
    const uint32_t grid = (uint32_t)((n_blocks + block - 1) / block);
    cudaStream_t s = (cudaStream_t)stream;
    const uint8_t *w = (const uint8_t *)d_weights;

    switch ((OcGgufQuantizationType)qtype) {
    case OC_QUANT_Q4_K_S:
    case OC_QUANT_Q4_K_M:
        k_mmq_getrow_q4k<<<grid, block, 0, s>>>(w, token, d_out, cols, row_bytes);
        break;
    case OC_QUANT_Q5_K_S:
    case OC_QUANT_Q5_K_M:
        k_mmq_getrow_q5k<<<grid, block, 0, s>>>(w, token, d_out, cols, row_bytes);
        break;
    case OC_QUANT_Q6_K:
        k_mmq_getrow_q6k<<<grid, block, 0, s>>>(w, token, d_out, cols, row_bytes);
        break;
    case OC_QUANT_Q8_0:
    case OC_QUANT_AL8:
        k_mmq_getrow_q8_0<<<grid, block, 0, s>>>(w, token, d_out, cols, row_bytes);
        break;
    case OC_QUANT_Q4_0:
    case OC_QUANT_AL5:
        k_mmq_getrow_q4_0<<<grid, block, 0, s>>>(w, token, d_out, cols, row_bytes);
        break;
    case OC_QUANT_Q5_0:
    case OC_QUANT_AL6:
        k_mmq_getrow_q5_0<<<grid, block, 0, s>>>(w, token, d_out, cols, row_bytes);
        break;
    case OC_QUANT_AL5_XS:
        k_mmq_getrow_al5_xs<<<grid, block, 0, s>>>(w, token, d_out, cols, row_bytes);
        break;
    case OC_QUANT_IQ4_XS:
        k_mmq_getrow_iq4_xs<<<grid, block, 0, s>>>(w, token, d_out, cols, row_bytes);
        break;
    default:
        return false;
    }
    return cudaGetLastError() == cudaSuccess;
}
