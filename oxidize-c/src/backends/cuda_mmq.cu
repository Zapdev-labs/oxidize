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

/* Fast-path variants: the row index comes from device memory (an on-device
 * argmax result or a prompt-ids array), never the host. */
#define MQ_DEFINE_GETROW_DEV(NAME, BLOCK_BYTES, VALS, EXPAND)                \
__global__ void NAME(const uint8_t *w, const uint32_t *d_token,              \
                     float *out, size_t cols, size_t row_bytes)              \
{                                                                            \
    const uint8_t *rw = w + (size_t)(*d_token) * row_bytes;                  \
    const size_t n_blocks = cols / (VALS);                                   \
    for (size_t b = blockIdx.x * blockDim.x + threadIdx.x; b < n_blocks;     \
         b += (size_t)gridDim.x * blockDim.x)                                \
        EXPAND(rw + b * (BLOCK_BYTES), out + b * (VALS));                    \
}

#define MQ_DEFINE_GETROW_BATCH(NAME, BLOCK_BYTES, VALS, EXPAND)              \
__global__ void NAME(const uint8_t *w, const uint32_t *d_ids, float *out,    \
                     size_t cols, size_t row_bytes, size_t out_stride)       \
{                                                                            \
    const uint8_t *rw = w + (size_t)d_ids[blockIdx.y] * row_bytes;           \
    const size_t n_blocks = cols / (VALS);                                   \
    float *o = out + (size_t)blockIdx.y * out_stride;                        \
    for (size_t b = blockIdx.x; b < n_blocks; b += (size_t)gridDim.x)        \
        EXPAND(rw + b * (BLOCK_BYTES), o + b * (VALS));                      \
}

MQ_DEFINE_GETROW_DEV(k_mmq_getrow_dev_q4k,    MQ_DEV_Q4_K_SIZE,   MQ_QK_K,  mq_q4k_expand)
MQ_DEFINE_GETROW_DEV(k_mmq_getrow_dev_q5k,    MQ_DEV_Q5_K_SIZE,   MQ_QK_K,  mq_q5k_expand)
MQ_DEFINE_GETROW_DEV(k_mmq_getrow_dev_q6k,    MQ_DEV_Q6_K_SIZE,   MQ_QK_K,  mq_q6k_expand)
MQ_DEFINE_GETROW_DEV(k_mmq_getrow_dev_q8_0,   MQ_DEV_Q8_0_SIZE,   MQ_QK8_0, mq_q8_0_expand)
MQ_DEFINE_GETROW_DEV(k_mmq_getrow_dev_q4_0,   MQ_DEV_Q4_0_SIZE,   MQ_QK8_0, mq_q4_0_expand)
MQ_DEFINE_GETROW_DEV(k_mmq_getrow_dev_q5_0,   MQ_DEV_Q5_0_SIZE,   MQ_QK8_0, mq_q5_0_expand)
MQ_DEFINE_GETROW_DEV(k_mmq_getrow_dev_al5_xs, MQ_DEV_AL5_XS_SIZE, MQ_QK8_0, mq_al5_xs_expand)
MQ_DEFINE_GETROW_DEV(k_mmq_getrow_dev_iq4_xs, MQ_DEV_IQ4_XS_SIZE, MQ_QK_K,  mq_iq4_xs_expand)

MQ_DEFINE_GETROW_BATCH(k_mmq_getrow_batch_q4k,    MQ_DEV_Q4_K_SIZE,   MQ_QK_K,  mq_q4k_expand)
MQ_DEFINE_GETROW_BATCH(k_mmq_getrow_batch_q5k,    MQ_DEV_Q5_K_SIZE,   MQ_QK_K,  mq_q5k_expand)
MQ_DEFINE_GETROW_BATCH(k_mmq_getrow_batch_q6k,    MQ_DEV_Q6_K_SIZE,   MQ_QK_K,  mq_q6k_expand)
MQ_DEFINE_GETROW_BATCH(k_mmq_getrow_batch_q8_0,   MQ_DEV_Q8_0_SIZE,   MQ_QK8_0, mq_q8_0_expand)
MQ_DEFINE_GETROW_BATCH(k_mmq_getrow_batch_q4_0,   MQ_DEV_Q4_0_SIZE,   MQ_QK8_0, mq_q4_0_expand)
MQ_DEFINE_GETROW_BATCH(k_mmq_getrow_batch_q5_0,   MQ_DEV_Q5_0_SIZE,   MQ_QK8_0, mq_q5_0_expand)
MQ_DEFINE_GETROW_BATCH(k_mmq_getrow_batch_al5_xs, MQ_DEV_AL5_XS_SIZE, MQ_QK8_0, mq_al5_xs_expand)
MQ_DEFINE_GETROW_BATCH(k_mmq_getrow_batch_iq4_xs, MQ_DEV_IQ4_XS_SIZE, MQ_QK_K,  mq_iq4_xs_expand)

/* ─── Batched matvec (prompt prefill) ─────────────────────────────────────
 *
 * Same warp-per-row structure as the single-token matvec, but the warp then
 * walks n_tok activation vectors against the row it already fetched. The
 * row's packed bytes re-read from L1/L2 for tokens 2..n_tok instead of from
 * DRAM once per token, so the whole prompt chunk costs ~one weight sweep.
 * The token loop is outermost to keep the accumulator scalar (a per-token
 * register array would spill); the row stays cache-hot across iterations. */
#define MQ_DEFINE_MATVEC_BATCH(NAME, DOT)                                    \
__global__ void NAME(const uint8_t *w, const float *x, float *out,           \
                     size_t rows, uint32_t n_groups, size_t row_bytes,       \
                     uint32_t n_tok, size_t x_stride, size_t out_stride)     \
{                                                                            \
    const uint32_t lane = threadIdx.x & 31u;                                 \
    const size_t row = (size_t)blockIdx.x * MQ_WARPS + (threadIdx.x >> 5);   \
    if (row >= rows) return;                                                 \
    const uint8_t *rw = w + row * row_bytes;                                 \
    for (uint32_t t = 0u; t < n_tok; t++) {                                  \
        const float *xt = x + (size_t)t * x_stride;                          \
        float partial = 0.0f;                                                \
        for (uint32_t g = lane; g < n_groups; g += 32u)                      \
            partial += DOT(rw, g, xt + (size_t)g * 32u);                     \
        partial = mq_warp_reduce(partial);                                   \
        if (lane == 0u) out[(size_t)t * out_stride + row] = partial;         \
    }                                                                        \
}

MQ_DEFINE_MATVEC_BATCH(k_mmq_mvb_q4k,    mq_q4k_group_dot)
MQ_DEFINE_MATVEC_BATCH(k_mmq_mvb_q5k,    mq_q5k_group_dot)
MQ_DEFINE_MATVEC_BATCH(k_mmq_mvb_q6k,    mq_q6k_group_dot)
MQ_DEFINE_MATVEC_BATCH(k_mmq_mvb_q8_0,   mq_q8_0_group_dot)
MQ_DEFINE_MATVEC_BATCH(k_mmq_mvb_q4_0,   mq_q4_0_group_dot)
MQ_DEFINE_MATVEC_BATCH(k_mmq_mvb_q5_0,   mq_q5_0_group_dot)
MQ_DEFINE_MATVEC_BATCH(k_mmq_mvb_al5_xs, mq_al5_xs_group_dot)
MQ_DEFINE_MATVEC_BATCH(k_mmq_mvb_iq4_xs, mq_iq4_xs_group_dot)

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

extern "C" bool oc_cuda_mmq_get_row_dev(uint32_t qtype,
                                        const void *d_weights,
                                        const uint32_t *d_token,
                                        float *d_out, size_t cols,
                                        void *stream)
{
    if (!d_weights || !d_token || !d_out) return false;
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
        k_mmq_getrow_dev_q4k<<<grid, block, 0, s>>>(w, d_token, d_out, cols, row_bytes);
        break;
    case OC_QUANT_Q5_K_S:
    case OC_QUANT_Q5_K_M:
        k_mmq_getrow_dev_q5k<<<grid, block, 0, s>>>(w, d_token, d_out, cols, row_bytes);
        break;
    case OC_QUANT_Q6_K:
        k_mmq_getrow_dev_q6k<<<grid, block, 0, s>>>(w, d_token, d_out, cols, row_bytes);
        break;
    case OC_QUANT_Q8_0:
    case OC_QUANT_AL8:
        k_mmq_getrow_dev_q8_0<<<grid, block, 0, s>>>(w, d_token, d_out, cols, row_bytes);
        break;
    case OC_QUANT_Q4_0:
    case OC_QUANT_AL5:
        k_mmq_getrow_dev_q4_0<<<grid, block, 0, s>>>(w, d_token, d_out, cols, row_bytes);
        break;
    case OC_QUANT_Q5_0:
    case OC_QUANT_AL6:
        k_mmq_getrow_dev_q5_0<<<grid, block, 0, s>>>(w, d_token, d_out, cols, row_bytes);
        break;
    case OC_QUANT_AL5_XS:
        k_mmq_getrow_dev_al5_xs<<<grid, block, 0, s>>>(w, d_token, d_out, cols, row_bytes);
        break;
    case OC_QUANT_IQ4_XS:
        k_mmq_getrow_dev_iq4_xs<<<grid, block, 0, s>>>(w, d_token, d_out, cols, row_bytes);
        break;
    default:
        return false;
    }
    return cudaGetLastError() == cudaSuccess;
}

extern "C" bool oc_cuda_mmq_get_row_batch(uint32_t qtype,
                                          const void *d_weights,
                                          const uint32_t *d_ids,
                                          float *d_out, size_t cols,
                                          size_t out_stride,
                                          uint32_t n_tok, void *stream)
{
    if (!d_weights || !d_ids || !d_out || n_tok == 0) return false;
    const size_t row_bytes = oc_cuda_mmq_row_bytes(qtype, cols);
    if (row_bytes == 0) return false;

    size_t sb = 0, db = 0, vals = 0;
    mq_layout(qtype, &sb, &db, &vals);
    const size_t n_blocks = cols / vals;
    const uint32_t block = MQ_BLOCK_THREADS;
    const uint32_t gx = (uint32_t)((n_blocks + block - 1) / block);
    dim3 grid(gx, n_tok);
    cudaStream_t s = (cudaStream_t)stream;
    const uint8_t *w = (const uint8_t *)d_weights;

    switch ((OcGgufQuantizationType)qtype) {
    case OC_QUANT_Q4_K_S:
    case OC_QUANT_Q4_K_M:
        k_mmq_getrow_batch_q4k<<<grid, block, 0, s>>>(w, d_ids, d_out, cols, row_bytes, out_stride);
        break;
    case OC_QUANT_Q5_K_S:
    case OC_QUANT_Q5_K_M:
        k_mmq_getrow_batch_q5k<<<grid, block, 0, s>>>(w, d_ids, d_out, cols, row_bytes, out_stride);
        break;
    case OC_QUANT_Q6_K:
        k_mmq_getrow_batch_q6k<<<grid, block, 0, s>>>(w, d_ids, d_out, cols, row_bytes, out_stride);
        break;
    case OC_QUANT_Q8_0:
    case OC_QUANT_AL8:
        k_mmq_getrow_batch_q8_0<<<grid, block, 0, s>>>(w, d_ids, d_out, cols, row_bytes, out_stride);
        break;
    case OC_QUANT_Q5_0:
    case OC_QUANT_AL6:
        k_mmq_getrow_batch_q5_0<<<grid, block, 0, s>>>(w, d_ids, d_out, cols, row_bytes, out_stride);
        break;
    case OC_QUANT_Q4_0:
    case OC_QUANT_AL5:
        k_mmq_getrow_batch_q4_0<<<grid, block, 0, s>>>(w, d_ids, d_out, cols, row_bytes, out_stride);
        break;
    case OC_QUANT_AL5_XS:
        k_mmq_getrow_batch_al5_xs<<<grid, block, 0, s>>>(w, d_ids, d_out, cols, row_bytes, out_stride);
        break;
    case OC_QUANT_IQ4_XS:
        k_mmq_getrow_batch_iq4_xs<<<grid, block, 0, s>>>(w, d_ids, d_out, cols, row_bytes, out_stride);
        break;
    default:
        return false;
    }
    return cudaGetLastError() == cudaSuccess;
}

extern "C" bool oc_cuda_mmq_matvec_batch(uint32_t qtype,
                                         const void *d_weights,
                                         const float *d_x, float *d_out,
                                         uint32_t n_tok, size_t x_stride,
                                         size_t out_stride,
                                         size_t rows, size_t cols,
                                         void *stream)
{
    if (!d_weights || !d_x || !d_out || rows == 0 || n_tok == 0) return false;
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
        k_mmq_mvb_q4k<<<grid, block, 0, s>>>(w, d_x, d_out, rows, n_groups, row_bytes, n_tok, x_stride, out_stride);
        break;
    case OC_QUANT_Q5_K_S:
    case OC_QUANT_Q5_K_M:
        k_mmq_mvb_q5k<<<grid, block, 0, s>>>(w, d_x, d_out, rows, n_groups, row_bytes, n_tok, x_stride, out_stride);
        break;
    case OC_QUANT_Q6_K:
        k_mmq_mvb_q6k<<<grid, block, 0, s>>>(w, d_x, d_out, rows, n_groups, row_bytes, n_tok, x_stride, out_stride);
        break;
    case OC_QUANT_Q8_0:
    case OC_QUANT_AL8:
        k_mmq_mvb_q8_0<<<grid, block, 0, s>>>(w, d_x, d_out, rows, n_groups, row_bytes, n_tok, x_stride, out_stride);
        break;
    case OC_QUANT_Q4_0:
    case OC_QUANT_AL5:
        k_mmq_mvb_q4_0<<<grid, block, 0, s>>>(w, d_x, d_out, rows, n_groups, row_bytes, n_tok, x_stride, out_stride);
        break;
    case OC_QUANT_Q5_0:
    case OC_QUANT_AL6:
        k_mmq_mvb_q5_0<<<grid, block, 0, s>>>(w, d_x, d_out, rows, n_groups, row_bytes, n_tok, x_stride, out_stride);
        break;
    case OC_QUANT_AL5_XS:
        k_mmq_mvb_al5_xs<<<grid, block, 0, s>>>(w, d_x, d_out, rows, n_groups, row_bytes, n_tok, x_stride, out_stride);
        break;
    case OC_QUANT_IQ4_XS:
        k_mmq_mvb_iq4_xs<<<grid, block, 0, s>>>(w, d_x, d_out, rows, n_groups, row_bytes, n_tok, x_stride, out_stride);
        break;
    default:
        return false;
    }
    return cudaGetLastError() == cudaSuccess;
}

/* ─── dp4a path: int8-quantized activations ────────────────────────────────
 *
 * The f32 group-dots above spend ~6 instructions per element on nibble
 * extraction, int→float conversion and scalar FMA — the kernels run at full
 * SM occupancy yet move ~35% of peak DRAM bandwidth on a T4 because they are
 * issue-bound, not memory-bound. This section removes that ceiling the same
 * way every fast int matvec does: quantize the activation to int8 once per
 * 32-element group (with the group's fp32 scale and exact fp32 sum kept
 * alongside), then accumulate the weight dot with __dp4a, which retires four
 * byte-multiply-adds per instruction. The per-group work drops roughly 7x.
 *
 * Numerics: the activation is approximated as x ≈ xs*xq (int8, per-group
 * scale), so results differ slightly from the f32 path — the same class of
 * approximation llama.cpp's mmq makes. Offset formats (Q4_0's -8, Q5_0's
 * -16, Q6_K's -32) are corrected with the exact integer activation sums, so
 * the only loss is the activation's int8 rounding, not the weights'.
 *
 * Supported: Q4_K/Q5_K/Q6_K/Q8_0/Q4_0/Q5_0 and aliases. AL5_XS/IQ4_XS keep
 * the f32 kernels (check oc_cuda_mmq_xq_supported before dispatching). */

#define MQX_MAX_TOK 16u

/* CUDA exposes only __dp4a(int,int,int) and (uint,uint,uint); mixed-sign
 * calls are ambiguous. Our packed weight bytes are always <= 63 (nibbles
 * plus carried high bits), so they are positive int8 and the signed
 * overload computes exactly u8*w × s8*x. */
__device__ __forceinline__ int32_t mq_dp4a(uint32_t w, uint32_t x, int32_t c)
{
    return __dp4a((int)w, (int)x, c);
}

/* Load one 32-byte int8 activation group as 8 words via two 16B loads. */
__device__ __forceinline__ void mqx_load8(const int8_t *xq, uint32_t t,
                                          uint32_t ng, uint32_t g,
                                          uint32_t v[8])
{
    const uint4 a = *reinterpret_cast<const uint4 *>(
        xq + ((size_t)t * ng + g) * 32u);
    const uint4 b = *reinterpret_cast<const uint4 *>(
        xq + ((size_t)t * ng + g) * 32u + 16u);
    v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w;
    v[4] = b.x; v[5] = b.y; v[6] = b.z; v[7] = b.w;
}

/* Quantize one [n_tok][n_groups*32] activation tile. One warp per group:
 * lane l owns element g*32+l, so the amax/sum reductions are pure shuffles.
 * Produces, per group: xq int8 (amax→127), xs fp32 scale, xsum fp32 exact
 * group sum (K-quant min term), xqsum int8 sums per 16-elem half (0-quant
 * offsets). Grid.x covers groups, grid.y tokens. */
__global__ void k_mmq_quant_x(const float *x, int8_t *xq, float2 *xsc,
                              int32_t *xqsum,
                              uint32_t n_groups, size_t x_stride)
{
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t g = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (g >= n_groups) return;
    const size_t gi = (size_t)blockIdx.y * n_groups + g;
    const float *xr = x + (size_t)blockIdx.y * x_stride + (size_t)g * 32u;

    float v = xr[lane];
    float amax = fabsf(v);
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        amax = fmaxf(amax, __shfl_down_sync(0xffffffffu, amax, o));
    amax = __shfl_sync(0xffffffffu, amax, 0);
    const float s = amax > 0.0f ? amax * (1.0f / 127.0f) : 1.0f;

    int32_t qi = __float2int_rn(v / s);
    qi = qi > 127 ? 127 : (qi < -127 ? -127 : qi);
    xq[gi * 32u + lane] = (int8_t)qi;

    float sm = v;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        sm += __shfl_down_sync(0xffffffffu, sm, o);
    if (lane == 0u) xsc[gi] = make_float2(s, sm);

    int32_t iq = qi;
    #pragma unroll
    for (int o = 8; o > 0; o >>= 1)
        iq += __shfl_down_sync(0xffffffffu, iq, o, 16);
    if (lane == 0u || lane == 16u)
        xqsum[gi * 2u + (lane >> 4)] = iq;
}

/* ── Per-group int dots. Each returns the group's float contribution for
 * every token: the weight bytes are loaded once and the token loop stays
 * inside so prefill amortizes the fetch. `acc` is [MQX_MAX_TOK] registers;
 * the unrolled loop keeps its indices constant so it never spills. */

/* Q4_K: nibbles are unsigned (bias lives in the min term).
 * out = d*sc*xs*Σxq·nib − mn*m*xsum. */
__device__ __forceinline__ void mxd_q4k(
    const uint8_t *row, uint32_t g,
    const int8_t *xq, const float2 *xsc,
    const int32_t * /*xqsum*/, uint32_t t0, uint32_t n_tok,
    uint32_t ng, float *acc)
{
    const uint8_t *blk = row + (size_t)(g >> 3) * MQ_DEV_Q4_K_SIZE;
    const uint32_t gw = g & 7u;
    const float d  = mq_f16(blk[0], blk[1]);
    const float mn = mq_f16(blk[2], blk[3]);
    uint8_t sc, m;
    mq_scale_min_k4(gw, blk + 4, &sc, &m);
    const uint8_t *qs = blk + 16 + (size_t)(gw >> 1) * 32u;
    const bool hi = (gw & 1u) != 0u;
    const uint4 w0 = *reinterpret_cast<const uint4 *>(qs);
    const uint4 w1 = *reinterpret_cast<const uint4 *>(qs + 16);
    const uint32_t wd[8] = { w0.x, w0.y, w0.z, w0.w,
                             w1.x, w1.y, w1.z, w1.w };
    const float dsc = d * (float)sc;
    const float mnm = mn * (float)m;
    #pragma unroll
    for (uint32_t ti = 0u; ti < MQX_MAX_TOK; ti++) {
        const uint32_t t = t0 + ti;
        if (t >= n_tok) break;
        uint32_t xv[8];
        mqx_load8(xq, t, ng, g, xv);
        int32_t qx = 0;
        #pragma unroll
        for (uint32_t i = 0u; i < 8u; i++) {
            const uint32_t nib = hi ? ((wd[i] >> 4) & 0x0F0F0F0Fu)
                                    : (wd[i] & 0x0F0F0F0Fu);
            qx = mq_dp4a(nib, xv[i], qx);
        }
        const size_t gi = (size_t)t * ng + g;
        const float2 sc2 = xsc[gi];
        acc[ti] += dsc * (sc2.x * (float)qx) - mnm * sc2.y;
    }
}

/* Q5_K: nibble | (bit gw of qh byte) << 4 — unsigned 0..31, same
 * scale/min factoring as Q4_K. */
__device__ __forceinline__ void mxd_q5k(
    const uint8_t *row, uint32_t g,
    const int8_t *xq, const float2 *xsc,
    const int32_t * /*xqsum*/, uint32_t t0, uint32_t n_tok,
    uint32_t ng, float *acc)
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
    const uint4 w0 = *reinterpret_cast<const uint4 *>(qs);
    const uint4 w1 = *reinterpret_cast<const uint4 *>(qs + 16);
    const uint4 h0 = *reinterpret_cast<const uint4 *>(qh);
    const uint4 h1 = *reinterpret_cast<const uint4 *>(qh + 16);
    const uint32_t wd[8] = { w0.x, w0.y, w0.z, w0.w,
                             w1.x, w1.y, w1.z, w1.w };
    const uint32_t hd[8] = { h0.x, h0.y, h0.z, h0.w,
                             h1.x, h1.y, h1.z, h1.w };
    const float dsc = d * (float)sc;
    const float mnm = mn * (float)m;
    #pragma unroll
    for (uint32_t ti = 0u; ti < MQX_MAX_TOK; ti++) {
        const uint32_t t = t0 + ti;
        if (t >= n_tok) break;
        uint32_t xv[8];
        mqx_load8(xq, t, ng, g, xv);
        int32_t qx = 0;
        #pragma unroll
        for (uint32_t i = 0u; i < 8u; i++) {
            const uint32_t nib = hi ? ((wd[i] >> 4) & 0x0F0F0F0Fu)
                                    : (wd[i] & 0x0F0F0F0Fu);
            const uint32_t hb = ((hd[i] >> gw) & 0x01010101u) << 4;
            qx = mq_dp4a(nib | hb, xv[i], qx);
        }
        const size_t gi = (size_t)t * ng + g;
        const float2 sc2 = xsc[gi];
        acc[ti] += dsc * (sc2.x * (float)qx) - mnm * sc2.y;
    }
}

/* Q6_K: 6-bit fields (ql nibble | qh 2 bits << 4) − 32, with the int8 scale
 * switching at element 16. Two half accumulators honor that boundary; the
 * −32 offset cancels against the per-half activation sums. */
__device__ __forceinline__ void mxd_q6k(
    const uint8_t *row, uint32_t g,
    const int8_t *xq, const float2 *xsc,
    const int32_t *xqsum, uint32_t t0, uint32_t n_tok,
    uint32_t ng, float *acc)
{
    const uint8_t *blk = row + (size_t)(g >> 3) * MQ_DEV_Q6_K_SIZE;
    const uint32_t G = g & 7u;
    const uint32_t half = G >> 2;
    const uint32_t j = G & 3u;
    const float d = mq_f16(blk[208], blk[209]);
    const uint8_t *ql = blk + (size_t)half * 64u + ((j & 1u) ? 32u : 0u);
    const uint8_t *qh = blk + 128 + (size_t)half * 32u;
    const int8_t  *sc = (const int8_t *)(blk + 192) + (size_t)half * 8u
                      + (size_t)j * 2u;
    const uint32_t shift = j * 2u;
    const uint4 l0 = *reinterpret_cast<const uint4 *>(ql);
    const uint4 l1 = *reinterpret_cast<const uint4 *>(ql + 16);
    const uint4 h0 = *reinterpret_cast<const uint4 *>(qh);
    const uint4 h1 = *reinterpret_cast<const uint4 *>(qh + 16);
    const uint32_t lw[8] = { l0.x, l0.y, l0.z, l0.w, l1.x, l1.y, l1.z, l1.w };
    const uint32_t hw[8] = { h0.x, h0.y, h0.z, h0.w, h1.x, h1.y, h1.z, h1.w };
    const float s0 = (float)sc[0], s1 = (float)sc[1];
    #pragma unroll
    for (uint32_t ti = 0u; ti < MQX_MAX_TOK; ti++) {
        const uint32_t t = t0 + ti;
        if (t >= n_tok) break;
        uint32_t xv[8];
        mqx_load8(xq, t, ng, g, xv);
        int32_t pa = 0, pb = 0;
        #pragma unroll
        for (uint32_t i = 0u; i < 8u; i++) {
            const uint32_t lo = (j < 2u) ? (lw[i] & 0x0F0F0F0Fu)
                                        : ((lw[i] >> 4) & 0x0F0F0F0Fu);
            const uint32_t hb = ((hw[i] >> shift) & 0x03030303u) << 4;
            if (i < 4u) pa = mq_dp4a(lo | hb, xv[i], pa);
            else        pb = mq_dp4a(lo | hb, xv[i], pb);
        }
        const size_t gi = (size_t)t * ng + g;
        const float xsg = xsc[gi].x;
        const int2 q2 = *reinterpret_cast<const int2 *>(xqsum + gi * 2u);
        acc[ti] += d * xsg * (s0 * (float)(pa - 32 * q2.x)
                           + s1 * (float)(pb - 32 * q2.y));
    }
}

/* Q8_0: weights are already int8 — a pure s8·s8 dot. */
__device__ __forceinline__ void mxd_q8_0(
    const uint8_t *row, uint32_t g,
    const int8_t *xq, const float2 *xsc,
    const int32_t * /*xqsum*/, uint32_t t0, uint32_t n_tok,
    uint32_t ng, float *acc)
{
    const uint8_t *blk = row + (size_t)g * MQ_DEV_Q8_0_SIZE;
    const float d = mq_f16(blk[0], blk[1]);
    const uint4 q0 = *reinterpret_cast<const uint4 *>(blk + 2);
    const uint4 q1 = *reinterpret_cast<const uint4 *>(blk + 18);
    const uint32_t wd[8] = { q0.x, q0.y, q0.z, q0.w,
                             q1.x, q1.y, q1.z, q1.w };
    #pragma unroll
    for (uint32_t ti = 0u; ti < MQX_MAX_TOK; ti++) {
        const uint32_t t = t0 + ti;
        if (t >= n_tok) break;
        uint32_t xv[8];
        mqx_load8(xq, t, ng, g, xv);
        int32_t qx = 0;
        #pragma unroll
        for (uint32_t i = 0u; i < 8u; i++)
            qx = mq_dp4a(wd[i], xv[i], qx);
        const size_t gi = (size_t)t * ng + g;
        acc[ti] += d * xsc[gi].x * (float)qx;
    }
}

/* Q4_0: value = (nibble − 8)*d. Low nibble of byte j → element j, high →
 * element j+16, matching the fp kernel's pairing. The −8 offset folds into
 * the whole-group activation sum. */
__device__ __forceinline__ void mxd_q4_0(
    const uint8_t *row, uint32_t g,
    const int8_t *xq, const float2 *xsc,
    const int32_t *xqsum, uint32_t t0, uint32_t n_tok,
    uint32_t ng, float *acc)
{
    const uint8_t *blk = row + (size_t)g * MQ_DEV_Q4_0_SIZE;
    const float d = mq_f16(blk[0], blk[1]);
    const uint4 q0 = *reinterpret_cast<const uint4 *>(blk + 2);
    const uint32_t wd[4] = { q0.x, q0.y, q0.z, q0.w };
    #pragma unroll
    for (uint32_t ti = 0u; ti < MQX_MAX_TOK; ti++) {
        const uint32_t t = t0 + ti;
        if (t >= n_tok) break;
        uint32_t xv[8];
        mqx_load8(xq, t, ng, g, xv);
        int32_t qx = 0;
        #pragma unroll
        for (uint32_t i = 0u; i < 4u; i++) {
            qx = mq_dp4a(wd[i] & 0x0F0F0F0Fu, xv[i], qx);
            qx = mq_dp4a((wd[i] >> 4) & 0x0F0F0F0Fu, xv[i + 4u], qx);
        }
        const size_t gi = (size_t)t * ng + g;
        const int2 q2 = *reinterpret_cast<const int2 *>(xqsum + gi * 2u);
        acc[ti] += d * xsc[gi].x * (float)(qx - 8 * (q2.x + q2.y));
    }
}

/* Q5_0: value = ((nibble | hbit<<4) − 16)*d. The 32 high bits live in a
 * 4-byte field; mask*0x00204081 scatters bits 0..3 into byte lanes. */
__device__ __forceinline__ void mxd_q5_0(
    const uint8_t *row, uint32_t g,
    const int8_t *xq, const float2 *xsc,
    const int32_t *xqsum, uint32_t t0, uint32_t n_tok,
    uint32_t ng, float *acc)
{
    const uint8_t *blk = row + (size_t)g * MQ_DEV_Q5_0_SIZE;
    const float d = mq_f16(blk[0], blk[1]);
    const uint32_t qh = (uint32_t)blk[2] | ((uint32_t)blk[3] << 8)
                      | ((uint32_t)blk[4] << 16) | ((uint32_t)blk[5] << 24);
    const uint4 q0 = *reinterpret_cast<const uint4 *>(blk + 6);
    const uint32_t wd[4] = { q0.x, q0.y, q0.z, q0.w };
    #pragma unroll
    for (uint32_t ti = 0u; ti < MQX_MAX_TOK; ti++) {
        const uint32_t t = t0 + ti;
        if (t >= n_tok) break;
        uint32_t xv[8];
        mqx_load8(xq, t, ng, g, xv);
        int32_t qx = 0;
        #pragma unroll
        for (uint32_t i = 0u; i < 4u; i++) {
            const uint32_t hbl =
                (((qh >> (4u * i)) & 0xFu) * 0x00204081u) & 0x01010101u;
            const uint32_t hbh =
                (((qh >> (16u + 4u * i)) & 0xFu) * 0x00204081u) & 0x01010101u;
            qx = mq_dp4a((wd[i] & 0x0F0F0F0Fu) | (hbl << 4), xv[i], qx);
            qx = mq_dp4a(((wd[i] >> 4) & 0x0F0F0F0Fu) | (hbh << 4),
                         xv[i + 4u], qx);
        }
        const size_t gi = (size_t)t * ng + g;
        const int2 q2 = *reinterpret_cast<const int2 *>(xqsum + gi * 2u);
        acc[ti] += d * xsc[gi].x * (float)(qx - 16 * (q2.x + q2.y));
    }
}

/* One warp per output row, same shape as the f32 kernels; the lane strides
 * groups and keeps a per-token int8/dot accumulator set in registers. */
#define MQ_DEFINE_MATVEC_XQ(NAME, GRPDOT)                                    \
__global__ void NAME(const uint8_t *w, const int8_t *xq, const float2 *xsc,\
                     const int32_t *xqsum, float *out,                       \
                     uint32_t n_tok, size_t out_stride,                      \
                     size_t rows, uint32_t n_groups, size_t row_bytes)       \
{                                                                            \
    const uint32_t lane = threadIdx.x & 31u;                                 \
    const uint32_t t0 = blockIdx.x * MQX_MAX_TOK;                            \
    if (t0 >= n_tok) return;                                                 \
    const size_t row = (size_t)blockIdx.y * MQ_WARPS + (threadIdx.x >> 5);   \
    if (row >= rows) return;                                                 \
    const uint8_t *rw = w + row * row_bytes;                                 \
    float acc[MQX_MAX_TOK];                                                  \
    _Pragma("unroll")                                                        \
    for (uint32_t i = 0u; i < MQX_MAX_TOK; i++) acc[i] = 0.0f;               \
    for (uint32_t g = lane; g < n_groups; g += 32u)                          \
        GRPDOT(rw, g, xq, xsc, xqsum, t0, n_tok, n_groups, acc);             \
    _Pragma("unroll")                                                        \
    for (uint32_t i = 0u; i < MQX_MAX_TOK; i++) {                            \
        const uint32_t t = t0 + i;                                           \
        if (t >= n_tok) break;                                               \
        const float v = mq_warp_reduce(acc[i]);                              \
        if (lane == 0u) out[(size_t)t * out_stride + row] = v;               \
    }                                                                        \
}

MQ_DEFINE_MATVEC_XQ(k_mmq_mvx_q4k,  mxd_q4k)
MQ_DEFINE_MATVEC_XQ(k_mmq_mvx_q5k,  mxd_q5k)
MQ_DEFINE_MATVEC_XQ(k_mmq_mvx_q6k,  mxd_q6k)
MQ_DEFINE_MATVEC_XQ(k_mmq_mvx_q8_0, mxd_q8_0)
MQ_DEFINE_MATVEC_XQ(k_mmq_mvx_q4_0, mxd_q4_0)
MQ_DEFINE_MATVEC_XQ(k_mmq_mvx_q5_0, mxd_q5_0)

extern "C" bool oc_cuda_mmq_xq_supported(uint32_t qtype, size_t cols)
{
    if (cols == 0 || (cols % 32u) != 0u) return false;
    switch ((OcGgufQuantizationType)qtype) {
    case OC_QUANT_Q4_K_S: case OC_QUANT_Q4_K_M:
    case OC_QUANT_Q5_K_S: case OC_QUANT_Q5_K_M:
    case OC_QUANT_Q6_K:
    case OC_QUANT_Q8_0:   case OC_QUANT_AL8:
    case OC_QUANT_Q4_0:   case OC_QUANT_AL5:
    case OC_QUANT_Q5_0:   case OC_QUANT_AL6:
        /* K-quants additionally need whole super-blocks per row. */
        return oc_cuda_mmq_supported(qtype, cols);
    default:
        return false;
    }
}

extern "C" bool oc_cuda_mmq_quantize_x(const float *d_x, int8_t *d_xq,
                                       float *d_xsc,
                                       int32_t *d_xqsum,
                                       uint32_t n_tok, size_t cols,
                                       size_t x_stride, void *stream)
{
    if (!d_x || !d_xq || !d_xsc || !d_xqsum || n_tok == 0 ||
        cols == 0 || (cols % 32u) != 0u || n_tok > 4u * MQX_MAX_TOK)
        return false;
    const uint32_t n_groups = (uint32_t)(cols / 32u);
    dim3 grid((n_groups + 3u) / 4u, n_tok);
    k_mmq_quant_x<<<grid, 128u, 0, (cudaStream_t)stream>>>(
        d_x, d_xq, (float2 *)d_xsc, d_xqsum, n_groups, x_stride);
    return cudaGetLastError() == cudaSuccess;
}

extern "C" bool oc_cuda_mmq_matvec_xq(uint32_t qtype, const void *d_weights,
                                      const int8_t *d_xq, const float *d_xsc,
                                      const int32_t *d_xqsum,
                                      float *d_out, uint32_t n_tok,
                                      size_t out_stride, size_t rows,
                                      size_t cols, void *stream)
{
    if (!d_weights || !d_xq || !d_xsc || !d_xqsum || !d_out ||
        rows == 0 || n_tok == 0 || n_tok > 4u * MQX_MAX_TOK)
        return false;
    const size_t row_bytes = oc_cuda_mmq_row_bytes(qtype, cols);
    if (row_bytes == 0) return false;

    const uint32_t n_groups = (uint32_t)(cols / 32u);
    const uint32_t block = MQ_WARPS * 32u;
    const dim3 grid((n_tok + MQX_MAX_TOK - 1u) / MQX_MAX_TOK,
                    (unsigned)((rows + MQ_WARPS - 1u) / MQ_WARPS));
    cudaStream_t s = (cudaStream_t)stream;
    const uint8_t *w = (const uint8_t *)d_weights;

    switch ((OcGgufQuantizationType)qtype) {
    case OC_QUANT_Q4_K_S:
    case OC_QUANT_Q4_K_M:
        k_mmq_mvx_q4k<<<grid, block, 0, s>>>(w, d_xq, (const float2 *)d_xsc, d_xqsum,
            d_out, n_tok, out_stride, rows, n_groups, row_bytes);
        break;
    case OC_QUANT_Q5_K_S:
    case OC_QUANT_Q5_K_M:
        k_mmq_mvx_q5k<<<grid, block, 0, s>>>(w, d_xq, (const float2 *)d_xsc, d_xqsum,
            d_out, n_tok, out_stride, rows, n_groups, row_bytes);
        break;
    case OC_QUANT_Q6_K:
        k_mmq_mvx_q6k<<<grid, block, 0, s>>>(w, d_xq, (const float2 *)d_xsc, d_xqsum,
            d_out, n_tok, out_stride, rows, n_groups, row_bytes);
        break;
    case OC_QUANT_Q8_0:
    case OC_QUANT_AL8:
        k_mmq_mvx_q8_0<<<grid, block, 0, s>>>(w, d_xq, (const float2 *)d_xsc, d_xqsum,
            d_out, n_tok, out_stride, rows, n_groups, row_bytes);
        break;
    case OC_QUANT_Q4_0:
    case OC_QUANT_AL5:
        k_mmq_mvx_q4_0<<<grid, block, 0, s>>>(w, d_xq, (const float2 *)d_xsc, d_xqsum,
            d_out, n_tok, out_stride, rows, n_groups, row_bytes);
        break;
    case OC_QUANT_Q5_0:
    case OC_QUANT_AL6:
        k_mmq_mvx_q5_0<<<grid, block, 0, s>>>(w, d_xq, (const float2 *)d_xsc, d_xqsum,
            d_out, n_tok, out_stride, rows, n_groups, row_bytes);
        break;
    default:
        return false;
    }
    return cudaGetLastError() == cudaSuccess;
}
