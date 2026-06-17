// CUDA kernels for the oxidize GPU backend.
//
// Compiled to PTX by `build.rs` via `nvcc` at build time (the CUDA toolkit is
// already required by the `cuda` feature). Generating PTX from source — rather
// than checking in hand-written PTX — guarantees validity for the installed
// toolkit and forward-JIT-compatibility with newer GPUs (e.g. Blackwell /
// sm_120, where deprecated PTX `.target` levels are rejected).
//
// GPU-first design: quantized weights are uploaded once, dequantized on the GPU
// to half precision (f16), and kept resident in VRAM. Storing weights as f16
// (2 bytes/param) keeps large models within VRAM (a 32B model ~= 64 GB) and the
// matmul reads them directly. Half values are carried as raw `unsigned short`
// bit patterns so the Rust side only needs `DeviceBuffer<u16>` (no f16 Rust type).
//
// Entry-point names must stay in sync with the *_KERNEL_NAME constants in
// `backends/cuda.rs`.

#include <cuda_fp16.h>

// --------------------------------------------------------------------------
// Dense matrix-vector products: output[row] = sum_c matrix[row*cols + c] * vector[c]
// --------------------------------------------------------------------------

// One warp (32 lanes) per output row. Lanes stride across `cols` so that
// adjacent lanes touch adjacent weights — fully coalesced VRAM reads — then a
// shuffle reduction sums the partials. Decode-time gemv is bandwidth-bound, so
// coalescing is what gets close to the GPU's peak memory throughput.
//
// Launch with `blockDim.x` a multiple of 32 and enough total warps to cover
// every row (grid sized as `ceil(rows * 32 / blockDim.x)`).

__device__ __forceinline__ float warp_reduce_sum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_down_sync(0xffffffffu, v, offset);
    return v;
}

// NOTE: This kernel is currently unused by the CUDA backend; cuBLAS sgemv is
// preferred for f32 GEMV.  It is kept in the PTX so that existing tests and
// the GEMV_KERNEL_NAME constant remain valid.
extern "C" __global__ void gemv_f32_kernel(
    const float* matrix, const float* vector, float* output,
    unsigned int rows, unsigned int cols)
{
    unsigned int global_thread = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int row = global_thread >> 5;     // one warp per row
    unsigned int lane = threadIdx.x & 31u;
    if (row >= rows) return;

    const float* w = matrix + (size_t)row * cols;
    float sum = 0.0f;
    for (unsigned int c = lane; c < cols; c += 32u)
        sum += w[c] * vector[c];

    sum = warp_reduce_sum(sum);
    if (lane == 0u) output[row] = sum;
}

// f16-weight variant: `matrix` holds half-precision weights as raw u16 bits.
// Processes two half weights per iteration with half2 + float2 loads.
extern "C" __global__ void gemv_f16_kernel(
    const unsigned short* matrix, const float* vector, float* output,
    unsigned int rows, unsigned int cols)
{
    unsigned int global_thread = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int row = global_thread >> 5;
    unsigned int lane = threadIdx.x & 31u;
    if (row >= rows) return;

    const __half* w = reinterpret_cast<const __half*>(matrix) + (size_t)row * cols;
    const float* v = vector;
    float sum = 0.0f;

    unsigned int c = lane * 2u;
    for (; c + 1u < cols; c += 64u) {
        __half2 wh = *reinterpret_cast<const __half2*>(w + c);
        float2 vf = *reinterpret_cast<const float2*>(v + c);
        float2 wf = __half22float2(wh);
        sum = fmaf(wf.x, vf.x, sum);
        sum = fmaf(wf.y, vf.y, sum);
    }
    if ((cols & 1u) != 0u && c < cols)
        sum = fmaf(__half2float(w[c]), v[c], sum);

    sum = warp_reduce_sum(sum);
    if (lane == 0u) output[row] = sum;
}

// --------------------------------------------------------------------------
// GPU dequantization kernels: one thread per quantized block, writing f16 bits.
// Block math mirrors the CPU reference in compute/quantization.rs exactly.
// --------------------------------------------------------------------------

// Q8_0: 34-byte block = f16 scale + 32 int8 weights -> 32 values.
extern "C" __global__ void dequant_q8_0_kernel(
    const unsigned char* in, unsigned short* out, unsigned int nblocks)
{
    unsigned int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nblocks) return;
    const unsigned char* blk = in + (size_t)b * 34;
    float d = __half2float(*reinterpret_cast<const __half*>(blk));
    const signed char* q = reinterpret_cast<const signed char*>(blk + 2);
    __half* o = reinterpret_cast<__half*>(out) + (size_t)b * 32;
    for (int i = 0; i < 32; i++)
        o[i] = __float2half(d * (float)q[i]);
}

// Q4_K scale/min unpack (mirrors get_scale_min_k4).
__device__ __forceinline__ void q4k_scale_min(
    int j, const unsigned char* scales, unsigned char* sc, unsigned char* mn)
{
    if (j < 4) {
        *sc = scales[j] & 63;
        *mn = scales[j + 4] & 63;
    } else {
        *sc = (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4);
        *mn = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
    }
}

// Q4_K: 144-byte block = f16 d + f16 min + 12 scale bytes + 128 quant bytes -> 256 values.
extern "C" __global__ void dequant_q4_k_kernel(
    const unsigned char* in, unsigned short* out, unsigned int nblocks)
{
    unsigned int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nblocks) return;
    const unsigned char* blk = in + (size_t)b * 144;
    float d = __half2float(*reinterpret_cast<const __half*>(blk));
    float mn = __half2float(*reinterpret_cast<const __half*>(blk + 2));
    const unsigned char* scales = blk + 4;
    const unsigned char* qs = blk + 16;
    __half* o = reinterpret_cast<__half*>(out) + (size_t)b * 256;

    unsigned int out_ptr = 0;
    int is = 0;
    for (int gp = 0; gp < 4; gp++) {
        unsigned int q_base = gp * 32;
        unsigned char sc1, m1, sc2, m2;
        q4k_scale_min(is, scales, &sc1, &m1);
        q4k_scale_min(is + 1, scales, &sc2, &m2);
        float d1 = d * (float)sc1, min1 = mn * (float)m1;
        float d2 = d * (float)sc2, min2 = mn * (float)m2;
        for (int l = 0; l < 32; l++)
            o[out_ptr + l] = __float2half(d1 * (float)(qs[q_base + l] & 0xF) - min1);
        for (int l = 0; l < 32; l++)
            o[out_ptr + 32 + l] = __float2half(d2 * (float)(qs[q_base + l] >> 4) - min2);
        out_ptr += 64;
        is += 2;
    }
}

// Q6_K: 210-byte block = 128 ql + 64 qh + 16 int8 scales + f16 d -> 256 values.
extern "C" __global__ void dequant_q6_k_kernel(
    const unsigned char* in, unsigned short* out, unsigned int nblocks)
{
    unsigned int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nblocks) return;
    const unsigned char* blk = in + (size_t)b * 210;
    const unsigned char* ql = blk;
    const unsigned char* qh = blk + 128;
    const signed char* sc = reinterpret_cast<const signed char*>(blk + 192);
    float d = __half2float(*reinterpret_cast<const __half*>(blk + 208));
    __half* o = reinterpret_cast<__half*>(out) + (size_t)b * 256;

    unsigned int q_ptr = 0;
    for (int g = 0; g < 2; g++) {
        int ql_off = g * 64, qh_off = g * 32, sc_off = g * 8;
        for (int l = 0; l < 32; l++) {
            int is = l / 16;
            int q1 = ((ql[ql_off + l] & 0xF) | (((qh[qh_off + l] & 3)) << 4)) - 32;
            int q2 = ((ql[ql_off + l + 32] & 0xF) | ((((qh[qh_off + l] >> 2) & 3)) << 4)) - 32;
            int q3 = ((ql[ql_off + l] >> 4) | ((((qh[qh_off + l] >> 4) & 3)) << 4)) - 32;
            int q4 = ((ql[ql_off + l + 32] >> 4) | ((((qh[qh_off + l] >> 6) & 3)) << 4)) - 32;
            o[q_ptr + l]      = __float2half(d * (float)sc[sc_off + is]     * (float)q1);
            o[q_ptr + 32 + l] = __float2half(d * (float)sc[sc_off + is + 2] * (float)q2);
            o[q_ptr + 64 + l] = __float2half(d * (float)sc[sc_off + is + 4] * (float)q3);
            o[q_ptr + 96 + l] = __float2half(d * (float)sc[sc_off + is + 6] * (float)q4);
        }
        q_ptr += 128;
    }
}

// --------------------------------------------------------------------------
// On-the-fly quantized GEMV kernels (no f16 materialization)
//
// These kernels read quantized weights directly and dequantize inside the
// dot-product loop.  This keeps VRAM usage at the compressed size (~1 byte
// per param for Q8_0, ~0.56 for Q4_0) instead of expanding to 2-byte f16.
// Essential for running 70B models on 4GB GPUs.
// --------------------------------------------------------------------------

// Q8_0 GEMV: each lane reads Q8_0 blocks [scale_f16 + 32 int8] and accumulates
// directly.  No intermediate f16 buffer.
extern "C" __global__ void gemv_q8_0_kernel(
    const unsigned char* matrix, const float* vector, float* output,
    unsigned int rows, unsigned int cols)
{
    unsigned int global_thread = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int row = global_thread >> 5;     // one warp per row
    unsigned int lane = threadIdx.x & 31u;
    if (row >= rows) return;

    unsigned int blocks_per_row = cols >> 5;   // cols / 32
    const unsigned char* row_blocks = matrix + (size_t)row * blocks_per_row * 34;

    float sum = 0.0f;
    for (unsigned int b = lane; b < blocks_per_row; b += 32u) {
        const unsigned char* blk = row_blocks + (size_t)b * 34;
        float d = __half2float(*reinterpret_cast<const __half*>(blk));
        const signed char* q = reinterpret_cast<const signed char*>(blk + 2);
        unsigned int vec_base = b << 5;  // b * 32
#pragma unroll
        for (int i = 0; i < 32; i++) {
            sum += d * (float)q[i] * vector[vec_base + i];
        }
    }

    sum = warp_reduce_sum(sum);
    if (lane == 0u) output[row] = sum;
}

// Q4_0 GEMV: 18-byte block = f16 scale + 32 nibbles (4 bits each).
// Memory: 18 bytes / 32 values = 0.5625 bytes/param.
extern "C" __global__ void gemv_q4_0_kernel(
    const unsigned char* matrix, const float* vector, float* output,
    unsigned int rows, unsigned int cols)
{
    unsigned int global_thread = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int row = global_thread >> 5;
    unsigned int lane = threadIdx.x & 31u;
    if (row >= rows) return;

    unsigned int blocks_per_row = cols >> 5;   // cols / 32
    const unsigned char* row_blocks = matrix + (size_t)row * blocks_per_row * 18;

    float sum = 0.0f;
    for (unsigned int b = lane; b < blocks_per_row; b += 32u) {
        const unsigned char* blk = row_blocks + (size_t)b * 18;
        float d = __half2float(*reinterpret_cast<const __half*>(blk));
        const unsigned char* q = blk + 2;
        unsigned int vec_base = b << 5;
#pragma unroll
        for (int i = 0; i < 16; i++) {
            unsigned char qb = q[i];
            float v0 = d * ((float)(qb & 0xF) - 8.0f);
            float v1 = d * ((float)(qb >> 4) - 8.0f);
            sum += v0 * vector[vec_base + i * 2];
            sum += v1 * vector[vec_base + i * 2 + 1];
        }
    }

    sum = warp_reduce_sum(sum);
    if (lane == 0u) output[row] = sum;
}

// --------------------------------------------------------------------------
// Q4_K × Q8_K direct GEMV (OXK GPU path)
//
// Mirrors the CPU OXK kernels: quantize the activation vector to Q8_K once,
// then stream compressed Q4_K weights without expanding to f16 in VRAM.
// One warp per output row; lanes stripe across super-blocks.
// --------------------------------------------------------------------------

__device__ __forceinline__ int q8k_bsum_i16(const unsigned char* bsums, int index) {
    const unsigned char* p = bsums + (size_t)index * 2u;
    return (int)(short)((unsigned int)p[0] | ((unsigned int)p[1] << 8));
}

__device__ float q4k_q8k_block_dot(const unsigned char* w_blk, const unsigned char* q8_blk) {
    float d_w = __half2float(*reinterpret_cast<const __half*>(w_blk));
    float dmin_w = __half2float(*reinterpret_cast<const __half*>(w_blk + 2));
    float d_q8 = *reinterpret_cast<const float*>(q8_blk);
    const unsigned char* scales = w_blk + 4;
    const unsigned char* qs = w_blk + 16;
    const signed char* q8 = reinterpret_cast<const signed char*>(q8_blk + 4);
    const unsigned char* bsums = q8_blk + 4 + 256;

    int pos = 0;
    int min_acc = 0;
    for (int gp = 0; gp < 4; gp++) {
        int g1 = gp * 2;
        int g2 = g1 + 1;
        unsigned char sc1, mn1, sc2, mn2;
        q4k_scale_min(g1, scales, &sc1, &mn1);
        q4k_scale_min(g2, scales, &sc2, &mn2);
        int sum1 = 0;
        int sum2 = 0;
#pragma unroll
        for (int i = 0; i < 32; i++) {
            unsigned char byte = qs[gp * 32 + i];
            sum1 += (int)(byte & 0xF) * (int)q8[g1 * 32 + i];
            sum2 += (int)(byte >> 4) * (int)q8[g2 * 32 + i];
        }
        pos += (int)sc1 * sum1 + (int)sc2 * sum2;
        int bs1 = q8k_bsum_i16(bsums, g1 * 2) + q8k_bsum_i16(bsums, g1 * 2 + 1);
        int bs2 = q8k_bsum_i16(bsums, g2 * 2) + q8k_bsum_i16(bsums, g2 * 2 + 1);
        min_acc += (int)mn1 * bs1 + (int)mn2 * bs2;
    }
    return d_w * d_q8 * (float)pos - dmin_w * d_q8 * (float)min_acc;
}

// Q4_K GEMV: matrix rows are `blocks_per_row` × 144-byte blocks; q8k holds
// one Q8_K block (292 bytes) per super-block along the shared dimension.
extern "C" __global__ void gemv_q4_k_kernel(
    const unsigned char* matrix, const unsigned char* q8k, float* output,
    unsigned int rows, unsigned int blocks_per_row)
{
    unsigned int global_thread = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int row = global_thread >> 5;
    unsigned int lane = threadIdx.x & 31u;
    if (row >= rows) return;

    const unsigned char* row_blocks = matrix + (size_t)row * blocks_per_row * 144u;
    float sum = 0.0f;
    for (unsigned int b = lane; b < blocks_per_row; b += 32u) {
        const unsigned char* w_blk = row_blocks + (size_t)b * 144u;
        const unsigned char* q8_blk = q8k + (size_t)b * 292u;
        sum += q4k_q8k_block_dot(w_blk, q8_blk);
    }

    sum = warp_reduce_sum(sum);
    if (lane == 0u) output[row] = sum;
}

// --------------------------------------------------------------------------
// IQ1_S / IQ1_M (TQ1 family) — on-the-fly ternary GEMV for ultra-low-bit GGUFs
// (e.g. freakyskittle/GLM-5.2-GGUF, Kimi-K2.7 on HF). Mirrors CPU reference.
// --------------------------------------------------------------------------

__device__ __forceinline__ void iq1s_grid_decode(unsigned short index, signed char* out8) {
    unsigned short idx = index;
    for (int i = 0; i < 8; i++) {
        unsigned int bits = idx & 3u;
        out8[i] = (bits == 0u) ? (signed char)-1 : ((bits == 1u) ? (signed char)0 : (signed char)1);
        idx >>= 2;
        if (i == 3) idx = index >> 8;
    }
}

__device__ __forceinline__ float iq1s_block_dot(const unsigned char* blk, const float* vector) {
    const float IQ1S_DELTA = 0.125f;
    float d = __half2float(*reinterpret_cast<const __half*>(blk));
    const unsigned char* qs = blk + 2;
    const unsigned short* qh = reinterpret_cast<const unsigned short*>(blk + 34);
    float sum = 0.0f;
    signed char grid_vals[8];
    unsigned int out_ptr = 0;
    for (int ib = 0; ib < 8; ib++) {
        float dl = d * (2.0f * (float)((qh[ib] >> 12) & 7u) + 1.0f);
        float delta = (qh[ib] & 0x8000u) ? -IQ1S_DELTA : IQ1S_DELTA;
        for (int l = 0; l < 4; l++) {
            unsigned short grid_idx = (unsigned short)qs[l + ib * 4]
                | (unsigned short)(((qh[ib] >> (3 * l)) & 7u) << 8);
            iq1s_grid_decode(grid_idx, grid_vals);
            for (int j = 0; j < 8; j++) {
                sum += dl * ((float)grid_vals[j] + delta) * vector[out_ptr + j];
            }
            out_ptr += 8;
        }
    }
    return sum;
}

extern "C" __global__ void gemv_iq1_s_kernel(
    const unsigned char* matrix, const float* vector, float* output,
    unsigned int rows, unsigned int blocks_per_row)
{
    unsigned int global_thread = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int row = global_thread >> 5;
    unsigned int lane = threadIdx.x & 31u;
    if (row >= rows) return;

    const unsigned char* row_blocks = matrix + (size_t)row * blocks_per_row * 50u;
    float sum = 0.0f;
    for (unsigned int b = lane; b < blocks_per_row; b += 32u) {
        sum += iq1s_block_dot(row_blocks + (size_t)b * 50u, vector + (size_t)b * 256u);
    }
    sum = warp_reduce_sum(sum);
    if (lane == 0u) output[row] = sum;
}

__device__ __forceinline__ float iq1m_block_dot(const unsigned char* blk, const float* vector) {
    const float IQ1S_DELTA = 0.125f;
    const unsigned char* qs = blk;
    const unsigned char* qh = blk + 32;
    const unsigned char* scales = blk + 48;
    float sum = 0.0f;
    signed char grid_vals[8];
    unsigned int out_ptr = 0;
    for (int ib = 0; ib < 8; ib++) {
        unsigned short sc = (unsigned short)scales[ib * 2]
            | ((unsigned short)scales[ib * 2 + 1] << 8);
        float dl = __half2float(*reinterpret_cast<const __half*>(&sc));
        for (int l = 0; l < 4; l++) {
            unsigned short idxs[4] = {
                (unsigned short)qs[l + ib * 4] | (unsigned short)(((qh[l + ib * 4] >> 0) & 7u) << 8),
                (unsigned short)qs[l + ib * 4] | (unsigned short)(((qh[l + ib * 4] >> 3) & 7u) << 8),
                (unsigned short)qs[l + ib * 4] | (unsigned short)(((qh[l + ib * 4] >> 6) & 7u) << 8),
                (unsigned short)qs[l + ib * 4 + 32] | (unsigned short)(((qh[l + ib * 4] >> 1) & 7u) << 8),
            };
            float deltas[4] = {
                (qh[l + ib * 4] & 1u) ? -IQ1S_DELTA : IQ1S_DELTA,
                (qh[l + ib * 4] & 2u) ? -IQ1S_DELTA : IQ1S_DELTA,
                (qh[l + ib * 4] & 4u) ? -IQ1S_DELTA : IQ1S_DELTA,
                (qh[l + ib * 4 + 32] & 1u) ? -IQ1S_DELTA : IQ1S_DELTA,
            };
            for (int g = 0; g < 4; g++) {
                iq1s_grid_decode(idxs[g], grid_vals);
                for (int j = 0; j < 8; j++) {
                    sum += dl * ((float)grid_vals[j] + deltas[g]) * vector[out_ptr + j];
                }
                out_ptr += 8;
            }
        }
    }
    return sum;
}

extern "C" __global__ void gemv_iq1_m_kernel(
    const unsigned char* matrix, const float* vector, float* output,
    unsigned int rows, unsigned int blocks_per_row)
{
    unsigned int global_thread = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int row = global_thread >> 5;
    unsigned int lane = threadIdx.x & 31u;
    if (row >= rows) return;

    const unsigned char* row_blocks = matrix + (size_t)row * blocks_per_row * 56u;
    float sum = 0.0f;
    for (unsigned int b = lane; b < blocks_per_row; b += 32u) {
        sum += iq1m_block_dot(row_blocks + (size_t)b * 56u, vector + (size_t)b * 256u);
    }
    sum = warp_reduce_sum(sum);
    if (lane == 0u) output[row] = sum;
}

extern "C" __global__ void dequant_q2_k_kernel(
    const unsigned char* in, unsigned short* out, unsigned int nblocks)
{
    unsigned int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nblocks) return;
    const unsigned char* blk = in + (size_t)b * 84u;
    float d = __half2float(*reinterpret_cast<const __half*>(blk + 80));
    float mn = __half2float(*reinterpret_cast<const __half*>(blk + 82));
    const unsigned char* scales = blk;
    const unsigned char* qs = blk + 16;
    __half* o = reinterpret_cast<__half*>(out) + (size_t)b * 256u;
    unsigned int q_ptr = 0;
    int is = 0;
    for (int outer = 0; outer < 2; outer++) {
        unsigned int qs_base = outer * 32u;
        for (int inner = 0; inner < 4; inner++) {
            unsigned char sc1 = scales[is++];
            float dl1 = d * (float)(sc1 & 0xF);
            float ml1 = mn * (float)(sc1 >> 4);
            unsigned char sc2 = scales[is++];
            float dl2 = d * (float)(sc2 & 0xF);
            float ml2 = mn * (float)(sc2 >> 4);
            for (int l = 0; l < 32; l++) {
                unsigned char qbyte = qs[qs_base + l];
                o[q_ptr + l] = __float2half(dl1 * (float)(qbyte & 3) - ml1);
                o[q_ptr + 32 + l] = __float2half(dl2 * (float)((qbyte >> 2) & 3) - ml2);
            }
            q_ptr += 64;
        }
    }
}

__device__ __constant__ float E2M1_DOUBLED[16] = {
    0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 6.0f, 8.0f, 12.0f,
    0.0f, -1.0f, -2.0f, -3.0f, -4.0f, -6.0f, -8.0f, -12.0f
};

__device__ __forceinline__ float ue4m3_to_f32(unsigned char b) {
    unsigned int sign = (b >> 7) & 1u;
    unsigned int exp = (b >> 3) & 0xFu;
    unsigned int mant = b & 7u;
    float v = (exp == 0u)
        ? (float)mant * exp2f(-9.0f)
        : (1.0f + (float)mant / 8.0f) * exp2f((float)exp - 7.0f);
    return sign != 0u ? -v : v;
}

extern "C" __global__ void dequant_nvfp4_kernel(
    const unsigned char* in, unsigned short* out, unsigned int nblocks)
{
    unsigned int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nblocks) return;
    const unsigned char* blk = in + (size_t)b * 36u;
    __half* o = reinterpret_cast<__half*>(out) + (size_t)b * 64u;
    for (int sub = 0; sub < 4; sub++) {
        float scale = ue4m3_to_f32(blk[sub]);
        unsigned int q_base = 4u + (unsigned int)sub * 8u;
        unsigned int out_base = (unsigned int)sub * 16u;
        for (int j = 0; j < 8; j++) {
            unsigned char packed = blk[q_base + j];
            o[out_base + j] = __float2half(scale * E2M1_DOUBLED[packed & 0xF]);
            o[out_base + j + 8] = __float2half(scale * E2M1_DOUBLED[packed >> 4]);
        }
    }
}

extern "C" __global__ void gemv_nvfp4_kernel(
    const unsigned char* matrix, const float* vector, float* output,
    unsigned int rows, unsigned int blocks_per_row)
{
    unsigned int global_thread = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int row = global_thread >> 5;
    unsigned int lane = threadIdx.x & 31u;
    if (row >= rows) return;

    const unsigned char* row_blocks = matrix + (size_t)row * blocks_per_row * 36u;
    float sum = 0.0f;
    for (unsigned int b = lane; b < blocks_per_row; b += 32u) {
        const unsigned char* blk = row_blocks + (size_t)b * 36u;
        const float* v = vector + (size_t)b * 64u;
        for (int sub = 0; sub < 4; sub++) {
            float scale = ue4m3_to_f32(blk[sub]);
            unsigned int q_base = 4u + (unsigned int)sub * 8u;
            unsigned int v_base = (unsigned int)sub * 16u;
            for (int j = 0; j < 8; j++) {
                unsigned char packed = blk[q_base + j];
                sum += scale * E2M1_DOUBLED[packed & 0xF] * v[v_base + j];
                sum += scale * E2M1_DOUBLED[packed >> 4] * v[v_base + j + 8];
            }
        }
    }
    sum = warp_reduce_sum(sum);
    if (lane == 0u) output[row] = sum;
}
