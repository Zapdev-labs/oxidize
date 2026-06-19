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
// Wavefront / warp width abstraction.
//
// This file is compiled by BOTH nvcc (the `cuda` feature) and hipcc (the
// `rocm` feature). The GEMV kernels follow a "one lane group computes one
// output row" contract: lanes stripe across the columns/blocks of a row with a
// stride equal to the lane-group width, then a shuffle reduction sums the
// per-lane partials.
//
// On NVIDIA the lane group is a 32-lane warp; on AMD it is a 64-lane wavefront
// (GCN/CDNA always, and RDNA when compiled `-mwavefrontsize64`, which build.rs
// pins by default). OX_WAVE is the SINGLE source of truth for that width so the
// device stride, the lane mask, the row index shift, the reduction tree depth,
// and the host launch grid (rocm::GEMV_LANES_PER_ROW) cannot drift apart.
//
// IMPORTANT: the quantization BLOCK SIZE (32 values per Q8_0/Q4_0 block, the
// `cols >> 5` super-block counts, `b << 5` block offsets, and the inner
// `i < 32` dequant loops) is a SEPARATE, architecture-independent constant and
// is intentionally NOT routed through OX_WAVE.
#if defined(__HIP_PLATFORM_AMD__)
  #ifndef OX_WAVE
    #define OX_WAVE 64u
  #endif
  #define OX_WAVE_LOG2 6u   // log2(64)
#else
  #ifndef OX_WAVE
    #define OX_WAVE 32u
  #endif
  #define OX_WAVE_LOG2 5u   // log2(32)
#endif
#define OX_LANE_MASK (OX_WAVE - 1u)   // 31u on CUDA, 63u on AMD

// --------------------------------------------------------------------------
// Dense matrix-vector products: output[row] = sum_c matrix[row*cols + c] * vector[c]
// --------------------------------------------------------------------------

// One lane group (OX_WAVE lanes: 32 on CUDA, 64 on AMD) per output row. Lanes
// stride across `cols` so that adjacent lanes touch adjacent weights — fully
// coalesced VRAM reads — then a shuffle reduction sums the partials. Decode-time
// gemv is bandwidth-bound, so coalescing is what gets close to the GPU's peak
// memory throughput.
//
// Launch with `blockDim.x` a multiple of OX_WAVE and enough total lane groups to
// cover every row (grid sized as `ceil(rows * OX_WAVE / blockDim.x)`; the host
// uses rocm::GEMV_LANES_PER_ROW for the OX_WAVE factor).

__device__ __forceinline__ float warp_reduce_sum(float v) {
#if defined(__HIP_PLATFORM_AMD__)
    // HIP: __shfl_down is wavefront-width aware and takes no lane mask. Start at
    // OX_WAVE/2 (=32) so lanes 32..63 are folded in (6 steps for wave64).
    for (int offset = (int)(OX_WAVE >> 1); offset > 0; offset >>= 1)
        v += __shfl_down(v, offset);
#else
    // CUDA: OX_WAVE>>1 == 16, identical to the original 5-step, full-mask loop.
    for (int offset = (int)(OX_WAVE >> 1); offset > 0; offset >>= 1)
        v += __shfl_down_sync(0xffffffffu, v, offset);
#endif
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
    unsigned int row = global_thread >> OX_WAVE_LOG2;     // one warp per row
    unsigned int lane = threadIdx.x & OX_LANE_MASK;
    if (row >= rows) return;

    const float* w = matrix + (size_t)row * cols;
    float sum = 0.0f;
    for (unsigned int c = lane; c < cols; c += OX_WAVE)
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
    unsigned int row = global_thread >> OX_WAVE_LOG2;
    unsigned int lane = threadIdx.x & OX_LANE_MASK;
    if (row >= rows) return;

    const __half* w = reinterpret_cast<const __half*>(matrix) + (size_t)row * cols;
    const float* v = vector;
    float sum = 0.0f;

    unsigned int c = lane * 2u;
    for (; c + 1u < cols; c += (OX_WAVE * 2u)) {   // wave lanes * 2 halves/lane
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
    unsigned int row = global_thread >> OX_WAVE_LOG2;     // one warp per row
    unsigned int lane = threadIdx.x & OX_LANE_MASK;
    if (row >= rows) return;

    unsigned int blocks_per_row = cols >> 5;   // cols / 32
    const unsigned char* row_blocks = matrix + (size_t)row * blocks_per_row * 34;

    float sum = 0.0f;
    for (unsigned int b = lane; b < blocks_per_row; b += OX_WAVE) {
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
    unsigned int row = global_thread >> OX_WAVE_LOG2;
    unsigned int lane = threadIdx.x & OX_LANE_MASK;
    if (row >= rows) return;

    unsigned int blocks_per_row = cols >> 5;   // cols / 32
    const unsigned char* row_blocks = matrix + (size_t)row * blocks_per_row * 18;

    float sum = 0.0f;
    for (unsigned int b = lane; b < blocks_per_row; b += OX_WAVE) {
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
// Q4_K × F32 direct GEMV — GPU-native activation path
//
// Takes a plain float32 input vector (already on GPU as part of the activation
// buffer) instead of a pre-quantized Q8_K vector.  Eliminates the CPU-side
// Q8_K quantization step and the H2D upload, enabling the fully GPU-resident
// forward pass where the hidden state never leaves the GPU between layers.
//
// One lane group (OX_WAVE threads) per output row.  Lanes stripe across blocks.
// --------------------------------------------------------------------------

__device__ float q4k_f32in_block_dot(const unsigned char* w_blk,
                                      const float* x,
                                      unsigned int block_offset)
{
    float d_w    = __half2float(*reinterpret_cast<const __half*>(w_blk));
    float dmin_w = __half2float(*reinterpret_cast<const __half*>(w_blk + 2));
    const unsigned char* scales = w_blk + 4;
    const unsigned char* qs     = w_blk + 16;   // 128 bytes = 256 nibbles

    const float* xb = x + (size_t)block_offset * 256u;

    float pos_acc = 0.0f;
    float min_acc = 0.0f;

    for (int gp = 0; gp < 4; gp++) {
        unsigned char sc1, mn1, sc2, mn2;
        q4k_scale_min(gp * 2,     scales, &sc1, &mn1);
        q4k_scale_min(gp * 2 + 1, scales, &sc2, &mn2);

        float sum1 = 0.0f, sum2 = 0.0f;
        float xsum1 = 0.0f, xsum2 = 0.0f;

        const unsigned char* gp_qs = qs + gp * 32;
        const float* xb1 = xb + gp * 64;
        const float* xb2 = xb + gp * 64 + 32;

#pragma unroll
        for (int i = 0; i < 32; i++) {
            unsigned char byte = gp_qs[i];
            float lo = (float)(byte & 0xFu);
            float hi = (float)(byte >> 4u);
            float x1 = xb1[i];
            float x2 = xb2[i];
            sum1  += lo * x1;
            sum2  += hi * x2;
            xsum1 += x1;
            xsum2 += x2;
        }

        pos_acc += (float)sc1 * sum1 + (float)sc2 * sum2;
        min_acc += (float)mn1 * xsum1 + (float)mn2 * xsum2;
    }

    return d_w * pos_acc - dmin_w * min_acc;
}

// Q4K GEMV with GPU-resident float32 input (GPU-native activation path).
extern "C" __global__ void gemv_q4k_f32in_kernel(
    const unsigned char* __restrict__ matrix,  // Q4K weights [rows × blocks_per_row × 144 B]
    const float*          __restrict__ x,      // F32 input  [blocks_per_row * 256]
    float*                __restrict__ output, // F32 output [rows]
    unsigned int rows,
    unsigned int blocks_per_row)
{
    unsigned int global_thread = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int row  = global_thread >> OX_WAVE_LOG2;
    unsigned int lane = threadIdx.x & OX_LANE_MASK;
    if (row >= rows) return;

    const unsigned char* row_blocks = matrix + (size_t)row * blocks_per_row * 144u;
    float sum = 0.0f;

    for (unsigned int b = lane; b < blocks_per_row; b += OX_WAVE)
        sum += q4k_f32in_block_dot(row_blocks + (size_t)b * 144u, x, b);

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
    unsigned int row = global_thread >> OX_WAVE_LOG2;
    unsigned int lane = threadIdx.x & OX_LANE_MASK;
    if (row >= rows) return;

    const unsigned char* row_blocks = matrix + (size_t)row * blocks_per_row * 144u;
    float sum = 0.0f;
    for (unsigned int b = lane; b < blocks_per_row; b += OX_WAVE) {
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
    unsigned int row = global_thread >> OX_WAVE_LOG2;
    unsigned int lane = threadIdx.x & OX_LANE_MASK;
    if (row >= rows) return;

    const unsigned char* row_blocks = matrix + (size_t)row * blocks_per_row * 50u;
    float sum = 0.0f;
    for (unsigned int b = lane; b < blocks_per_row; b += OX_WAVE) {
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
    unsigned int row = global_thread >> OX_WAVE_LOG2;
    unsigned int lane = threadIdx.x & OX_LANE_MASK;
    if (row >= rows) return;

    const unsigned char* row_blocks = matrix + (size_t)row * blocks_per_row * 56u;
    float sum = 0.0f;
    for (unsigned int b = lane; b < blocks_per_row; b += OX_WAVE) {
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
    unsigned int row = global_thread >> OX_WAVE_LOG2;
    unsigned int lane = threadIdx.x & OX_LANE_MASK;
    if (row >= rows) return;

    const unsigned char* row_blocks = matrix + (size_t)row * blocks_per_row * 36u;
    float sum = 0.0f;
    for (unsigned int b = lane; b < blocks_per_row; b += OX_WAVE) {
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

// --------------------------------------------------------------------------
// GPU-resident forward-pass utility kernels
// These are used to keep the hidden state (residual stream) on device across
// the full transformer forward pass, eliminating per-layer CPU<->GPU trips.
// --------------------------------------------------------------------------

// gemv_q6k_f32in_kernel: Q6_K matrix × F32 input → F32 output
//
// Q6_K block layout (block_q6_K, 210 bytes per 256 values):
//   ql[128]     — lower 4 bits, interleaved: byte l holds lo4 for values l and l+32
//   qh[64]      — upper 2 bits, interleaved: byte l holds hi2 for values l, l+32, l+64, l+96
//   scales[16]  — int8 scales, one per 16 values
//   d (2 bytes) — half-float superblock scale
//
// Layout matches ggml block_q6_K; the block has TWO sub-groups of 128 values each.
// Within each sub-group, for l=0..31: q(l), q(l+32), q(l+64), q(l+96) share qh[l].
//
// One lane group (OX_WAVE threads) computes one output row; threads stride over blocks.
__device__ float q6k_f32in_block_dot(
    const unsigned char* __restrict__ w_blk,
    const float* __restrict__ x,
    unsigned int block_offset)
{
    float d  = __half2float(*reinterpret_cast<const __half*>(w_blk + 208));
    const int8_t*          sc = reinterpret_cast<const int8_t*>(w_blk + 192);
    const unsigned char*   ql = w_blk;        // [128 bytes]
    const unsigned char*   qh = w_blk + 128;  // [64 bytes]
    const float*           xb = x + (size_t)block_offset * 256u;
    float sum = 0.0f;
    // Two sub-groups of 128 values (n=0 → values 0..127, n=1 → values 128..255)
    for (int n = 0; n < 2; n++) {
        const unsigned char* qln = ql + n * 64;   // 64 bytes per sub-group
        const unsigned char* qhn = qh + n * 32;   // 32 bytes per sub-group
        const int8_t*        scn = sc + n * 8;    // 8 scales per sub-group
        const float*         xn  = xb + n * 128;
        for (int l = 0; l < 32; l++) {
            int is = l >> 4;  // 0 for l=0..15, 1 for l=16..31
            int8_t q1 = (int8_t)(((qln[l     ] & 0xFu) | (((qhn[l] >> 0) & 3u) << 4)) - 32);
            int8_t q2 = (int8_t)(((qln[l + 32] & 0xFu) | (((qhn[l] >> 2) & 3u) << 4)) - 32);
            int8_t q3 = (int8_t)(((qln[l     ] >>    4) | (((qhn[l] >> 4) & 3u) << 4)) - 32);
            int8_t q4 = (int8_t)(((qln[l + 32] >>    4) | (((qhn[l] >> 6) & 3u) << 4)) - 32);
            sum += d * (float)scn[is    ] * (float)q1 * xn[l     ];
            sum += d * (float)scn[is + 2] * (float)q2 * xn[l + 32];
            sum += d * (float)scn[is + 4] * (float)q3 * xn[l + 64];
            sum += d * (float)scn[is + 6] * (float)q4 * xn[l + 96];
        }
    }
    return sum;
}

extern "C" __global__ void gemv_q6k_f32in_kernel(
    const unsigned char* __restrict__ matrix,
    const float*          __restrict__ x,
    float*                __restrict__ output,
    unsigned int rows, unsigned int blocks_per_row)
{
    unsigned int global_thread = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int row  = global_thread >> OX_WAVE_LOG2;
    unsigned int lane = threadIdx.x & OX_LANE_MASK;
    if (row >= rows) return;
    const unsigned char* row_blocks = matrix + (size_t)row * blocks_per_row * 210u;
    float sum = 0.0f;
    for (unsigned int b = lane; b < blocks_per_row; b += OX_WAVE)
        sum += q6k_f32in_block_dot(row_blocks + (size_t)b * 210u, x, b);
    sum = warp_reduce_sum(sum);
    if (lane == 0u) output[row] = sum;
}

// rms_norm_f32_kernel: y[i] = x[i] / rms(x) * weight[i]
//
// One block per row. All threads in the block cooperate to compute the
// sum-of-squares via shared-memory parallel reduction, then each thread
// applies the normalised weight scale. Supports hidden sizes up to 16 384
// (512 threads * 32 elements per thread) which covers every model class in
// the oxidize weight set.
//
// Launch: gridDim.x = batch (typically 1 for decode), blockDim.x = 256 (tunable).
// Dynamic shared memory: blockDim.x * sizeof(float) bytes.
extern "C" __global__ void rms_norm_f32_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    float* __restrict__ y,
    int hidden_size, float eps)
{
    extern __shared__ float sdata[];  // blockDim.x floats

    int tid = threadIdx.x;
    int row = blockIdx.x;
    const float* xrow = x + (size_t)row * hidden_size;
    float* yrow = y + (size_t)row * hidden_size;

    // Each thread accumulates its partial sum of squares.
    float partial = 0.0f;
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float v = xrow[i];
        partial += v * v;
    }
    sdata[tid] = partial;
    __syncthreads();

    // Tree reduction in shared memory.
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride)
            sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }

    // sdata[0] now holds the full sum-of-squares.
    float inv_rms = rsqrtf(sdata[0] / (float)hidden_size + eps);

    // Apply scale.
    for (int i = tid; i < hidden_size; i += blockDim.x)
        yrow[i] = xrow[i] * inv_rms * weight[i];
}

// residual_add_f32_kernel: x[i] += delta[i]  (in-place residual stream update)
//
// Launch: blockDim.x = 256, gridDim.x = ceil(n / 256).
extern "C" __global__ void residual_add_f32_kernel(
    float* __restrict__ x,
    const float* __restrict__ delta,
    int n)
{
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n)
        x[i] += delta[i];
}

// silu_mul_f32_kernel: out[i] = silu(gate[i]) * up[i]
//
// This is the SwiGLU non-linearity used by the FFN block of Llama / Mistral /
// Qwen / Gemma models.  gate and up are separate f32 arrays of length n.
// out may alias gate (in-place update is safe because each thread writes to
// exactly the same index it reads from both inputs).
//
// Launch: blockDim.x = 256, gridDim.x = ceil(n / 256).
extern "C" __global__ void silu_mul_f32_kernel(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float* __restrict__ out,
    int n)
{
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) {
        float g = gate[i];
        // silu(g) = g * sigmoid(g) = g / (1 + exp(-g))
        float silu_g = g / (1.0f + expf(-g));
        out[i] = silu_g * up[i];
    }
}

// softmax_f32_kernel: in-place numerically stable softmax over n elements.
//
// Runs as a single block. Uses shared memory to find the maximum (for
// numerical stability) and the normalisation sum, then writes the final
// probabilities back in-place. The primary consumer is final-logit softmax
// (vocab_size ~ 32k-128k); for attention scores use the flash-attention path.
//
// Launch: gridDim.x = 1, blockDim.x = 256.
// Dynamic shared memory: 2 * blockDim.x * sizeof(float) bytes.
extern "C" __global__ void softmax_f32_kernel(float* __restrict__ x, int n)
{
    extern __shared__ float smem[];
    int tid = threadIdx.x;
    int bdx = blockDim.x;
    float* max_buf = smem;
    float* sum_buf = smem + bdx;

    // Pass 1: find max for numerical stability.
    float local_max = -3.402823466e+38f;  // -FLT_MAX
    for (int i = tid; i < n; i += bdx) {
        float v = x[i];
        if (v > local_max) local_max = v;
    }
    max_buf[tid] = local_max;
    __syncthreads();
    for (int stride = bdx >> 1; stride > 0; stride >>= 1) {
        if (tid < stride && max_buf[tid + stride] > max_buf[tid])
            max_buf[tid] = max_buf[tid + stride];
        __syncthreads();
    }
    float gmax = max_buf[0];

    // Pass 2: compute exp(x - max) in-place and accumulate sum.
    float local_sum = 0.0f;
    for (int i = tid; i < n; i += bdx) {
        float v = expf(x[i] - gmax);
        x[i] = v;
        local_sum += v;
    }
    sum_buf[tid] = local_sum;
    __syncthreads();
    for (int stride = bdx >> 1; stride > 0; stride >>= 1) {
        if (tid < stride)
            sum_buf[tid] += sum_buf[tid + stride];
        __syncthreads();
    }
    float inv_sum = 1.0f / sum_buf[0];

    // Pass 3: normalise.
    for (int i = tid; i < n; i += bdx)
        x[i] *= inv_sum;
}

// cast_f32_to_f16_kernel: convert n f32 values to f16 (stored as u16 bit patterns).
//
// Used to upload f32 activations to device as f16 to halve bandwidth on the
// activation->GEMV path when the model is running in mixed-precision mode.
//
// Launch: blockDim.x = 256, gridDim.x = ceil(n / 256).
extern "C" __global__ void cast_f32_to_f16_kernel(
    const float* __restrict__ in,
    unsigned short* __restrict__ out,
    int n)
{
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) {
        __half h = __float2half(in[i]);
        // Store as raw u16 bits - matches DeviceBuffer<u16> on the Rust side.
        out[i] = *reinterpret_cast<const unsigned short*>(&h);
    }
}

// cast_f16_to_f32_kernel: convert n f16 values (stored as u16 bit patterns) to f32.
//
// Used to download device f16 activations back to f32 for CPU-side operations
// that do not yet have a GPU implementation.
//
// Launch: blockDim.x = 256, gridDim.x = ceil(n / 256).
extern "C" __global__ void cast_f16_to_f32_kernel(
    const unsigned short* __restrict__ in,
    float* __restrict__ out,
    int n)
{
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) {
        __half h = *reinterpret_cast<const __half*>(&in[i]);
        out[i] = __half2float(h);
    }
}
