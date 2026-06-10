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
extern "C" __global__ void gemv_f16_kernel(
    const unsigned short* matrix, const float* vector, float* output,
    unsigned int rows, unsigned int cols)
{
    unsigned int global_thread = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int row = global_thread >> 5;     // one warp per row
    unsigned int lane = threadIdx.x & 31u;
    if (row >= rows) return;

    const __half* w = reinterpret_cast<const __half*>(matrix) + (size_t)row * cols;
    float sum = 0.0f;
    for (unsigned int c = lane; c < cols; c += 32u)
        sum += __half2float(w[c]) * vector[c];

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
