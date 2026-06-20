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
// Accurate RoPE trig under --use_fast_math.
//
// build.rs compiles this file with `--use_fast_math` (and hipcc `-ffast-math`).
// Fast-math lowers `sinf`/`cosf` to the hardware `__sinf`/`__cosf` intrinsics,
// which SKIP large-argument range reduction and are only accurate for roughly
// |x| < pi. RoPE computes `angle = pos * freq`; when seeding the prompt KV
// cache `pos` reaches the prompt length (hundreds), so for the low-frequency
// lanes `angle` is hundreds of radians and the fast intrinsics return garbage
// cos/sin. That corrupts every cached key's rotation (and the decode query),
// deterministically flipping the OX_GPU_ATTN token-0 argmax vs the CPU
// reference (which uses libm `f32::cos/sin` with full range reduction).
//
// Fix: reduce the angle modulo 2*pi in DOUBLE precision first, then evaluate
// double-precision sin/cos on the small reduced argument. Double-precision trig
// is not replaced by the low-accuracy float device intrinsics under fast-math,
// and after the reduction the argument is in [-pi, pi], so the result matches
// the CPU rotation to f32 ULPs regardless of `pos`. Only the OX_GPU_ATTN rope
// kernel uses this; the default GPU path never calls it.
__device__ __forceinline__ void rope_sincos(double angle, float* s_out, float* c_out) {
    // 1/(2*pi) and 2*pi as exact-as-f64 constants for the modulo reduction.
    const double inv_two_pi = 0.15915494309189535;   // 1 / (2*pi)
    const double two_pi     = 6.283185307179586;      // 2*pi
    // k = round(angle / 2pi); reduced = angle - k*2pi  -> in [-pi, pi].
    double k = floor(angle * inv_two_pi + 0.5);
    double reduced = angle - k * two_pi;
    double s, c;
    sincos(reduced, &s, &c);
    *s_out = (float)s;
    *c_out = (float)c;
}

// --------------------------------------------------------------------------
// Bit-exact f32 -> f16 store for the KV cache.
//
// The CPU flash-attention path reads the KV cache via an exact f16 -> f32
// widening (matches __half2float), but it WRITES the cache with the host
// `f32_to_f16_bits` (compute/kv_cache/storage.rs:378-423), which rounds
// half-UP on normals, has an explicit subnormal path, and carries a mantissa
// overflow into the exponent. `__float2half` instead rounds to-nearest-EVEN,
// so for halfway cases and subnormals it produces a different bit pattern.
//
// Under OX_GPU_ATTN the device F16 KV cache must be argmax-identical to the CPU
// reference, so the on-device store MUST mirror the host routine bit-for-bit.
// This is pure integer arithmetic — unaffected by --use_fast_math and
// compute_75-safe. Returns the raw f16 bit pattern (the cache holds u16 bits).
__device__ __forceinline__ unsigned short f32_to_f16_bits_dev(float value) {
    unsigned int x = __float_as_uint(value);
    unsigned short sign = (unsigned short)((x >> 16) & 0x8000u);
    int exp = (int)((x >> 23) & 0xFFu);
    unsigned int frac = x & 0x7FFFFFu;

    if (exp == 0xFF) {                          // Inf / NaN
        if (frac == 0u) return sign | 0x7C00u;
        unsigned short nan = (unsigned short)(frac >> 13);
        return sign | 0x7C00u | nan | 1u;
    }

    int exp16 = exp - 127 + 15;
    if (exp16 <= 0) {                           // subnormal / underflow
        if (exp16 < -10) return sign;
        unsigned int mant = frac | 0x800000u;
        unsigned int shift = (unsigned int)(14 - exp16);   // shift in [14, 24]
        unsigned short half_frac = (unsigned short)(mant >> shift);
        if (((mant >> (shift - 1u)) & 1u) != 0u) {
            half_frac = (unsigned short)(half_frac + 1u);
        }
        return sign | half_frac;
    }

    if (exp16 >= 0x1F) return sign | 0x7C00u;   // overflow -> Inf

    unsigned short half_exp = (unsigned short)(((unsigned int)exp16) << 10);
    unsigned short half_frac = (unsigned short)(frac >> 13);
    if ((frac & 0x1000u) != 0u) {               // round half UP
        half_frac = (unsigned short)(half_frac + 1u);
        if ((half_frac & 0x0400u) != 0u) {      // mantissa carry -> bump exponent
            half_frac = 0;
            half_exp = (unsigned short)(half_exp + 0x0400u);
            if (half_exp >= 0x7C00u) return sign | 0x7C00u;
        }
    }
    return sign | half_exp | (unsigned short)(half_frac & 0x03FFu);
}

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
// Activation int8 quantization: F32 → Q8_K (one super-block per CUDA block).
//
// The f32in GEMV kernels below no longer dequantize the WEIGHTS to f32 per
// nibble (compute/latency-bound). Instead the shared ACTIVATION vector is
// quantized ONCE to Q8_K and the weight×activation dot runs as an int8 dp4a
// accumulation — the same memory-bound contract as `gemv_q4_k_kernel`. Because
// one `normed` buffer feeds Q/K/V (and one `ffn_norm` feeds gate+up), this
// quantization is amortized over 3-4 large matmuls per launch.
//
// Output layout is the on-GPU Q8_K block `q4k_q8k_block_dot`/`q6k` consume
// (292 bytes per 256-value super-block): float d (+0), 256 int8 q (+4), 16
// int16 bsums (+260) — bsums[s] = Σ of the 16 int8 q in sub-block s.
//
// Launch: gridDim.x = blocks_per_row (one super-block per block), blockDim.x =
// 256 (one thread per value). Mirrors the CPU OXK Q8_K quantizer (amax/127).
// --------------------------------------------------------------------------
extern "C" __global__ void quantize_f32_to_q8k_kernel(
    const float* __restrict__ x,           // [nblocks * 256]
    unsigned char* __restrict__ q8k_out,   // [nblocks * 292]
    unsigned int nblocks)
{
    __shared__ float s_amax[256];
    __shared__ float s_d;

    unsigned int blk = blockIdx.x;
    if (blk >= nblocks) return;
    unsigned int tid = threadIdx.x;        // 0..255, one per value

    const float* xb = x + (size_t)blk * 256u;
    float v = xb[tid];

    // Block-wide amax reduction (256 threads).
    s_amax[tid] = fabsf(v);
    __syncthreads();
    for (unsigned int stride = 128u; stride > 0u; stride >>= 1) {
        if (tid < stride) {
            float other = s_amax[tid + stride];
            if (other > s_amax[tid]) s_amax[tid] = other;
        }
        __syncthreads();
    }
    if (tid == 0u) {
        float amax = s_amax[0];
        s_d = amax / 127.0f;               // Q8_K activation scale
    }
    __syncthreads();

    float d = s_d;
    float inv_d = (d != 0.0f) ? (1.0f / d) : 0.0f;
    // round-to-nearest, matching the CPU OXK quantizer.
    int qi = (int)lrintf(v * inv_d);
    if (qi > 127) qi = 127;
    if (qi < -128) qi = -128;

    unsigned char* out = q8k_out + (size_t)blk * 292u;
    if (tid == 0u) {
        *reinterpret_cast<float*>(out) = d;
    }
    signed char* q8 = reinterpret_cast<signed char*>(out + 4);
    q8[tid] = (signed char)qi;

    // bsums[s] = Σ of the 16 int8 q in sub-block s (s = tid>>4). Accumulate via
    // a 16-thread shared reduction per sub-block.
    __shared__ int s_bsum[256];
    s_bsum[tid] = qi;
    __syncthreads();
    // Reduce within each contiguous group of 16 (sub-block) without crossing
    // group boundaries: only fold offsets that stay inside the sub-block.
    for (unsigned int stride = 8u; stride > 0u; stride >>= 1) {
        if ((tid & 15u) < stride) {
            s_bsum[tid] += s_bsum[tid + stride];
        }
        __syncthreads();
    }
    if ((tid & 15u) == 0u) {
        int sub = (int)(tid >> 4);         // 0..15
        short* bsums = reinterpret_cast<short*>(out + 4 + 256);
        bsums[sub] = (short)s_bsum[tid];
    }
}

// --------------------------------------------------------------------------
// Q4_K × Q8_K direct GEMV — GPU-native activation path (memory-bound, dp4a).
//
// The activation vector is pre-quantized to Q8_K by `quantize_f32_to_q8k_kernel`
// ONCE per shared buffer, then this kernel streams the compressed Q4_K weights
// and accumulates the dot with int8 `__dp4a` MACs (4 MACs/instruction, no
// per-weight f32 dequant). This is bit-for-bit the `gemv_q4_k_kernel` (Q8_K)
// reference modulo f32 summation order — NOT the old per-nibble f32 path.
//
// SPLIT-K: each row is computed by `n_splits` warps, each owning a 1/n_splits
// stride of the blocks_per_row super-blocks; their f32 partials are summed in
// shared memory (deterministic per-row, no atomics).
//
// dp4a is available on sm_61+ (compute_75-safe). OX_WAVE-portable: lanes stripe
// the 128-byte qs payload coalesced; `__dp4a` works identically on AMD (HIP
// provides it / build.rs maps it). One warp cooperates per block.
// --------------------------------------------------------------------------

// Per-block Q4_K × Q8_K dot, lanes cooperative + dp4a.
//
// The 128-byte `qs` payload holds 4 groups of 32 bytes; group gp's low nibbles
// are sub-block 2*gp, high nibbles are sub-block 2*gp+1. Lanes stripe the 32
// bytes of each group as 4-byte chunks so each `__dp4a` consumes 4 contiguous
// weights against 4 contiguous Q8 activations. The integer sub-block scales
// (`sc`) and mins (`mn`) fold once per block; d/dmin/d_q8 apply once.
__device__ float q4k_q8kin_block_dot_dp4a(const unsigned char* __restrict__ w_blk,
                                          const unsigned char* __restrict__ q8_blk,
                                          unsigned int lane)
{
    float d_w    = __half2float(*reinterpret_cast<const __half*>(w_blk));
    float dmin_w = __half2float(*reinterpret_cast<const __half*>(w_blk + 2));
    float d_q8   = *reinterpret_cast<const float*>(q8_blk);
    const unsigned char* scales = w_blk + 4;
    const unsigned char* qs     = w_blk + 16;        // 128 bytes
    const signed char*   q8     = reinterpret_cast<const signed char*>(q8_blk + 4);
    const unsigned char* bsums  = q8_blk + 4 + 256;

    // Each lane keeps a per-sub-block int32 partial (Σ q4·q8 within its stride).
    int pos[8];
#pragma unroll
    for (int s = 0; s < 8; s++) pos[s] = 0;

    // 128 bytes = 32 4-byte chunks. Lane L processes chunks L, L+OX_WAVE, ...
    // (OX_WAVE=32 → 1 chunk/lane; OX_WAVE=64 → first 32 lanes active). Chunk c
    // covers bytes [c*4 .. c*4+4) → group gp = (c*4)>>5 = c>>3, within-group
    // byte offset i0 = (c*4)&31.
#pragma unroll
    for (unsigned int c = lane; c < 32u; c += OX_WAVE) {
        unsigned int byte0 = c * 4u;
        unsigned int gp = byte0 >> 5;                // 0..3
        unsigned int i0 = byte0 & 31u;               // 0,4,...,28
        // 4 packed weight bytes (low+high nibble each).
        unsigned int packed = *reinterpret_cast<const unsigned int*>(qs + byte0);
        // low nibbles → 4 int8 in [0..15]; high nibbles → 4 int8 in [0..15].
        unsigned int lo4 = packed & 0x0F0F0F0Fu;
        unsigned int hi4 = (packed >> 4) & 0x0F0F0F0Fu;
        // matching Q8 activations: low nibble for value gp*64 + i0, high nibble
        // for value gp*64 + 32 + i0 (4 contiguous each).
        unsigned int x_lo = *reinterpret_cast<const unsigned int*>(q8 + gp * 64u + i0);
        unsigned int x_hi = *reinterpret_cast<const unsigned int*>(q8 + gp * 64u + 32u + i0);
        pos[gp * 2u]      = __dp4a(lo4, x_lo, pos[gp * 2u]);
        pos[gp * 2u + 1u] = __dp4a(hi4, x_hi, pos[gp * 2u + 1u]);
    }

    // Fold this lane's partials with the shared integer scales; mins from bsums.
    int pos_acc = 0;
    int min_acc = 0;
#pragma unroll
    for (int gp = 0; gp < 4; gp++) {
        unsigned char sc1, mn1, sc2, mn2;
        q4k_scale_min(gp * 2,     scales, &sc1, &mn1);
        q4k_scale_min(gp * 2 + 1, scales, &sc2, &mn2);
        pos_acc += (int)sc1 * pos[gp * 2] + (int)sc2 * pos[gp * 2 + 1];
        int bs1 = q8k_bsum_i16(bsums, gp * 4)     + q8k_bsum_i16(bsums, gp * 4 + 1);
        int bs2 = q8k_bsum_i16(bsums, gp * 4 + 2) + q8k_bsum_i16(bsums, gp * 4 + 3);
        min_acc += (int)mn1 * bs1 + (int)mn2 * bs2;
    }

    return d_w * d_q8 * (float)pos_acc - dmin_w * d_q8 * (float)min_acc;
}

// Q4K GEMV with a GPU-resident Q8_K-quantized input (GPU-native path).
// Split-K: `n_splits` warps per row; one warp cooperates per block.
extern "C" __global__ void gemv_q4k_f32in_kernel(
    const unsigned char* __restrict__ matrix,  // Q4K weights [rows × blocks_per_row × 144 B]
    const unsigned char* __restrict__ xq8k,    // Q8_K input  [blocks_per_row * 292 B]
    float*                __restrict__ output, // F32 output [rows]
    unsigned int rows,
    unsigned int blocks_per_row,
    unsigned int n_splits)
{
    extern __shared__ float s_partial[];       // blockDim.x / OX_WAVE floats
    unsigned int global_thread = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int warp_id  = global_thread >> OX_WAVE_LOG2;
    unsigned int lane     = threadIdx.x & OX_LANE_MASK;
    unsigned int warp_in_block = threadIdx.x >> OX_WAVE_LOG2;
    unsigned int row   = warp_id / n_splits;
    unsigned int kpart = warp_id % n_splits;
    bool active = (row < rows);

    float sum = 0.0f;
    if (active) {
        const unsigned char* row_blocks = matrix + (size_t)row * blocks_per_row * 144u;
        for (unsigned int b = kpart; b < blocks_per_row; b += n_splits) {
            const unsigned char* w_blk = row_blocks + (size_t)b * 144u;
            const unsigned char* q8_blk = xq8k + (size_t)b * 292u;
            sum += q4k_q8kin_block_dot_dp4a(w_blk, q8_blk, lane);
        }
    }
    sum = warp_reduce_sum(sum);

    // Stage-2: the n_splits warps for a row live contiguously in this block
    // (block holds blockDim.x/OX_WAVE warps; launcher guarantees that count is a
    // multiple of n_splits). Lane 0 of each warp posts its partial; the first
    // warp of each row group sums them and writes output[row].
    if (lane == 0u) s_partial[warp_in_block] = sum;
    __syncthreads();
    if (active && kpart == 0u && lane == 0u) {
        float acc = 0.0f;
        for (unsigned int j = 0; j < n_splits; j++)
            acc += s_partial[warp_in_block + j];
        output[row] = acc;
    }
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
// Coalesced Q6_K × F32 block dot (one warp per row; ALL OX_WAVE lanes cooperate
// on the same block at once).
//
// COALESCING STRATEGY: there are 64 work-items `(n,l)` (2 sub-groups × 32 `l`),
// each producing 4 output values (q1..q4). Lanes stripe the items as
// `it = lane; it < 64; it += OX_WAVE` (OX_WAVE=32 → 2 items/lane; OX_WAVE=64 →
// 1 item/lane). With `l = it & 31`, consecutive lanes (consecutive `l` within a
// sub-group) read consecutive bytes of `qln[l]` / `qln[l+32]` / `qhn[l]`, turning
// the old 210-B-per-lane scatter into contiguous coalesced streams.
//
// NUMERIC PARITY: identical ql/qh/sc/d decode, identical `-32` bias, identical
// scale-index map (`is = l>>4`, +2/+4/+6), identical per-term `d·sc·q·x` multiply
// structure. Each lane accumulates its items into a lane-local partial; the final
// cross-lane sum is the single warp_reduce_sum in the kernel. Only f32 summation
// ORDER changes (acceptable per the argmax verification gate).
__device__ float q6k_f32in_block_dot_coop(
    const unsigned char* __restrict__ w_blk,
    const float* __restrict__ xb,
    unsigned int lane)
{
    float d  = __half2float(*reinterpret_cast<const __half*>(w_blk + 208));
    const int8_t*          sc = reinterpret_cast<const int8_t*>(w_blk + 192);
    const unsigned char*   ql = w_blk;        // [128 bytes]
    const unsigned char*   qh = w_blk + 128;  // [64 bytes]
    float sum = 0.0f;

    // 64 (n,l) items: item = n*32 + l. Lanes stripe contiguously over l.
    for (unsigned int it = lane; it < 64u; it += OX_WAVE) {
        unsigned int n = it >> 5;          // sub-group 0/1
        unsigned int l = it & 31u;         // 0..31
        const unsigned char* qln = ql + n * 64u;   // 64 bytes per sub-group
        const unsigned char* qhn = qh + n * 32u;   // 32 bytes per sub-group
        const int8_t*        scn = sc + n * 8u;     // 8 scales per sub-group
        const float*         xn  = xb + n * 128u;

        int is = (int)(l >> 4);  // 0 for l=0..15, 1 for l=16..31
        int8_t q1 = (int8_t)(((qln[l     ] & 0xFu) | (((qhn[l] >> 0) & 3u) << 4)) - 32);
        int8_t q2 = (int8_t)(((qln[l + 32] & 0xFu) | (((qhn[l] >> 2) & 3u) << 4)) - 32);
        int8_t q3 = (int8_t)(((qln[l     ] >>    4) | (((qhn[l] >> 4) & 3u) << 4)) - 32);
        int8_t q4 = (int8_t)(((qln[l + 32] >>    4) | (((qhn[l] >> 6) & 3u) << 4)) - 32);
        sum += d * (float)scn[is    ] * (float)q1 * xn[l     ];
        sum += d * (float)scn[is + 2] * (float)q2 * xn[l + 32];
        sum += d * (float)scn[is + 4] * (float)q3 * xn[l + 64];
        sum += d * (float)scn[is + 6] * (float)q4 * xn[l + 96];
    }
    return sum;
}

// Coalesced: one warp per row, lanes cooperatively stream each block's payload.
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
    // Loop blocks sequentially; the whole warp cooperates on each block so the
    // 192-byte ql+qh payload streams coalesced (vs. the old 210-B-per-lane stride).
    for (unsigned int b = 0; b < blocks_per_row; b++) {
        const unsigned char* w_blk = row_blocks + (size_t)b * 210u;
        const float* xb = x + (size_t)b * 256u;
        sum += q6k_f32in_block_dot_coop(w_blk, xb, lane);
    }
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

// ==========================================================================
// On-device attention (OX_GPU_ATTN path).
//
// These three kernels keep the decode hidden state GPU-resident across all
// layers, eliminating the per-layer CPU-attention round trip (RoPE + flash
// attention on host). They reproduce the CPU semantics exactly:
//   * NeoX / split-half partial RoPE (`apply_rope_f32`).
//   * F16 KV cache (raw IEEE half bits in `unsigned short`), matching the host
//     `f32_to_f16_bits` / `f16_le_to_f32` round-trip via `__float2half` /
//     `__half2float`.
//   * GQA decode flash attention with online (Milakov-Gimelshein) softmax,
//     causal-by-construction (single newest query attends to all kept keys),
//     and optional sliding window via a `base_row` offset into the cache.
//
// All three are sm_75-safe (no cooperative groups, no sm_80+ intrinsics).
// Entry-point names must stay in sync with the *_KERNEL_NAME constants in
// `backends/cuda.rs`.
// ==========================================================================

// rope_f32_kernel: in-place partial NeoX RoPE on F32 Q and K.
//
// Mirrors `apply_rope_f32` (compute/tensor/kernels/activation.rs): split-half
// rotation pairing lane `i` with `half + i`, where `half = rope_dim/2`. Only
// the leading `rope_dim` lanes of each head are rotated; `[rope_dim..head_dim]`
// pass through untouched. `pos == 0` is an identity (no rotation).
//
// Both Q (n_q_heads) and K (n_kv_heads) are processed in one launch: blocks
// [0, n_q_heads) rotate Q, blocks [n_q_heads, n_q_heads+n_kv_heads) rotate K.
//
// freq_i = theta^(-2*i/rope_dim), matching the CPU geometric recurrence
// freq_{i+1} = freq_i * theta^(-2/rope_dim). Computed per-lane in DOUBLE
// precision (--use_fast_math degrades the float powf/sinf/cosf paths; the
// frequency and the range-reduced sin/cos are done in double so the rotation
// matches the CPU reference even for the large `pos` values used when seeding
// the prompt KV cache). See rope_sincos.
//
// Launch: gridDim.x = n_q_heads + n_kv_heads, blockDim.x >= rope_dim/2.
extern "C" __global__ void rope_f32_kernel(
    float* q,
    float* k,
    unsigned int pos,
    unsigned int n_q_heads,
    unsigned int n_kv_heads,
    unsigned int head_dim,
    unsigned int rope_dim,
    float theta)
{
    // pos == 0 is the identity copy (values already in place); nothing to do.
    if (pos == 0u) return;

    unsigned int head = blockIdx.x;
    unsigned int total_heads = n_q_heads + n_kv_heads;
    if (head >= total_heads) return;

    // Select Q or K buffer and the head index within that buffer.
    float* buf;
    unsigned int head_local;
    if (head < n_q_heads) {
        buf = q;
        head_local = head;
    } else {
        buf = k;
        head_local = head - n_q_heads;
    }

    unsigned int half = rope_dim >> 1;
    unsigned int i = threadIdx.x;
    if (i >= half) return;

    double inv = 1.0 / (double)rope_dim;
    // freq_i = theta^(-2*i/rope_dim) == (theta^(-2/rope_dim))^i. Compute in
    // double: powf is also degraded by --use_fast_math (-> __powf), and the
    // angle = pos*freq feeds the range-reduced trig below, so a precise freq
    // keeps the rotation matching the CPU recurrence at large pos.
    double freq = pow((double)theta, -2.0 * (double)i * inv);
    double angle = (double)pos * freq;
    float c, s;
    // Range-reduced sin/cos: --use_fast_math would otherwise lower sinf/cosf to
    // the hardware intrinsics, which are inaccurate for the large angles seen
    // when seeding the prompt cache (pos up to the prompt length). The angle is
    // kept in double through the reduction so no low bits are lost before the
    // modulo-2pi. See rope_sincos above.
    rope_sincos(angle, &s, &c);

    unsigned int base = head_local * head_dim;
    float x0 = buf[base + i];
    float x1 = buf[base + half + i];
    buf[base + i]        = x0 * c - x1 * s;
    buf[base + half + i] = x0 * s + x1 * c;
}

// kv_append_f16_kernel: cast post-RoPE F32 K/V into the device F16 cache row.
//
// Writes one token row at physical position `pos` (the caller passes
// `pos % context_size`). The cache layout is row-major [pos][kv_head*head_dim+d]
// with row stride `kv_len`. Half values are stored as raw u16 bit patterns,
// matching `cast_f32_to_f16_kernel` and the host `f32_to_f16_bits`.
//
// Launch: blockDim.x = 256, gridDim.x = ceil(kv_len / 256).
extern "C" __global__ void kv_append_f16_kernel(
    const float* k,
    const float* v,
    unsigned short* kc,
    unsigned short* vc,
    unsigned int pos,
    unsigned int kv_len)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < kv_len) {
        size_t row = (size_t)pos * (size_t)kv_len + (size_t)i;
        // Bit-exact mirror of the host f32_to_f16_bits (round-half-up + subnormal
        // path), NOT __float2half (round-to-even), so the device KV cache matches
        // the CPU reference for the OX_GPU_ATTN argmax parity gate.
        kc[row] = f32_to_f16_bits_dev(k[i]);
        vc[row] = f32_to_f16_bits_dev(v[i]);
    }
}

// flash_attn_decode_kernel: GQA decode flash attention (F16 cache -> F32 out).
//
// Reproduces `flash_attention_decode_heads_impl` (compute/flash_attention.rs):
//   * scale = 1/sqrt(head_dim) (passed in).
//   * GQA: query head `qh` reads kv head `qh / (q_heads/kv_heads)`.
//   * Online (Milakov-Gimelshein) softmax over `seq_len` keys; the single
//     newest query attends to all kept keys (causal-by-construction at decode).
//   * Sliding window handled by the caller via `base_row`: keys/values are read
//     from logical rows [base_row, base_row + seq_len), so passing
//     base_row = full_seq_len - window and seq_len = window yields the
//     windowed-causal mask (RoPE encodes absolute positions).
//
// `q` = [q_heads*head_dim] (post-RoPE single query token). `k_cache`/`v_cache`
// are the layer's cache base pointers (row stride kv_len = kv_heads*head_dim).
// `attn_out` = [q_heads*head_dim].
//
// One block per query head: gridDim.x = q_heads, blockDim.x = head_dim
// (head_dim <= 256 for all target models, and head_dim is a power-of-two /
// even multiple so the tree reduction is exact). Thread `d` owns output lane
// `d`; the per-key score dot is reduced across the head_dim threads in shared
// memory, so all threads observe the same online-softmax scalar state.
//
// Dynamic shared memory: (2*head_dim + 1) * sizeof(float) bytes
//   [0 .. head_dim)          -> qsh:   the query vector (cached once)
//   [head_dim .. 2*head_dim) -> redux: per-key dot-product reduction scratch
//   [2*head_dim]             -> score: broadcast reduced score for the key
extern "C" __global__ void flash_attn_decode_kernel(
    const float* q,
    const unsigned short* k_cache,
    const unsigned short* v_cache,
    float* attn_out,
    unsigned int seq_len,
    unsigned int base_row,
    unsigned int q_heads,
    unsigned int kv_heads,
    unsigned int head_dim,
    float scale)
{
    extern __shared__ float smem[];
    float* qsh   = smem;                    // [head_dim] cached query
    float* redux = smem + head_dim;         // [head_dim] dot reduction scratch
    float* score = smem + 2u * head_dim;    // [1] broadcast score

    unsigned int qh = blockIdx.x;
    unsigned int d  = threadIdx.x;
    if (qh >= q_heads || d >= head_dim) return;

    unsigned int group_size = q_heads / kv_heads;
    unsigned int kv_head    = qh / group_size;
    unsigned int kv_len     = kv_heads * head_dim;
    unsigned int kv_off     = kv_head * head_dim;

    // Cache the query head into shared memory once.
    qsh[d] = q[qh * head_dim + d];
    __syncthreads();

    // Online-softmax running state (replicated; identical across all threads).
    float running_max = -3.402823466e+38f;  // -FLT_MAX
    float running_sum = 0.0f;
    float acc = 0.0f;  // this thread's output accumulator for lane d

    for (unsigned int t = 0; t < seq_len; ++t) {
        size_t krow = ((size_t)base_row + (size_t)t) * (size_t)kv_len + (size_t)kv_off;

        // Per-lane partial: q[d] * k[d]; tree-reduce across the head_dim lanes.
        __half hk = *reinterpret_cast<const __half*>(&k_cache[krow + d]);
        redux[d] = qsh[d] * __half2float(hk);
        __syncthreads();
        for (unsigned int stride = head_dim >> 1; stride > 0; stride >>= 1) {
            if (d < stride) redux[d] += redux[d + stride];
            __syncthreads();
        }
        if (d == 0) score[0] = redux[0] * scale;
        __syncthreads();

        float s = score[0];

        // Online softmax update (Milakov-Gimelshein).
        float new_max = running_max > s ? running_max : s;
        float corr = expf(running_max - new_max);
        float p    = expf(s - new_max);

        __half hv = *reinterpret_cast<const __half*>(&v_cache[krow + d]);
        acc = acc * corr + p * __half2float(hv);

        running_sum = running_sum * corr + p;
        running_max = new_max;
        __syncthreads();  // make redux reuse on the next key race-free
    }

    float inv = running_sum > 0.0f ? (1.0f / running_sum) : 0.0f;
    attn_out[qh * head_dim + d] = acc * inv;
}

// Split-K GQA decode attention. Each (query head, KV split) block writes an
// unnormalized online-softmax state. The split interval is identical to the
// host split_k_range: the first seq_len % split_count splits receive one extra
// token. Launch with grid (q_heads, split_count) and block head_dim.
extern "C" __global__ void flash_attn_decode_splitk_kernel(
    const float* q,
    const unsigned short* k_cache,
    const unsigned short* v_cache,
    float* partial_max,
    float* partial_sum,
    float* partial_acc,
    unsigned int seq_len,
    unsigned int base_row,
    unsigned int q_heads,
    unsigned int kv_heads,
    unsigned int head_dim,
    unsigned int split_count,
    float scale)
{
    extern __shared__ float smem[];
    float* qsh = smem;
    float* redux = smem + head_dim;
    float* score = smem + 2u * head_dim;
    unsigned int qh = blockIdx.x;
    unsigned int split_idx = blockIdx.y;
    unsigned int d = threadIdx.x;
    if (qh >= q_heads || split_idx >= split_count || d >= head_dim) return;

    unsigned int base_len = seq_len / split_count;
    unsigned int remainder = seq_len % split_count;
    unsigned int split_start = split_idx * base_len +
        (split_idx < remainder ? split_idx : remainder);
    unsigned int split_len = base_len + (split_idx < remainder ? 1u : 0u);
    unsigned int split_end = split_start + split_len;
    unsigned int group_size = q_heads / kv_heads;
    unsigned int kv_head = qh / group_size;
    unsigned int kv_len = kv_heads * head_dim;
    unsigned int kv_off = kv_head * head_dim;
    size_t partial_idx = (size_t)qh * (size_t)split_count + (size_t)split_idx;

    qsh[d] = q[qh * head_dim + d];
    __syncthreads();
    float running_max = -3.402823466e+38f;
    float running_sum = 0.0f;
    float acc = 0.0f;

    for (unsigned int t = split_start; t < split_end; ++t) {
        size_t krow = ((size_t)base_row + (size_t)t) * (size_t)kv_len + (size_t)kv_off;
        __half hk = *reinterpret_cast<const __half*>(&k_cache[krow + d]);
        redux[d] = qsh[d] * __half2float(hk);
        __syncthreads();
        for (unsigned int stride = head_dim >> 1; stride > 0; stride >>= 1) {
            if (d < stride) redux[d] += redux[d + stride];
            __syncthreads();
        }
        if (d == 0) score[0] = redux[0] * scale;
        __syncthreads();

        float s = score[0];
        float new_max = running_max > s ? running_max : s;
        float corr = expf(running_max - new_max);
        float p = expf(s - new_max);
        __half hv = *reinterpret_cast<const __half*>(&v_cache[krow + d]);
        acc = acc * corr + p * __half2float(hv);
        running_sum = running_sum * corr + p;
        running_max = new_max;
        __syncthreads();
    }

    if (d == 0) {
        partial_max[partial_idx] = running_max;
        partial_sum[partial_idx] = running_sum;
    }
    partial_acc[partial_idx * (size_t)head_dim + (size_t)d] = acc;
    // Make this block's partial writes explicitly visible before exit. Single-
    // stream ordering already serializes this kernel before the reduce kernel
    // (see gpu_native_forward.rs), but the explicit block-scope fence documents
    // the write-visibility intent and is robust if this is ever ported to a
    // multi-stream / reordered-flush launch.
    __threadfence_block();
}

// Exact associative merge of split-local online-softmax states. Empty
// partitions have local_sum == 0 and are skipped to avoid an indeterminate
// exp(-FLT_MAX - -FLT_MAX).
extern "C" __global__ void flash_attn_decode_reduce_kernel(
    const float* partial_max,
    const float* partial_sum,
    const float* partial_acc,
    float* attn_out,
    unsigned int q_heads,
    unsigned int head_dim,
    unsigned int split_count)
{
    unsigned int qh = blockIdx.x;
    unsigned int d = threadIdx.x;
    if (qh >= q_heads || d >= head_dim) return;

    size_t partial_base = (size_t)qh * (size_t)split_count;
    float global_max = -3.402823466e+38f;
    for (unsigned int split_idx = 0; split_idx < split_count; ++split_idx) {
        size_t idx = partial_base + (size_t)split_idx;
        if (partial_sum[idx] > 0.0f && partial_max[idx] > global_max) {
            global_max = partial_max[idx];
        }
    }
    // Explicit barrier between the two passes. Every thread computes the SAME
    // global_max from the same global reads (each in its own register), so this
    // is not a data-race fix; it makes the cross-warp instruction ordering
    // intent explicit before the partials are consumed in the second loop.
    __syncthreads();

    float denominator = 0.0f;
    float numerator = 0.0f;
    for (unsigned int split_idx = 0; split_idx < split_count; ++split_idx) {
        size_t idx = partial_base + (size_t)split_idx;
        float local_sum = partial_sum[idx];
        if (local_sum > 0.0f) {
            float correction = expf(partial_max[idx] - global_max);
            denominator += correction * local_sum;
            numerator += correction * partial_acc[idx * (size_t)head_dim + (size_t)d];
        }
    }

    attn_out[qh * head_dim + d] = denominator > 0.0f ? numerator / denominator : 0.0f;
}
