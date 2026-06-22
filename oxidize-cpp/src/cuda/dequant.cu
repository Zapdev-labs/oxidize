// GPU dequantization kernels: one CUDA thread decodes one quantized block into
// __half output values, byte/bit-faithful to the CPU dequant kernels.
//
// Ported from:
//   oxidize-cpp/src/quant.cpp  (dequant_q8_0, dequant_q2_k, dequant_q4_k,
//     dequant_q6_k, get_scale_min_k4, f16_le_to_f32) — which itself ports
//     oxidize-core/src/compute/quantization.rs
//   oxidize-core/src/backends/cuda.rs  (dequant_kernel_for block geometry:
//     Q8_0=34B/32, Q4_K=144B/256, Q6_K=210B/256, Q2_K=84B/256). Note Q8_0 here
//     is the GGUF 2+32=34 byte block layout, matching the resident dequant path
//     in cuda.rs (the CPU BLOCK_Q8_0_SIZE is also 34).
//
// Layout constants are identical to include/oxidize/quant.hpp.

#include "cuda_common.cuh"

namespace oxidize {
namespace cuda {

namespace {

constexpr int QK8_0 = 32;
constexpr int BLOCK_Q8_0 = 34;   // 2 (f16 d) + 32 (int8)
constexpr int QK_K = 256;
constexpr int BLOCK_Q2_K = 84;
constexpr int BLOCK_Q4_K = 144;
constexpr int BLOCK_Q6_K = 210;

// IEEE f16 (little-endian bytes) -> f32. Bit-exact port of
// quant.cpp::f16_le_to_f32 (uses the hardware path on device for identical
// rounding, which __half2float provides).
__device__ __forceinline__ float f16_le_to_f32(const uint8_t* p) {
  uint16_t bits = static_cast<uint16_t>(p[0]) |
                  (static_cast<uint16_t>(p[1]) << 8);
  __half h;
  // __half stores the same 16-bit IEEE-754 binary16 representation.
  *reinterpret_cast<uint16_t*>(&h) = bits;
  return __half2float(h);
}

// get_scale_min_k4 (quant.cpp / quantization.rs) — 6-bit scale/min for K-quants.
__device__ __forceinline__ void get_scale_min_k4(int j, const uint8_t* scales,
                                                 uint8_t& sc, uint8_t& m) {
  if (j < 4) {
    sc = scales[j] & 63;
    m = scales[j + 4] & 63;
  } else {
    sc = (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4);
    m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
  }
}

// --- Q8_0 -----------------------------------------------------------------
__global__ void dequant_q8_0_kernel(const uint8_t* __restrict__ in,
                                    __half* __restrict__ out,
                                    unsigned n_blocks) {
  unsigned b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= n_blocks) return;
  const uint8_t* block = in + static_cast<size_t>(b) * BLOCK_Q8_0;
  __half* o = out + static_cast<size_t>(b) * QK8_0;
  float d = f16_le_to_f32(block);
  for (int i = 0; i < QK8_0; ++i) {
    int8_t q = static_cast<int8_t>(block[2 + i]);
    o[i] = __float2half(static_cast<float>(q) * d);
  }
}

// --- Q2_K -----------------------------------------------------------------
__global__ void dequant_q2_k_kernel(const uint8_t* __restrict__ in,
                                    __half* __restrict__ out,
                                    unsigned n_blocks) {
  unsigned b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= n_blocks) return;
  const uint8_t* block = in + static_cast<size_t>(b) * BLOCK_Q2_K;
  __half* o = out + static_cast<size_t>(b) * QK_K;
  float d = f16_le_to_f32(block + 80);
  float mn = f16_le_to_f32(block + 82);
  const uint8_t* scales = block;       // [0..16]
  const uint8_t* qs = block + 16;      // [16..80]
  int q_ptr = 0;
  int is = 0;
  for (int outer = 0; outer < 2; ++outer) {
    int qs_base = outer * 32;
    for (int k = 0; k < 4; ++k) {
      uint8_t sc1 = scales[is];
      float dl1 = d * static_cast<float>(sc1 & 0xF);
      float ml1 = mn * static_cast<float>(sc1 >> 4);
      ++is;
      uint8_t sc2 = scales[is];
      float dl2 = d * static_cast<float>(sc2 & 0xF);
      float ml2 = mn * static_cast<float>(sc2 >> 4);
      ++is;
      int shift = ((is / 2 - 1) % 4) * 2;
      for (int l = 0; l < 16; ++l) {
        o[q_ptr + l] =
            __float2half(dl1 * static_cast<float>((qs[qs_base + l] >> shift) & 3) - ml1);
      }
      for (int l = 0; l < 16; ++l) {
        o[q_ptr + 16 + l] =
            __float2half(dl2 * static_cast<float>((qs[qs_base + 16 + l] >> shift) & 3) - ml2);
      }
      q_ptr += 32;
    }
  }
}

// --- Q4_K -----------------------------------------------------------------
__global__ void dequant_q4_k_kernel(const uint8_t* __restrict__ in,
                                    __half* __restrict__ out,
                                    unsigned n_blocks) {
  unsigned b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= n_blocks) return;
  const uint8_t* block = in + static_cast<size_t>(b) * BLOCK_Q4_K;
  __half* o = out + static_cast<size_t>(b) * QK_K;
  float d = f16_le_to_f32(block);
  float mn = f16_le_to_f32(block + 2);
  const uint8_t* scales = block + 4;   // [4..16]
  const uint8_t* qs = block + 16;      // [16..144]
  int out_ptr = 0;
  int is = 0;
  for (int gp = 0; gp < 4; ++gp) {
    int q_base = gp * 32;
    uint8_t sc1, m1, sc2, m2;
    get_scale_min_k4(is, scales, sc1, m1);
    get_scale_min_k4(is + 1, scales, sc2, m2);
    float d1 = d * static_cast<float>(sc1);
    float min1 = mn * static_cast<float>(m1);
    float d2 = d * static_cast<float>(sc2);
    float min2 = mn * static_cast<float>(m2);
    for (int l = 0; l < 32; ++l) {
      o[out_ptr + l] = __float2half(d1 * static_cast<float>(qs[q_base + l] & 0xF) - min1);
    }
    for (int l = 0; l < 32; ++l) {
      o[out_ptr + 32 + l] = __float2half(d2 * static_cast<float>(qs[q_base + l] >> 4) - min2);
    }
    out_ptr += 64;
    is += 2;
  }
}

// --- Q6_K -----------------------------------------------------------------
__global__ void dequant_q6_k_kernel(const uint8_t* __restrict__ in,
                                    __half* __restrict__ out,
                                    unsigned n_blocks) {
  unsigned b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= n_blocks) return;
  const uint8_t* block = in + static_cast<size_t>(b) * BLOCK_Q6_K;
  __half* o = out + static_cast<size_t>(b) * QK_K;
  float d = f16_le_to_f32(block + 208);
  const uint8_t* ql = block;           // [0..128]
  const uint8_t* qh = block + 128;     // [128..192]
  const int8_t* sc = reinterpret_cast<const int8_t*>(block + 192);  // [192..208]
  int q_ptr = 0;
  for (int group = 0; group < 2; ++group) {
    int ql_off = group * 64;
    int qh_off = group * 32;
    int sc_off = group * 8;
    for (int l = 0; l < 32; ++l) {
      int is = l / 16;
      int q1 = (static_cast<int>(ql[ql_off + l] & 0xF) |
                (static_cast<int>(qh[qh_off + l] & 3) << 4)) - 32;
      int q2 = (static_cast<int>(ql[ql_off + l + 32] & 0xF) |
                (static_cast<int>((qh[qh_off + l] >> 2) & 3) << 4)) - 32;
      int q3 = (static_cast<int>(ql[ql_off + l] >> 4) |
                (static_cast<int>((qh[qh_off + l] >> 4) & 3) << 4)) - 32;
      int q4 = (static_cast<int>(ql[ql_off + l + 32] >> 4) |
                (static_cast<int>((qh[qh_off + l] >> 6) & 3) << 4)) - 32;
      o[q_ptr + l] = __float2half(d * static_cast<float>(sc[sc_off + is]) * static_cast<float>(q1));
      o[q_ptr + 32 + l] = __float2half(d * static_cast<float>(sc[sc_off + is + 2]) * static_cast<float>(q2));
      o[q_ptr + 64 + l] = __float2half(d * static_cast<float>(sc[sc_off + is + 4]) * static_cast<float>(q3));
      o[q_ptr + 96 + l] = __float2half(d * static_cast<float>(sc[sc_off + is + 6]) * static_cast<float>(q4));
    }
    q_ptr += 128;
  }
}

}  // namespace

void launch_dequant_q8_0(const uint8_t* src, __half* dst, unsigned n_blocks,
                         cudaStream_t stream) {
  int grid = grid_for(n_blocks, kBlockSize);
  dequant_q8_0_kernel<<<grid, kBlockSize, 0, stream>>>(src, dst, n_blocks);
  CUDA_CHECK_KERNEL();
}

void launch_dequant_q4_k(const uint8_t* src, __half* dst, unsigned n_blocks,
                         cudaStream_t stream) {
  int grid = grid_for(n_blocks, kBlockSize);
  dequant_q4_k_kernel<<<grid, kBlockSize, 0, stream>>>(src, dst, n_blocks);
  CUDA_CHECK_KERNEL();
}

void launch_dequant_q6_k(const uint8_t* src, __half* dst, unsigned n_blocks,
                         cudaStream_t stream) {
  int grid = grid_for(n_blocks, kBlockSize);
  dequant_q6_k_kernel<<<grid, kBlockSize, 0, stream>>>(src, dst, n_blocks);
  CUDA_CHECK_KERNEL();
}

void launch_dequant_q2_k(const uint8_t* src, __half* dst, unsigned n_blocks,
                         cudaStream_t stream) {
  int grid = grid_for(n_blocks, kBlockSize);
  dequant_q2_k_kernel<<<grid, kBlockSize, 0, stream>>>(src, dst, n_blocks);
  CUDA_CHECK_KERNEL();
}

}  // namespace cuda
}  // namespace oxidize
