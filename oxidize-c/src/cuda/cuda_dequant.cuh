/* Device-side per-value dequant, shared by the gemma4 and llama CUDA backends.
 *
 * dqv<T>(row, i) = value i of a quantized row, ported one-for-one from the
 * scalar decoders in quant.c (dequant_q4_k_block & co) — the reference-tested
 * ones. Per-VALUE, not per-block: it re-derives the block scales for every
 * element it touches.
 * ponytail: that is a few extra ALU ops and L1 hits per weight. Decode is bound
 * on the weight stream from VRAM, so it does not show; if a batched prefill
 * kernel ever lands here, dequant each block into shared memory once instead.
 *
 * These are the EXACT decoders held to tests/cuda_equiv.c — the same bytes are
 * decoded here and by the CPU forward and every logit must agree. Both CUDA
 * backends #include this so a quant type proven for gemma4 is the same code path
 * for llama. Includers must have already pulled in <cuda_fp16.h> and quant.h
 * (for the OC_* type ids and OC_BLK_* block sizes). */
#ifndef OC_CUDA_DEQUANT_CUH
#define OC_CUDA_DEQUANT_CUH

#include <stdint.h>

__device__ __forceinline__ float dh(const uint8_t* p) { /* unaligned f16 load */
  uint16_t h = (uint16_t)p[0] | ((uint16_t)p[1] << 8);
  return __half2float(__ushort_as_half(h));
}

/* K-quant 6-bit scale/min unpack — get_scale_min_k4() in quant_impl.h. */
__device__ __forceinline__ void ksm(int j, const uint8_t* s, int* sc, int* m) {
  if (j < 4) {
    *sc = s[j] & 63;
    *m = s[j + 4] & 63;
  } else {
    *sc = (s[j + 4] & 0xF) | ((s[j - 4] >> 6) << 4);
    *m = (s[j + 4] >> 4) | ((s[j] >> 6) << 4);
  }
}

template <int T>
__device__ __forceinline__ float dqv(const uint8_t* row, int i);

template <>
__device__ __forceinline__ float dqv<OC_F32>(const uint8_t* row, int i) {
  return ((const float*)row)[i];
}
template <>
__device__ __forceinline__ float dqv<OC_F16>(const uint8_t* row, int i) {
  return dh(row + 2 * (size_t)i);
}
template <>
__device__ __forceinline__ float dqv<OC_Q8_0>(const uint8_t* row, int i) {
  const uint8_t* b = row + (size_t)(i >> 5) * OC_BLK_Q8_0;
  return (float)(int)(int8_t)b[2 + (i & 31)] * dh(b);
}
template <>
__device__ __forceinline__ float dqv<OC_Q4_0>(const uint8_t* row, int i) {
  const uint8_t* b = row + (size_t)(i >> 5) * OC_BLK_Q4_0;
  int j = i & 31; /* ggml split order: j and j+16 share a byte */
  uint8_t p = b[2 + (j & 15)];
  int q = j < 16 ? (p & 0xF) : (p >> 4);
  return (float)(q - 8) * dh(b);
}
template <>
__device__ __forceinline__ float dqv<OC_AL5_XS>(const uint8_t* row, int i) {
  const uint8_t* b = row + (size_t)(i >> 5) * OC_BLK_AL5_XS;
  int bit = 3 * (i & 31);
  const uint8_t* qs = b + 2;
  int byte = bit >> 3, off = bit & 7;
  uint32_t v = (uint32_t)qs[byte] >> off;
  if (off > 5) v |= (uint32_t)qs[byte + 1] << (8 - off);
  return ((float)(int)(v & 7u) - 4.0f) * dh(b);
}
template <>
__device__ __forceinline__ float dqv<OC_Q4_K>(const uint8_t* row, int i) {
  const uint8_t* b = row + (size_t)(i >> 8) * OC_BLK_Q4_K;
  int j = i & 255, gp = j >> 6, l = j & 63;
  int sc, m;
  ksm(2 * gp + (l >> 5), b + 4, &sc, &m);
  uint8_t p = b[16 + gp * 32 + (l & 31)];
  int q = l < 32 ? (p & 0xF) : (p >> 4);
  return dh(b) * (float)sc * (float)q - dh(b + 2) * (float)m;
}
template <>
__device__ __forceinline__ float dqv<OC_Q5_K>(const uint8_t* row, int i) {
  const uint8_t* b = row + (size_t)(i >> 8) * OC_BLK_Q5_K;
  int j = i & 255, gp = j >> 6, l = j & 63, ll = l & 31;
  int sc, m;
  ksm(2 * gp + (l >> 5), b + 4, &sc, &m);
  uint8_t lo = b[48 + gp * 32 + ll], hi = b[16 + ll];
  int u = (l < 32 ? 1 : 2) << (2 * gp); /* u1 = 1<<2gp, u2 = 2<<2gp */
  int q = (l < 32 ? (lo & 0xF) : (lo >> 4)) + ((hi & u) ? 16 : 0);
  return dh(b) * (float)sc * (float)q - dh(b + 2) * (float)m;
}
template <>
__device__ __forceinline__ float dqv<OC_Q6_K>(const uint8_t* row, int i) {
  const uint8_t* b = row + (size_t)(i >> 8) * OC_BLK_Q6_K;
  int j = i & 255, g = j >> 7, r = j & 127, sub = r >> 5, l = r & 31;
  const uint8_t* ql = b + g * 64;
  uint8_t hb = b[128 + g * 32 + l];
  const int8_t* sc = (const int8_t*)(b + 192 + g * 8);
  int lo = (sub & 1) ? ql[l + 32] : ql[l]; /* sub 0,2 -> l; sub 1,3 -> l+32 */
  int q = ((sub < 2) ? (lo & 0xF) : (lo >> 4)) | (((hb >> (2 * sub)) & 3) << 4);
  return dh(b + 208) * (float)sc[(l >> 4) + 2 * sub] * (float)(q - 32);
}

#endif
