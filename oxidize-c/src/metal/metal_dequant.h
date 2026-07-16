/* ============================================================================
 * UNVERIFIED — this file has NEVER been compiled or run. It was written BLIND
 * against the verified CUDA reference (src/cuda/cuda_dequant.cuh) and the Rust
 * Metal backend (oxidize-core/src/backends/metal.rs). It requires macOS + Xcode
 * (the Metal toolchain: `xcrun metal`) and an Apple GPU to compile and validate.
 * It MAY NOT COMPILE. No equivalence gate (the CUDA one is tests/cuda_equiv.c)
 * has ever been run against it. Treat every logit it would produce as unproven.
 * ============================================================================
 *
 * Device-side per-value dequant for the Metal gemma4 and llama backends — a
 * one-for-one MSL port of dqv<T>()/dh()/ksm() from src/cuda/cuda_dequant.cuh,
 * which are themselves the reference-tested scalar decoders from quant.c.
 *
 * Per-VALUE, not per-block: dqv<T>(row, i) re-derives the block scales for every
 * element it touches. Decode is bound on the weight stream from VRAM (Apple
 * unified memory here), so the extra ALU does not show. Included by llama.metal
 * and gemma4.metal so a quant type is decoded the SAME way on both paths.
 *
 * The OC_* type ids and OC_BLK_* block sizes are duplicated from quant.h /
 * quant_impl.h because MSL cannot include the C headers. THEY MUST STAY IN SYNC
 * with quant.h — if a block layout changes there and not here, every logit is
 * silently wrong. (The CUDA side gets these from quant.h via the extern "C"
 * include; MSL has no such luxury.) */
#ifndef OC_METAL_DEQUANT_H
#define OC_METAL_DEQUANT_H

#include <metal_stdlib>
using namespace metal;

/* --- ggml type ids (mirror enum in quant.h) --- */
#define OC_F32    0
#define OC_F16    1
#define OC_Q4_0   2
#define OC_Q8_0   8
#define OC_Q4_K   12
#define OC_Q5_K   13
#define OC_Q6_K   14
#define OC_AL5_XS 243

/* --- block byte sizes (mirror #defines in quant.h) --- */
#define OC_BLK_Q4_0   18  /* f16 d + 16 nibble bytes */
#define OC_BLK_Q8_0   34  /* f16 d + 32 int8 */
#define OC_BLK_Q4_K   144
#define OC_BLK_Q5_K   176
#define OC_BLK_Q6_K   210
#define OC_BLK_AL5_XS 14  /* f16 scale + 12 bytes of 3-bit codes (32 codes) */

/* unaligned f16 load — mirrors dh() in cuda_dequant.cuh */
static inline float dh(device const uchar* p) {
  ushort h = (ushort)p[0] | ((ushort)p[1] << 8);
  return float(as_type<half>(h));
}

/* K-quant 6-bit scale/min unpack — get_scale_min_k4() / ksm() */
static inline void ksm(int j, device const uchar* s, thread int* sc,
                       thread int* m) {
  if (j < 4) {
    *sc = s[j] & 63;
    *m = s[j + 4] & 63;
  } else {
    *sc = (s[j + 4] & 0xF) | ((s[j - 4] >> 6) << 4);
    *m = (s[j + 4] >> 4) | ((s[j] >> 6) << 4);
  }
}

/* Per-value dequant, templated on the ggml type id. Exact port of dqv<T>. */
template <int T>
static inline float dqv(device const uchar* row, int i);

template <>
inline float dqv<OC_F32>(device const uchar* row, int i) {
  return ((device const float*)row)[i];
}
template <>
inline float dqv<OC_F16>(device const uchar* row, int i) {
  return dh(row + 2 * (ulong)i);
}
template <>
inline float dqv<OC_Q8_0>(device const uchar* row, int i) {
  device const uchar* b = row + (ulong)(i >> 5) * OC_BLK_Q8_0;
  return (float)(int)(char)b[2 + (i & 31)] * dh(b);
}
template <>
inline float dqv<OC_Q4_0>(device const uchar* row, int i) {
  device const uchar* b = row + (ulong)(i >> 5) * OC_BLK_Q4_0;
  int j = i & 31; /* ggml split order: j and j+16 share a byte */
  uchar p = b[2 + (j & 15)];
  int q = j < 16 ? (p & 0xF) : (p >> 4);
  return (float)(q - 8) * dh(b);
}
template <>
inline float dqv<OC_AL5_XS>(device const uchar* row, int i) {
  device const uchar* b = row + (ulong)(i >> 5) * OC_BLK_AL5_XS;
  int bit = 3 * (i & 31);
  device const uchar* qs = b + 2;
  int byte = bit >> 3, off = bit & 7;
  uint v = (uint)qs[byte] >> off;
  if (off > 5) v |= (uint)qs[byte + 1] << (8 - off);
  return ((float)(int)(v & 7u) - 4.0f) * dh(b);
}
template <>
inline float dqv<OC_Q4_K>(device const uchar* row, int i) {
  device const uchar* b = row + (ulong)(i >> 8) * OC_BLK_Q4_K;
  int j = i & 255, gp = j >> 6, l = j & 63;
  int sc, m;
  ksm(2 * gp + (l >> 5), b + 4, &sc, &m);
  uchar p = b[16 + gp * 32 + (l & 31)];
  int q = l < 32 ? (p & 0xF) : (p >> 4);
  return dh(b) * (float)sc * (float)q - dh(b + 2) * (float)m;
}
template <>
inline float dqv<OC_Q5_K>(device const uchar* row, int i) {
  device const uchar* b = row + (ulong)(i >> 8) * OC_BLK_Q5_K;
  int j = i & 255, gp = j >> 6, l = j & 63, ll = l & 31;
  int sc, m;
  ksm(2 * gp + (l >> 5), b + 4, &sc, &m);
  uchar lo = b[48 + gp * 32 + ll], hi = b[16 + ll];
  int u = (l < 32 ? 1 : 2) << (2 * gp); /* u1 = 1<<2gp, u2 = 2<<2gp */
  int q = (l < 32 ? (lo & 0xF) : (lo >> 4)) + ((hi & u) ? 16 : 0);
  return dh(b) * (float)sc * (float)q - dh(b + 2) * (float)m;
}
template <>
inline float dqv<OC_Q6_K>(device const uchar* row, int i) {
  device const uchar* b = row + (ulong)(i >> 8) * OC_BLK_Q6_K;
  int j = i & 255, g = j >> 7, r = j & 127, sub = r >> 5, l = r & 31;
  device const uchar* ql = b + g * 64;
  uchar hb = b[128 + g * 32 + l];
  device const char* sc = (device const char*)(b + 192 + g * 8);
  int lo = (sub & 1) ? ql[l + 32] : ql[l]; /* sub 0,2 -> l; sub 1,3 -> l+32 */
  int q = ((sub < 2) ? (lo & 0xF) : (lo >> 4)) | (((hb >> (2 * sub)) & 3) << 4);
  return dh(b + 208) * (float)sc[(l >> 4) + 2 * sub] * (float)(q - 32);
}

#endif
