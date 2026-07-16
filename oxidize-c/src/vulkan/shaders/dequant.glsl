/* ======================================================================
 * UNVERIFIED — this file has NEVER been compiled or run.
 * Written BLIND against src/cuda/cuda_dequant.cuh (the reference-tested
 * device dequant) + the Rust vulkan backend. Requires a Vulkan 1.1+ driver,
 * a GPU, and the Vulkan SDK (glslangValidator) to compile to SPIR-V and
 * validate. IT MAY NOT COMPILE. No claim of correctness is made.
 * ======================================================================
 *
 * Per-VALUE dequant, a one-for-one port of dqv<T>() in cuda_dequant.cuh
 * (which is itself a port of the scalar decoders in quant.c). Ported to GLSL:
 *   - CUDA templates<int T>  ->  a runtime `qtype` switch (the includer passes
 *     a specialization constant so glslang can const-fold the switch).
 *   - byte pointers into VRAM  ->  byte offsets into a `uint data[]` SSBO named
 *     `Wbuf` that the includer MUST declare before #include-ing this file.
 *     GLSL cannot pass buffers to functions, so the decoders reference the
 *     global Wbuf directly (exactly as the CUDA code reads through `row`).
 *   - __half loads  ->  unpackHalf2x16 (core since GLSL 4.2 / Vulkan).
 *
 * The block-size and type-id constants below MUST stay in lockstep with
 * src/quant.h (OC_BLK_* and the OC_* type ids). They are duplicated here
 * because a .comp has no C preprocessor include of quant.h.
 */
#ifndef OC_VK_DEQUANT_GLSL
#define OC_VK_DEQUANT_GLSL

/* ---- OC_* quant type ids (mirror src/quant.h) ---- */
#define OC_F32    0
#define OC_F16    1
#define OC_Q4_0   2
#define OC_Q8_0   8
#define OC_Q4_K   12
#define OC_Q5_K   13
#define OC_Q6_K   14
#define OC_AL5_XS 243

/* ---- block byte sizes (mirror src/quant.h OC_BLK_*) ---- */
#define OC_BLK_Q4_0   18
#define OC_BLK_Q8_0   34
#define OC_BLK_Q4_K   144
#define OC_BLK_Q5_K   176
#define OC_BLK_Q6_K   210
#define OC_BLK_AL5_XS 14

/* The includer declares, before #include:
 *   layout(std430, binding = 0) readonly buffer WB { uint data[]; } Wbuf;
 * All offsets below are BYTE offsets into Wbuf, matching the CUDA uint8_t*. */

/* one byte at byte-offset b */
uint oc_u8(uint b) {
  return (Wbuf.data[b >> 2u] >> ((b & 3u) * 8u)) & 0xFFu;
}

/* unaligned f16 load — dh() in cuda_dequant.cuh */
float oc_dh(uint b) {
  uint h = oc_u8(b) | (oc_u8(b + 1u) << 8u);
  return unpackHalf2x16(h).x;
}

/* signed int8 at byte-offset b */
int oc_s8(uint b) {
  int v = int(oc_u8(b));
  return (v >= 128) ? v - 256 : v;
}

/* K-quant 6-bit scale/min unpack — get_scale_min_k4()/ksm(). sbase is the
 * byte offset of the 12 scale bytes. Returns sc in .x, m in .y. */
ivec2 oc_ksm(int j, uint sbase) {
  int sc, m;
  if (j < 4) {
    sc = int(oc_u8(sbase + uint(j))) & 63;
    m  = int(oc_u8(sbase + uint(j + 4))) & 63;
  } else {
    sc = (int(oc_u8(sbase + uint(j + 4))) & 0xF) | ((int(oc_u8(sbase + uint(j - 4))) >> 6) << 4);
    m  = (int(oc_u8(sbase + uint(j + 4))) >> 4)   | ((int(oc_u8(sbase + uint(j)))     >> 6) << 4);
  }
  return ivec2(sc, m);
}

/* dqv<T>(row_base, i): value i of a quantized row whose bytes start at byte
 * offset `row_base` in Wbuf. `qtype` is the OC_* type id (a spec constant at
 * the call site so the switch folds away). */
float oc_dqv(int qtype, uint row_base, int i) {
  uint ui = uint(i);
  if (qtype == OC_F32) {
    return uintBitsToFloat(Wbuf.data[(row_base >> 2u) + ui]);
  } else if (qtype == OC_F16) {
    return oc_dh(row_base + 2u * ui);
  } else if (qtype == OC_Q8_0) {
    uint b = row_base + uint(i >> 5) * uint(OC_BLK_Q8_0);
    return float(oc_s8(b + 2u + uint(i & 31))) * oc_dh(b);
  } else if (qtype == OC_Q4_0) {
    uint b = row_base + uint(i >> 5) * uint(OC_BLK_Q4_0);
    int j = i & 31;                       /* ggml split: j and j+16 share a byte */
    uint p = oc_u8(b + 2u + uint(j & 15));
    int q = (j < 16) ? int(p & 0xFu) : int(p >> 4u);
    return float(q - 8) * oc_dh(b);
  } else if (qtype == OC_AL5_XS) {
    uint b = row_base + uint(i >> 5) * uint(OC_BLK_AL5_XS);
    int bit = 3 * (i & 31);
    uint qs = b + 2u;
    int byte = bit >> 3, off = bit & 7;
    uint v = oc_u8(qs + uint(byte)) >> uint(off);
    if (off > 5) v |= oc_u8(qs + uint(byte + 1)) << uint(8 - off);
    return (float(int(v & 7u)) - 4.0) * oc_dh(b);
  } else if (qtype == OC_Q4_K) {
    uint b = row_base + uint(i >> 8) * uint(OC_BLK_Q4_K);
    int j = i & 255, gp = j >> 6, l = j & 63;
    ivec2 sm = oc_ksm(2 * gp + (l >> 5), b + 4u);
    uint p = oc_u8(b + 16u + uint(gp * 32) + uint(l & 31));
    int q = (l < 32) ? int(p & 0xFu) : int(p >> 4u);
    return oc_dh(b) * float(sm.x) * float(q) - oc_dh(b + 2u) * float(sm.y);
  } else if (qtype == OC_Q5_K) {
    uint b = row_base + uint(i >> 8) * uint(OC_BLK_Q5_K);
    int j = i & 255, gp = j >> 6, l = j & 63, ll = l & 31;
    ivec2 sm = oc_ksm(2 * gp + (l >> 5), b + 4u);
    uint lo = oc_u8(b + 48u + uint(gp * 32) + uint(ll));
    uint hi = oc_u8(b + 16u + uint(ll));
    int u = (l < 32 ? 1 : 2) << (2 * gp);              /* u1 = 1<<2gp, u2 = 2<<2gp */
    int q = (l < 32 ? int(lo & 0xFu) : int(lo >> 4u)) + (((hi & uint(u)) != 0u) ? 16 : 0);
    return oc_dh(b) * float(sm.x) * float(q) - oc_dh(b + 2u) * float(sm.y);
  } else if (qtype == OC_Q6_K) {
    uint b = row_base + uint(i >> 8) * uint(OC_BLK_Q6_K);
    int j = i & 255, g = j >> 7, r = j & 127, sub = r >> 5, l = r & 31;
    uint ql = b + uint(g * 64);
    uint hb = oc_u8(b + 128u + uint(g * 32) + uint(l));
    uint scbase = b + 192u + uint(g * 8);
    uint lo = (sub & 1) != 0 ? oc_u8(ql + uint(l + 32)) : oc_u8(ql + uint(l));
    int q = ((sub < 2) ? int(lo & 0xFu) : int(lo >> 4u)) | int(((hb >> uint(2 * sub)) & 3u) << 4u);
    int sc = oc_s8(scbase + uint((l >> 4) + 2 * sub));
    return oc_dh(b + 208u) * float(sc) * float(q - 32);
  }
  return 0.0; /* unreachable: host check_type() gates every weight */
}

#endif
