// ============================================================================
// UNVERIFIED — NEVER COMPILED OR RUN. Written BLIND.
// Ported one-for-one from src/cuda/cuda_dequant.cuh (dh/ksm/dqv<T>), which is
// itself the reference-tested scalar decoder from quant.c. Requires a
// WebGPU-capable browser (or Dawn/wgpu native) + Emscripten to validate.
// MAY NOT COMPILE. No claim of correctness is made — this is a blind translation
// of CUDA device code into WGSL and has never been checked against
// tests/cuda_equiv.c or any real weights.
//
// This file is a PRELUDE: the host (webgpu_backend.c) prepends it to the shaders
// that need per-value dequant (matvec.wgsl, embed.wgsl). It declares the weight
// buffer W at @binding(0) and the byte/f16/dqv helpers. Shaders that include it
// MUST bind their weight blob at group(0) binding(0) and put everything else at
// binding(1)+.
//
// WebGPU has no u8 storage arrays, so W is an array<u32> and every access
// re-derives the byte from the containing word. This is the single most
// error-prone part of the blind port: the CUDA code indexes a uint8_t* directly.
// Block byte sizes are the OC_BLK_* constants from src/quant.h.
// ============================================================================

@group(0) @binding(0) var<storage, read> W : array<u32>;

// byte at global byte-index bi within W
fn wb(bi : u32) -> u32 {
  let word = W[bi >> 2u];
  return (word >> ((bi & 3u) * 8u)) & 0xffu;
}

// signed int8 at byte bi (arithmetic shift sign-extends)
fn ws8(bi : u32) -> i32 {
  return (i32(wb(bi)) << 24u) >> 24u;
}

// little-endian f16 at byte bi -> f32 (unpack2x16float decodes the low half-word)
fn wf16(bi : u32) -> f32 {
  let h = wb(bi) | (wb(bi + 1u) << 8u);
  return unpack2x16float(h).x;
}

// little-endian f32 at byte bi (bi need not be 4-aligned)
fn wf32(bi : u32) -> f32 {
  let v = wb(bi) | (wb(bi + 1u) << 8u) | (wb(bi + 2u) << 16u) | (wb(bi + 3u) << 24u);
  return bitcast<f32>(v);
}

// K-quant 6-bit scale/min unpack — get_scale_min_k4() in quant_impl.h.
// returns vec2(sc, m)
fn ksm(j : u32, sBase : u32) -> vec2<i32> {
  var sc : i32;
  var m : i32;
  if (j < 4u) {
    sc = i32(wb(sBase + j) & 63u);
    m = i32(wb(sBase + j + 4u) & 63u);
  } else {
    sc = i32((wb(sBase + j + 4u) & 0xFu) | ((wb(sBase + j - 4u) >> 6u) << 4u));
    m = i32((wb(sBase + j + 4u) >> 4u) | ((wb(sBase + j) >> 6u) << 4u));
  }
  return vec2<i32>(sc, m);
}

// OC_* type ids (mirror src/quant.h). Kept as WGSL consts for the switch below.
const OC_F32 : u32 = 0u;
const OC_F16 : u32 = 1u;
const OC_Q4_0 : u32 = 2u;
const OC_Q8_0 : u32 = 8u;
const OC_Q4_K : u32 = 12u;
const OC_Q5_K : u32 = 13u;
const OC_Q6_K : u32 = 14u;
const OC_AL5_XS : u32 = 243u;

// OC_BLK_* block byte sizes (mirror src/quant.h)
const BLK_Q4_0 : u32 = 18u;
const BLK_Q8_0 : u32 = 34u;
const BLK_Q4_K : u32 = 144u;
const BLK_Q5_K : u32 = 176u;
const BLK_Q6_K : u32 = 210u;
const BLK_AL5_XS : u32 = 14u;

// dqv(qtype, rowByteBase, i): value i of the quantized row at byte offset
// rowByteBase. One-for-one with cuda_dequant.cuh dqv<T>. rowByteBase is a BYTE
// offset into W (row * rowbytes).
fn dqv(qtype : u32, base : u32, i : u32) -> f32 {
  switch (qtype) {
    case 0u: { // F32
      return wf32(base + 4u * i);
    }
    case 1u: { // F16
      return wf16(base + 2u * i);
    }
    case 8u: { // Q8_0
      let b = base + (i >> 5u) * BLK_Q8_0;
      return f32(ws8(b + 2u + (i & 31u))) * wf16(b);
    }
    case 2u: { // Q4_0
      let b = base + (i >> 5u) * BLK_Q4_0;
      let j = i & 31u;
      let p = wb(b + 2u + (j & 15u));
      var q : i32;
      if (j < 16u) { q = i32(p & 0xFu); } else { q = i32(p >> 4u); }
      return f32(q - 8) * wf16(b);
    }
    case 243u: { // AL5_XS (3-bit codes)
      let b = base + (i >> 5u) * BLK_AL5_XS;
      let bit = 3u * (i & 31u);
      let qs = b + 2u;
      let byte = bit >> 3u;
      let off = bit & 7u;
      var v = wb(qs + byte) >> off;
      if (off > 5u) { v = v | (wb(qs + byte + 1u) << (8u - off)); }
      return (f32(i32(v & 7u)) - 4.0) * wf16(b);
    }
    case 12u: { // Q4_K
      let b = base + (i >> 8u) * BLK_Q4_K;
      let j = i & 255u;
      let gp = j >> 6u;
      let l = j & 63u;
      let scm = ksm(2u * gp + (l >> 5u), b + 4u);
      let p = wb(b + 16u + gp * 32u + (l & 31u));
      var q : i32;
      if (l < 32u) { q = i32(p & 0xFu); } else { q = i32(p >> 4u); }
      return wf16(b) * f32(scm.x) * f32(q) - wf16(b + 2u) * f32(scm.y);
    }
    case 13u: { // Q5_K
      let b = base + (i >> 8u) * BLK_Q5_K;
      let j = i & 255u;
      let gp = j >> 6u;
      let l = j & 63u;
      let ll = l & 31u;
      let scm = ksm(2u * gp + (l >> 5u), b + 4u);
      let lo = wb(b + 48u + gp * 32u + ll);
      let hi = wb(b + 16u + ll);
      let u = select(2u, 1u, l < 32u) << (2u * gp); // (l<32?1:2)<<(2*gp)
      var q : i32;
      if (l < 32u) { q = i32(lo & 0xFu); } else { q = i32(lo >> 4u); }
      if ((hi & u) != 0u) { q = q + 16; }
      return wf16(b) * f32(scm.x) * f32(q) - wf16(b + 2u) * f32(scm.y);
    }
    case 14u: { // Q6_K
      let b = base + (i >> 8u) * BLK_Q6_K;
      let j = i & 255u;
      let g = j >> 7u;
      let r = j & 127u;
      let sub = r >> 5u;
      let l = r & 31u;
      let ql = b + g * 64u;
      let hb = wb(b + 128u + g * 32u + l);
      let sc = ws8(b + 192u + g * 8u + (l >> 4u) + 2u * sub);
      var lo : u32;
      if ((sub & 1u) != 0u) { lo = wb(ql + l + 32u); } else { lo = wb(ql + l); }
      var q : i32;
      if (sub < 2u) { q = i32(lo & 0xFu); } else { q = i32(lo >> 4u); }
      q = q | (i32((hb >> (2u * sub)) & 3u) << 4u);
      return wf16(b + 208u) * f32(sc) * f32(q - 32);
    }
    default: {
      return 0.0; // unreachable: host check_type() gates every upload
    }
  }
}
