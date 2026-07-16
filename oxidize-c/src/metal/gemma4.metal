/* ============================================================================
 * UNVERIFIED — this file has NEVER been compiled or run. Written BLIND against
 * the verified CUDA reference (src/cuda/gemma4_cuda.cu) and the Rust Metal
 * backend (oxidize-core/src/backends/metal.rs). Requires macOS + Xcode's Metal
 * toolchain (`xcrun metal`) and an Apple GPU to compile and validate. It MAY NOT
 * COMPILE. No equivalence gate has ever been run against it.
 * ============================================================================
 *
 * GPU-resident Gemma 4 decode kernels — an MSL port of the `k_*` kernels in
 * src/cuda/gemma4_cuda.cu (prefixed `gk_` here so the single metallib has no
 * host_name collision with llama.metal's `lk_*`).
 *
 * PORTED: the f16-KV decode path — matvec (+ AL5_XS hand-fused variant), embed
 * (with sqrt(hidden) scale), sandwich RMSNorms, rope (with the global-layer
 * rope_freqs divisor), f16 KV store, fused causal attention, GeGLU, the
 * (ffn+attn)*output_scale residual, final tanh softcap, and a two-stage argmax.
 *
 * NOT PORTED (the gemma4_metal.mm host REFUSES these loudly at init):
 *   - the rotoquant int4 KV cache (k_fht / q4_row / k_kv_store_q4 / k_attn_q4).
 *     Metal has no direct analog written/tested; f16 KV is the fallback.
 *   - multi-GPU layer-split pipeline (Apple Silicon is single unified-memory
 *     device; --gpus > 1 is refused). */
#include "metal_dequant.h"

#define GK_MAX_BATCH 8

/* ---- fused dequant matvec, one SIMD-group per row (decode nb == 1). */
template <int T>
kernel void gk_matvec(device float* y            [[buffer(0)]],
                      device const uchar* W      [[buffer(1)]],
                      constant int& rows         [[buffer(2)]],
                      constant int& cols         [[buffer(3)]],
                      device const float* x      [[buffer(4)]],
                      constant uint& rowbytes    [[buffer(5)]],
                      uint tg_pos   [[threadgroup_position_in_grid]],
                      uint sg_id    [[simdgroup_index_in_threadgroup]],
                      uint sgs_per  [[simdgroups_per_threadgroup]],
                      uint lane     [[thread_index_in_simdgroup]]) {
  int row = (int)(tg_pos * sgs_per + sg_id);
  if (row >= rows) return;
  device const uchar* rp = W + (ulong)row * rowbytes;
  float acc = 0.0f;
  for (int i = (int)lane; i < cols; i += 32) acc += dqv<T>(rp, i) * x[i];
  for (uint o = 16; o > 0; o >>= 1) acc += simd_shuffle_down(acc, o);
  if (lane == 0) y[row] = acc;
}
#define MV_INST(name, T) \
  template [[host_name(name)]] kernel void gk_matvec<T>( \
      device float*, device const uchar*, constant int&, constant int&, \
      device const float*, constant uint&, uint, uint, uint, uint);
MV_INST("gk_matvec_f32", OC_F32)
MV_INST("gk_matvec_f16", OC_F16)
MV_INST("gk_matvec_q4_0", OC_Q4_0)
MV_INST("gk_matvec_q8_0", OC_Q8_0)
MV_INST("gk_matvec_q4_k", OC_Q4_K)
MV_INST("gk_matvec_q5_k", OC_Q5_K)
MV_INST("gk_matvec_q6_k", OC_Q6_K)

/* AL5_XS hand-fused matvec (3-bit codes unpacked 32 at a time), mirroring
 * k_matvec_al5xs. Same result as gk_matvec<OC_AL5_XS>; decode-speed variant. */
kernel void gk_matvec_al5xs(device float* y           [[buffer(0)]],
                            device const uchar* W     [[buffer(1)]],
                            constant int& rows        [[buffer(2)]],
                            constant int& cols        [[buffer(3)]],
                            device const float* x     [[buffer(4)]],
                            uint tg_pos   [[threadgroup_position_in_grid]],
                            uint sg_id    [[simdgroup_index_in_threadgroup]],
                            uint sgs_per  [[simdgroups_per_threadgroup]],
                            uint lane     [[thread_index_in_simdgroup]]) {
  int row = (int)(tg_pos * sgs_per + sg_id);
  if (row >= rows) return;
  int nblk = cols / 32;
  device const uchar* rp = W + (ulong)row * (ulong)nblk * OC_BLK_AL5_XS;
  float acc = 0.0f;
  for (int blk = (int)lane; blk < nblk; blk += 32) {
    device const uchar* bp = rp + (ulong)blk * OC_BLK_AL5_XS;
    float d = dh(bp);
    device const uchar* q = bp + 2;
    uint ww[4];
    ww[0] = (uint)q[0] | ((uint)q[1] << 8) | ((uint)q[2] << 16) | ((uint)q[3] << 24);
    ww[1] = (uint)q[4] | ((uint)q[5] << 8) | ((uint)q[6] << 16) | ((uint)q[7] << 24);
    ww[2] = (uint)q[8] | ((uint)q[9] << 8) | ((uint)q[10] << 16) | ((uint)q[11] << 24);
    ww[3] = 0;
    device const float* xb = x + blk * 32;
    float pa = 0.0f;
    for (int i = 0; i < 32; ++i) {
      int bit = 3 * i;
      int word = bit >> 5, off = bit & 31;
      uint v = ww[word] >> off;
      if (off > 29) v |= ww[word + 1] << (32 - off);
      float qq = (float)(int)(v & 7u) - 4.0f;
      pa += qq * xb[i];
    }
    acc += d * pa;
  }
  for (uint o = 16; o > 0; o >>= 1) acc += simd_shuffle_down(acc, o);
  if (lane == 0) y[row] = acc;
}

/* ---- embedding row dequant * emb_scale into x. */
template <int T>
kernel void gk_embed(device float* x         [[buffer(0)]],
                     device const uchar* row [[buffer(1)]],
                     constant int& n         [[buffer(2)]],
                     constant float& scale   [[buffer(3)]],
                     uint gid [[thread_position_in_grid]]) {
  if ((int)gid < n) x[gid] = dqv<T>(row, (int)gid) * scale;
}
#define EMB_INST(name, T) \
  template [[host_name(name)]] kernel void gk_embed<T>( \
      device float*, device const uchar*, constant int&, constant float&, uint);
EMB_INST("gk_embed_f32", OC_F32)
EMB_INST("gk_embed_f16", OC_F16)
EMB_INST("gk_embed_q4_0", OC_Q4_0)
EMB_INST("gk_embed_q8_0", OC_Q8_0)
EMB_INST("gk_embed_q4_k", OC_Q4_K)
EMB_INST("gk_embed_q5_k", OC_Q5_K)
EMB_INST("gk_embed_q6_k", OC_Q6_K)
EMB_INST("gk_embed_al5xs", OC_AL5_XS)

/* ---- RMSNorm: grid.x vectors of length `per`; has_w==0 is the scale-less V. */
kernel void gk_rmsnorm(device float* out        [[buffer(0)]],
                       device const float* x    [[buffer(1)]],
                       device const float* w     [[buffer(2)]],
                       constant int& per        [[buffer(3)]],
                       constant float& eps      [[buffer(4)]],
                       constant int& has_w      [[buffer(5)]],
                       uint blk [[threadgroup_position_in_grid]],
                       uint tid [[thread_position_in_threadgroup]],
                       uint nth [[threads_per_threadgroup]]) {
  x += (ulong)blk * per;
  out += (ulong)blk * per;
  threadgroup float red[256];
  float s = 0.0f;
  for (int i = (int)tid; i < per; i += (int)nth) s += x[i] * x[i];
  red[tid] = s;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint o = nth / 2; o > 0; o >>= 1) {
    if (tid < o) red[tid] += red[tid + o];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  float inv = rsqrt(red[0] / (float)per + eps);
  for (int i = (int)tid; i < per; i += (int)nth)
    out[i] = x[i] * inv * (has_w ? w[i] : 1.0f);
}

/* ---- NeoX split-half RoPE with optional per-i freqs divisor (global layers).
 * has_freqs==0 => plain angle; ==1 => angle /= freqs[i]. */
kernel void gk_rope(device float* vec         [[buffer(0)]],
                    constant int& head_dim    [[buffer(1)]],
                    constant int& pos         [[buffer(2)]],
                    constant float& theta     [[buffer(3)]],
                    constant int& rope_len    [[buffer(4)]],
                    device const float* freqs [[buffer(5)]],
                    constant int& has_freqs   [[buffer(6)]],
                    uint blk [[threadgroup_position_in_grid]],
                    uint tid [[thread_position_in_threadgroup]],
                    uint nth [[threads_per_threadgroup]]) {
  device float* p = vec + (ulong)blk * head_dim;
  int half = rope_len / 2;
  for (int i = (int)tid; i < half; i += (int)nth) {
    float freq = pow(theta, -2.0f * (float)i / (float)rope_len);
    float angle = (float)pos * freq;
    if (has_freqs) angle /= freqs[i];
    float c = cos(angle), s = sin(angle);
    float x0 = p[i], x1 = p[half + i];
    p[i] = x0 * c - x1 * s;
    p[half + i] = x0 * s + x1 * c;
  }
}

/* ---- store K/V rows (f32) into the f16 ring caches at `slot`. */
kernel void gk_kv_store(device half* kc        [[buffer(0)]],
                        device half* vc        [[buffer(1)]],
                        device const float* k  [[buffer(2)]],
                        device const float* v  [[buffer(3)]],
                        constant int& k_len    [[buffer(4)]],
                        constant int& v_len    [[buffer(5)]],
                        constant uint& slot    [[buffer(6)]],
                        uint gid [[thread_position_in_grid]]) {
  int i = (int)gid;
  if (i < k_len) kc[(ulong)slot * k_len + i] = (half)k[i];
  if (i < v_len) vc[(ulong)slot * v_len + i] = (half)v[i];
}

/* ---- fused decode attention: one threadgroup per q head (f16 KV ring). */
kernel void gk_attn(device float* out        [[buffer(0)]],
                    device const float* q     [[buffer(1)]],
                    device const half* kc     [[buffer(2)]],
                    device const half* vc     [[buffer(3)]],
                    constant int& hd          [[buffer(4)]],
                    constant int& vd          [[buffer(5)]],
                    constant int& group       [[buffer(6)]],
                    constant int& cache_cap   [[buffer(7)]],
                    constant int& t0          [[buffer(8)]],
                    constant int& t1          [[buffer(9)]],
                    constant float& scale     [[buffer(10)]],
                    threadgroup float* sm      [[threadgroup(0)]],
                    uint blk [[threadgroup_position_in_grid]],
                    uint ngrid [[threadgroups_per_grid]],
                    uint tid [[thread_position_in_threadgroup]],
                    uint nth [[threads_per_threadgroup]]) {
  int h = (int)blk;
  int kvh = h / group;
  int n_kv = (int)ngrid / group;
  int k_row = n_kv * hd, v_row = n_kv * vd;
  int count = t1 - t0;
  threadgroup float* sq = sm;
  threadgroup float* sp = sm + hd;
  threadgroup float* red = sp + count;

  for (int i = (int)tid; i < hd; i += (int)nth) sq[i] = q[h * hd + i];
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (int t = t0 + (int)tid; t < t1; t += (int)nth) {
    device const half* krow = kc + (ulong)(t % cache_cap) * k_row + kvh * hd;
    float dot = 0.0f;
    for (int d = 0; d < hd; ++d) dot += sq[d] * (float)krow[d];
    sp[t - t0] = dot * scale;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float mx = -INFINITY;
  for (int i = (int)tid; i < count; i += (int)nth) mx = fmax(mx, sp[i]);
  red[tid] = mx;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint o = nth / 2; o > 0; o >>= 1) {
    if (tid < o) red[tid] = fmax(red[tid], red[tid + o]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  mx = red[0];
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float s = 0.0f;
  for (int i = (int)tid; i < count; i += (int)nth) {
    float e = exp(sp[i] - mx);
    sp[i] = e;
    s += e;
  }
  red[tid] = s;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint o = nth / 2; o > 0; o >>= 1) {
    if (tid < o) red[tid] += red[tid + o];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  float inv = red[0] > 0.0f ? 1.0f / red[0] : 0.0f;

  for (int d = (int)tid; d < vd; d += (int)nth) {
    float acc = 0.0f;
    for (int t = t0; t < t1; ++t) {
      device const half* vrow = vc + (ulong)(t % cache_cap) * v_row + kvh * vd;
      acc += sp[t - t0] * (float)vrow[d];
    }
    out[(ulong)h * vd + d] = acc * inv;
  }
}

/* ---- GeGLU: gate = gelu_tanh(gate) * up. */
kernel void gk_geglu(device float* gate      [[buffer(0)]],
                     device const float* up  [[buffer(1)]],
                     constant int& n         [[buffer(2)]],
                     uint gid [[thread_position_in_grid]]) {
  if ((int)gid >= n) return;
  const float K = 0.797884560f; /* sqrt(2/pi) */
  float g = gate[gid];
  float gelu = 0.5f * g * (1.0f + tanh(K * (g + 0.044715f * g * g * g)));
  gate[gid] = gelu * up[gid];
}

/* ---- c[i] += x[i]. */
kernel void gk_add(device float* c        [[buffer(0)]],
                   device const float* x  [[buffer(1)]],
                   constant int& n        [[buffer(2)]],
                   uint gid [[thread_position_in_grid]]) {
  if ((int)gid < n) c[gid] += x[gid];
}

/* ---- x[i] = (ffn[i] + attn[i]) * s   (blk.N.layer_output_scale). */
kernel void gk_resid_out(device float* x         [[buffer(0)]],
                         device const float* ffn [[buffer(1)]],
                         device const float* attn[[buffer(2)]],
                         constant float& s        [[buffer(3)]],
                         constant int& n          [[buffer(4)]],
                         uint gid [[thread_position_in_grid]]) {
  if ((int)gid < n) x[gid] = (ffn[gid] + attn[gid]) * s;
}

/* ---- final softcap: l = c * tanh(l / c). */
kernel void gk_softcap(device float* l   [[buffer(0)]],
                       constant float& c [[buffer(1)]],
                       constant int& n   [[buffer(2)]],
                       uint gid [[thread_position_in_grid]]) {
  if ((int)gid < n) l[gid] = c * tanh(l[gid] / c);
}

/* ---- two-stage argmax over n floats. */
kernel void gk_argmax_stage1(device const float* v [[buffer(0)]],
                             constant int& n        [[buffer(1)]],
                             device float* bmax     [[buffer(2)]],
                             device int* bidx       [[buffer(3)]],
                             uint gid   [[thread_position_in_grid]],
                             uint gsz   [[threads_per_grid]],
                             uint blk   [[threadgroup_position_in_grid]],
                             uint tid   [[thread_position_in_threadgroup]],
                             uint nth   [[threads_per_threadgroup]]) {
  threadgroup float rmax[256];
  threadgroup int ridx[256];
  float mx = -INFINITY;
  int mi = 0;
  for (int i = (int)gid; i < n; i += (int)gsz)
    if (v[i] > mx) { mx = v[i]; mi = i; }
  rmax[tid] = mx;
  ridx[tid] = mi;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint o = nth / 2; o > 0; o >>= 1) {
    if (tid < o && rmax[tid + o] > rmax[tid]) {
      rmax[tid] = rmax[tid + o];
      ridx[tid] = ridx[tid + o];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (tid == 0) { bmax[blk] = rmax[0]; bidx[blk] = ridx[0]; }
}

kernel void gk_argmax_stage2(device const float* bmax [[buffer(0)]],
                             device const int* bidx    [[buffer(1)]],
                             constant int& n           [[buffer(2)]],
                             device int* out           [[buffer(3)]],
                             uint tid [[thread_position_in_threadgroup]],
                             uint nth [[threads_per_threadgroup]]) {
  threadgroup float rmax[256];
  threadgroup int ridx[256];
  float mx = -INFINITY;
  int mi = 0;
  for (int i = (int)tid; i < n; i += (int)nth)
    if (bmax[i] > mx) { mx = bmax[i]; mi = bidx[i]; }
  rmax[tid] = mx;
  ridx[tid] = mi;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint o = nth / 2; o > 0; o >>= 1) {
    if (tid < o && rmax[tid + o] > rmax[tid]) {
      rmax[tid] = rmax[tid + o];
      ridx[tid] = ridx[tid + o];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (tid == 0) *out = ridx[0];
}
