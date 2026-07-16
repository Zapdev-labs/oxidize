/* ============================================================================
 * UNVERIFIED — this file has NEVER been compiled or run. Written BLIND against
 * the verified CUDA reference (src/cuda/llama_cuda.cu) and the Rust Metal
 * backend (oxidize-core/src/backends/metal.rs). Requires macOS + Xcode's Metal
 * toolchain (`xcrun metal`) and an Apple GPU to compile and validate. It MAY NOT
 * COMPILE. No equivalence gate has ever been run against it.
 * ============================================================================
 *
 * GPU-resident llama-family dense decode kernels — an MSL port of the `lk_*`
 * kernels in src/cuda/llama_cuda.cu. One CUDA warp -> one Metal SIMD-group
 * (both 32-wide on Apple GPUs). __shfl_down_sync -> simd_shuffle_down,
 * __syncthreads -> threadgroup_barrier, __shared__ -> threadgroup, extern
 * __shared__ -> a dynamic threadgroup buffer bound at index 0.
 *
 * The fused dequant matvec uses the SHARED dqv<T> (metal_dequant.h), the same
 * decoder the gemma4 Metal path and the CPU forward use. host_name-suffixed
 * template instantiations give the host one pipeline per weight type, mirroring
 * the CUDA matvec() switch. */
#include "metal_dequant.h"

/* ---- fused dequant matvec, one SIMD-group per row: y[r] = dot(dq(Wrow), x). */
template <int T>
kernel void lk_matvec(device float* y            [[buffer(0)]],
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
  template [[host_name(name)]] kernel void lk_matvec<T>( \
      device float*, device const uchar*, constant int&, constant int&, \
      device const float*, constant uint&, uint, uint, uint, uint);
MV_INST("lk_matvec_f32", OC_F32)
MV_INST("lk_matvec_f16", OC_F16)
MV_INST("lk_matvec_q4_0", OC_Q4_0)
MV_INST("lk_matvec_q8_0", OC_Q8_0)
MV_INST("lk_matvec_q4_k", OC_Q4_K)
MV_INST("lk_matvec_q5_k", OC_Q5_K)
MV_INST("lk_matvec_q6_k", OC_Q6_K)
MV_INST("lk_matvec_al5xs", OC_AL5_XS)

/* ---- embedding row dequant into x (llama does NOT scale embeddings). */
template <int T>
kernel void lk_embed(device float* x         [[buffer(0)]],
                     device const uchar* row [[buffer(1)]],
                     constant int& n         [[buffer(2)]],
                     uint gid [[thread_position_in_grid]]) {
  if ((int)gid < n) x[gid] = dqv<T>(row, (int)gid);
}
#define EMB_INST(name, T) \
  template [[host_name(name)]] kernel void lk_embed<T>( \
      device float*, device const uchar*, constant int&, uint);
EMB_INST("lk_embed_f32", OC_F32)
EMB_INST("lk_embed_f16", OC_F16)
EMB_INST("lk_embed_q4_0", OC_Q4_0)
EMB_INST("lk_embed_q8_0", OC_Q8_0)
EMB_INST("lk_embed_q4_k", OC_Q4_K)
EMB_INST("lk_embed_q5_k", OC_Q5_K)
EMB_INST("lk_embed_q6_k", OC_Q6_K)
EMB_INST("lk_embed_al5xs", OC_AL5_XS)

/* ---- RMSNorm: grid.x independent vectors of length `per`, weight w shared.
 * out may alias x. w == NULL (buffer bound but flagged has_w==0) => scale 1. */
kernel void lk_rmsnorm(device float* out        [[buffer(0)]],
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

/* ---- NeoX split-half RoPE, grid.x = heads. Pairs (p[i], p[half+i]). */
kernel void lk_rope_neox(device float* vec   [[buffer(0)]],
                         constant int& head_dim [[buffer(1)]],
                         constant int& pos      [[buffer(2)]],
                         constant float& theta  [[buffer(3)]],
                         constant int& rope_len [[buffer(4)]],
                         uint blk [[threadgroup_position_in_grid]],
                         uint tid [[thread_position_in_threadgroup]],
                         uint nth [[threads_per_threadgroup]]) {
  device float* p = vec + (ulong)blk * head_dim;
  int half = rope_len / 2;
  for (int i = (int)tid; i < half; i += (int)nth) {
    float freq = pow(theta, -2.0f * (float)i / (float)rope_len);
    float angle = (float)pos * freq;
    float c = cos(angle), s = sin(angle);
    float x0 = p[i], x1 = p[half + i];
    p[i] = x0 * c - x1 * s;
    p[half + i] = x0 * s + x1 * c;
  }
}

/* ---- ggml NORMAL RoPE: rotates ADJACENT pairs (p[2i], p[2i+1]). */
kernel void lk_rope_normal(device float* vec   [[buffer(0)]],
                           constant int& head_dim [[buffer(1)]],
                           constant int& pos      [[buffer(2)]],
                           constant float& theta  [[buffer(3)]],
                           constant int& rope_len [[buffer(4)]],
                           uint blk [[threadgroup_position_in_grid]],
                           uint tid [[thread_position_in_threadgroup]],
                           uint nth [[threads_per_threadgroup]]) {
  device float* p = vec + (ulong)blk * head_dim;
  int half = rope_len / 2;
  for (int i = (int)tid; i < half; i += (int)nth) {
    float freq = pow(theta, -2.0f * (float)i / (float)rope_len);
    float angle = (float)pos * freq;
    float c = cos(angle), s = sin(angle);
    float x0 = p[2 * i], x1 = p[2 * i + 1];
    p[2 * i] = x0 * c - x1 * s;
    p[2 * i + 1] = x0 * s + x1 * c;
  }
}

/* ---- store one token's K/V rows (f32) into the f16 caches at `slot`. */
kernel void lk_kv_store(device half* kc        [[buffer(0)]],
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

/* ---- fused decode attention: one threadgroup per q head, full causal [t0,t1).
 * Dynamic threadgroup buffer sm = [hd] q | [count] scores | [nth] reduce. */
kernel void lk_attn(device float* out        [[buffer(0)]],
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

/* ---- c[i] += x[i]: residual adds and the optional bias adds. */
kernel void lk_add(device float* c        [[buffer(0)]],
                   device const float* x  [[buffer(1)]],
                   constant int& n        [[buffer(2)]],
                   uint gid [[thread_position_in_grid]]) {
  if ((int)gid < n) c[gid] += x[gid];
}

/* ---- SwiGLU: gate = silu(gate) * up  (SiLU = x*sigmoid(x)). */
kernel void lk_silu_mul(device float* gate      [[buffer(0)]],
                        device const float* up  [[buffer(1)]],
                        constant int& n         [[buffer(2)]],
                        uint gid [[thread_position_in_grid]]) {
  if ((int)gid >= n) return;
  float g = gate[gid];
  gate[gid] = (g / (1.0f + exp(-g))) * up[gid];
}
