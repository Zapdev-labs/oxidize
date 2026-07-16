// ============================================================================
// UNVERIFIED — NEVER COMPILED OR RUN. Written BLIND against lk_rope_neox /
// lk_rope_normal (llama_cuda.cu) and k_rope (gemma4_cuda.cu). Requires WebGPU +
// Emscripten. MAY NOT COMPILE.
//
// RoPE, one workgroup per head (dispatch grid.x = heads). mode 0 = NeoX
// split-half pairs (p[i], p[half+i]); mode 1 = ggml NORMAL adjacent pairs
// (p[2i], p[2i+1]). dims [rope_len, head_dim) pass through (partial rotary).
// freq = theta^(-2i/rope_len). has_freqs!=0 divides the angle by freqs[i]
// (gemma4 rope_freqs.weight on global/non-SWA layers). Host skips the dispatch
// at pos==0 (identity). freqs is bound as a dummy 1-elem buffer when unused.
// ============================================================================

struct RopeParams {
  head_dim : u32,
  rope_len : u32,
  pos : u32,
  mode : u32,       // 0 = neox, 1 = normal
  theta : f32,
  has_freqs : u32,
  _pad0 : u32,
  _pad1 : u32,
};

@group(0) @binding(0) var<uniform> P : RopeParams;
@group(0) @binding(1) var<storage, read_write> vec_buf : array<f32>;
@group(0) @binding(2) var<storage, read> freqs : array<f32>;

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid : vec3<u32>,
        @builtin(local_invocation_id) lid : vec3<u32>) {
  let base = wid.x * P.head_dim;
  let half = P.rope_len / 2u;
  var i = lid.x;
  loop {
    if (i >= half) { break; }
    let freq = pow(P.theta, -2.0 * f32(i) / f32(P.rope_len));
    var angle = f32(P.pos) * freq;
    if (P.has_freqs != 0u) { angle = angle / freqs[i]; }
    let c = cos(angle);
    let s = sin(angle);
    if (P.mode == 0u) {
      let x0 = vec_buf[base + i];
      let x1 = vec_buf[base + half + i];
      vec_buf[base + i] = x0 * c - x1 * s;
      vec_buf[base + half + i] = x0 * s + x1 * c;
    } else {
      let x0 = vec_buf[base + 2u * i];
      let x1 = vec_buf[base + 2u * i + 1u];
      vec_buf[base + 2u * i] = x0 * c - x1 * s;
      vec_buf[base + 2u * i + 1u] = x0 * s + x1 * c;
    }
    i = i + 128u;
  }
}
