// ============================================================================
// UNVERIFIED — NEVER COMPILED OR RUN. Written BLIND against lk_rmsnorm
// (llama_cuda.cu) / k_rmsnorm (gemma4_cuda.cu). Requires WebGPU + Emscripten.
// MAY NOT COMPILE.
//
// RMSNorm: one workgroup per vector (dispatch grid.x vectors), length `per`,
// shared weight w. out may alias x. Mirrors oc_rms_norm; the +1 is already baked
// into the GGUF norm weight. has_w==0 is the scale-less form (gemma4 V heads).
// ============================================================================

struct RmsParams {
  per : u32,
  has_w : u32,
  eps : f32,
  _pad : u32,
};

@group(0) @binding(0) var<uniform> P : RmsParams;
@group(0) @binding(1) var<storage, read> xin : array<f32>;
@group(0) @binding(2) var<storage, read> w : array<f32>;
@group(0) @binding(3) var<storage, read_write> outv : array<f32>;

const WG : u32 = 256u;
var<workgroup> red : array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid : vec3<u32>,
        @builtin(local_invocation_id) lid : vec3<u32>) {
  let off = wid.x * P.per;
  var s = 0.0;
  var i = lid.x;
  loop {
    if (i >= P.per) { break; }
    let v = xin[off + i];
    s = s + v * v;
    i = i + WG;
  }
  red[lid.x] = s;
  workgroupBarrier();
  var o = WG >> 1u;
  loop {
    if (o == 0u) { break; }
    if (lid.x < o) { red[lid.x] = red[lid.x] + red[lid.x + o]; }
    workgroupBarrier();
    o = o >> 1u;
  }
  let inv = inverseSqrt(red[0] / f32(P.per) + P.eps);
  i = lid.x;
  loop {
    if (i >= P.per) { break; }
    var g = 1.0;
    if (P.has_w != 0u) { g = w[i]; }
    outv[off + i] = xin[off + i] * inv * g;
    i = i + WG;
  }
}
