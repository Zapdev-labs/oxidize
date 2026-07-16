// ============================================================================
// UNVERIFIED — NEVER COMPILED OR RUN. Written BLIND against lk_attn/k_attn
// (src/cuda/*.cu) + src/vulkan/shaders/attn.comp. Requires WebGPU + Dawn/
// Emscripten. MAY NOT COMPILE.
//
// Fused decode attention: ONE WORKGROUP PER Q HEAD, full causal over [t0, t1).
// GQA: kv head = h / group. Ring index (t % cache_cap). KV caches are packed
// f16 in array<u32> (see kv_store.wgsl) — matches CUDA, NOT Vulkan's FP32.
//
// Scores live in a global scratch SSBO (binding 5), one cache_cap-stride region
// per head: scratch[h*cache_cap + (t-t0)]. q is cached in a fixed 256-float
// workgroup array (host MUST refuse head_dim > 256).
// ============================================================================

struct AttnParams {
  hd : u32,
  vd : u32,
  group : u32,
  cache_cap : u32,
  t0 : u32,
  t1 : u32,
  n_heads : u32, /* = dispatch workgroups.x; used to derive n_kv */
  _pad : u32,
  scale : f32,
  _pad1 : u32,
  _pad2 : u32,
  _pad3 : u32,
};

@group(0) @binding(0) var<uniform> P : AttnParams;
@group(0) @binding(1) var<storage, read_write> outv : array<f32>;
@group(0) @binding(2) var<storage, read> q : array<f32>;
@group(0) @binding(3) var<storage, read> kc : array<u32>; /* packed f16 */
@group(0) @binding(4) var<storage, read> vc : array<u32>;
@group(0) @binding(5) var<storage, read_write> scratch : array<f32>;

const WG : u32 = 128u;
var<workgroup> sq : array<f32, 256>;
var<workgroup> red : array<f32, 128>;

fn load_f16(buf : ptr<storage, array<u32>, read>, idx : u32) -> f32 {
  let word = (*buf)[idx >> 1u];
  let shift = (idx & 1u) * 16u;
  let h = (word >> shift) & 0xffffu;
  return unpack2x16float(h).x;
}

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid : vec3<u32>,
        @builtin(local_invocation_id) lid : vec3<u32>) {
  let h = wid.x;
  let tt = lid.x;
  let n_kv = P.n_heads / P.group;
  let kvh = h / P.group;
  let k_row = n_kv * P.hd;
  let v_row = n_kv * P.vd;
  let count = P.t1 - P.t0;
  let sbase = h * P.cache_cap;

  var i = tt;
  loop {
    if (i >= P.hd) { break; }
    sq[i] = q[h * P.hd + i];
    i = i + WG;
  }
  workgroupBarrier();

  /* scores */
  var t = P.t0 + tt;
  loop {
    if (t >= P.t1) { break; }
    let krow = (t % P.cache_cap) * k_row + kvh * P.hd;
    var dot = 0.0;
    var d = 0u;
    loop {
      if (d >= P.hd) { break; }
      dot = dot + sq[d] * load_f16(&kc, krow + d);
      d = d + 1u;
    }
    scratch[sbase + (t - P.t0)] = dot * P.scale;
    t = t + WG;
  }
  workgroupBarrier();

  /* max */
  var mx = -1.0e30;
  i = tt;
  loop {
    if (i >= count) { break; }
    mx = max(mx, scratch[sbase + i]);
    i = i + WG;
  }
  red[tt] = mx;
  workgroupBarrier();
  var o = WG >> 1u;
  loop {
    if (o == 0u) { break; }
    if (tt < o) { red[tt] = max(red[tt], red[tt + o]); }
    workgroupBarrier();
    o = o >> 1u;
  }
  mx = red[0];
  workgroupBarrier();

  /* exp + sum */
  var s = 0.0;
  i = tt;
  loop {
    if (i >= count) { break; }
    let e = exp(scratch[sbase + i] - mx);
    scratch[sbase + i] = e;
    s = s + e;
    i = i + WG;
  }
  red[tt] = s;
  workgroupBarrier();
  o = WG >> 1u;
  loop {
    if (o == 0u) { break; }
    if (tt < o) { red[tt] = red[tt] + red[tt + o]; }
    workgroupBarrier();
    o = o >> 1u;
  }
  var inv = 0.0;
  if (red[0] > 0.0) { inv = 1.0 / red[0]; }
  workgroupBarrier();

  /* weighted V */
  var d = tt;
  loop {
    if (d >= P.vd) { break; }
    var acc = 0.0;
    t = P.t0;
    loop {
      if (t >= P.t1) { break; }
      let vrow = (t % P.cache_cap) * v_row + kvh * P.vd;
      acc = acc + scratch[sbase + (t - P.t0)] * load_f16(&vc, vrow + d);
      t = t + 1u;
    }
    outv[h * P.vd + d] = acc * inv;
    d = d + WG;
  }
}
