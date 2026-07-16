// ============================================================================
// UNVERIFIED — NEVER COMPILED OR RUN. Written BLIND against k_argmax_stage1/
// k_argmax_stage2 (src/cuda/gemma4_cuda.cu). Requires WebGPU + Dawn/
// Emscripten. MAY NOT COMPILE.
//
// Two-pass argmax over n floats. Pass 1 (dispatch ARGMAX_BLOCKS workgroups):
// each workgroup writes its local (max, idx) into bmax/bidx. Pass 2 (dispatch
// 1 workgroup): reduces those into a single i32 at out[0]. Softcap is monotonic
// so gemma greedy can skip softcap for argmax (host path).
//
// mode 0 = stage1, mode 1 = stage2. Host rebinds the same pipeline with
// different uniforms/buffers.
// ============================================================================

struct ArgmaxParams {
  n : u32,      /* stage1: vocab; stage2: ARGMAX_BLOCKS */
  mode : u32,   /* 0 = stage1, 1 = stage2 */
  _pad0 : u32,
  _pad1 : u32,
};

@group(0) @binding(0) var<uniform> P : ArgmaxParams;
@group(0) @binding(1) var<storage, read> v : array<f32>;          /* logits or bmax */
@group(0) @binding(2) var<storage, read> bidx_in : array<i32>;    /* stage2 only; dummy 1-elem for stage1 */
@group(0) @binding(3) var<storage, read_write> bmax : array<f32>; /* stage1 out / unused stage2 */
@group(0) @binding(4) var<storage, read_write> bidx : array<i32>; /* stage1 out OR stage2 final out[0] */

const WG : u32 = 256u;
var<workgroup> rmax : array<f32, 256>;
var<workgroup> ridx : array<i32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid : vec3<u32>,
        @builtin(local_invocation_id) lid : vec3<u32>,
        @builtin(num_workgroups) nwg : vec3<u32>) {
  var mx = -1.0e30;
  var mi : i32 = 0;
  let tt = lid.x;

  if (P.mode == 0u) {
    /* stage1: grid-stride over logits */
    var i = wid.x * WG + tt;
    loop {
      if (i >= P.n) { break; }
      if (v[i] > mx) { mx = v[i]; mi = i32(i); }
      i = i + nwg.x * WG;
    }
  } else {
    /* stage2: reduce over the stage1 partials (v=bmax, bidx_in=bidx) */
    var i = tt;
    loop {
      if (i >= P.n) { break; }
      if (v[i] > mx) { mx = v[i]; mi = bidx_in[i]; }
      i = i + WG;
    }
  }

  rmax[tt] = mx;
  ridx[tt] = mi;
  workgroupBarrier();
  var o = WG >> 1u;
  loop {
    if (o == 0u) { break; }
    if (tt < o && rmax[tt + o] > rmax[tt]) {
      rmax[tt] = rmax[tt + o];
      ridx[tt] = ridx[tt + o];
    }
    workgroupBarrier();
    o = o >> 1u;
  }
  if (tt == 0u) {
    if (P.mode == 0u) {
      bmax[wid.x] = rmax[0];
      bidx[wid.x] = ridx[0];
    } else {
      bidx[0] = ridx[0];
    }
  }
}
