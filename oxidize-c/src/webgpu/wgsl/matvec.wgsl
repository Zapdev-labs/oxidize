// ============================================================================
// UNVERIFIED — NEVER COMPILED OR RUN. Written BLIND against src/cuda/llama_cuda.cu
// (lk_matvec) + src/cuda/gemma4_cuda.cu (k_matvec). Requires WebGPU + Emscripten
// to validate. MAY NOT COMPILE. PRELUDE (prelude.wgsl) is prepended by the host,
// so W (weights) is at binding(0) and dqv() is in scope.
//
// Fused dequant matvec: y[r] = dot(dequant(W row r), x). CUDA used one WARP per
// row with __shfl_down_sync; WebGPU has no subgroup shuffle in core WGSL, so this
// uses one WORKGROUP per row with a shared-memory tree reduction. Decode is nb==1
// (single token); the CUDA batched variant is not ported.
// ============================================================================

struct MvParams {
  rows : u32,
  cols : u32,
  qtype : u32,
  rowbytes : u32, // bytes per weight row (oc_row_bytes(type, cols))
};

@group(0) @binding(1) var<uniform> P : MvParams;
@group(0) @binding(2) var<storage, read> x : array<f32>;
@group(0) @binding(3) var<storage, read_write> y : array<f32>;

const WG : u32 = 64u;
var<workgroup> red : array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid : vec3<u32>,
        @builtin(local_invocation_id) lid : vec3<u32>) {
  let row = wid.x;
  if (row >= P.rows) { return; }
  let base = row * P.rowbytes;
  var acc = 0.0;
  var i = lid.x;
  loop {
    if (i >= P.cols) { break; }
    acc = acc + dqv(P.qtype, base, i) * x[i];
    i = i + WG;
  }
  red[lid.x] = acc;
  workgroupBarrier();
  var o = WG >> 1u;
  loop {
    if (o == 0u) { break; }
    if (lid.x < o) { red[lid.x] = red[lid.x] + red[lid.x + o]; }
    workgroupBarrier();
    o = o >> 1u;
  }
  if (lid.x == 0u) { y[row] = red[0]; }
}
