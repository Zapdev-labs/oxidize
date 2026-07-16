// ============================================================================
// UNVERIFIED — NEVER COMPILED OR RUN. Written BLIND against k_resid_out
// (src/cuda/gemma4_cuda.cu) and src/vulkan/shaders/resid_out.comp. Requires
// WebGPU + Dawn/Emscripten. MAY NOT COMPILE.
//
// x[i] = (ffn[i] + attn[i]) * s  (blk.N.layer_output_scale).
// ============================================================================

struct ResidParams {
  n : u32,
  _pad : u32,
  s : f32,
  _pad1 : u32,
};

@group(0) @binding(0) var<uniform> P : ResidParams;
@group(0) @binding(1) var<storage, read_write> x : array<f32>;
@group(0) @binding(2) var<storage, read> ffn : array<f32>;
@group(0) @binding(3) var<storage, read> attn : array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i < P.n) { x[i] = (ffn[i] + attn[i]) * P.s; }
}
