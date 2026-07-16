// ============================================================================
// UNVERIFIED — NEVER COMPILED OR RUN. Written BLIND against k_softcap
// (src/cuda/gemma4_cuda.cu) and src/vulkan/shaders/softcap.comp. Requires
// WebGPU + Dawn/Emscripten. MAY NOT COMPILE.
//
// Final logit softcap: l = c * tanh(l / c).
// ============================================================================

struct SoftcapParams {
  n : u32,
  _pad : u32,
  c : f32,
  _pad1 : u32,
};

@group(0) @binding(0) var<uniform> P : SoftcapParams;
@group(0) @binding(1) var<storage, read_write> logits : array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i < P.n) { logits[i] = P.c * tanh(logits[i] / P.c); }
}
