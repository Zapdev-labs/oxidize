// ============================================================================
// UNVERIFIED — NEVER COMPILED OR RUN. Written BLIND against lk_add/k_add
// (src/cuda/*.cu) and src/vulkan/shaders/add.comp. Requires WebGPU + Dawn/
// Emscripten. MAY NOT COMPILE.
//
// c[i] += x[i]: residual adds and optional bias adds.
// ============================================================================

struct AddParams {
  n : u32,
  _pad0 : u32,
  _pad1 : u32,
  _pad2 : u32,
};

@group(0) @binding(0) var<uniform> P : AddParams;
@group(0) @binding(1) var<storage, read_write> c : array<f32>;
@group(0) @binding(2) var<storage, read> x : array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i < P.n) { c[i] = c[i] + x[i]; }
}
