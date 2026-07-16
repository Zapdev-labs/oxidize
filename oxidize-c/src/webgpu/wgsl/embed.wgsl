// ============================================================================
// UNVERIFIED — NEVER COMPILED OR RUN. Written BLIND against lk_embed
// (llama_cuda.cu) and k_embed (gemma4_cuda.cu). Requires WebGPU + Emscripten.
// MAY NOT COMPILE. PRELUDE prepended: W at binding(0), dqv() in scope.
//
// Embedding row dequant into x, times `scale`. llama passes scale=1.0; gemma4
// passes sqrt(hidden) (m->emb_scale). The whole tok_embd blob is bound as W and
// `rowbase` is the token row byte offset (tk * oc_row_bytes) — WebGPU dynamic
// buffer offsets must be 256-aligned, which GGUF rows are not, so the offset is
// passed as a uniform instead of by re-binding.
// ============================================================================

struct EmbParams {
  n : u32,
  qtype : u32,
  rowbase : u32, // token row byte offset into W
  scale : f32,
};

@group(0) @binding(1) var<uniform> P : EmbParams;
@group(0) @binding(2) var<storage, read_write> x : array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i < P.n) { x[i] = dqv(P.qtype, P.rowbase, i) * P.scale; }
}
