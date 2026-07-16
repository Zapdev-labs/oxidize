// ============================================================================
// UNVERIFIED — NEVER COMPILED OR RUN. Written BLIND against lk_kv_store /
// k_kv_store (src/cuda/*.cu). Requires WebGPU + Dawn/Emscripten. MAY NOT
// COMPILE.
//
// Store one token's K/V rows into the f16 caches at `slot`. Mirrors CUDA's
// __half KV path (NOT the Vulkan FP32 divergence). Gemma4 rotoquant int4 is
// NOT ported — host refuses kv_quant models at init.
//
// Storage is array<u32> packing two little-endian f16s per word (CUDA-dense
// layout). Each invocation owns ONE word of K and ONE word of V to avoid the
// read-modify-write race of two threads sharing a u32. Odd trailing elements
// leave the high half zeroed.
// ============================================================================

struct KvParams {
  k_len : u32,
  v_len : u32,
  slot : u32,
  _pad : u32,
};

@group(0) @binding(0) var<uniform> P : KvParams;
@group(0) @binding(1) var<storage, read_write> kc : array<u32>; /* packed f16 */
@group(0) @binding(2) var<storage, read_write> vc : array<u32>;
@group(0) @binding(3) var<storage, read> k : array<f32>;
@group(0) @binding(4) var<storage, read> v : array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let wi = gid.x; /* word index within the row */

  let k_words = (P.k_len + 1u) >> 1u;
  if (wi < k_words) {
    let i0 = wi * 2u;
    let lo = pack2x16float(vec2<f32>(k[i0], 0.0)) & 0xffffu;
    var hi = 0u;
    if (i0 + 1u < P.k_len) {
      hi = pack2x16float(vec2<f32>(k[i0 + 1u], 0.0)) & 0xffffu;
    }
    let base = (P.slot * P.k_len) >> 1u; /* word base; requires even k_len*slot
                                           * — host pads KV rows to even length */
    /* Safer absolute element addressing: convert element offset to word. */
    let elem = P.slot * P.k_len + i0;
    kc[elem >> 1u] = lo | (hi << 16u);
  }

  let v_words = (P.v_len + 1u) >> 1u;
  if (wi < v_words) {
    let i0 = wi * 2u;
    let lo = pack2x16float(vec2<f32>(v[i0], 0.0)) & 0xffffu;
    var hi = 0u;
    if (i0 + 1u < P.v_len) {
      hi = pack2x16float(vec2<f32>(v[i0 + 1u], 0.0)) & 0xffffu;
    }
    let elem = P.slot * P.v_len + i0;
    vc[elem >> 1u] = lo | (hi << 16u);
  }
}
