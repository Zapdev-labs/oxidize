/* Minimal f32 tensor ops + persistent pthread pool. Ported from
 * oxidize-cpp/src/tensor_cpu.cpp scalar reference paths.
 * These functions are the backend seam: a CUDA backend replaces them. */
#ifndef OC_TENSOR_H
#define OC_TENSOR_H

#include <stddef.h>
#include <stdint.h>

/* Persistent thread pool. n_threads <= 0 => number of online CPUs.
 * The OC_THREADS env var, if a positive integer, overrides n_threads. The
 * barrier is a spin barrier, so OC_THREADS=1 (no workers, no spinning) is the
 * way to keep sanitizer/valgrind runs from burning every core. */
void oc_pool_init(int n_threads);
void oc_pool_free(void);
int oc_pool_size(void);

/* Replicate a read-only weight region into node-local copies on every NUMA
 * node and pin pool workers round-robin across nodes; matvec then reads its
 * node's copy. No-op on single-node systems or when the process affinity
 * mask is confined to one node (e.g. under numactl -N 0). Costs one copy of
 * `size` bytes per extra node. Call once, after oc_pool_init. */
void oc_numa_replicate(const void* base, size_t size);

typedef void (*oc_range_fn)(void* ctx, size_t i0, size_t i1);
/* Runs fn over [0, n) split statically across pool threads (blocking).
 * ponytail: one job at a time, process-wide — the pool has a single job slot,
 * so two threads must not call this concurrently. Fine for one forward pass at
 * a time (including a draft + target pair run back to back); a server serving
 * two requests at once needs a real task queue here. */
void oc_parallel_for(size_t n, oc_range_fn fn, void* ctx);

/* ---- op context ------------------------------------------------------------
 * Scratch the ops below need but the CALLER owns: the int8 activation buffer
 * for oc_matvec and the packed panels for oc_matmul. These were file-statics,
 * which quietly made the runtime one-model-per-process — two models sharing one
 * buffer is a silent data race, not a crash.
 *
 * One OcCtx serves one forward pass at a time; that is the same contract the
 * model's own scratch vectors (m->q, m->normed, ...) already have, so a model
 * owning one OcCtx is exactly right. Buffers grow on demand. Must not be NULL.
 *
 * The thread pool and the NUMA replica table stay process-wide singletons on
 * purpose: they are machine state (cores, physical pages), not model state, and
 * one pool per model would oversubscribe every core N times over. */
typedef struct OcCtx OcCtx;
OcCtx* oc_ctx_new(void); /* NULL on OOM */
void oc_ctx_free(OcCtx* c);

/* y[r] = dot(W_row_r, x). W is a quantized (or f32/f16) row-major matrix of
 * ggml type `ggml_type`; threaded over rows via the pool. */
void oc_matvec(OcCtx* c, float* y, uint32_t ggml_type, const uint8_t* W,
               size_t rows, size_t cols, const float* x);

/* Y[t][r] = dot(W_row_r, X[t]) for every t in [0, n_tokens): the batched form
 * of oc_matvec. X is [n_tokens][cols] and Y is [n_tokens][rows], both row-major
 * and f32. n_tokens == 1 calls oc_matvec unchanged.
 *
 * NOT a loop over oc_matvec: each weight row is dequantized ONCE and reused for
 * all n_tokens, so W is read from DRAM once per batch instead of once per
 * token. That is what stops prefill costing the same per token as decode.
 * Aborts on a type/cols combination it has no kernel for — a wrong number here
 * would silently corrupt every logit downstream. */
void oc_matmul(OcCtx* c, float* Y, uint32_t ggml_type, const uint8_t* W,
               size_t rows, size_t cols, const float* X, size_t n_tokens);


/* out[i] = x[i] * inv_rms * w[i]  (Gemma GGUF norm weights already carry +1,
 * mirroring oxidize-cpp which uses plus_one=false for Gemma). */
void oc_rms_norm(float* out, const float* x, const float* w, size_t n, float eps);

void oc_softmax(float* x, size_t n);

/* NeoX split-half RoPE: pairs (h[i], h[i+rope_dim/2]); rope_dim==0 => full
 * head_dim; dims [rope_dim, head_dim) pass through (partial RoPE).
 * freqs (optional, len rope_dim/2): proportional divisors, angle_i /= freqs[i]
 * (Gemma4 global-layer rope_freqs.weight). NULL = none. */
void oc_rope(float* vec, size_t head_dim, size_t num_heads, size_t pos,
             float theta, size_t rope_dim, const float* freqs);

/* NORMAL (ggml non-NeoX) RoPE: rotates ADJACENT pairs (p[2i], p[2i+1]) instead
 * of split halves. llama.cpp permutes q/k for this mode on llama/mistral/yi
 * GGUFs, so applying it to those stored (permuted) weights reproduces HF/NeoX
 * rope on the natural layout. Same partial-rope + frequency recurrence as
 * oc_rope; no freqs input (only Gemma's NeoX layers use rope_freqs). */
void oc_rope_normal(float* vec, size_t head_dim, size_t num_heads, size_t pos,
                    float theta, size_t rope_dim);

/* out = gelu_tanh(gate) * up (GeGLU); out may alias gate. */
void oc_geglu(float* gate, const float* up, float* out, size_t n);

float oc_dot_f32(const float* a, const float* b, size_t n);

/* ---- KV-cache precision --------------------------------------------------
 * The element type the models store K/V as. Chosen once at startup (like
 * oc_force_isa) and read by each model loader to size and select the cache
 * path. Same value drives batched prefill AND sequential decode, so both round
 * to the same bits and the batched==sequential invariant is preserved.
 *   F32  exact, 4 B/value (default).
 *   F16  ~zero-loss, 2 B/value — half the DRAM traffic and footprint.
 *   Q8   symmetric int8 with one scale per (position, kv-head), 1 B/value.
 *   Q4   the rotated int4 rotoquant (gemma4 only; llama falls back to F16). */
typedef enum { OC_KV_F32 = 0, OC_KV_F16 = 1, OC_KV_Q8 = 2, OC_KV_Q4 = 3 } OcKvType;
/* Largest head_dim the f16/q8 attention decodes into a stack buffer; a head_dim
 * above this falls back to f32. 512 covers every shipping model (Gemma's 256 is
 * the widest) with room to spare. */
#define OC_KV_MAX_HEAD 512
void oc_kv_set_type(OcKvType t);
OcKvType oc_kv_get_type(void);
const char* oc_kv_type_name(OcKvType t);
/* Bytes per cached value for the byte-buffer path: f32=4 f16=2 q8=1. Q4 packs
 * two values per byte and carries scale+min meta, so it returns 0 and is stored
 * through the oc_kvq_* codec, not this one. */
size_t oc_kv_elem_bytes(OcKvType t);

/* Per-head KV codec for F16 and Q8 (Q4 uses oc_kvq_encode/decode in quant.h).
 * encode: n floats -> out (n*oc_kv_elem_bytes bytes); Q8 writes one scale.
 * decode: the inverse. decode(encode(x)) is what both the cache store/load and
 * the batch panel round-trip run, so prefill and decode observe identical
 * values. `out`/`in` need only 1-byte alignment (F16 halves are written byte-
 * wise). scale is unused (may be NULL) for F16. */
void oc_kv_encode(OcKvType t, const float* x, size_t n, uint8_t* out, float* scale);
void oc_kv_decode(OcKvType t, const uint8_t* in, size_t n, float scale, float* out);

#endif
