/* ======================================================================
 * UNVERIFIED — this file has NEVER been compiled or run.
 * Written BLIND against src/cuda/ (verified CUDA backend), src/vulkan/
 * vk_common.h, and the Dawn/emdawn <webgpu/webgpu.h> C API. Requires a
 * WebGPU-capable browser (emdawnwebgpu) or native Dawn to build/validate.
 * IT MAY NOT COMPILE and MAY BE WRONG. No verification was performed.
 * ======================================================================
 *
 * Thin WebGPU compute host: the boilerplate CUDA gives for free (device/
 * queue, malloc/memcpy, kernel launch, stream ordering). gemma4_webgpu.c and
 * llama_webgpu.c sit on top of this exactly as the .cu files sit on the CUDA
 * runtime.
 *
 * Design choices:
 *   - One adapter / one device / one queue. Multi-GPU is REFUSED at the
 *     model-backend init (WebGPU has no peer-copy pipeline).
 *   - Weights live in GPU storage buffers, uploaded once via MAP_WRITE staging
 *     (or queue.WriteBuffer). Scratch/activations are STORAGE buffers.
 *   - WebGPU has NO push constants: every kernel takes a small UNIFORM buffer
 *     (16-byte-aligned structs matching the .wgsl Param structs).
 *   - A command encoder is (re)recorded per token. Between dependent dispatches
 *     the encoder's natural ordering stands in for CUDA stream order; a coarse
 *     buffer barrier is inserted via the pass boundary (one compute pass per
 *     dispatch for simplicity — ponytail: coalesce into fewer passes if a
 *     profiler says the pass overhead costs tok/s).
 *   - One queue.Submit + map-async (or wait) per token == CUDA's one sync.
 *
 * Shader loading: .wgsl source from shader_dir (default "src/webgpu/wgsl").
 * matvec.wgsl and embed.wgsl have prelude.wgsl PREPENDED (declares W + dqv).
 */
#ifndef OC_WEBGPU_COMMON_H
#define OC_WEBGPU_COMMON_H

#include <stddef.h>
#include <stdint.h>

/* Dawn / emdawnwebgpu ship <webgpu/webgpu.h>. Path is ASSUMED; validators must
 * confirm against their install (may be <webgpu.h> or via -I). */
#include <webgpu/webgpu.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  WGPUBuffer buf;
  uint64_t size;
  WGPUBufferUsage usage;
} WgpuBuf;

typedef struct {
  WGPUInstance instance;
  WGPUAdapter adapter;
  WGPUDevice device;
  WGPUQueue queue;
  /* Optional: retained for teardown ordering. */
  int ok;
} WgpuCtx;

/* One compute pipeline + bind-group layout. n_entries describes the layout
 * the host builds when creating bind groups (uniform/storage mix). */
typedef struct {
  WGPUComputePipeline pipe;
  WGPUBindGroupLayout bgl;
  int n_entries; /* layout entry count */
  int has_prelude; /* 1 if shader was built with prelude.wgsl prepended */
} WgpuPipe;

/* Pipelines. matvec/embed are shared across quant types: qtype is a uniform
 * field (unlike Vulkan's per-type specialized pipelines). */
typedef struct {
  WgpuPipe matvec;
  WgpuPipe embed;
  WgpuPipe rmsnorm;
  WgpuPipe rope;
  WgpuPipe kv_store;
  WgpuPipe attn;
  WgpuPipe add;
  WgpuPipe silu_mul;
  WgpuPipe geglu;
  WgpuPipe resid_out;
  WgpuPipe softcap;
  WgpuPipe argmax;
} WgpuKernels;

/* Per-token command recorder. */
typedef struct {
  WgpuCtx* ctx;
  WGPUCommandEncoder enc;
} WgpuRec;

/* Supported OC_* quant type ids (same gate as cuda check_type). Returns 0 ok,
 * -1 if unsupported. */
int wgpu_check_type(uint32_t ggml_type, const char* what, char* err,
                    size_t errlen);

int wgpu_ctx_init(WgpuCtx* c, char* err, size_t errlen);
void wgpu_ctx_free(WgpuCtx* c);

/* STORAGE | COPY_DST | COPY_SRC device buffer of `n` bytes (n rounded up to 4). */
int wgpu_buf_device(WgpuCtx* c, uint64_t n, WgpuBuf* out);
/* UNIFORM | COPY_DST buffer (n rounded up to 16 for alignment). */
int wgpu_buf_uniform(WgpuCtx* c, uint64_t n, WgpuBuf* out);
int wgpu_upload(WgpuCtx* c, WgpuBuf* dst, const void* src, uint64_t n);
int wgpu_zero(WgpuCtx* c, WgpuBuf* b, uint64_t n);
/* Blocking download (map-async + wait). The ONE sync primitive for logits. */
int wgpu_download(WgpuCtx* c, const WgpuBuf* src, void* dst, uint64_t n);
void wgpu_buf_free(WgpuCtx* c, WgpuBuf* b);

/* shader_dir holds the .wgsl sources (default "src/webgpu/wgsl", overridable
 * via $OXIDIZE_WEBGPU_WGSL). */
int wgpu_kernels_init(WgpuCtx* c, WgpuKernels* k, const char* shader_dir,
                      char* err, size_t errlen);
void wgpu_kernels_free(WgpuCtx* c, WgpuKernels* k);

int wgpu_rec_begin(WgpuCtx* c, WgpuRec* r);
/* Bind `p` over `bufs[0..p->n_entries)`, write `uniform` bytes into the first
 * buffer if it is the uniform slot (caller owns uniform contents — pass a
 * dedicated UNIFORM WgpuBuf already uploaded, OR pass uniforms via the
 * `ubo` argument which is uploaded inline before dispatch). Prefer the
 * dedicated ubo path: upload `ubo_bytes` into `ubo` then dispatch.
 *
 * gx/gy/gz = workgroup counts (NOT global size). */
void wgpu_dispatch(WgpuRec* r, const WgpuPipe* p, WgpuBuf* const* bufs,
                   uint32_t gx, uint32_t gy, uint32_t gz);
/* Submit and wait for the queue to go idle (one sync per token when a
 * download follows; prefill may skip download and only call this optionally). */
int wgpu_rec_submit_wait(WgpuRec* r);

/* Convenience: upload a POD uniform blob into `ubo` (must be UNIFORM usage). */
int wgpu_upload_uniform(WgpuCtx* c, WgpuBuf* ubo, const void* src, uint64_t n);

#ifdef __cplusplus
}
#endif

#endif /* OC_WEBGPU_COMMON_H */
