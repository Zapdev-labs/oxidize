/* ============================================================================
 * UNVERIFIED — this file has NEVER been compiled or run. It was written BLIND
 * against the verified CUDA reference (src/cuda/) and the Rust Metal backend
 * (oxidize-core/src/backends/metal.rs). It requires macOS + Xcode (Metal /
 * Foundation frameworks, `xcrun metal`) and an Apple GPU to compile and
 * validate. It MAY NOT COMPILE. No equivalence gate has ever been run against
 * it. Treat every buffer / pipeline / dispatch helper here as unproven.
 * ============================================================================
 *
 * Thin Metal compute host: the boilerplate the CUDA runtime gives for free
 * (device/queue, malloc/memcpy, kernel launch, stream ordering). gemma4_metal.mm
 * and llama_metal.mm sit on top of this exactly as the .cu files sit on the
 * CUDA runtime.
 *
 * Design choices (mirroring the CUDA backend where it matters):
 *   - One MTLDevice / one MTLCommandQueue. Multi-GPU (gemma4 CUDA's layer-split
 *     pipeline) is NOT ported — refused at init by the arch backends.
 *   - Weights stay in their GGUF-quantized form in shared MTLBuffers (Apple
 *     unified memory); the MSL kernels fuse dqv<T> with the matvec, same
 *     contract as cuda_dequant.cuh.
 *   - A command buffer is (re)recorded per token — constants (pos, slot, ...)
 *     change every step, exactly as CUDA re-enqueues its stream.
 *   - One [cmd commit] + waitUntilCompleted per token == the CUDA "one sync".
 *
 * ObjC id<> handles are stored as void* so this header stays C-includable from
 * the public gemma4_metal.h / llama_metal.h surface. The .mm translation units
 * cast them back.
 */
#ifndef OC_METAL_COMMON_H
#define OC_METAL_COMMON_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  size_t width, height, depth;
} MtSize;

typedef struct {
  void* device;  /* id<MTLDevice> */
  void* queue;   /* id<MTLCommandQueue> */
  void* library; /* id<MTLLibrary> from the compiled metallib */
  void* pipes;   /* internal NSMutableDictionary name -> PSO (ARC) */
  void* dummy;   /* tiny shared MTLBuffer for unused device slots */
} MtCtx;

typedef struct {
  void* buf; /* id<MTLBuffer> */
  size_t size;
} MtBuf;

typedef struct {
  void* pso; /* id<MTLComputePipelineState> */
} MtPipe;

/* Command recorder: begin -> N dispatches -> commit_wait. */
typedef struct {
  MtCtx* ctx;
  void* cmd; /* id<MTLCommandBuffer> */
  void* enc; /* id<MTLComputeCommandEncoder> */
} MtRec;

/* One slot in the kernel's [[buffer(i)]] table. Device buffers and setBytes
 * constants are interleaved the way gemma4.metal / llama.metal declare them. */
typedef enum { MT_BIND_BUF = 0, MT_BIND_BYTES = 1 } MtBindKind;
typedef struct {
  MtBindKind kind;
  const MtBuf* buf; /* MT_BIND_BUF (NULL => ctx dummy) */
  size_t offset;    /* byte offset into buf */
  const void* bytes;
  size_t nbytes; /* MT_BIND_BYTES */
} MtBind;

#define MT_BUF(b) \
  ((MtBind){MT_BIND_BUF, (b), 0, NULL, 0})
#define MT_BUF_OFF(b, off) \
  ((MtBind){MT_BIND_BUF, (b), (off), NULL, 0})
#define MT_BYTES(ptr, n) \
  ((MtBind){MT_BIND_BYTES, NULL, 0, (ptr), (n)})

/* OC_* quant type id -> compact 0..7 index (or -1 if unsupported). */
int mt_qidx(uint32_t ggml_type);

/* Default metallib path: $OXIDIZE_METAL_LIB, else "src/metal/build/oxidize.metallib". */
int mt_ctx_init(MtCtx* c, const char* metallib_path, char* err, size_t errlen);
void mt_ctx_free(MtCtx* c);

int mt_buf_alloc(MtCtx* c, size_t n, MtBuf* out);
int mt_buf_upload(MtCtx* c, MtBuf* dst, const void* src, size_t n);
int mt_buf_zero(MtCtx* c, MtBuf* b, size_t n);
int mt_buf_download(MtCtx* c, const MtBuf* src, void* dst, size_t n);
void mt_buf_free(MtCtx* c, MtBuf* b);

int mt_pipe_get(MtCtx* c, const char* name, MtPipe* out, char* err,
                size_t errlen);

const char* mt_matvec_name(const char* prefix, uint32_t ggml_type);
const char* mt_embed_name(const char* prefix, uint32_t ggml_type);

int mt_rec_begin(MtCtx* c, MtRec* r);

/* Bind PSO, apply binds[i] at buffer index i, optional threadgroup mem at
 * index 0, dispatch. */
void mt_dispatch(MtRec* r, const MtPipe* p, const MtBind* binds, int nbinds,
                 size_t threadgroup_mem, MtSize grid, MtSize tgs);

int mt_rec_commit_wait(MtRec* r);
int mt_rec_commit(MtRec* r);

MtSize mt_tgs_1d(size_t n);
MtSize mt_grid_1d(size_t n, size_t tgs);
MtSize mt_matvec_tgs(void);
MtSize mt_matvec_grid(int rows);

#ifdef __cplusplus
}
#endif

#endif /* OC_METAL_COMMON_H */
