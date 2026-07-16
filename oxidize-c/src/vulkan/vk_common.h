/* ======================================================================
 * UNVERIFIED — this file has NEVER been compiled or run.
 * Written BLIND against src/cuda/ (the verified CUDA backend) + the Rust
 * vulkan backend (oxidize-core/src/backends/vulkan.rs). Requires a Vulkan
 * 1.1+ driver, a GPU, libvulkan, and the Vulkan SDK to build/validate.
 * IT MAY NOT COMPILE and MAY BE WRONG. No verification was performed.
 * ======================================================================
 *
 * Thin Vulkan-compute host: the boilerplate the CUDA runtime gives for free
 * (device/queue, malloc/memcpy, kernel launch, stream ordering). The gemma4
 * and llama Vulkan backends sit on top of this exactly as the .cu files sit
 * on the CUDA runtime.
 *
 * Design choices (mirroring the CUDA backend where it matters):
 *   - One logical device / one compute queue. Multi-GPU (gemma4's layer-split
 *     pipeline) is NOT ported — refused at init.
 *   - Weights live in DEVICE_LOCAL storage buffers, uploaded once via a
 *     staging buffer. Scratch/activations are DEVICE_LOCAL storage buffers.
 *   - A command buffer is (re)recorded per token — push constants (pos, slot,
 *     ...) change every step, exactly as the CUDA code re-enqueues its stream.
 *     Between dependent dispatches a COARSE global memory barrier stands in for
 *     CUDA's implicit stream ordering.
 *     ponytail: coarse whole-pipeline barrier; swap for per-buffer barriers if
 *     a profiler ever says the false dependencies cost tok/s.
 *   - One fence wait per token == the CUDA "one sync per token".
 */
#ifndef OC_VK_COMMON_H
#define OC_VK_COMMON_H

#include <stddef.h>
#include <stdint.h>

#include <vulkan/vulkan.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  VkBuffer buf;
  VkDeviceMemory mem;
  VkDeviceSize size;
} VkBuf;

typedef struct {
  VkInstance instance;
  VkPhysicalDevice phys;
  VkDevice dev;
  uint32_t qfam;
  VkQueue queue;
  VkCommandPool cmdpool;
  VkDescriptorPool descpool;
  VkFence fence;
  VkCommandBuffer xfer_cmd; /* reused for one-shot transfers */
  VkCommandBuffer tok_cmd;  /* reused for the per-token compute graph */
  VkPhysicalDeviceMemoryProperties memprops;
} VkCtx;

typedef struct {
  VkPipeline pipe;
  VkPipelineLayout layout;
  VkDescriptorSetLayout dsl;
  int nbufs;
} VkPipe;

/* One compute pipeline per quant type for the dequant-fused ops, plus the
 * fixed-function ops. matvec[]/embed[] are indexed by vk_qidx(). */
typedef struct {
  VkPipe matvec[8];
  VkPipe embed[8];
  VkPipe rmsnorm;
  VkPipe rope_neox;   /* mode spec = 0 */
  VkPipe rope_normal; /* mode spec = 1 */
  VkPipe kv_store;
  VkPipe attn;
  VkPipe add;
  VkPipe silu_mul;
  VkPipe geglu;
  VkPipe resid_out;
  VkPipe softcap;
  VkPipe argmax; /* optional; greedy decode over logits */
} VkKernels;

/* Command recorder: begin -> N dispatches -> submit_wait. */
typedef struct {
  VkCtx* ctx;
  VkCommandBuffer cmd;
} VkRec;

/* OC_* quant type id -> compact 0..7 index (or -1 if unsupported). */
int vk_qidx(uint32_t ggml_type);

int vk_ctx_init(VkCtx* c, char* err, size_t errlen);
void vk_ctx_free(VkCtx* c);

int vk_buf_device(VkCtx* c, VkDeviceSize n, VkBuf* out);
int vk_upload(VkCtx* c, VkBuf* dst, const void* src, VkDeviceSize n);
int vk_zero(VkCtx* c, VkBuf* b, VkDeviceSize n);
int vk_download(VkCtx* c, const VkBuf* src, void* dst, VkDeviceSize n);
void vk_buf_free(VkCtx* c, VkBuf* b);

/* shader_dir holds the .spv produced by src/vulkan/Makefile (default
 * "src/vulkan/shaders", overridable via $OXIDIZE_VK_SHADERS). */
int vk_kernels_init(VkCtx* c, VkKernels* k, const char* shader_dir, char* err,
                    size_t errlen);
void vk_kernels_free(VkCtx* c, VkKernels* k);

int vk_rec_begin(VkCtx* c, VkRec* r);
/* Bind `p`, allocate+write a descriptor set over bufs[0..p->nbufs) (each bound
 * whole), push `pushsize` bytes, dispatch (gx,gy,gz), then a global barrier. */
void vk_dispatch(VkRec* r, const VkPipe* p, const VkBuf* const* bufs,
                 const void* push, uint32_t pushsize, uint32_t gx, uint32_t gy,
                 uint32_t gz);
/* Device-local buffer copy + coarse barrier (K=V layers, etc.). */
void vk_rec_copy(VkRec* r, const VkBuf* dst, const VkBuf* src, VkDeviceSize n);
/* Submit, wait the fence (the one sync per token), reset the descriptor pool.
 * Returns 0 on success, -1 on device-lost / submit error. */
int vk_rec_submit_wait(VkRec* r);

#ifdef __cplusplus
}
#endif

#endif
