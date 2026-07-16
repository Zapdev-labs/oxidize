/* ======================================================================
 * UNVERIFIED — this file has NEVER been compiled or run.
 * Written BLIND against src/cuda/ + oxidize-core/src/backends/vulkan.rs.
 * Requires a Vulkan 1.1+ driver, a GPU, libvulkan, and the Vulkan SDK.
 * IT MAY NOT COMPILE and MAY BE WRONG. No verification was performed.
 * Known-blind risks: memory-type selection, descriptor-pool sizing, barrier
 * scopes, and SPIR-V loading are all plausible-but-untested.
 * ====================================================================== */
#include "vk_common.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../quant.h" /* OC_* type ids */

#define VK_CHECK(expr)                                                     \
  do {                                                                     \
    VkResult r_ = (expr);                                                  \
    if (r_ != VK_SUCCESS) {                                                \
      if (err && errlen)                                                   \
        snprintf(err, errlen, "vulkan: %s failed (%d) at %s:%d", #expr,    \
                 (int)r_, __FILE__, __LINE__);                            \
      return -1;                                                           \
    }                                                                      \
  } while (0)

static const uint32_t OC_TYPES[8] = {OC_F32, OC_F16,  OC_Q4_0, OC_Q8_0,
                                     OC_Q4_K, OC_Q5_K, OC_Q6_K, OC_AL5_XS};

int vk_qidx(uint32_t t) {
  for (int i = 0; i < 8; ++i)
    if (OC_TYPES[i] == t) return i;
  return -1;
}

static uint32_t find_mem(const VkCtx* c, uint32_t type_bits,
                         VkMemoryPropertyFlags want) {
  for (uint32_t i = 0; i < c->memprops.memoryTypeCount; ++i)
    if ((type_bits & (1u << i)) &&
        (c->memprops.memoryTypes[i].propertyFlags & want) == want)
      return i;
  return UINT32_MAX;
}

/* ---------------- context ---------------- */

int vk_ctx_init(VkCtx* c, char* err, size_t errlen) {
  memset(c, 0, sizeof(*c));

  VkApplicationInfo app = {VK_STRUCTURE_TYPE_APPLICATION_INFO};
  app.pApplicationName = "oxidize-c";
  app.apiVersion = VK_API_VERSION_1_1;
  VkInstanceCreateInfo ici = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
  ici.pApplicationInfo = &app;
  VK_CHECK(vkCreateInstance(&ici, NULL, &c->instance));

  uint32_t ndev = 0;
  VK_CHECK(vkEnumeratePhysicalDevices(c->instance, &ndev, NULL));
  if (ndev == 0) {
    if (err && errlen) snprintf(err, errlen, "vulkan: no physical device");
    return -1;
  }
  VkPhysicalDevice* devs = calloc(ndev, sizeof(*devs));
  if (!devs) return -1;
  VK_CHECK(vkEnumeratePhysicalDevices(c->instance, &ndev, devs));
  /* Prefer a discrete GPU; fall back to whatever is first. */
  c->phys = devs[0];
  for (uint32_t i = 0; i < ndev; ++i) {
    VkPhysicalDeviceProperties p;
    vkGetPhysicalDeviceProperties(devs[i], &p);
    if (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
      c->phys = devs[i];
      break;
    }
  }
  free(devs);
  vkGetPhysicalDeviceMemoryProperties(c->phys, &c->memprops);

  /* pick a queue family with COMPUTE */
  uint32_t nqf = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(c->phys, &nqf, NULL);
  VkQueueFamilyProperties* qf = calloc(nqf, sizeof(*qf));
  if (!qf) return -1;
  vkGetPhysicalDeviceQueueFamilyProperties(c->phys, &nqf, qf);
  c->qfam = UINT32_MAX;
  for (uint32_t i = 0; i < nqf; ++i)
    if (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
      c->qfam = i;
      break;
    }
  free(qf);
  if (c->qfam == UINT32_MAX) {
    if (err && errlen) snprintf(err, errlen, "vulkan: no compute queue family");
    return -1;
  }

  float prio = 1.0f;
  VkDeviceQueueCreateInfo qci = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
  qci.queueFamilyIndex = c->qfam;
  qci.queueCount = 1;
  qci.pQueuePriorities = &prio;
  VkDeviceCreateInfo dci = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
  dci.queueCreateInfoCount = 1;
  dci.pQueueCreateInfos = &qci;
  VK_CHECK(vkCreateDevice(c->phys, &dci, NULL, &c->dev));
  vkGetDeviceQueue(c->dev, c->qfam, 0, &c->queue);

  VkCommandPoolCreateInfo pci = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  pci.queueFamilyIndex = c->qfam;
  VK_CHECK(vkCreateCommandPool(c->dev, &pci, NULL, &c->cmdpool));

  VkCommandBufferAllocateInfo cbi = {
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  cbi.commandPool = c->cmdpool;
  cbi.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cbi.commandBufferCount = 1;
  VK_CHECK(vkAllocateCommandBuffers(c->dev, &cbi, &c->xfer_cmd));
  VK_CHECK(vkAllocateCommandBuffers(c->dev, &cbi, &c->tok_cmd));

  /* Descriptor pool: generously sized, reset once per token. */
  VkDescriptorPoolSize ps = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 65536};
  VkDescriptorPoolCreateInfo dpi = {
      VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  dpi.maxSets = 16384;
  dpi.poolSizeCount = 1;
  dpi.pPoolSizes = &ps;
  VK_CHECK(vkCreateDescriptorPool(c->dev, &dpi, NULL, &c->descpool));

  VkFenceCreateInfo fci = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  VK_CHECK(vkCreateFence(c->dev, &fci, NULL, &c->fence));
  return 0;
}

void vk_ctx_free(VkCtx* c) {
  if (!c->dev) return;
  vkDeviceWaitIdle(c->dev);
  if (c->fence) vkDestroyFence(c->dev, c->fence, NULL);
  if (c->descpool) vkDestroyDescriptorPool(c->dev, c->descpool, NULL);
  if (c->cmdpool) vkDestroyCommandPool(c->dev, c->cmdpool, NULL);
  vkDestroyDevice(c->dev, NULL);
  if (c->instance) vkDestroyInstance(c->instance, NULL);
  memset(c, 0, sizeof(*c));
}

/* ---------------- buffers ---------------- */

static int make_buf(VkCtx* c, VkDeviceSize n, VkBufferUsageFlags usage,
                    VkMemoryPropertyFlags props, VkBuf* out) {
  char* err = NULL; size_t errlen = 0; /* VK_CHECK prints via return */
  memset(out, 0, sizeof(*out));
  if (n == 0) n = 4;
  VkBufferCreateInfo bci = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  bci.size = n;
  bci.usage = usage;
  bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  VK_CHECK(vkCreateBuffer(c->dev, &bci, NULL, &out->buf));
  VkMemoryRequirements mr;
  vkGetBufferMemoryRequirements(c->dev, out->buf, &mr);
  uint32_t mt = find_mem(c, mr.memoryTypeBits, props);
  if (mt == UINT32_MAX) {
    vkDestroyBuffer(c->dev, out->buf, NULL);
    out->buf = VK_NULL_HANDLE;
    return -1;
  }
  VkMemoryAllocateInfo mai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  mai.allocationSize = mr.size;
  mai.memoryTypeIndex = mt;
  VK_CHECK(vkAllocateMemory(c->dev, &mai, NULL, &out->mem));
  VK_CHECK(vkBindBufferMemory(c->dev, out->buf, out->mem, 0));
  out->size = n;
  return 0;
}

int vk_buf_device(VkCtx* c, VkDeviceSize n, VkBuf* out) {
  return make_buf(c, n,
                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                      VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                      VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, out);
}

void vk_buf_free(VkCtx* c, VkBuf* b) {
  if (b->buf) vkDestroyBuffer(c->dev, b->buf, NULL);
  if (b->mem) vkFreeMemory(c->dev, b->mem, NULL);
  memset(b, 0, sizeof(*b));
}

/* one-shot copy submit on the shared xfer command buffer */
static int submit_xfer(VkCtx* c, char* err, size_t errlen) {
  VK_CHECK(vkEndCommandBuffer(c->xfer_cmd));
  VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
  si.commandBufferCount = 1;
  si.pCommandBuffers = &c->xfer_cmd;
  vkResetFences(c->dev, 1, &c->fence);
  VK_CHECK(vkQueueSubmit(c->queue, 1, &si, c->fence));
  VK_CHECK(vkWaitForFences(c->dev, 1, &c->fence, VK_TRUE, UINT64_MAX));
  return 0;
}

int vk_upload(VkCtx* c, VkBuf* dst, const void* src, VkDeviceSize n) {
  char* err = NULL; size_t errlen = 0;
  VkBuf stg;
  if (make_buf(c, n, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                   VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
               &stg) != 0)
    return -1;
  void* map = NULL;
  if (vkMapMemory(c->dev, stg.mem, 0, n, 0, &map) != VK_SUCCESS) {
    vk_buf_free(c, &stg);
    return -1;
  }
  memcpy(map, src, n);
  vkUnmapMemory(c->dev, stg.mem);

  VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkResetCommandBuffer(c->xfer_cmd, 0);
  if (vkBeginCommandBuffer(c->xfer_cmd, &bi) != VK_SUCCESS) {
    vk_buf_free(c, &stg);
    return -1;
  }
  VkBufferCopy cp = {0, 0, n};
  vkCmdCopyBuffer(c->xfer_cmd, stg.buf, dst->buf, 1, &cp);
  int rc = submit_xfer(c, err, errlen);
  vk_buf_free(c, &stg);
  return rc;
}

int vk_zero(VkCtx* c, VkBuf* b, VkDeviceSize n) {
  char* err = NULL; size_t errlen = 0;
  VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkResetCommandBuffer(c->xfer_cmd, 0);
  if (vkBeginCommandBuffer(c->xfer_cmd, &bi) != VK_SUCCESS) return -1;
  vkCmdFillBuffer(c->xfer_cmd, b->buf, 0, n, 0);
  return submit_xfer(c, err, errlen);
}

int vk_download(VkCtx* c, const VkBuf* src, void* dst, VkDeviceSize n) {
  char* err = NULL; size_t errlen = 0;
  VkBuf stg;
  if (make_buf(c, n, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                   VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
               &stg) != 0)
    return -1;
  VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkResetCommandBuffer(c->xfer_cmd, 0);
  if (vkBeginCommandBuffer(c->xfer_cmd, &bi) != VK_SUCCESS) {
    vk_buf_free(c, &stg);
    return -1;
  }
  VkBufferCopy cp = {0, 0, n};
  vkCmdCopyBuffer(c->xfer_cmd, src->buf, stg.buf, 1, &cp);
  if (submit_xfer(c, err, errlen) != 0) {
    vk_buf_free(c, &stg);
    return -1;
  }
  void* map = NULL;
  if (vkMapMemory(c->dev, stg.mem, 0, n, 0, &map) != VK_SUCCESS) {
    vk_buf_free(c, &stg);
    return -1;
  }
  memcpy(dst, map, n);
  vkUnmapMemory(c->dev, stg.mem);
  vk_buf_free(c, &stg);
  return 0;
}

/* ---------------- pipelines ---------------- */

static int read_file(const char* path, uint32_t** out, size_t* nbytes) {
  FILE* f = fopen(path, "rb");
  if (!f) return -1;
  fseek(f, 0, SEEK_END);
  long sz = ftell(f);
  fseek(f, 0, SEEK_SET);
  if (sz <= 0 || (sz % 4) != 0) { fclose(f); return -1; }
  uint32_t* buf = malloc((size_t)sz);
  if (!buf) { fclose(f); return -1; }
  if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) {
    free(buf);
    fclose(f);
    return -1;
  }
  fclose(f);
  *out = buf;
  *nbytes = (size_t)sz;
  return 0;
}

/* Create a compute pipeline from `dir/name.spv`. nbufs storage bindings
 * (0..nbufs-1), a push-constant block of pushsize bytes, and an optional
 * int specialization constant at constant_id 0 (spec_used != 0). */
static int create_pipe(VkCtx* c, const char* dir, const char* name, int nbufs,
                       uint32_t pushsize, int spec_used, int32_t spec_val,
                       VkPipe* out, char* err, size_t errlen) {
  memset(out, 0, sizeof(*out));
  out->nbufs = nbufs;

  char path[1024];
  snprintf(path, sizeof(path), "%s/%s.spv", dir, name);
  uint32_t* code = NULL;
  size_t codebytes = 0;
  if (read_file(path, &code, &codebytes) != 0) {
    if (err && errlen)
      snprintf(err, errlen, "vulkan: cannot read SPIR-V %s", path);
    return -1;
  }
  VkShaderModuleCreateInfo smi = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
  smi.codeSize = codebytes;
  smi.pCode = code;
  VkShaderModule mod;
  VkResult mr = vkCreateShaderModule(c->dev, &smi, NULL, &mod);
  free(code);
  if (mr != VK_SUCCESS) {
    if (err && errlen)
      snprintf(err, errlen, "vulkan: shader module %s failed (%d)", name, (int)mr);
    return -1;
  }

  VkDescriptorSetLayoutBinding binds[8];
  for (int i = 0; i < nbufs; ++i) {
    binds[i] = (VkDescriptorSetLayoutBinding){0};
    binds[i].binding = (uint32_t)i;
    binds[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    binds[i].descriptorCount = 1;
    binds[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  }
  VkDescriptorSetLayoutCreateInfo dli = {
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  dli.bindingCount = (uint32_t)nbufs;
  dli.pBindings = binds;
  VK_CHECK(vkCreateDescriptorSetLayout(c->dev, &dli, NULL, &out->dsl));

  VkPushConstantRange pcr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, pushsize};
  VkPipelineLayoutCreateInfo pli = {
      VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  pli.setLayoutCount = 1;
  pli.pSetLayouts = &out->dsl;
  pli.pushConstantRangeCount = pushsize ? 1 : 0;
  pli.pPushConstantRanges = pushsize ? &pcr : NULL;
  VK_CHECK(vkCreatePipelineLayout(c->dev, &pli, NULL, &out->layout));

  VkSpecializationMapEntry sme = {0, 0, sizeof(int32_t)};
  VkSpecializationInfo spec = {1, &sme, sizeof(int32_t), &spec_val};

  VkComputePipelineCreateInfo cpi = {
      VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  cpi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  cpi.stage.module = mod;
  cpi.stage.pName = "main";
  cpi.stage.pSpecializationInfo = spec_used ? &spec : NULL;
  cpi.layout = out->layout;
  VkResult pr =
      vkCreateComputePipelines(c->dev, VK_NULL_HANDLE, 1, &cpi, NULL, &out->pipe);
  vkDestroyShaderModule(c->dev, mod, NULL);
  if (pr != VK_SUCCESS) {
    if (err && errlen)
      snprintf(err, errlen, "vulkan: pipeline %s failed (%d)", name, (int)pr);
    return -1;
  }
  return 0;
}

static void free_pipe(VkCtx* c, VkPipe* p) {
  if (p->pipe) vkDestroyPipeline(c->dev, p->pipe, NULL);
  if (p->layout) vkDestroyPipelineLayout(c->dev, p->layout, NULL);
  if (p->dsl) vkDestroyDescriptorSetLayout(c->dev, p->dsl, NULL);
  memset(p, 0, sizeof(*p));
}

int vk_kernels_init(VkCtx* c, VkKernels* k, const char* dir, char* err,
                    size_t errlen) {
  memset(k, 0, sizeof(*k));
#define P(field, name, nb, push, spec_used, spec_val)                      \
  if (create_pipe(c, dir, name, nb, push, spec_used, spec_val, &k->field, \
                  err, errlen) != 0)                                       \
    goto fail

  /* push-constant block sizes must match the .comp declarations. */
  for (int i = 0; i < 8; ++i) {
    /* matvec: {int rows; int cols; uint rowbytes;} = 12 bytes; spec = OC type */
    if (create_pipe(c, dir, "matvec", 3, 12, 1, (int32_t)OC_TYPES[i],
                    &k->matvec[i], err, errlen) != 0)
      goto fail;
    /* embed: {int n; uint row_off; float scale;} = 12 bytes */
    if (create_pipe(c, dir, "embed", 2, 12, 1, (int32_t)OC_TYPES[i],
                    &k->embed[i], err, errlen) != 0)
      goto fail;
  }
  P(rmsnorm, "rmsnorm", 3, 12, 0, 0);   /* {int per; float eps; int has_w;} */
  P(rope_neox, "rope", 2, 20, 1, 0);    /* {int hd;int pos;int rl;float th;int hf;} */
  P(rope_normal, "rope", 2, 20, 1, 1);
  P(kv_store, "kv_store", 4, 12, 0, 0); /* {int k_len; int v_len; uint slot;} */
  P(attn, "attn", 5, 28, 0, 0);         /* 6 int + 1 float */
  P(add, "add", 2, 4, 0, 0);            /* {int n;} */
  P(silu_mul, "silu_mul", 2, 4, 0, 0);
  P(geglu, "geglu", 2, 4, 0, 0);
  P(resid_out, "resid_out", 3, 8, 0, 0); /* {int n; float s;} */
  P(softcap, "softcap", 1, 8, 0, 0);     /* {int n; float c;} */
  P(argmax, "argmax", 2, 4, 0, 0);       /* {int n;} -> int out[0] */
#undef P
  return 0;
fail:
  vk_kernels_free(c, k);
  return -1;
}

void vk_kernels_free(VkCtx* c, VkKernels* k) {
  for (int i = 0; i < 8; ++i) {
    free_pipe(c, &k->matvec[i]);
    free_pipe(c, &k->embed[i]);
  }
  free_pipe(c, &k->rmsnorm);
  free_pipe(c, &k->rope_neox);
  free_pipe(c, &k->rope_normal);
  free_pipe(c, &k->kv_store);
  free_pipe(c, &k->attn);
  free_pipe(c, &k->add);
  free_pipe(c, &k->silu_mul);
  free_pipe(c, &k->geglu);
  free_pipe(c, &k->resid_out);
  free_pipe(c, &k->softcap);
  free_pipe(c, &k->argmax);
}

/* ---------------- recording ---------------- */

int vk_rec_begin(VkCtx* c, VkRec* r) {
  r->ctx = c;
  r->cmd = c->tok_cmd;
  if (!r->cmd) return -1;
  vkResetCommandBuffer(r->cmd, 0);
  VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  if (vkBeginCommandBuffer(r->cmd, &bi) != VK_SUCCESS) return -1;
  return 0;
}

void vk_dispatch(VkRec* r, const VkPipe* p, const VkBuf* const* bufs,
                 const void* push, uint32_t pushsize, uint32_t gx, uint32_t gy,
                 uint32_t gz) {
  VkCtx* c = r->ctx;

  VkDescriptorSetAllocateInfo ai = {
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  ai.descriptorPool = c->descpool;
  ai.descriptorSetCount = 1;
  ai.pSetLayouts = &p->dsl;
  VkDescriptorSet set;
  if (vkAllocateDescriptorSets(c->dev, &ai, &set) != VK_SUCCESS) return;

  VkDescriptorBufferInfo bi[8];
  VkWriteDescriptorSet w[8];
  for (int i = 0; i < p->nbufs; ++i) {
    bi[i] = (VkDescriptorBufferInfo){bufs[i]->buf, 0, VK_WHOLE_SIZE};
    w[i] = (VkWriteDescriptorSet){VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w[i].dstSet = set;
    w[i].dstBinding = (uint32_t)i;
    w[i].descriptorCount = 1;
    w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    w[i].pBufferInfo = &bi[i];
  }
  vkUpdateDescriptorSets(c->dev, (uint32_t)p->nbufs, w, 0, NULL);

  vkCmdBindPipeline(r->cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p->pipe);
  vkCmdBindDescriptorSets(r->cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p->layout, 0,
                          1, &set, 0, NULL);
  if (pushsize)
    vkCmdPushConstants(r->cmd, p->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       pushsize, push);
  vkCmdDispatch(r->cmd, gx, gy, gz);

  /* coarse global barrier: this dispatch's writes visible to the next. */
  VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
  vkCmdPipelineBarrier(r->cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL,
                       0, NULL);
}

void vk_rec_copy(VkRec* r, const VkBuf* dst, const VkBuf* src, VkDeviceSize n) {
  VkBufferCopy cp = {0, 0, n};
  vkCmdCopyBuffer(r->cmd, src->buf, dst->buf, 1, &cp);
  VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
  vkCmdPipelineBarrier(r->cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL,
                       0, NULL);
}

int vk_rec_submit_wait(VkRec* r) {
  VkCtx* c = r->ctx;
  if (vkEndCommandBuffer(r->cmd) != VK_SUCCESS) return -1;
  VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
  si.commandBufferCount = 1;
  si.pCommandBuffers = &r->cmd;
  vkResetFences(c->dev, 1, &c->fence);
  if (vkQueueSubmit(c->queue, 1, &si, c->fence) != VK_SUCCESS) return -1;
  if (vkWaitForFences(c->dev, 1, &c->fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS)
    return -1;
  vkResetDescriptorPool(c->dev, c->descpool, 0);
  return 0;
}
