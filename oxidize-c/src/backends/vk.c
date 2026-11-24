/*
 * vk.c — Vulkan compute runtime; see vk.h.
 */
#define _POSIX_C_SOURCE 200809L
#include "oxidize/vk.h"

#ifdef OC_VULKAN

#include "oxidize/log.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

#define N_STAMPS 4096

struct OcVk {
    VkInstance       inst;
    VkPhysicalDevice pd;
    VkDevice         dev;
    VkQueue          queue;
    uint32_t         qf;
    VkCommandPool    pool;
    VkCommandBuffer  cb;
    VkFence          fence;
    VkQueryPool      qp;
    VkPhysicalDeviceMemoryProperties mp;
    int              mt[3];
    bool             recording;
    VkPipeline       bound;
    uint64_t         stamps[N_STAMPS];
    uint32_t         n_stamped;
    OcVkInfo         info;
};

struct OcVkPipe {
    VkShaderModule   mod;
    VkPipelineLayout layout;
    VkPipeline       pipe;
    uint32_t         push;
};

typedef struct {
    VkBuffer       buf;
    VkDeviceMemory mem;
} BufImpl;

static OcError vkerr(VkResult r, const char *what)
{
    if (r == VK_SUCCESS) return OC_OK;
    oc_log(OC_LOG_ERROR, "vulkan: %s failed (%d)", what, (int)r);
    return r == VK_ERROR_OUT_OF_DEVICE_MEMORY || r == VK_ERROR_OUT_OF_HOST_MEMORY ? OC_ERR_OOM : OC_ERR_BACKEND;
}

static int find_type(const VkPhysicalDeviceMemoryProperties *mp, VkMemoryPropertyFlags want,
                     VkMemoryPropertyFlags avoid)
{
    for (uint32_t i = 0; i < mp->memoryTypeCount; i++) {
        VkMemoryPropertyFlags f = mp->memoryTypes[i].propertyFlags;
        if ((f & want) == want && !(f & avoid)) return (int)i;
    }
    return -1;
}

OcError oc_vk_init(OcVk **out)
{
    *out = NULL;
    OcVk *vk = calloc(1, sizeof *vk);
    if (!vk) return OC_ERR_OOM;
    OcError e;

    const VkApplicationInfo app = { .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .pApplicationName = "oxidize-c",
                                    .apiVersion = VK_API_VERSION_1_3 };
    const VkInstanceCreateInfo ici = { .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &app };
    if ((e = vkerr(vkCreateInstance(&ici, NULL, &vk->inst), "vkCreateInstance")) != OC_OK) goto fail;

    uint32_t n_pd = 0;
    vkEnumeratePhysicalDevices(vk->inst, &n_pd, NULL);
    VkPhysicalDevice pds[16];
    if (n_pd > 16) n_pd = 16;
    vkEnumeratePhysicalDevices(vk->inst, &n_pd, pds);
    const char *want = getenv("OC_VK_DEVICE");
    int pick = -1;
    for (uint32_t i = 0; i < n_pd; i++) {
        VkPhysicalDeviceProperties pr;
        vkGetPhysicalDeviceProperties(pds[i], &pr);
        if (want ? (uint32_t)atoi(want) == i : pr.deviceType != VK_PHYSICAL_DEVICE_TYPE_CPU) { pick = (int)i; break; }
    }
    if (pick < 0) { oc_log(OC_LOG_ERROR, "vulkan: no GPU found"); e = OC_ERR_BACKEND; goto fail; }
    vk->pd = pds[pick];

    VkPhysicalDeviceSubgroupProperties sgp = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES };
    VkPhysicalDeviceMaintenance3Properties m3 = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_3_PROPERTIES,
                                                  .pNext = &sgp };
    VkPhysicalDeviceProperties2 pr2 = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &m3 };
    vkGetPhysicalDeviceProperties2(vk->pd, &pr2);
    if (pr2.properties.apiVersion < VK_API_VERSION_1_3) {
        oc_log(OC_LOG_ERROR, "vulkan: %s lacks Vulkan 1.3", pr2.properties.deviceName);
        e = OC_ERR_BACKEND;
        goto fail;
    }
    snprintf(vk->info.name, sizeof vk->info.name, "%s", pr2.properties.deviceName);
    vk->info.subgroup_size = sgp.subgroupSize;
    vk->info.max_alloc = m3.maxMemoryAllocationSize;
    vk->info.integrated = pr2.properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU;
    vk->info.timestamp_ns = pr2.properties.limits.timestampComputeAndGraphics ? pr2.properties.limits.timestampPeriod : 0.0;

    VkPhysicalDeviceVulkan13Features h13 = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES };
    VkPhysicalDeviceVulkan12Features h12 = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, .pNext = &h13 };
    VkPhysicalDeviceVulkan11Features h11 = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES, .pNext = &h12 };
    VkPhysicalDeviceFeatures2 hf = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &h11 };
    vkGetPhysicalDeviceFeatures2(vk->pd, &hf);
    if (!h12.bufferDeviceAddress || !h12.shaderInt8 || !h12.storageBuffer8BitAccess || !h11.storageBuffer16BitAccess ||
        !h13.shaderIntegerDotProduct || !hf.features.shaderInt64 || !h12.shaderFloat16) {
        oc_log(OC_LOG_ERROR, "vulkan: %s lacks a required feature (BDA, int8, int64, fp16, integer dot)", vk->info.name);
        e = OC_ERR_BACKEND;
        goto fail;
    }

    uint32_t n_qf = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(vk->pd, &n_qf, NULL);
    VkQueueFamilyProperties qfs[16];
    if (n_qf > 16) n_qf = 16;
    vkGetPhysicalDeviceQueueFamilyProperties(vk->pd, &n_qf, qfs);
    /* a compute-only family when there is one: it stays off the graphics
     * ring the desktop draws on, and long jobs there do not stall it */
    vk->qf = UINT32_MAX;
    for (uint32_t i = 0; i < n_qf; i++)
        if ((qfs[i].queueFlags & VK_QUEUE_COMPUTE_BIT) && !(qfs[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) { vk->qf = i; break; }
    for (uint32_t i = 0; i < n_qf && vk->qf == UINT32_MAX; i++)
        if (qfs[i].queueFlags & VK_QUEUE_COMPUTE_BIT) vk->qf = i;
    if (vk->qf == UINT32_MAX) { e = OC_ERR_BACKEND; goto fail; }

    VkPhysicalDeviceVulkan13Features f13 = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
                                             .shaderIntegerDotProduct = VK_TRUE,
                                             .subgroupSizeControl = h13.subgroupSizeControl,
                                             .computeFullSubgroups = h13.computeFullSubgroups };
    VkPhysicalDeviceVulkan12Features f12 = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, .pNext = &f13,
                                             .bufferDeviceAddress = VK_TRUE, .shaderInt8 = VK_TRUE,
                                             .storageBuffer8BitAccess = VK_TRUE, .shaderFloat16 = VK_TRUE };
    VkPhysicalDeviceVulkan11Features f11 = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES, .pNext = &f12,
                                             .storageBuffer16BitAccess = VK_TRUE };
    VkPhysicalDeviceFeatures2 ff = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &f11,
                                     .features = { .shaderInt64 = VK_TRUE } };
    const float prio = 1.0f;
    const VkDeviceQueueCreateInfo qci = { .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueFamilyIndex = vk->qf,
                                          .queueCount = 1, .pQueuePriorities = &prio };
    const VkDeviceCreateInfo dci = { .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .pNext = &ff,
                                     .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci };
    if ((e = vkerr(vkCreateDevice(vk->pd, &dci, NULL, &vk->dev), "vkCreateDevice")) != OC_OK) goto fail;
    vkGetDeviceQueue(vk->dev, vk->qf, 0, &vk->queue);

    vkGetPhysicalDeviceMemoryProperties(vk->pd, &vk->mp);
    const VkMemoryPropertyFlags HV = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    vk->mt[OC_VK_MEM_DEVICE] = find_type(&vk->mp, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | HV, 0);
    if (vk->mt[OC_VK_MEM_DEVICE] < 0) vk->mt[OC_VK_MEM_DEVICE] = find_type(&vk->mp, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0);
    vk->mt[OC_VK_MEM_HOST] = find_type(&vk->mp, HV | VK_MEMORY_PROPERTY_HOST_CACHED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (vk->mt[OC_VK_MEM_HOST] < 0) vk->mt[OC_VK_MEM_HOST] = find_type(&vk->mp, HV | VK_MEMORY_PROPERTY_HOST_CACHED_BIT, 0);
    if (vk->mt[OC_VK_MEM_HOST] < 0) vk->mt[OC_VK_MEM_HOST] = find_type(&vk->mp, HV, 0);
    vk->mt[OC_VK_MEM_STREAM] = find_type(&vk->mp, HV, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
    if (vk->mt[OC_VK_MEM_STREAM] < 0) vk->mt[OC_VK_MEM_STREAM] = vk->mt[OC_VK_MEM_HOST];
    if (vk->mt[OC_VK_MEM_DEVICE] < 0 || vk->mt[OC_VK_MEM_HOST] < 0) { e = OC_ERR_BACKEND; goto fail; }
    vk->info.heap_device = vk->mp.memoryHeaps[vk->mp.memoryTypes[vk->mt[OC_VK_MEM_DEVICE]].heapIndex].size;
    vk->info.heap_host = vk->mp.memoryHeaps[vk->mp.memoryTypes[vk->mt[OC_VK_MEM_HOST]].heapIndex].size;

    const VkCommandPoolCreateInfo pci = { .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                                          .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
                                          .queueFamilyIndex = vk->qf };
    if ((e = vkerr(vkCreateCommandPool(vk->dev, &pci, NULL, &vk->pool), "vkCreateCommandPool")) != OC_OK) goto fail;
    const VkCommandBufferAllocateInfo cai = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                                              .commandPool = vk->pool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                                              .commandBufferCount = 1 };
    if ((e = vkerr(vkAllocateCommandBuffers(vk->dev, &cai, &vk->cb), "vkAllocateCommandBuffers")) != OC_OK) goto fail;
    const VkFenceCreateInfo fci = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    if ((e = vkerr(vkCreateFence(vk->dev, &fci, NULL, &vk->fence), "vkCreateFence")) != OC_OK) goto fail;
    if (vk->info.timestamp_ns > 0.0) {
        const VkQueryPoolCreateInfo qpi = { .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
                                            .queryType = VK_QUERY_TYPE_TIMESTAMP, .queryCount = N_STAMPS };
        if (vkCreateQueryPool(vk->dev, &qpi, NULL, &vk->qp) != VK_SUCCESS) vk->info.timestamp_ns = 0.0;
    }
    *out = vk;
    return OC_OK;

fail:
    oc_vk_free(vk);
    return e;
}

void oc_vk_free(OcVk *vk)
{
    if (!vk) return;
    if (vk->dev) {
        vkDeviceWaitIdle(vk->dev);
        if (vk->qp) vkDestroyQueryPool(vk->dev, vk->qp, NULL);
        if (vk->fence) vkDestroyFence(vk->dev, vk->fence, NULL);
        if (vk->pool) vkDestroyCommandPool(vk->dev, vk->pool, NULL);
        vkDestroyDevice(vk->dev, NULL);
    }
    if (vk->inst) vkDestroyInstance(vk->inst, NULL);
    free(vk);
}

const OcVkInfo *oc_vk_info(const OcVk *vk) { return &vk->info; }

OcError oc_vk_buf_new(OcVk *vk, size_t size, OcVkMemKind kind, OcVkBuf *out)
{
    memset(out, 0, sizeof *out);
    BufImpl *bi = calloc(1, sizeof *bi);
    if (!bi) return OC_ERR_OOM;
    const VkBufferCreateInfo bci = { .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = size ? size : 4,
                                     .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                                              VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                              VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
                                     .sharingMode = VK_SHARING_MODE_EXCLUSIVE };
    OcError e = vkerr(vkCreateBuffer(vk->dev, &bci, NULL, &bi->buf), "vkCreateBuffer");
    if (e != OC_OK) { free(bi); return e; }
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(vk->dev, bi->buf, &req);
    int mt = vk->mt[kind];
    if (!(req.memoryTypeBits & (1u << mt))) {
        const VkMemoryPropertyFlags f = vk->mp.memoryTypes[mt].propertyFlags;
        mt = -1;
        for (uint32_t i = 0; i < vk->mp.memoryTypeCount; i++)
            if ((req.memoryTypeBits & (1u << i)) && (vk->mp.memoryTypes[i].propertyFlags & f) == f) { mt = (int)i; break; }
    }
    if (mt < 0) { vkDestroyBuffer(vk->dev, bi->buf, NULL); free(bi); return OC_ERR_BACKEND; }
    const VkMemoryAllocateFlagsInfo fl = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO,
                                           .flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT };
    const VkMemoryAllocateInfo mai = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .pNext = &fl,
                                       .allocationSize = req.size, .memoryTypeIndex = (uint32_t)mt };
    e = vkerr(vkAllocateMemory(vk->dev, &mai, NULL, &bi->mem), "vkAllocateMemory");
    if (e != OC_OK) { vkDestroyBuffer(vk->dev, bi->buf, NULL); free(bi); return e; }
    vkBindBufferMemory(vk->dev, bi->buf, bi->mem, 0);
    if (vk->mp.memoryTypes[mt].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
        if (vkMapMemory(vk->dev, bi->mem, 0, VK_WHOLE_SIZE, 0, &out->map) != VK_SUCCESS) out->map = NULL;
    }
    const VkBufferDeviceAddressInfo ai = { .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, .buffer = bi->buf };
    out->addr = vkGetBufferDeviceAddress(vk->dev, &ai);
    out->size = size;
    out->impl = bi;
    return OC_OK;
}

void oc_vk_buf_free(OcVk *vk, OcVkBuf *b)
{
    if (!b || !b->impl) return;
    BufImpl *bi = b->impl;
    if (b->map) vkUnmapMemory(vk->dev, bi->mem);
    vkDestroyBuffer(vk->dev, bi->buf, NULL);
    vkFreeMemory(vk->dev, bi->mem, NULL);
    free(bi);
    memset(b, 0, sizeof *b);
}

OcError oc_vk_pipe_new(OcVk *vk, const uint32_t *spv, size_t spv_bytes, uint32_t push_bytes,
                       const uint32_t *spec, uint32_t n_spec, OcVkPipe **out)
{
    *out = NULL;
    if (n_spec > 8) return OC_ERR_INVALID_ARG;
    VkSpecializationMapEntry ent[8];
    for (uint32_t i = 0; i < n_spec; i++) ent[i] = (VkSpecializationMapEntry){ i, i * 4u, 4 };
    const VkSpecializationInfo si = { n_spec, ent, n_spec * 4u, spec };
    OcVkPipe *p = calloc(1, sizeof *p);
    if (!p) return OC_ERR_OOM;
    p->push = push_bytes;
    const VkShaderModuleCreateInfo smi = { .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = spv_bytes,
                                           .pCode = spv };
    OcError e = vkerr(vkCreateShaderModule(vk->dev, &smi, NULL, &p->mod), "vkCreateShaderModule");
    if (e != OC_OK) goto fail;
    const VkPushConstantRange pr = { .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .size = (push_bytes + 3u) & ~3u };
    const VkPipelineLayoutCreateInfo lci = { .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                                             .pushConstantRangeCount = push_bytes ? 1u : 0u, .pPushConstantRanges = &pr };
    if ((e = vkerr(vkCreatePipelineLayout(vk->dev, &lci, NULL, &p->layout), "vkCreatePipelineLayout")) != OC_OK) goto fail;
    const VkComputePipelineCreateInfo cpi = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                   .module = p->mod, .pName = "main", .pSpecializationInfo = n_spec ? &si : NULL },
        .layout = p->layout,
    };
    if ((e = vkerr(vkCreateComputePipelines(vk->dev, VK_NULL_HANDLE, 1, &cpi, NULL, &p->pipe), "vkCreateComputePipelines")) != OC_OK)
        goto fail;
    *out = p;
    return OC_OK;
fail:
    oc_vk_pipe_free(vk, p);
    return e;
}

void oc_vk_pipe_free(OcVk *vk, OcVkPipe *p)
{
    if (!p) return;
    if (p->pipe) vkDestroyPipeline(vk->dev, p->pipe, NULL);
    if (p->layout) vkDestroyPipelineLayout(vk->dev, p->layout, NULL);
    if (p->mod) vkDestroyShaderModule(vk->dev, p->mod, NULL);
    free(p);
}

OcError oc_vk_begin(OcVk *vk)
{
    vkResetCommandBuffer(vk->cb, 0);
    const VkCommandBufferBeginInfo bi = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                          .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
    OcError e = vkerr(vkBeginCommandBuffer(vk->cb, &bi), "vkBeginCommandBuffer");
    if (e != OC_OK) return e;
    if (vk->qp) vkCmdResetQueryPool(vk->cb, vk->qp, 0, N_STAMPS);
    vk->n_stamped = 0;
    vk->recording = true;
    vk->bound = VK_NULL_HANDLE;
    return OC_OK;
}

static void bind(OcVk *vk, const OcVkPipe *p, const void *push)
{
    if (vk->bound != p->pipe) {
        vkCmdBindPipeline(vk->cb, VK_PIPELINE_BIND_POINT_COMPUTE, p->pipe);
        vk->bound = p->pipe;
    }
    if (p->push) vkCmdPushConstants(vk->cb, p->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, (p->push + 3u) & ~3u, push);
}

void oc_vk_dispatch(OcVk *vk, const OcVkPipe *p, const void *push, uint32_t gx, uint32_t gy, uint32_t gz)
{
    if (!gx || !gy || !gz) return;
    bind(vk, p, push);
    vkCmdDispatch(vk->cb, gx, gy, gz);
}

void oc_vk_dispatch_indirect(OcVk *vk, const OcVkPipe *p, const void *push, const OcVkBuf *buf, size_t offset)
{
    bind(vk, p, push);
    vkCmdDispatchIndirect(vk->cb, ((const BufImpl *)buf->impl)->buf, offset);
}

void oc_vk_barrier(OcVk *vk)
{
    const VkMemoryBarrier mb = { .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                                 .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
                                 .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
                                                  VK_ACCESS_INDIRECT_COMMAND_READ_BIT };
    vkCmdPipelineBarrier(vk->cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT |
                         VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
}

void oc_vk_fill(OcVk *vk, const OcVkBuf *b, size_t off, size_t bytes, uint32_t value)
{
    vkCmdFillBuffer(vk->cb, ((const BufImpl *)b->impl)->buf, off, bytes, value);
}

void oc_vk_stamp(OcVk *vk, uint32_t slot)
{
    if (!vk->qp || slot >= N_STAMPS) return;
    vkCmdWriteTimestamp(vk->cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, vk->qp, slot);
    if (slot + 1 > vk->n_stamped) vk->n_stamped = slot + 1;
}

OcError oc_vk_submit(OcVk *vk)
{
    if (!vk->recording) return OC_ERR_INVALID_ARG;
    const VkMemoryBarrier mb = { .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                                 .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
                                 .dstAccessMask = VK_ACCESS_HOST_READ_BIT };
    vkCmdPipelineBarrier(vk->cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
    vk->recording = false;
    OcError e = vkerr(vkEndCommandBuffer(vk->cb), "vkEndCommandBuffer");
    if (e != OC_OK) return e;
    const VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &vk->cb };
    if ((e = vkerr(vkQueueSubmit(vk->queue, 1, &si, vk->fence), "vkQueueSubmit")) != OC_OK) return e;
    e = vkerr(vkWaitForFences(vk->dev, 1, &vk->fence, VK_TRUE, UINT64_MAX), "vkWaitForFences");
    vkResetFences(vk->dev, 1, &vk->fence);
    if (e == OC_OK && vk->qp && vk->n_stamped)
        vkGetQueryPoolResults(vk->dev, vk->qp, 0, vk->n_stamped, vk->n_stamped * sizeof(uint64_t), vk->stamps,
                              sizeof(uint64_t), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
    return e;
}

double oc_vk_stamp_ns(OcVk *vk, uint32_t slot)
{
    return slot < N_STAMPS ? (double)vk->stamps[slot] * vk->info.timestamp_ns : 0.0;
}

#else /* !OC_VULKAN */

OcError oc_vk_init(OcVk **out) { *out = NULL; return OC_ERR_BACKEND; }
void oc_vk_free(OcVk *vk) { (void)vk; }
const OcVkInfo *oc_vk_info(const OcVk *vk) { (void)vk; return NULL; }
OcError oc_vk_buf_new(OcVk *vk, size_t size, OcVkMemKind kind, OcVkBuf *out)
{
    (void)vk; (void)size; (void)kind;
    *out = (OcVkBuf){ 0 };
    return OC_ERR_BACKEND;
}
void oc_vk_buf_free(OcVk *vk, OcVkBuf *b) { (void)vk; (void)b; }
OcError oc_vk_pipe_new(OcVk *vk, const uint32_t *spv, size_t spv_bytes, uint32_t push_bytes,
                       const uint32_t *spec, uint32_t n_spec, OcVkPipe **out)
{
    (void)vk; (void)spv; (void)spv_bytes; (void)push_bytes; (void)spec; (void)n_spec;
    *out = NULL;
    return OC_ERR_BACKEND;
}
void oc_vk_pipe_free(OcVk *vk, OcVkPipe *p) { (void)vk; (void)p; }
OcError oc_vk_begin(OcVk *vk) { (void)vk; return OC_ERR_BACKEND; }
void oc_vk_dispatch(OcVk *vk, const OcVkPipe *p, const void *push, uint32_t gx, uint32_t gy, uint32_t gz)
{
    (void)vk; (void)p; (void)push; (void)gx; (void)gy; (void)gz;
}
void oc_vk_dispatch_indirect(OcVk *vk, const OcVkPipe *p, const void *push, const OcVkBuf *buf, size_t offset)
{
    (void)vk; (void)p; (void)push; (void)buf; (void)offset;
}
void oc_vk_barrier(OcVk *vk) { (void)vk; }
void oc_vk_fill(OcVk *vk, const OcVkBuf *b, size_t off, size_t bytes, uint32_t value)
{
    (void)vk; (void)b; (void)off; (void)bytes; (void)value;
}
void oc_vk_stamp(OcVk *vk, uint32_t slot) { (void)vk; (void)slot; }
OcError oc_vk_submit(OcVk *vk) { (void)vk; return OC_ERR_BACKEND; }
double oc_vk_stamp_ns(OcVk *vk, uint32_t slot) { (void)vk; (void)slot; return 0.0; }

#endif
