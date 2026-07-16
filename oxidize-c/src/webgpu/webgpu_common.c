/* ======================================================================
 * UNVERIFIED — this file has NEVER been compiled or run.
 * Written BLIND against src/cuda/ + src/vulkan/vk_common.c + Dawn/emdawn
 * <webgpu/webgpu.h>. Requires a WebGPU device (browser via emdawnwebgpu, or
 * native Dawn). IT MAY NOT COMPILE and MAY BE WRONG. Known-blind risks:
 * adapter request callbacks, buffer map async, bind-group layout mismatch
 * vs .wgsl, and uniform alignment are all plausible-but-untested.
 * ====================================================================== */
#include "webgpu_common.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../quant.h"

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#endif

#define ARGMAX_BLOCKS 256

/* ---- tiny helpers ---- */

int wgpu_check_type(uint32_t t, const char* what, char* err, size_t errlen) {
  switch (t) {
    case OC_F32: case OC_F16: case OC_Q4_0: case OC_Q8_0:
    case OC_Q4_K: case OC_Q5_K: case OC_Q6_K: case OC_AL5_XS:
      return 0;
    default: break;
  }
  if (err && errlen)
    snprintf(err, errlen,
             "webgpu: %s has quant type %u; kernels exist for "
             "F32/F16/Q4_0/Q8_0/Q4_K/Q5_K/Q6_K/AL5_XS only",
             what, t);
  return -1;
}

static uint64_t round4(uint64_t n) { return n ? ((n + 3u) & ~3ull) : 4ull; }
static uint64_t round16(uint64_t n) { return n ? ((n + 15u) & ~15ull) : 16ull; }

/* Busy-wait pump for Dawn/emdawn async. UNVERIFIED shape. */
static void wgpu_pump(WgpuCtx* c) {
  (void)c;
#ifdef __EMSCRIPTEN__
  emscripten_sleep(0);
#else
  /* Native Dawn: wgpuInstanceProcessEvents if available (Dawn extension).
   * ASSUMED symbol; may need wgpuDeviceTick / poll. */
#  if defined(WGPU_WIRE_CLIENT) || defined(WGPU_DAWN)
  /* no-op fallback */
#  endif
#endif
}

/* ---- async request state ---- */

typedef struct {
  WGPUAdapter adapter;
  int done;
  WGPURequestAdapterStatus status;
} AdapterReq;

typedef struct {
  WGPUDevice device;
  int done;
  WGPURequestDeviceStatus status;
  char msg[256];
} DeviceReq;

static void on_adapter(WGPURequestAdapterStatus status, WGPUAdapter adapter,
                       WGPUStringView message, void* userdata1, void* userdata2) {
  (void)message; (void)userdata2;
  AdapterReq* r = (AdapterReq*)userdata1;
  r->status = status;
  r->adapter = adapter;
  r->done = 1;
}

static void on_device(WGPURequestDeviceStatus status, WGPUDevice device,
                      WGPUStringView message, void* userdata1, void* userdata2) {
  (void)userdata2;
  DeviceReq* r = (DeviceReq*)userdata1;
  r->status = status;
  r->device = device;
  r->done = 1;
  if (message.data && message.length)
    snprintf(r->msg, sizeof(r->msg), "%.*s", (int)message.length, message.data);
  else
    r->msg[0] = 0;
}

static void on_device_lost(WGPUDevice const* device, WGPUDeviceLostReason reason,
                           WGPUStringView message, void* userdata1,
                           void* userdata2) {
  (void)device; (void)reason; (void)userdata1; (void)userdata2;
  fprintf(stderr, "webgpu: device lost: %.*s\n",
          (int)(message.length ? message.length : 0),
          message.data ? message.data : "");
}

static void on_uncaptured(WGPUDevice const* device, WGPUErrorType type,
                          WGPUStringView message, void* userdata1,
                          void* userdata2) {
  (void)device; (void)userdata1; (void)userdata2;
  fprintf(stderr, "webgpu: uncaptured error (%d): %.*s\n", (int)type,
          (int)(message.length ? message.length : 0),
          message.data ? message.data : "");
}

/* ---- context ---- */

int wgpu_ctx_init(WgpuCtx* c, char* err, size_t errlen) {
  memset(c, 0, sizeof(*c));

  WGPUInstanceDescriptor idesc = {0};
  c->instance = wgpuCreateInstance(&idesc);
  if (!c->instance) {
    if (err && errlen) snprintf(err, errlen, "webgpu: wgpuCreateInstance failed");
    return -1;
  }

  AdapterReq areq = {0};
  WGPURequestAdapterOptions opts = {0};
  opts.powerPreference = WGPUPowerPreference_HighPerformance;

  /* Dawn 2024+ uses callback-info structs. ASSUMED API shape — may need
   * adjustment for the installed webgpu.h generation. */
  WGPURequestAdapterCallbackInfo acbi = {0};
  acbi.mode = WGPUCallbackMode_AllowSpontaneous;
  acbi.callback = on_adapter;
  acbi.userdata1 = &areq;
  wgpuInstanceRequestAdapter(c->instance, &opts, acbi);

  int spins = 0;
  while (!areq.done && spins++ < 100000) wgpu_pump(c);
  if (!areq.done || areq.status != WGPURequestAdapterStatus_Success ||
      !areq.adapter) {
    if (err && errlen)
      snprintf(err, errlen, "webgpu: no adapter (status %d)", (int)areq.status);
    wgpu_ctx_free(c);
    return -1;
  }
  c->adapter = areq.adapter;

  DeviceReq dreq = {0};
  WGPUDeviceDescriptor ddesc = {0};
  ddesc.label = (WGPUStringView){"oxidize-c-webgpu", 16};
  WGPUDeviceLostCallbackInfo lost = {0};
  lost.mode = WGPUCallbackMode_AllowSpontaneous;
  lost.callback = on_device_lost;
  ddesc.deviceLostCallbackInfo = lost;
  WGPUUncapturedErrorCallbackInfo unc = {0};
  unc.callback = on_uncaptured;
  ddesc.uncapturedErrorCallbackInfo = unc;

  /* Request shader-f16 if the header exposes the feature enum — optional. */
  WGPURequestDeviceCallbackInfo dcbi = {0};
  dcbi.mode = WGPUCallbackMode_AllowSpontaneous;
  dcbi.callback = on_device;
  dcbi.userdata1 = &dreq;
  wgpuAdapterRequestDevice(c->adapter, &ddesc, dcbi);

  spins = 0;
  while (!dreq.done && spins++ < 100000) wgpu_pump(c);
  if (!dreq.done || dreq.status != WGPURequestDeviceStatus_Success ||
      !dreq.device) {
    if (err && errlen)
      snprintf(err, errlen, "webgpu: requestDevice failed: %s",
               dreq.msg[0] ? dreq.msg : "(no message)");
    wgpu_ctx_free(c);
    return -1;
  }
  c->device = dreq.device;
  c->queue = wgpuDeviceGetQueue(c->device);
  c->ok = 1;
  return 0;
}

void wgpu_ctx_free(WgpuCtx* c) {
  if (!c) return;
  if (c->queue) wgpuQueueRelease(c->queue);
  if (c->device) wgpuDeviceRelease(c->device);
  if (c->adapter) wgpuAdapterRelease(c->adapter);
  if (c->instance) wgpuInstanceRelease(c->instance);
  memset(c, 0, sizeof(*c));
}

/* ---- buffers ---- */

static int make_buf(WgpuCtx* c, uint64_t n, WGPUBufferUsage usage, WgpuBuf* out) {
  memset(out, 0, sizeof(*out));
  n = round4(n);
  WGPUBufferDescriptor bd = {0};
  bd.size = n;
  bd.usage = usage;
  bd.mappedAtCreation = 0;
  out->buf = wgpuDeviceCreateBuffer(c->device, &bd);
  if (!out->buf) return -1;
  out->size = n;
  out->usage = usage;
  return 0;
}

int wgpu_buf_device(WgpuCtx* c, uint64_t n, WgpuBuf* out) {
  return make_buf(c, n,
                  WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst |
                      WGPUBufferUsage_CopySrc,
                  out);
}

int wgpu_buf_uniform(WgpuCtx* c, uint64_t n, WgpuBuf* out) {
  return make_buf(c, round16(n),
                  WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst, out);
}

void wgpu_buf_free(WgpuCtx* c, WgpuBuf* b) {
  (void)c;
  if (b && b->buf) {
    wgpuBufferRelease(b->buf);
    memset(b, 0, sizeof(*b));
  }
}

int wgpu_upload(WgpuCtx* c, WgpuBuf* dst, const void* src, uint64_t n) {
  if (!dst || !dst->buf || !src) return -1;
  if (n > dst->size) n = dst->size;
  wgpuQueueWriteBuffer(c->queue, dst->buf, 0, src, (size_t)n);
  return 0;
}

int wgpu_upload_uniform(WgpuCtx* c, WgpuBuf* ubo, const void* src, uint64_t n) {
  return wgpu_upload(c, ubo, src, n);
}

int wgpu_zero(WgpuCtx* c, WgpuBuf* b, uint64_t n) {
  if (!b || !b->buf) return -1;
  if (n > b->size) n = b->size;
  /* queue.WriteBuffer of zeros — fine for KV init; large caches may want a
   * clear shader later. */
  uint8_t* z = (uint8_t*)calloc((size_t)n, 1);
  if (!z) return -1;
  int rc = wgpu_upload(c, b, z, n);
  free(z);
  return rc;
}

typedef struct {
  int done;
  WGPUMapAsyncStatus status;
} MapReq;

static void on_mapped(WGPUMapAsyncStatus status, WGPUStringView message,
                      void* userdata1, void* userdata2) {
  (void)message; (void)userdata2;
  MapReq* r = (MapReq*)userdata1;
  r->status = status;
  r->done = 1;
}

int wgpu_download(WgpuCtx* c, const WgpuBuf* src, void* dst, uint64_t n) {
  if (!src || !src->buf || !dst) return -1;
  if (n > src->size) n = src->size;
  n = round4(n);

  /* Staging buffer with MAP_READ. */
  WgpuBuf stg;
  if (make_buf(c, n, WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead, &stg) !=
      0)
    return -1;

  WGPUCommandEncoder enc =
      wgpuDeviceCreateCommandEncoder(c->device, NULL);
  wgpuCommandEncoderCopyBufferToBuffer(enc, src->buf, 0, stg.buf, 0, n);
  WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(enc, NULL);
  wgpuCommandEncoderRelease(enc);
  wgpuQueueSubmit(c->queue, 1, &cmd);
  wgpuCommandBufferRelease(cmd);

  MapReq mreq = {0};
  WGPUBufferMapCallbackInfo mci = {0};
  mci.mode = WGPUCallbackMode_AllowSpontaneous;
  mci.callback = on_mapped;
  mci.userdata1 = &mreq;
  wgpuBufferMapAsync(stg.buf, WGPUMapMode_Read, 0, n, mci);

  int spins = 0;
  while (!mreq.done && spins++ < 1000000) wgpu_pump(c);
  if (!mreq.done || mreq.status != WGPUMapAsyncStatus_Success) {
    wgpu_buf_free(c, &stg);
    return -1;
  }
  const void* mapped = wgpuBufferGetConstMappedRange(stg.buf, 0, n);
  if (!mapped) {
    wgpuBufferUnmap(stg.buf);
    wgpu_buf_free(c, &stg);
    return -1;
  }
  memcpy(dst, mapped, (size_t)n);
  wgpuBufferUnmap(stg.buf);
  wgpu_buf_free(c, &stg);
  return 0;
}

/* ---- shaders / pipelines ---- */

static char* read_text(const char* path, size_t* out_len) {
  FILE* f = fopen(path, "rb");
  if (!f) return NULL;
  fseek(f, 0, SEEK_END);
  long sz = ftell(f);
  fseek(f, 0, SEEK_SET);
  if (sz < 0) { fclose(f); return NULL; }
  char* buf = (char*)malloc((size_t)sz + 1);
  if (!buf) { fclose(f); return NULL; }
  if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) {
    free(buf);
    fclose(f);
    return NULL;
  }
  fclose(f);
  buf[sz] = 0;
  if (out_len) *out_len = (size_t)sz;
  return buf;
}

static char* load_wgsl(const char* dir, const char* name, int with_prelude,
                       char* err, size_t errlen) {
  char path[1024];
  char* body = NULL;
  char* prelude = NULL;
  size_t blen = 0, plen = 0;

  snprintf(path, sizeof(path), "%s/%s", dir, name);
  body = read_text(path, &blen);
  if (!body) {
    if (err && errlen) snprintf(err, errlen, "webgpu: cannot read %s", path);
    return NULL;
  }
  if (!with_prelude) return body;

  snprintf(path, sizeof(path), "%s/prelude.wgsl", dir);
  prelude = read_text(path, &plen);
  if (!prelude) {
    free(body);
    if (err && errlen) snprintf(err, errlen, "webgpu: cannot read %s", path);
    return NULL;
  }
  char* out = (char*)malloc(plen + blen + 4);
  if (!out) {
    free(body);
    free(prelude);
    return NULL;
  }
  memcpy(out, prelude, plen);
  out[plen] = '\n';
  memcpy(out + plen + 1, body, blen + 1);
  free(body);
  free(prelude);
  return out;
}

/* Binding layout descriptors — MUST match the .wgsl @binding indices.
 * kind: 0=uniform, 1=read storage, 2=read_write storage. */
typedef struct {
  int binding;
  int kind;
} BindSpec;

static int create_pipe(WgpuCtx* c, const char* dir, const char* fname,
                       int with_prelude, const BindSpec* specs, int nspecs,
                       WgpuPipe* out, char* err, size_t errlen) {
  memset(out, 0, sizeof(*out));
  out->n_entries = nspecs;
  out->has_prelude = with_prelude;

  char* src = load_wgsl(dir, fname, with_prelude, err, errlen);
  if (!src) return -1;

  WGPUShaderSourceWGSL wgsl = {0};
  wgsl.chain.sType = WGPUSType_ShaderSourceWGSL;
  wgsl.code = (WGPUStringView){src, strlen(src)};

  WGPUShaderModuleDescriptor md = {0};
  md.nextInChain = (WGPUChainedStruct*)&wgsl;
  WGPUShaderModule mod = wgpuDeviceCreateShaderModule(c->device, &md);
  free(src);
  if (!mod) {
    if (err && errlen)
      snprintf(err, errlen, "webgpu: shader module %s failed", fname);
    return -1;
  }

  WGPUBindGroupLayoutEntry entries[8];
  memset(entries, 0, sizeof(entries));
  for (int i = 0; i < nspecs; ++i) {
    entries[i].binding = (uint32_t)specs[i].binding;
    entries[i].visibility = WGPUShaderStage_Compute;
    if (specs[i].kind == 0) {
      entries[i].buffer.type = WGPUBufferBindingType_Uniform;
      entries[i].buffer.minBindingSize = 0;
    } else if (specs[i].kind == 1) {
      entries[i].buffer.type = WGPUBufferBindingType_ReadOnlyStorage;
      entries[i].buffer.minBindingSize = 0;
    } else {
      entries[i].buffer.type = WGPUBufferBindingType_Storage;
      entries[i].buffer.minBindingSize = 0;
    }
  }
  WGPUBindGroupLayoutDescriptor bgld = {0};
  bgld.entryCount = (uint32_t)nspecs;
  bgld.entries = entries;
  out->bgl = wgpuDeviceCreateBindGroupLayout(c->device, &bgld);
  if (!out->bgl) {
    wgpuShaderModuleRelease(mod);
    if (err && errlen)
      snprintf(err, errlen, "webgpu: bind group layout %s failed", fname);
    return -1;
  }

  WGPUPipelineLayoutDescriptor pld = {0};
  pld.bindGroupLayoutCount = 1;
  pld.bindGroupLayouts = &out->bgl;
  WGPUPipelineLayout layout = wgpuDeviceCreatePipelineLayout(c->device, &pld);
  if (!layout) {
    wgpuShaderModuleRelease(mod);
    if (err && errlen)
      snprintf(err, errlen, "webgpu: pipeline layout %s failed", fname);
    return -1;
  }

  WGPUComputePipelineDescriptor cpd = {0};
  cpd.layout = layout;
  cpd.compute.module = mod;
  cpd.compute.entryPoint = (WGPUStringView){"main", 4};
  out->pipe = wgpuDeviceCreateComputePipeline(c->device, &cpd);
  wgpuPipelineLayoutRelease(layout);
  wgpuShaderModuleRelease(mod);
  if (!out->pipe) {
    if (err && errlen)
      snprintf(err, errlen, "webgpu: compute pipeline %s failed", fname);
    return -1;
  }
  return 0;
}

static void free_pipe(WgpuCtx* c, WgpuPipe* p) {
  (void)c;
  if (!p) return;
  if (p->pipe) wgpuComputePipelineRelease(p->pipe);
  if (p->bgl) wgpuBindGroupLayoutRelease(p->bgl);
  memset(p, 0, sizeof(*p));
}

int wgpu_kernels_init(WgpuCtx* c, WgpuKernels* k, const char* dir, char* err,
                      size_t errlen) {
  memset(k, 0, sizeof(*k));
  if (!dir || !dir[0]) dir = "src/webgpu/wgsl";

  /* kind: 0=uniform 1=ro storage 2=rw storage — binding order matches specs */
  const BindSpec mv[] = {{0, 1}, {1, 0}, {2, 1}, {3, 2}}; /* W,P,x,y — W is RO storage */
  /* matvec: W is read storage (prelude), but weights never written — kind 1.
   * Wait: prelude declares var<storage, read> W — yes kind 1. */
  const BindSpec emb[] = {{0, 1}, {1, 0}, {2, 2}};
  const BindSpec rms[] = {{0, 0}, {1, 1}, {2, 1}, {3, 2}};
  const BindSpec rope[] = {{0, 0}, {1, 2}, {2, 1}};
  const BindSpec kv[] = {{0, 0}, {1, 2}, {2, 2}, {3, 1}, {4, 1}};
  const BindSpec attn[] = {{0, 0}, {1, 2}, {2, 1}, {3, 1}, {4, 1}, {5, 2}};
  const BindSpec add[] = {{0, 0}, {1, 2}, {2, 1}};
  const BindSpec silu[] = {{0, 0}, {1, 2}, {2, 1}};
  const BindSpec geglu[] = {{0, 0}, {1, 2}, {2, 1}};
  const BindSpec resid[] = {{0, 0}, {1, 2}, {2, 1}, {3, 1}};
  const BindSpec soft[] = {{0, 0}, {1, 2}};
  const BindSpec argm[] = {{0, 0}, {1, 1}, {2, 1}, {3, 2}, {4, 2}};

#define P(field, file, prelude, specs)                                       \
  if (create_pipe(c, dir, file, prelude, specs,                              \
                  (int)(sizeof(specs) / sizeof((specs)[0])), &k->field, err, \
                  errlen) != 0)                                              \
    goto fail

  P(matvec, "matvec.wgsl", 1, mv);
  P(embed, "embed.wgsl", 1, emb);
  P(rmsnorm, "rmsnorm.wgsl", 0, rms);
  P(rope, "rope.wgsl", 0, rope);
  P(kv_store, "kv_store.wgsl", 0, kv);
  P(attn, "attn.wgsl", 0, attn);
  P(add, "add.wgsl", 0, add);
  P(silu_mul, "silu_mul.wgsl", 0, silu);
  P(geglu, "geglu.wgsl", 0, geglu);
  P(resid_out, "resid_out.wgsl", 0, resid);
  P(softcap, "softcap.wgsl", 0, soft);
  P(argmax, "argmax.wgsl", 0, argm);
#undef P
  (void)ARGMAX_BLOCKS;
  return 0;
fail:
  wgpu_kernels_free(c, k);
  return -1;
}

void wgpu_kernels_free(WgpuCtx* c, WgpuKernels* k) {
  if (!k) return;
  free_pipe(c, &k->matvec);
  free_pipe(c, &k->embed);
  free_pipe(c, &k->rmsnorm);
  free_pipe(c, &k->rope);
  free_pipe(c, &k->kv_store);
  free_pipe(c, &k->attn);
  free_pipe(c, &k->add);
  free_pipe(c, &k->silu_mul);
  free_pipe(c, &k->geglu);
  free_pipe(c, &k->resid_out);
  free_pipe(c, &k->softcap);
  free_pipe(c, &k->argmax);
}

/* ---- recording ---- */

int wgpu_rec_begin(WgpuCtx* c, WgpuRec* r) {
  r->ctx = c;
  r->enc = wgpuDeviceCreateCommandEncoder(c->device, NULL);
  return r->enc ? 0 : -1;
}

void wgpu_dispatch(WgpuRec* r, const WgpuPipe* p, WgpuBuf* const* bufs,
                   uint32_t gx, uint32_t gy, uint32_t gz) {
  if (!r || !r->enc || !p || !p->pipe) return;

  WGPUBindGroupEntry ents[8];
  memset(ents, 0, sizeof(ents));
  for (int i = 0; i < p->n_entries; ++i) {
    ents[i].binding = (uint32_t)i;
    ents[i].buffer = bufs[i]->buf;
    ents[i].offset = 0;
    ents[i].size = bufs[i]->size;
  }
  WGPUBindGroupDescriptor bgd = {0};
  bgd.layout = p->bgl;
  bgd.entryCount = (uint32_t)p->n_entries;
  bgd.entries = ents;
  WGPUBindGroup bg = wgpuDeviceCreateBindGroup(r->ctx->device, &bgd);
  if (!bg) return;

  /* One compute pass per dispatch (coarse barrier via pass boundary). */
  WGPUComputePassEncoder pass =
      wgpuCommandEncoderBeginComputePass(r->enc, NULL);
  wgpuComputePassEncoderSetPipeline(pass, p->pipe);
  wgpuComputePassEncoderSetBindGroup(pass, 0, bg, 0, NULL);
  wgpuComputePassEncoderDispatchWorkgroups(pass, gx, gy, gz);
  wgpuComputePassEncoderEnd(pass);
  wgpuComputePassEncoderRelease(pass);
  wgpuBindGroupRelease(bg);
}

int wgpu_rec_submit_wait(WgpuRec* r) {
  if (!r || !r->enc) return -1;
  WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(r->enc, NULL);
  wgpuCommandEncoderRelease(r->enc);
  r->enc = NULL;
  if (!cmd) return -1;
  wgpuQueueSubmit(r->ctx->queue, 1, &cmd);
  wgpuCommandBufferRelease(cmd);

  /* Best-effort idle. Dawn may expose wgpuDevicePoll / OnSubmittedWorkDone.
   * ASSUMED: spin a bit so subsequent mapAsync sees completed copies. */
  for (int i = 0; i < 100; ++i) wgpu_pump(r->ctx);
  return 0;
}
