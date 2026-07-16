/* ============================================================================
 * UNVERIFIED — this file has NEVER been compiled or run. It was written BLIND
 * against the verified CUDA reference (src/cuda/) and the Rust Metal backend
 * (oxidize-core/src/backends/metal.rs). It requires macOS + Xcode (Metal /
 * Foundation frameworks, `xcrun metal`) and an Apple GPU to compile and
 * validate. It MAY NOT COMPILE. No equivalence gate has ever been run against
 * it. Known-blind risks: metallib path resolution, setBytes constant packing
 * vs `constant T&` MSL bindings, threadgroup memory sizing for attn, and
 * pipeline-state caching are all plausible-but-untested.
 * ============================================================================ */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <CoreFoundation/CoreFoundation.h>

#include "metal_common.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern "C" {
#include "../quant.h"
}

static const uint32_t OC_TYPES[8] = {OC_F32, OC_F16,  OC_Q4_0, OC_Q8_0,
                                     OC_Q4_K, OC_Q5_K, OC_Q6_K, OC_AL5_XS};

int mt_qidx(uint32_t t) {
  for (int i = 0; i < 8; ++i)
    if (OC_TYPES[i] == t) return i;
  return -1;
}

static id<MTLDevice> dev_of(MtCtx* c) {
  return (__bridge id<MTLDevice>)c->device;
}
static id<MTLCommandQueue> queue_of(MtCtx* c) {
  return (__bridge id<MTLCommandQueue>)c->queue;
}
static id<MTLLibrary> lib_of(MtCtx* c) {
  return (__bridge id<MTLLibrary>)c->library;
}
static NSMutableDictionary* pipes_of(MtCtx* c) {
  return (__bridge NSMutableDictionary*)c->pipes;
}
static id<MTLBuffer> buf_of(const MtBuf* b) {
  return (__bridge id<MTLBuffer>)b->buf;
}

int mt_ctx_init(MtCtx* c, const char* metallib_path, char* err, size_t errlen) {
  memset(c, 0, sizeof(*c));
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
      if (err && errlen)
        snprintf(err, errlen, "metal: no MTLDevice (needs macOS + Apple GPU)");
      return -1;
    }
    id<MTLCommandQueue> queue = [device newCommandQueue];
    if (!queue) {
      if (err && errlen) snprintf(err, errlen, "metal: newCommandQueue failed");
      return -1;
    }

    const char* path = metallib_path;
    char envbuf[1024];
    if (!path || !path[0]) {
      const char* env = getenv("OXIDIZE_METAL_LIB");
      if (env && env[0])
        path = env;
      else {
        snprintf(envbuf, sizeof(envbuf), "src/metal/build/oxidize.metallib");
        path = envbuf;
      }
    }
    NSError* nserr = nil;
    NSString* nspath = [NSString stringWithUTF8String:path];
    NSURL* url = [NSURL fileURLWithPath:nspath];
    id<MTLLibrary> library = [device newLibraryWithURL:url error:&nserr];
    if (!library) {
      if (err && errlen)
        snprintf(err, errlen, "metal: failed to load metallib '%s': %s", path,
                 nserr ? [[nserr localizedDescription] UTF8String] : "?");
      return -1;
    }

    id<MTLBuffer> dummy =
        [device newBufferWithLength:64 options:MTLResourceStorageModeShared];
    if (!dummy) {
      if (err && errlen) snprintf(err, errlen, "metal: dummy buffer alloc failed");
      return -1;
    }
    memset([dummy contents], 0, 64);

    NSMutableDictionary* pipes = [NSMutableDictionary dictionary];

    c->device = (__bridge_retained void*)device;
    c->queue = (__bridge_retained void*)queue;
    c->library = (__bridge_retained void*)library;
    c->pipes = (__bridge_retained void*)pipes;
    c->dummy = (__bridge_retained void*)dummy;
  }
  return 0;
}

void mt_ctx_free(MtCtx* c) {
  if (!c) return;
  @autoreleasepool {
    if (c->dummy) CFRelease(c->dummy);
    if (c->pipes) CFRelease(c->pipes);
    if (c->library) CFRelease(c->library);
    if (c->queue) CFRelease(c->queue);
    if (c->device) CFRelease(c->device);
  }
  memset(c, 0, sizeof(*c));
}

int mt_buf_alloc(MtCtx* c, size_t n, MtBuf* out) {
  memset(out, 0, sizeof(*out));
  if (n == 0) n = 4;
  @autoreleasepool {
    id<MTLBuffer> b =
        [dev_of(c) newBufferWithLength:n options:MTLResourceStorageModeShared];
    if (!b) return -1;
    memset([b contents], 0, n);
    out->buf = (__bridge_retained void*)b;
    out->size = n;
  }
  return 0;
}

int mt_buf_upload(MtCtx* c, MtBuf* dst, const void* src, size_t n) {
  (void)c;
  if (!dst || !dst->buf || n > dst->size) return -1;
  @autoreleasepool {
    memcpy([buf_of(dst) contents], src, n);
  }
  return 0;
}

int mt_buf_zero(MtCtx* c, MtBuf* b, size_t n) {
  (void)c;
  if (!b || !b->buf) return -1;
  if (n > b->size) n = b->size;
  @autoreleasepool {
    memset([buf_of(b) contents], 0, n);
  }
  return 0;
}

int mt_buf_download(MtCtx* c, const MtBuf* src, void* dst, size_t n) {
  (void)c;
  if (!src || !src->buf || n > src->size) return -1;
  @autoreleasepool {
    memcpy(dst, [buf_of(src) contents], n);
  }
  return 0;
}

void mt_buf_free(MtCtx* c, MtBuf* b) {
  (void)c;
  if (!b) return;
  if (b->buf) CFRelease(b->buf);
  memset(b, 0, sizeof(*b));
}

int mt_pipe_get(MtCtx* c, const char* name, MtPipe* out, char* err,
                size_t errlen) {
  memset(out, 0, sizeof(*out));
  @autoreleasepool {
    NSString* key = [NSString stringWithUTF8String:name];
    NSMutableDictionary* pipes = pipes_of(c);
    id<MTLComputePipelineState> pso = pipes[key];
    if (!pso) {
      NSError* nserr = nil;
      id<MTLFunction> fn = [lib_of(c) newFunctionWithName:key];
      if (!fn) {
        if (err && errlen)
          snprintf(err, errlen, "metal: kernel '%s' not in metallib", name);
        return -1;
      }
      pso = [dev_of(c) newComputePipelineStateWithFunction:fn error:&nserr];
      if (!pso) {
        if (err && errlen)
          snprintf(err, errlen, "metal: PSO for '%s' failed: %s", name,
                   nserr ? [[nserr localizedDescription] UTF8String] : "?");
        return -1;
      }
      pipes[key] = pso;
    }
    /* Borrowed from the dictionary (retained by it for the ctx lifetime). */
    out->pso = (__bridge void*)pso;
  }
  return 0;
}

const char* mt_matvec_name(const char* prefix, uint32_t ggml_type) {
  /* gemma4 hand-fused AL5_XS; llama uses the templated al5xs instantiation. */
  if (ggml_type == OC_AL5_XS && prefix[0] == 'g') return "gk_matvec_al5xs";
  switch (ggml_type) {
    case OC_F32: return prefix[0] == 'g' ? "gk_matvec_f32" : "lk_matvec_f32";
    case OC_F16: return prefix[0] == 'g' ? "gk_matvec_f16" : "lk_matvec_f16";
    case OC_Q4_0: return prefix[0] == 'g' ? "gk_matvec_q4_0" : "lk_matvec_q4_0";
    case OC_Q8_0: return prefix[0] == 'g' ? "gk_matvec_q8_0" : "lk_matvec_q8_0";
    case OC_Q4_K: return prefix[0] == 'g' ? "gk_matvec_q4_k" : "lk_matvec_q4_k";
    case OC_Q5_K: return prefix[0] == 'g' ? "gk_matvec_q5_k" : "lk_matvec_q5_k";
    case OC_Q6_K: return prefix[0] == 'g' ? "gk_matvec_q6_k" : "lk_matvec_q6_k";
    case OC_AL5_XS: return "lk_matvec_al5xs";
    default: return NULL;
  }
}

const char* mt_embed_name(const char* prefix, uint32_t ggml_type) {
  switch (ggml_type) {
    case OC_F32: return prefix[0] == 'g' ? "gk_embed_f32" : "lk_embed_f32";
    case OC_F16: return prefix[0] == 'g' ? "gk_embed_f16" : "lk_embed_f16";
    case OC_Q4_0: return prefix[0] == 'g' ? "gk_embed_q4_0" : "lk_embed_q4_0";
    case OC_Q8_0: return prefix[0] == 'g' ? "gk_embed_q8_0" : "lk_embed_q8_0";
    case OC_Q4_K: return prefix[0] == 'g' ? "gk_embed_q4_k" : "lk_embed_q4_k";
    case OC_Q5_K: return prefix[0] == 'g' ? "gk_embed_q5_k" : "lk_embed_q5_k";
    case OC_Q6_K: return prefix[0] == 'g' ? "gk_embed_q6_k" : "lk_embed_q6_k";
    case OC_AL5_XS:
      return prefix[0] == 'g' ? "gk_embed_al5xs" : "lk_embed_al5xs";
    default: return NULL;
  }
}

int mt_rec_begin(MtCtx* c, MtRec* r) {
  memset(r, 0, sizeof(*r));
  r->ctx = c;
  @autoreleasepool {
    id<MTLCommandBuffer> cmd = [queue_of(c) commandBuffer];
    if (!cmd) return -1;
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    if (!enc) return -1;
    r->cmd = (__bridge_retained void*)cmd;
    r->enc = (__bridge_retained void*)enc;
  }
  return 0;
}

void mt_dispatch(MtRec* r, const MtPipe* p, const MtBind* binds, int nbinds,
                 size_t threadgroup_mem, MtSize grid, MtSize tgs) {
  @autoreleasepool {
    id<MTLComputeCommandEncoder> enc =
        (__bridge id<MTLComputeCommandEncoder>)r->enc;
    id<MTLComputePipelineState> pso =
        (__bridge id<MTLComputePipelineState>)p->pso;
    [enc setComputePipelineState:pso];
    for (int i = 0; i < nbinds; ++i) {
      if (binds[i].kind == MT_BIND_BYTES) {
        [enc setBytes:binds[i].bytes
               length:binds[i].nbytes
              atIndex:(NSUInteger)i];
      } else {
        id<MTLBuffer> b = binds[i].buf && binds[i].buf->buf
                              ? buf_of(binds[i].buf)
                              : (__bridge id<MTLBuffer>)r->ctx->dummy;
        [enc setBuffer:b offset:binds[i].offset atIndex:(NSUInteger)i];
      }
    }
    if (threadgroup_mem > 0)
      [enc setThreadgroupMemoryLength:threadgroup_mem atIndex:0];
    MTLSize g = MTLSizeMake(grid.width, grid.height, grid.depth);
    MTLSize t = MTLSizeMake(tgs.width, tgs.height, tgs.depth);
    [enc dispatchThreadgroups:g threadsPerThreadgroup:t];
  }
}

int mt_rec_commit_wait(MtRec* r) {
  if (!r || !r->cmd) return -1;
  int rc = 0;
  @autoreleasepool {
    id<MTLComputeCommandEncoder> enc =
        (__bridge_transfer id<MTLComputeCommandEncoder>)r->enc;
    id<MTLCommandBuffer> cmd = (__bridge_transfer id<MTLCommandBuffer>)r->cmd;
    r->enc = NULL;
    r->cmd = NULL;
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    if (cmd.status == MTLCommandBufferStatusError) {
      fprintf(stderr, "metal: command buffer error: %s\n",
              cmd.error ? [[cmd.error localizedDescription] UTF8String] : "?");
      rc = -1;
    }
  }
  return rc;
}

int mt_rec_commit(MtRec* r) {
  if (!r || !r->cmd) return -1;
  @autoreleasepool {
    id<MTLComputeCommandEncoder> enc =
        (__bridge_transfer id<MTLComputeCommandEncoder>)r->enc;
    id<MTLCommandBuffer> cmd = (__bridge_transfer id<MTLCommandBuffer>)r->cmd;
    r->enc = NULL;
    r->cmd = NULL;
    [enc endEncoding];
    [cmd commit];
  }
  return 0;
}

MtSize mt_tgs_1d(size_t n) {
  MtSize s = {n, 1, 1};
  return s;
}
MtSize mt_grid_1d(size_t n, size_t tgs) {
  size_t g = (n + tgs - 1) / tgs;
  if (g == 0) g = 1;
  MtSize s = {g, 1, 1};
  return s;
}
MtSize mt_matvec_tgs(void) {
  /* 8 SIMD-groups * 32 lanes = 256, matching CUDA WARPS=8. */
  MtSize s = {256, 1, 1};
  return s;
}
MtSize mt_matvec_grid(int rows) {
  const int WARPS = 8;
  int g = (rows + WARPS - 1) / WARPS;
  if (g < 1) g = 1;
  MtSize s = {(size_t)g, 1, 1};
  return s;
}
