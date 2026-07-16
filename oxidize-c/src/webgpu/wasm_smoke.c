/* ======================================================================
 * UNVERIFIED — NEVER COMPILED OR RUN. Minimal emscripten smoke entry for
 * make wasm. Only probes adapter/device via wgpu_ctx_init; does not load
 * weights or run a forward pass.
 * ====================================================================== */
#include "webgpu_common.h"

#include <stdio.h>

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#endif

int wgpu_smoke_main(void) {
  char err[512];
  err[0] = 0;
  WgpuCtx ctx;
  if (wgpu_ctx_init(&ctx, err, sizeof(err)) != 0) {
    fprintf(stderr, "webgpu smoke: init failed: %s\n", err);
    return 1;
  }
  fprintf(stderr, "webgpu smoke: adapter+device ok (UNVERIFIED path)\n");
  wgpu_ctx_free(&ctx);
  return 0;
}

#ifndef __EMSCRIPTEN__
int main(void) { return wgpu_smoke_main(); }
#endif
