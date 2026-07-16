# ======================================================================
# UNVERIFIED — this Makefile has NEVER been run successfully in CI.
# The WebGPU backend is NOT part of oxidize-c's default build. Build only
# from this directory, explicitly.
# ======================================================================
#
# Targets:
#   make            -> prints help (does NOT build)
#   make wasm       -> emcc build when $$EMSDK is set (emdawnwebgpu)
#   make check-env  -> report whether EMSDK / Dawn headers look present
#   make clean      -> remove local build artifacts under ./build/
#
# Never invoked from oxidize-c/Makefile.

SHELL := /bin/bash
ROOT := $(abspath ../..)
WGSL_DIR := $(abspath ./wgsl)
BUILD := $(abspath ./build)

# Optional: path to Dawn / emdawnwebgpu headers+libs. Override as needed.
WEBGPU_CFLAGS ?=
WEBGPU_LDFLAGS ?=

.PHONY: all help wasm check-env clean

all: help

help:
	@echo "oxidize-c WebGPU backend — UNVERIFIED, not in the default build."
	@echo ""
	@echo "  make wasm        Build WASM harness when EMSDK is set"
	@echo "  make check-env   Show toolchain detection"
	@echo "  make clean       Remove ./build"
	@echo ""
	@echo "WGSL sources: $(WGSL_DIR)"
	@echo "This Makefile is intentionally isolated from oxidize-c/Makefile."

check-env:
	@echo "EMSDK=$${EMSDK:-<unset>}"
	@echo "EMSCRIPTEN=$${EMSCRIPTEN:-<unset>}"
	@if [ -n "$${EMSDK}" ] && [ -f "$${EMSDK}/emsdk_env.sh" ]; then \
	  echo "emsdk_env.sh: found"; \
	else \
	  echo "emsdk_env.sh: NOT found (wasm target will refuse)"; \
	fi
	@echo "WGSL files:" && ls -1 $(WGSL_DIR)/*.wgsl 2>/dev/null | sed 's/^/  /'

# ---- WASM (emdawnwebgpu) -------------------------------------------------
# Requires: source $$EMSDK/emsdk_env.sh
# Links a minimal smoke object that only initializes a WgpuCtx (no model
# weights) so the browser harness can prove adapter acquisition. Full
# gemma4/llama objects need the rest of oxidize-c linked — out of scope for
# this isolated Makefile unless OXIDIZE_C_OBJS is provided by the caller.

WASM_OUT := $(BUILD)/oxidize_webgpu.js
WASM_HTML := $(abspath ./index.html)

wasm:
	@if [ -z "$${EMSDK}" ]; then \
	  echo "error: EMSDK is unset. source \$$EMSDK/emsdk_env.sh first." >&2; \
	  exit 1; \
	fi
	@mkdir -p $(BUILD)
	@echo "UNVERIFIED wasm build — may fail against your emdawnwebgpu revision."
	emcc -O0 -g \
	  -I$(ROOT)/src -I. \
	  -sUSE_WEBGPU=1 \
	  -sASYNCIFY \
	  -sALLOW_MEMORY_GROWTH=1 \
	  -sEXPORTED_FUNCTIONS='["_wgpu_smoke_main","_malloc","_free"]' \
	  -sEXPORTED_RUNTIME_METHODS='["ccall","cwrap"]' \
	  $(WEBGPU_CFLAGS) \
	  wasm_smoke.c webgpu_common.c \
	  -o $(WASM_OUT) \
	  $(WEBGPU_LDFLAGS)
	@echo "wrote $(WASM_OUT)"
	@echo "open $(WASM_HTML) via a local static server (not file://)."

clean:
	rm -rf $(BUILD)
