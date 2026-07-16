# ======================================================================
# UNVERIFIED — this Makefile / the targets it builds have NEVER been
# compiled or run against a real Vulkan driver. Blind scaffolding for
# src/vulkan/. IT MAY NOT COMPILE and MAY BE WRONG.
#
# NEVER wired into the top-level oxidize-c Makefile. Build ONLY from here:
#   make -C src/vulkan
# ======================================================================

# When `make -C src/vulkan` runs, CURDIR is .../oxidize-c/src/vulkan.
OXIDIZE_C := $(abspath $(CURDIR)/../..)

CC ?= gcc
CFLAGS ?= -std=c11 -O2 -Wall -Wextra -I$(OXIDIZE_C)/src
LDFLAGS ?= -lvulkan -lm

SHADER_DIR := $(CURDIR)/shaders
# Host code looks up SPIR-V via $OXIDIZE_VK_SHADERS (default src/vulkan/shaders
# relative to the process cwd). Export the absolute path for local smoke runs.
export OXIDIZE_VK_SHADERS ?= $(SHADER_DIR)

COMPS := matvec embed rmsnorm rope kv_store attn add silu_mul geglu \
         resid_out softcap argmax
SPVS  := $(addprefix $(SHADER_DIR)/,$(addsuffix .spv,$(COMPS)))

# dequant.glsl is #included by matvec/embed — glslangValidator needs -I.
GLSLANG ?= $(shell command -v glslangValidator 2>/dev/null)
GLSLC   ?= $(shell command -v glslc 2>/dev/null)

.PHONY: all shaders host clean help
all: shaders host

help:
	@echo "UNVERIFIED Vulkan backend — not part of the default oxidize-c build."
	@echo "  make shaders   # .comp -> .spv (glslangValidator or glslc)"
	@echo "  make host      # gemma4_vk.o llama_vk.o vk_common.o (needs -lvulkan)"
	@echo "  make clean"
	@echo "SPIR-V dir (OXIDIZE_VK_SHADERS): $(OXIDIZE_VK_SHADERS)"

shaders: $(SPVS)

$(SHADER_DIR)/%.spv: $(SHADER_DIR)/%.comp $(SHADER_DIR)/dequant.glsl
	@if [ -n "$(GLSLANG)" ]; then \
	  $(GLSLANG) -V -I$(SHADER_DIR) $< -o $@ ; \
	elif [ -n "$(GLSLC)" ]; then \
	  $(GLSLC) -fshader-stage=compute -I$(SHADER_DIR) $< -o $@ ; \
	else \
	  echo "error: need glslangValidator or glslc in PATH" >&2; exit 1; \
	fi

HOST_OBJS := vk_common.o gemma4_vk.o llama_vk.o

host: $(HOST_OBJS)
	@echo "UNVERIFIED: host objects built. No default link into oxidize-c."
	@echo "Objects: $(HOST_OBJS)"

vk_common.o: vk_common.c vk_common.h
	$(CC) $(CFLAGS) -c vk_common.c -o $@

gemma4_vk.o: gemma4_vk.c gemma4_vk.h vk_common.h
	$(CC) $(CFLAGS) -c gemma4_vk.c -o $@

llama_vk.o: llama_vk.c llama_vk.h vk_common.h
	$(CC) $(CFLAGS) -c llama_vk.c -o $@

# Optional smoke link of just the Vulkan objects (still needs the rest of
# oxidize-c to form a binary — this only checks the Vulkan TU symbols).
liboxidize_vk.a: $(HOST_OBJS)
	ar rcs $@ $(HOST_OBJS)

clean:
	rm -f $(HOST_OBJS) liboxidize_vk.a $(SPVS)
