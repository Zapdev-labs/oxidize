# oxidize-c — pure C11 CPU inference for Gemma 4 GGUF.
# `make`        -> build/oxidize-c: portable x86-64-v2 binary; the AVX2 / AVX-512
#                  / AVX-512-VNNI kernels are all compiled in and selected at
#                  RUNTIME from cpuid. Runs anywhere, dispatches upward.
# `make native` -> same, but the non-kernel code is also built -march=native.
#                  Only for the machine that built it.
# `make test`   -> build + run unit tests
# `make asan`   -> ASAN+UBSAN build + test run (single-threaded: the pool spins)
# `make fuzz`   -> libFuzzer GGUF target (needs clang); FUZZ_ARGS to tune
# `make cuda`   -> oxidize-c-cuda (GPU-resident decode; see CUDA.md)
#
# `./oxidize-c --version` prints the ISA it selected; OC_ISA=scalar|avx2|avx512
# forces one down.

CC      ?= gcc
# Target arch from the compiler triple, not `uname -m`: identical to before for a
# native x86 build, but a cross aarch64-*-gcc now selects the NEON path too (so
# the kernels can be built + run under qemu on an x86 CI box — see modal_arm.py).
CC_MACHINE := $(shell $(CC) -dumpmachine 2>/dev/null)

# Baseline for everything that is NOT a kernel. x86-64-v2 = SSE4.2/POPCNT:
# every 64-bit x86 since ~2009. NOT -march=native: that bakes the build host's
# ISA into every object and SIGILLs elsewhere.
ifneq (,$(findstring x86_64,$(CC_MACHINE)))
  ARCH_FLAGS   ?= -march=x86-64-v2
  ISA_SRCS     := src/quant_avx2.c src/quant_avx512.c src/quant_vnni.c
  AVX2_FLAGS   := -mavx2 -mfma -mf16c
  AVX512_FLAGS := -mavx2 -mfma -mf16c -mavx512f -mavx512bw -mavx512vl -mavx512dq
  VNNI_FLAGS   := $(AVX512_FLAGS) -mavx512vnni
else ifneq (,$(findstring aarch64,$(CC_MACHINE)))
  # NEON/ASIMD is armv8-a baseline, so quant_neon.c needs no extra -march flag;
  # the generic object rule builds it with plain CFLAGS. Nothing x86 is compiled.
  ARCH_FLAGS   ?=
  ISA_SRCS     := src/quant_neon.c
else
  ARCH_FLAGS   ?=
  ISA_SRCS     :=
endif
CFLAGS  ?= -std=c11 -D_DEFAULT_SOURCE -O3 $(ARCH_FLAGS) -Wall -Wextra -pthread
LDLIBS  := -lm -lpthread

BUILD   ?= build
BACKEND ?= cpu
TEST_ENV ?=
# Prefix for running the test binaries. Empty for a native build; set to
# `qemu-aarch64-static` to run a cross-built aarch64 binary under emulation.
RUNNER ?=
ISA_OBJS := $(ISA_SRCS:src/%.c=$(BUILD)/%.o)

# Backend seam: tensor.c implements the ops for the CPU backend; a CUDA
# backend supplies its own object for the same tensor.h interface.
BACKEND_OBJS_cpu  := $(BUILD)/tensor.o
COMMON_SRCS := src/gguf.c src/quant.c $(ISA_SRCS) src/tokenizer.c \
               src/model_gemma4.c src/model_qwen36.c src/model_llama.c \
               src/model_deepseek.c src/model.c src/sampler.c src/oxidize.c \
               src/vision.c
COMMON_OBJS := $(COMMON_SRCS:src/%.c=$(BUILD)/%.o)
OBJS        := $(COMMON_OBJS) $(BACKEND_OBJS_$(BACKEND))

all: oxidize-c

# Build for this machine only (bench box). The kernel TUs are unaffected: they
# always carry their own -m flags. Its own object dir, because make cannot see
# that CFLAGS changed and would otherwise reuse the portable objects.
native:
	$(MAKE) BUILD=build-native ARCH_FLAGS=-march=native

oxidize-c: $(OBJS) $(BUILD)/main.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

$(BUILD)/%.o: src/%.c $(wildcard src/*.h) | $(BUILD)
	$(CC) $(CFLAGS) -c -o $@ $<

# Per-ISA kernel objects: one TU per ISA level, each compiled with exactly the
# flags of that level, so the compiler may use the ISA throughout the kernel
# (F16C for the block scales, FMA in the reduction) and not only where an
# intrinsic spells it out. VNNI gets its own TU so the compiler cannot put a
# vpdpbusd into a kernel that has to run on an AVX-512 part without VNNI.
$(BUILD)/quant_avx2.o: src/quant_avx2.c $(wildcard src/*.h) | $(BUILD)
	$(CC) $(CFLAGS) $(AVX2_FLAGS) -c -o $@ $<
$(BUILD)/quant_avx512.o: src/quant_avx512.c $(wildcard src/*.h) | $(BUILD)
	$(CC) $(CFLAGS) $(AVX512_FLAGS) -c -o $@ $<
$(BUILD)/quant_vnni.o: src/quant_vnni.c $(wildcard src/*.h) | $(BUILD)
	$(CC) $(CFLAGS) $(VNNI_FLAGS) -c -o $@ $<

$(BUILD):
	mkdir -p $(BUILD)

# Embeddable C ABI (src/oxidize.h) as static + shared libraries. Both are built
# from a SEPARATE position-independent object tree ($(BUILD)/pic) so the CLI and
# the `native` bench binary keep their non-PIC objects — a shared object must be
# PIC, but the perf-critical CLI should not pay for it. Same sources as the CLI
# runtime (COMMON incl. the ISA kernels + oxidize.o) plus the CPU tensor backend.
PIC      := $(BUILD)/pic
LIB_OBJS := $(COMMON_SRCS:src/%.c=$(PIC)/%.o) $(PIC)/tensor.o

$(PIC)/%.o: src/%.c $(wildcard src/*.h) | $(PIC)
	$(CC) $(CFLAGS) -fPIC -c -o $@ $<
$(PIC)/quant_avx2.o: src/quant_avx2.c $(wildcard src/*.h) | $(PIC)
	$(CC) $(CFLAGS) $(AVX2_FLAGS) -fPIC -c -o $@ $<
$(PIC)/quant_avx512.o: src/quant_avx512.c $(wildcard src/*.h) | $(PIC)
	$(CC) $(CFLAGS) $(AVX512_FLAGS) -fPIC -c -o $@ $<
$(PIC)/quant_vnni.o: src/quant_vnni.c $(wildcard src/*.h) | $(PIC)
	$(CC) $(CFLAGS) $(VNNI_FLAGS) -fPIC -c -o $@ $<
$(PIC):
	mkdir -p $(PIC)

liboxidize.a: $(LIB_OBJS)
	$(AR) rcs $@ $^
liboxidize.so: $(LIB_OBJS)
	$(CC) $(CFLAGS) -shared -fPIC -o $@ $^ $(LDLIBS)

lib: liboxidize.a liboxidize.so

# Offline tools (tools/): one shared GGUF v3 writer + row encoders, three CLIs.
# The tool cores are also compiled -DOC_TOOLS_LIB (main() dropped) and linked
# into the unit-test binary, so tests/test_tools.c drives the real code.
$(BUILD)/tool_%.o: tools/%.c $(wildcard src/*.h) tools/gguf_write.h | $(BUILD)
	$(CC) $(CFLAGS) -c -o $@ $<
$(BUILD)/toollib_%.o: tools/%.c $(wildcard src/*.h) tools/gguf_write.h | $(BUILD)
	$(CC) $(CFLAGS) -DOC_TOOLS_LIB -c -o $@ $<

TOOL_RT   := $(BUILD)/gguf.o $(BUILD)/quant.o $(ISA_OBJS) $(BUILD)/tensor.o \
             $(BUILD)/tool_gguf_write.o
TOOL_LIB_OBJS := $(BUILD)/tool_gguf_write.o $(BUILD)/toollib_quantize.o \
                 $(BUILD)/toollib_prune.o $(BUILD)/toollib_merge.o \
                 $(BUILD)/toollib_convert.o

oxidize-c-quantize: $(TOOL_RT) $(BUILD)/tool_quantize.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)
oxidize-c-prune: $(TOOL_RT) $(BUILD)/tool_prune.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)
oxidize-c-merge: $(TOOL_RT) $(BUILD)/tool_merge.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)
# requant is quantize with its old default target (AL5_XS).
oxidize-c-requant: $(TOOL_RT) $(BUILD)/tool_quantize.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)
# convert: SafeTensors/HF -> GGUF (dense Llama-family; permutes q/k for llama).
oxidize-c-convert: $(TOOL_RT) $(BUILD)/tool_convert.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

quantize: oxidize-c-quantize
prune: oxidize-c-prune
merge: oxidize-c-merge
requant: oxidize-c-requant
convert: oxidize-c-convert
tools: quantize prune merge requant convert

TEST_OBJS := $(BUILD)/test_main.o $(BUILD)/test_quant.o $(BUILD)/test_model.o \
             $(BUILD)/test_sampler.o $(BUILD)/fuzz_gguf.o $(BUILD)/test_tools.o \
             $(BUILD)/test_vision.o $(TOOL_LIB_OBJS)

$(BUILD)/%.o: tests/%.c $(wildcard src/*.h) $(wildcard tests/*.h) | $(BUILD)
	$(CC) $(CFLAGS) -c -o $@ $<

$(BUILD)/oxidize-c-test: $(OBJS) $(TEST_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

# ABI acceptance test: its own main(), linked against the same runtime objects
# (which include oxidize.o) — drives src/oxidize.h end to end.
$(BUILD)/oxidize-c-abi-test: $(OBJS) $(BUILD)/test_abi.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

test: $(BUILD)/oxidize-c-test $(BUILD)/oxidize-c-abi-test
	$(TEST_ENV) $(RUNNER) $(BUILD)/oxidize-c-test ../oxidize-core/tests/fixtures/valid-v3.gguf
	$(TEST_ENV) $(RUNNER) $(BUILD)/oxidize-c-abi-test ../oxidize-core/tests/fixtures/valid-v3.gguf

# ASAN+UBSAN. OC_THREADS=1 is mandatory, not a nicety: the pool barrier spins
# 4096x before it yields, and a sanitized build spinning on every core takes
# minutes to do what it does in a second.
ASAN_CFLAGS := -std=c11 -D_DEFAULT_SOURCE -O1 -g $(ARCH_FLAGS) -Wall -Wextra \
               -pthread -fno-omit-frame-pointer \
               -fsanitize=address,undefined -fno-sanitize-recover=all
asan:
	$(MAKE) BUILD=build-asan CFLAGS="$(ASAN_CFLAGS)" \
	  TEST_ENV="OC_THREADS=1 UBSAN_OPTIONS=print_stacktrace=1" test

# libFuzzer over gguf_parse. `make fuzz FUZZ_ARGS="-runs=1000000"`.
FUZZ_ARGS ?= -max_total_time=60 -max_len=4096
fuzz: | $(BUILD)
	@command -v clang >/dev/null || { echo "make fuzz needs clang"; exit 1; }
	clang -std=c11 -D_DEFAULT_SOURCE -O1 -g -fno-omit-frame-pointer \
	  -fsanitize=fuzzer,address,undefined -fno-sanitize-recover=all \
	  -o $(BUILD)/fuzz-gguf src/gguf.c tests/fuzz_gguf.c
	mkdir -p $(BUILD)/corpus
	cp ../oxidize-core/tests/fixtures/*.gguf $(BUILD)/corpus/
	$(BUILD)/fuzz-gguf $(FUZZ_ARGS) $(BUILD)/corpus

$(BUILD)/pool-bench: $(BUILD)/tensor.o $(BUILD)/quant.o $(ISA_OBJS) $(BUILD)/pool_bench.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

pool-bench: $(BUILD)/pool-bench
	$(BUILD)/pool-bench

# oc_matmul vs n x oc_matvec at gemma-4-31B's real layer shapes, in RAM.
# `make gemm-bench BENCH_ARGS="q4k 8"`.
BENCH_ARGS ?=
$(BUILD)/gemm-bench: $(BUILD)/tensor.o $(BUILD)/quant.o $(ISA_OBJS) $(BUILD)/gemm_bench.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

gemm-bench: $(BUILD)/gemm-bench
	$(BUILD)/gemm-bench $(BENCH_ARGS)

# CUDA backend: nvcc for the kernels; gcc (pattern rule above) for main_cuda.c.
# The CPU objects are linked in too: gemma4_load() does all GGUF parsing and
# geometry, and the CUDA module uploads the still-quantized weights from mmap.
#
# MULTI-ARCH FATBIN, not -arch=sm_90a. nvcc cross-compiles happily, so a clean
# build proves nothing about what the binary can RUN: an sm_90a-only binary dies
# on anything but an H100 with "no kernel image is available for execution on the
# device". Ship SASS for every SM people actually have, plus a compute_90 PTX
# entry so a future SM JITs instead of failing. Override NVCC_GENCODE (or the
# whole NVCCFLAGS) to build for one arch and cut compile time.
NVCC      ?= nvcc
NVCC_GENCODE ?= -gencode arch=compute_70,code=sm_70 \
                -gencode arch=compute_75,code=sm_75 \
                -gencode arch=compute_80,code=sm_80 \
                -gencode arch=compute_86,code=sm_86 \
                -gencode arch=compute_89,code=sm_89 \
                -gencode arch=compute_90,code=sm_90 \
                -gencode arch=compute_90,code=compute_90
NVCCFLAGS ?= -O3 $(NVCC_GENCODE)

CUDA_HDRS := src/cuda/gemma4_cuda.h src/cuda/llama_cuda.h src/cuda/cuda_dequant.cuh
$(BUILD)/gemma4_cuda.o: src/cuda/gemma4_cuda.cu $(CUDA_HDRS) $(wildcard src/*.h) | $(BUILD)
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<
$(BUILD)/llama_cuda.o: src/cuda/llama_cuda.cu $(CUDA_HDRS) $(wildcard src/*.h) | $(BUILD)
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<

oxidize-c-cuda: $(COMMON_OBJS) $(BUILD)/tensor.o $(BUILD)/main_cuda.o \
                $(BUILD)/gemma4_cuda.o $(BUILD)/llama_cuda.o
	$(NVCC) $(NVCCFLAGS) -o $@ $^ -lm -lpthread

cuda: oxidize-c-cuda

# THE CUDA ACCEPTANCE TEST. Runs the same synthetic gemma4 model through
# gemma4_forward (CPU) and through the CUDA backend and compares every logit —
# per weight type, per -ngl split. Needs a real GPU: `modal run modal_cuda.py
# --action gputest`. A kernel that has not passed this is not done.
$(BUILD)/oxidize-c-cuda-test: $(COMMON_OBJS) $(BUILD)/tensor.o \
                              $(BUILD)/cuda_equiv.o $(BUILD)/gemma4_cuda.o \
                              $(BUILD)/llama_cuda.o
	$(NVCC) $(NVCCFLAGS) -o $@ $^ -lm -lpthread

cuda-test: $(BUILD)/oxidize-c-cuda-test
	$(BUILD)/oxidize-c-cuda-test

# Interactive chat TUI (Go, stdlib only) wrapping oxidize-c --chat.
tui: oxidize-c
	cd tui && go build -o oxidize-c-tui .

clean:
	rm -rf build build-asan build-native oxidize-c oxidize-c-cuda oxidize-c-requant \
	       oxidize-c-quantize oxidize-c-prune oxidize-c-merge oxidize-c-convert \
	       liboxidize.a liboxidize.so tui/oxidize-c-tui

.PHONY: all native test asan fuzz pool-bench gemm-bench cuda cuda-test requant quantize \
        prune merge convert tools tui lib clean
