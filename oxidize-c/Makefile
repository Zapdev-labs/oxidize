# oxidize-c Makefile — canonical build system.
#
# Targets:
#   make / make build   — default CPU-only, dependency-free binary
#   make lib           — static library liboxidize-c.a
#   make test          — build + run in-repo test_runner (ASan + UBSan)
#   make lint          — clang-tidy on library sources
#   make clean         — remove build artifacts
#   make avx512        — AVX-512 build (for .121 Cascade Lake)
#   make cuda          — CUDA backend build (OC_CUDA, for Modal L40S)
#
# C standard: C11 (-std=c11). No GNU extensions unless gated behind #ifdef.
# Baseline links only libc + libm + libpthread.

CC       ?= gcc
# Required public-header paths live in CPPFLAGS so a caller CFLAGS override
# cannot break the build.
CPPFLAGS += -Iinclude
CFLAGS   ?= -std=c11 -Wall -Wextra -Werror -O2
CFLAGS   += -ffp-contract=off
LDFLAGS  ?= -lm -lpthread
DEPFLAGS := -MMD -MP

# Library sources for the core, utilities, formats, model, compute, server,
# paged attention, backends, vision, mesh, and testable CLI arguments.
LIB_SRCS := $(wildcard src/core/*.c src/util/*.c src/format/*.c src/model/*.c \
                       src/compute/*.c src/autotune/*.c src/server/*.c \
                       src/paged/*.c src/backends/cuda_stub.c \
                       src/backends/gpu_dispatch.c \
                       src/backends/backend.c \
                       src/vision/*.c src/mesh/*.c \
                       src/cluster/*.c src/validation/*.c src/video/*.c) \
                       src/cli/args.c src/cli/commands.c
LIB_OBJS := $(LIB_SRCS:.c=.o)

# Binary source for the CLI entrypoint.
BIN_SRCS := src/cli/main.c
BIN_OBJS := $(BIN_SRCS:.c=.o)

# Default: build the binary.
build: oxidize-c

oxidize-c: $(LIB_OBJS) $(BIN_OBJS)
	$(CC) $(CPPFLAGS) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Static library (useful for downstream linking + the CLI/server features).
lib: liboxidize-c.a
liboxidize-c.a: $(LIB_OBJS)
	ar rcs $@ $^

# In-repo test framework (tests/framework.h) — dependency-free drop-in
# for the Criterion subset this suite uses. Each tests/test_<module>.c
# uses the Test(Suite, Case) macro for auto-registration;
# tests/framework_main.c provides main(). One test_<module>.c per
# src/<module>.c (see CONTRIBUTING.md).
TEST_FRAMEWORK_OBJ := tests/framework_main.o

TEST_SRCS := $(wildcard tests/test_*.c)
TEST_OBJS := $(TEST_SRCS:.c=.o)

# ASan + UBSan for leak/UB detection in tests (valgrind not installed locally).
# The runner needs pthreads and math. Linux additionally needs librt for
# POSIX shared memory; macOS provides the same APIs through libSystem and
# has no separate librt.
TEST_CFLAGS := -std=c11 -Wall -Wextra -Wno-format-zero-length \
               -Wno-override-init \
               -fsanitize=address,undefined -fno-omit-frame-pointer -g -O0 \
               -ffp-contract=off \
               -Itests
TEST_SYSTEM_LIBS := -lm -lpthread -ldl
ifneq ($(shell uname -s),Darwin)
TEST_SYSTEM_LIBS += -lrt
endif
TEST_LDFLAGS := $(TEST_SYSTEM_LIBS) -fsanitize=address,undefined

# ASan-instrumented library objects (stems: src/core/error.c -> src/core/error.asan.o).
LIB_ASAN_OBJS := $(LIB_SRCS:.c=.asan.o)
DEP_FILES := $(LIB_OBJS:.o=.d) $(BIN_OBJS:.o=.d) \
             $(TEST_OBJS:.o=.d) $(LIB_ASAN_OBJS:.o=.d) \
             $(TEST_FRAMEWORK_OBJ:.o=.d)

test_runner: $(TEST_OBJS) $(TEST_FRAMEWORK_OBJ) $(LIB_ASAN_OBJS)
	$(CC) $(CPPFLAGS) $(TEST_CFLAGS) -o $@ $(TEST_OBJS) $(TEST_FRAMEWORK_OBJ) $(LIB_ASAN_OBJS) $(TEST_LDFLAGS)

# Build library objects with ASan for the test_runner.
src/%.asan.o: src/%.c
	$(CC) $(CPPFLAGS) $(TEST_CFLAGS) $(DEPFLAGS) -c -o $@ $<

tests/%.o: tests/%.c
	$(CC) $(CPPFLAGS) $(TEST_CFLAGS) $(DEPFLAGS) -c -o $@ $<

# Default object rule (no ASan).
src/%.o: src/%.c
	$(CC) $(CPPFLAGS) $(CFLAGS) $(DEPFLAGS) -c -o $@ $<

-include $(DEP_FILES)

# Test filtering: `make test TEST_FILTER=<pattern>` runs only matching tests.
# Pattern conventions:
#   * suite/case or suite/case*     → passed through to the runner
#   * a suite name (error, kv_cache) → runs that suite
#   * test_<suite>_<case>            → test_ prefix stripped; each '_' may
#     stand in for the suite/case slash (test_kv_cache_init → kv_cache/init)
# When TEST_FILTER is empty (default), all tests run.
TEST_FILTER ?=
# --jobs is accepted by the runner but runs stay serial. Fork-per-test is
# the isolation boundary; parallel jobs deadlock under ASan on macOS.
TEST_RUNNER_ARGS :=

test: test_runner
	@if [ -z "$(TEST_FILTER)" ]; then \
		./test_runner $(TEST_RUNNER_ARGS); \
	else \
		f="$(TEST_FILTER)"; \
		f="$${f#test_}"; \
		./test_runner $(TEST_RUNNER_ARGS) --filter "$$f"; \
	fi

test-avx512: TEST_CFLAGS += -DOC_AVX512 -mavx512f -mavx512bw -mavx512dq \
                              -mavx512vl -mavx512vnni
test-avx512: clean test

# Lint: clang-tidy on library sources. Focused on bug-finding checks;
# stylistic checks (short names, braces-around-statements, etc.) are excluded
# to match the project's C style. The primary gate is -Wall -Wextra -Werror.
LINT_SOURCES := $(LIB_SRCS) $(BIN_SRCS)
LINT_CHECKS  := 'bugprone-*,clang-analyzer-*,performance-*,cert-*,misc-*,-misc-include-cleaner,-llvmlibc-*,-clang-analyzer-security.insecureAPI.DeprecatedOrUnsafeBufferHandling,-cert-err33-c,-bugprone-reserved-identifier,-bugprone-implicit-widening-of-multiplication-result,-bugprone-narrowing-conversions,-cert-dcl37-c,-cert-dcl51-cpp,-cert-int09-c,-performance-no-int-to-ptr'
lint:
	@if command -v clang-tidy >/dev/null 2>&1; then \
		echo "=== clang-tidy on $(LINT_SOURCES) ==="; \
		clang-tidy --checks=$(LINT_CHECKS) $(LINT_SOURCES) -- -std=c11 -Iinclude 2>&1; \
		status=$$?; \
		echo "=== clang-tidy exit: $$status ==="; \
		exit $$status; \
	else \
		echo "clang-tidy not found; build with -Wall -Wextra -Werror is the primary gate"; \
	fi

# AVX-512 build (for .121 Cascade Lake).
avx512: CFLAGS += -DOC_AVX512 -mavx512f -mavx512bw -mavx512dq -mavx512vl -mavx512vnni
avx512: clean oxidize-c

# CUDA build (for Modal L40S). NVCC compiles .cu files; cuda_stub.c excluded.
CUDA_LIB_SRCS := $(filter-out src/backends/cuda_stub.c,$(LIB_SRCS))
CUDA_LIB_OBJS := $(CUDA_LIB_SRCS:.c=.o)
CUDA_SRCS := $(wildcard src/backends/*.cu)
# `.cuo`, not `.cu.o`: the latter collides with make's builtin `%: %.o` link
# rule and produces "Circular src/backends/cuda.cu <- ...cu.o" warnings.
CUDA_OBJS := $(CUDA_SRCS:.cu=.cuo)

CUDA_HOME ?= /usr/local/cuda

# Target architectures. Without -gencode, nvcc emits sm_52 only, so every
# kernel is JIT-recompiled at load on a modern card and sm_61+ intrinsics
# (notably __dp4a, which the quantized matvec wants) are unavailable. Ship a
# fatbin covering the cards we actually run on: Pascal (61), T4 (75),
# A100 (80), A10G/A10 (86), L4/L40S/Ada (89), H100 (90). sm_86 would otherwise
# fall back to the sm_80 cubin via minor-version compatibility, which works but
# is not tuned for it.
# Override with e.g. `make cuda CUDA_ARCHS=89` for a faster single-arch build.
CUDA_ARCHS ?= 61 75 80 86 89 90
NVCC_GENCODE := $(foreach a,$(CUDA_ARCHS),-gencode arch=compute_$(a),code=sm_$(a))
# PTX from the *lowest* arch, not the highest: PTX JITs forward onto any newer
# card, so compute_61 PTX covers everything above it. Emitting it for the
# highest arch instead would only help cards newer than our newest target.
NVCC_GENCODE += -gencode arch=compute_$(firstword $(CUDA_ARCHS)),code=compute_$(firstword $(CUDA_ARCHS))

cuda: CFLAGS := -std=c11 -Wall -Wextra -Werror -O2 -ffp-contract=off -DOC_CUDA
# -std=c11 deliberately omitted from -Xcompiler: nvcc drives a C++ host
# compiler for .cu files, which warns that c11 is not a valid C++ standard.
# -Werror applies to the host side so `make cuda` is as strict as `make build`.
cuda: NVCC_FLAGS := -O2 -Iinclude -DOC_CUDA $(NVCC_GENCODE) \
                    -Xcompiler -Wall,-Wextra,-Werror
# -no-pie: nvcc emits non-PIC device-stub objects, but distributions that
# default gcc to PIE (Ubuntu 22.04, which the CUDA images are built on) then
# refuse to link them against the PIE C objects:
#   relocation R_X86_64_32 against `.rodata.str1.1' can not be used when
#   making a PIE object
# Linking the executable non-PIE is the right side to fix: it costs only ASLR
# on the binary itself, whereas rebuilding every C object as -fPIE just to
# accommodate the device stubs would apply to the CPU build too.
cuda: LDFLAGS := -lm -lpthread -L$(CUDA_HOME)/lib64 -lcudart -lstdc++ -no-pie
cuda: $(CUDA_LIB_OBJS) $(CUDA_OBJS) $(BIN_OBJS)
	gcc $(CPPFLAGS) $(CFLAGS) -no-pie -o oxidize-c $(CUDA_LIB_OBJS) $(CUDA_OBJS) $(BIN_OBJS) $(LDFLAGS)

%.cuo: %.cu
	nvcc $(NVCC_FLAGS) -c -o $@ $<

clean:
	rm -f $(LIB_OBJS) $(BIN_OBJS) $(TEST_OBJS) \
	      $(LIB_ASAN_OBJS) $(CUDA_OBJS) $(TEST_FRAMEWORK_OBJ) \
	      $(DEP_FILES) \
	      oxidize-c test_runner liboxidize-c.a

.PHONY: build lib test lint clean avx512 cuda
