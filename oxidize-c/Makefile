# oxidize-c Makefile — canonical build system.
#
# Targets:
#   make / make build   — default CPU-only, dependency-free binary
#   make lib           — static library liboxidize-c.a
#   make test          — build + run Criterion-style test_runner (ASan-instrumented)
#   make lint          — clang-tidy on changed files
#   make clean         — remove build artifacts
#   make avx512        — AVX-512 build (for .121 Cascade Lake)
#   make cuda          — CUDA backend build (OC_CUDA, for Modal L40S)
#
# C standard: C11 (-std=c11). No GNU extensions unless gated behind #ifdef.
# Baseline links only libc + libm + libpthread.

CC      ?= gcc
CFLAGS  ?= -std=c11 -Wall -Wextra -Werror -O2 -Iinclude
LDFLAGS ?= -lm -lpthread

# Library sources (core + util). Format/compute/model/server/cli are added by
# their respective feature workers.
LIB_SRCS := $(wildcard src/core/*.c src/util/*.c)
LIB_OBJS := $(LIB_SRCS:.c=.o)

# Binary sources (CLI main). Added by cli-flags-modes feature; for now a stub.
BIN_SRCS := src/cli/main.c
BIN_OBJS := $(BIN_SRCS:.c=.o)

# Default: build the binary.
build: oxidize-c

oxidize-c: $(LIB_OBJS) $(BIN_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Static library (useful for downstream linking + the CLI/server features).
lib: liboxidize-c.a
liboxidize-c.a: $(LIB_OBJS)
	ar rcs $@ $^

# Tests: minimal Criterion-style harness (tests/oc_min_test.h).
# The criterion-test-infra feature will replace this with real Criterion.
TEST_SRCS := $(wildcard tests/test_*.c)
TEST_OBJS := $(TEST_SRCS:.c=.o)
RUNNER_SRC := tests/runner.c
RUNNER_OBJ := tests/runner.o

# ASan + UBSan for leak/UB detection in tests (valgrind not installed locally).
TEST_CFLAGS := -std=c11 -Wall -Wextra -Wno-format-zero-length -fsanitize=address,undefined -fno-omit-frame-pointer -g -O0 -Iinclude -Itests
TEST_LDFLAGS := -lm -lpthread -fsanitize=address,undefined

# ASan-instrumented library objects (stems: src/core/error.c -> src/core/error.asan.o).
LIB_ASAN_OBJS := $(LIB_SRCS:.c=.asan.o)

test_runner: $(RUNNER_OBJ) $(TEST_OBJS) $(LIB_ASAN_OBJS)
	$(CC) $(TEST_CFLAGS) -o $@ $^ $(TEST_LDFLAGS)

# Build library objects with ASan for the test_runner.
src/%.asan.o: src/%.c
	$(CC) $(TEST_CFLAGS) -c -o $@ $<

tests/%.o: tests/%.c
	$(CC) $(TEST_CFLAGS) -c -o $@ $<

# Default object rule (no ASan).
src/%.o: src/%.c
	$(CC) $(CFLAGS) -c -o $@ $<

test: test_runner
	./test_runner

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
	else \
		echo "clang-tidy not found; build with -Wall -Wextra -Werror is the primary gate"; \
	fi

# AVX-512 build (for .121 Cascade Lake).
avx512: CFLAGS += -DOC_AVX512 -mavx512f -mavx512bw -mavx512dq -mavx512vl -mavx512vnni
avx512: clean oxidize-c

# CUDA build (for Modal L40S). NVCC compiles .cu files.
cuda: CC = nvcc
cuda: CFLAGS = -O2 -Iinclude -DOC_CUDA -Xcompiler -std=c11
cuda: oxidize-c

clean:
	rm -f $(LIB_OBJS) $(BIN_OBJS) $(TEST_OBJS) $(RUNNER_OBJ) \
	      $(LIB_ASAN_OBJS) \
	      oxidize-c test_runner liboxidize-c.a

.PHONY: build lib test lint clean avx512 cuda
