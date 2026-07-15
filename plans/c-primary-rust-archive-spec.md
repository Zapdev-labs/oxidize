# Spec: Make the Pure C Port the Primary Oxidize Runtime and Archive Rust Safely

## Status

Draft — planning only. This spec targets `oxidize-c/`, not `oxidize-cpp/`. It defines the work required before the Rust workspace can be archived. It does **not** authorize immediate deletion of Rust crates.

## Problem

Oxidize is still Rust-first. The Rust workspace owns the canonical core, CLI, server, Python bindings, conversion, pruning, merge, training, FFI, kernel, backend, and test surfaces.

The pure C port currently exists as `oxidize-c/`. It is promising, small, dependency-light, and closer to the desired minimal deployment shape, but it is not yet a full Rust replacement.

Current `oxidize-c/` evidence:

- `Makefile` describes it as: “pure C11 CPU inference for Gemma 4 GGUF.”
- Builds a CPU binary with `make`.
- Runs unit tests with `make test` against `../oxidize-core/tests/fixtures/valid-v3.gguf`.
- Has an experimental CUDA binary with `make cuda`.
- Has a Go stdlib TUI wrapper around `oxidize-c --chat`.
- Supports one-shot prompt mode, chat mode, metadata inspection, benchmark mode, sampling options, context/thread flags, and optional rotated int4 KV cache.
- Main model implementation is Gemma 4-specific: `model_gemma4.c` / `model_gemma4.h`.
- Includes GGUF parsing, quantization, tokenizer, sampler, tensor operations, unit tests, and a requant tool.

That is not enough to replace Rust today, but it can become enough if the C port is expanded into the primary product runtime.

## Goal

Make `oxidize-c/` the primary production implementation of Oxidize and eventually archive the Rust workspace after verified parity.

The target end state:

- `oxidize-c/` provides the default `oxidize` CLI/runtime.
- Rust is retained only as archived legacy/reference code.
- The default install/build/test path does not require Rust.
- Core inference, serving, model tooling, and supported binding workflows have C-based or explicitly supported non-Rust replacements.

## Non-goals

- Do not use `oxidize-cpp/` as the primary target for this migration.
- Do not delete Rust immediately.
- Do not remove Go or Python ports as part of this change.
- Do not require all historical experimental Rust features to survive if maintainers explicitly deprecate them.
- Do not claim C parity without tests, docs, and benchmarks.

## Product Decision

Choose `oxidize-c/` as the primary long-term runtime.

Rationale:

- Pure C11 is highly portable and easy to embed.
- The C port is simpler to distribute than Rust or C++.
- It already has the beginnings of the desired runtime: GGUF, tokenizer, sampler, tensor ops, Gemma 4 inference, chat, inspect, benchmark, requant, tests, and CUDA experiment.
- It can serve as a compact llama.cpp-style core while Go/Python wrappers remain separate.

Tradeoff:

- The C port is much less complete than Rust today. This migration is a substantial product rewrite, not a cleanup.

## Migration Principles

1. **C becomes primary only after gates pass**.
2. **Rust remains available during the migration**.
3. **Archive Rust, do not delete it blindly**.
4. **One user-facing `oxidize` command remains the goal**.
5. **Unsupported Rust-only features must be explicitly deprecated, not silently lost**.
6. **Keep fixtures and compatibility tests language-neutral before moving Rust**.
7. **Avoid C++ dependency creep inside the pure C runtime**.

## Required Parity Gates

### Gate 1 — Core Runtime Completeness

`oxidize-c/` must support production inference without relying on Rust.

Required:

- GGUF v2/v3 parser robustness.
- Safe error handling for malformed GGUF files.
- Tokenizer encode/decode from normal text prompts.
- Chat template support for supported model families.
- Streaming text generation.
- Reusable KV cache for interactive chat.
- Context overflow handling.
- Deterministic seed behavior.
- Thread control.
- Benchmark mode.
- Metadata inspection.

Sampling support must include:

- greedy
- temperature
- top-k
- top-p
- min-p
- repeat penalty
- frequency penalty if product-required
- presence penalty if product-required

### Gate 2 — Model Architecture Coverage

Current C is Gemma 4-specific. Rust cannot be archived until model support is intentionally resolved.

Minimum required decision:

- Either C supports all product-supported Rust model families, or the project formally narrows supported architectures.

Recommended migration sequence:

1. Keep Gemma 4 path stable.
2. Add Llama-family GGUF support.
3. Add architecture detection and clean unsupported-model errors.
4. Add a model dispatch layer so `main.c` does not bind directly to only `Gemma4Model`.
5. Add tests for architecture rejection and architecture selection.

The C API should move toward:

```c
typedef struct OxModel OxModel;

typedef enum {
  OX_ARCH_GEMMA4,
  OX_ARCH_LLAMA,
  OX_ARCH_UNSUPPORTED
} OxArchitecture;

int ox_model_open(OxModel** out, const char* path, const OxModelOptions* opts,
                  char* err, size_t errlen);
float* ox_model_forward(OxModel* model, int32_t token, size_t pos, bool need_logits);
void ox_model_free(OxModel* model);
```

This keeps model-specific internals behind a stable C runtime interface.

### Gate 3 — Quantization and Tensor Coverage

C must support the quantized formats needed by supported models.

Required:

- List every Rust-supported quantization type.
- List every C-supported quantization type.
- Decide which formats are required for primary support.
- Add tests for every supported dequant/dot path.
- Add failure messages for unsupported formats.

Current C already includes Q4_0 and AL5_XS tests. Expand coverage based on the final supported model set.

### Gate 4 — CLI Parity

The C binary must become the default `oxidize` UX.

Required CLI behavior:

- `oxidize --model path.gguf --prompt "..."`
- `oxidize --chat --model path.gguf`
- `oxidize --inspect --model path.gguf`
- `oxidize --bench --model path.gguf`
- `--max-tokens`
- `--ctx`
- `--threads`
- `--seed`
- sampling flags
- `--raw`
- clear usage/errors

If the Rust command shape `oxidize run <model>` is still desired, add compatibility aliases in C or provide a wrapper.

Recommended:

- Keep `oxidize-c` as the internal binary name during migration.
- Add an install target or wrapper that exposes it as `oxidize` only after CLI parity passes.

### Gate 5 — Server/API Replacement

Rust currently includes an OpenAI-compatible server. `oxidize-c/` does not yet provide an equivalent native C server.

Before Rust archival, choose one:

1. Implement an HTTP server in C.
2. Keep a Go/Python server wrapper around the C runtime.
3. Deprecate the built-in server and document an external wrapper.

If the product continues to promise OpenAI-compatible serving, required endpoints are:

- health/readiness endpoint
- `/v1/models`
- `/v1/chat/completions`
- streaming SSE for chat completions
- `/v1/completions` if still supported

Server contract tests must be added before Rust server archival.

### Gate 6 — Tooling Replacement

Rust crates currently provide more than inference. Every crate/tool needs a fate.

| Rust crate/tool | Required C-era decision |
|---|---|
| `oxidize-core` | Replace with `oxidize-c/src` runtime library |
| `oxidize-cli` | Replace with C CLI |
| `oxidize-server` | Replace, wrap, or deprecate |
| `oxidize-quantize` | Replace with C requant/quantize tools or deprecate |
| `oxidize-convert` | Replace SafeTensors/HF conversion or keep legacy |
| `oxidize-prune` | Replace or mark legacy |
| `oxidize-merge` | Replace or mark legacy |
| `oxidize-train` | Replace or mark legacy |
| `oxidize-finetuning` | Replace or mark legacy |
| `oxidize-py` | Replace with C bindings, pure Python, or legacy |
| `oxidize-ffi` | Replace with stable C ABI from `oxidize-c` |
| `oxidize-kernels` | Replace with C kernels or keep only as archived reference |

Acceptance criteria:

- No Rust-owned user workflow disappears without an explicit deprecation note.

### Gate 7 — C Library ABI

The C port should expose a stable embeddable library, not just a CLI.

Required:

- Public header, e.g. `include/oxidize.h` or `oxidize-c/src/oxidize.h`.
- Opaque model/session types.
- Stable error reporting.
- Options structs with version/size fields for ABI evolution.
- Streaming callback API.
- No global mutable configuration where avoidable.
- Thread-safety rules documented.

Example direction:

```c
typedef struct OxRuntime OxRuntime;
typedef struct OxModel OxModel;
typedef struct OxSession OxSession;

typedef struct {
  size_t struct_size;
  size_t ctx;
  int threads;
  uint64_t seed;
  int kv_quant;
} OxModelOptions;

typedef int (*OxTokenCallback)(const char* text, size_t len, void* user);
```

### Gate 8 — Python/FFI Replacement

Before archiving Rust:

- Decide whether Python should use pure `oxidize-python`, direct C FFI, or both.
- If direct C FFI is supported, add ctypes/cffi bindings and tests.
- Document installation without Rust.

### Gate 9 — Hardware Backend Story

Current C has CPU and experimental CUDA.

Before archival, decide each Rust backend fate:

- CUDA: implement/stabilize in C or mark experimental.
- Metal: replace or deprecate.
- Vulkan: replace or deprecate.
- MLX: replace or deprecate.
- WebGPU/WASM: replace or deprecate.

If the new primary product is CPU + CUDA only, update docs and release notes honestly.

### Gate 10 — Tests and CI

C must have CI that is strong enough to replace Rust CI.

Required C tests:

- GGUF fixture parse.
- Malformed GGUF rejection.
- Tokenizer round trips.
- Quantization/dequantization correctness.
- Tensor op correctness.
- Sampler determinism.
- CLI smoke tests.
- Chat-template tests.
- Architecture detection tests.
- At least one tiny runnable model integration test if licensing allows.
- Optional CUDA tests behind hardware-gated CI.

Required commands:

```bash
cd oxidize-c
make clean
make
make test
```

During migration, keep Rust validation:

```bash
sfw cargo test -p oxidize-core -p oxidize-cli -p oxidize-server -p oxidize-kernels
```

### Gate 11 — Benchmarks

Before making C primary:

- Benchmark C vs Rust for supported models.
- Benchmark prompt prefill and decode separately.
- Record model, quantization, CPU, threads, NUMA policy, compiler, flags, and commit.
- Benchmark memory usage.
- Benchmark chat/server path if server remains supported.

Minimum recommended threshold:

- C is no slower than Rust for the selected primary CPU model path.
- C memory usage is equal or better.
- C has no correctness regressions in tokenizer/sampling smoke tests.

## Implementation Plan

### Phase 0 — Formalize the C Target

Deliverables:

- Accept this spec or a revised version.
- Decide whether `oxidize-cpp/` remains experimental/performance-only.
- Create a detailed Rust-to-C parity checklist.
- Move shared fixtures out of Rust-owned paths if C/Go/Python depend on them.

Acceptance criteria:

- The project has an explicit written decision that `oxidize-c/` is the intended primary runtime.

### Phase 1 — C Runtime Library Boundary

Deliverables:

- Add a stable public C header.
- Introduce opaque model/session types.
- Refactor Gemma 4-specific CLI coupling behind a model dispatch layer.
- Keep existing CLI behavior working.

Acceptance criteria:

- CLI uses the public-ish runtime boundary instead of directly owning all model-specific flow.
- Tests still pass with `make test`.

### Phase 2 — Architecture Expansion

Deliverables:

- Add Llama-family model support or formally document why Gemma 4 is the only supported primary architecture.
- Add architecture detection.
- Add unsupported architecture errors.
- Add tests.

Acceptance criteria:

- Supported production model set is clear and tested.

### Phase 3 — CLI/Product Parity

Deliverables:

- Align C CLI with Rust product-critical flags.
- Add compatibility aliases where needed.
- Add CLI smoke tests.
- Add install target that can create an `oxidize` binary.

Acceptance criteria:

- Common Rust CLI workflows work through C or have documented replacements.

### Phase 4 — Server Decision and Implementation

Deliverables:

- Choose native C server, Go/Python wrapper, or deprecation.
- If preserving server, implement OpenAI-compatible contract tests.
- Add streaming support if required.

Acceptance criteria:

- Users have a supported non-Rust answer for API serving.

### Phase 5 — Tooling and Bindings

Deliverables:

- Replace or deprecate quantize/convert/prune/merge/train/ffi/py workflows.
- Add C requant/quantize docs and tests.
- Add Python/FFI strategy.

Acceptance criteria:

- Every Rust crate has a documented successor or deprecation decision.

### Phase 6 — Docs and Default Switch

Deliverables:

- Update README, QUICKSTART, HOW_TO_INSTALL, CONTRIBUTING, AGENTS.md, and Dockerfiles.
- Make C the default build/install path.
- Keep Rust legacy instructions.

Acceptance criteria:

- New users build and run C by default.
- Existing Rust users can still find legacy instructions.

### Phase 7 — Deprecation Window

Deliverables:

- Announce Rust legacy status.
- Keep Rust CI for critical crates for a defined period.
- Stop adding Rust features except critical fixes.

Acceptance criteria:

- No blocking user workflow remains Rust-only unless deprecated.

### Phase 8 — Archive Rust

Deliverables:

- Move Rust workspace to `archive/rust/` or preserve it on a legacy branch/tag.
- Remove Rust from default build and docs.
- Keep shared fixtures outside the archived tree.
- Keep migration notes.

Acceptance criteria:

- Main repo default flow is C-first and Rust-free.
- Archived Rust is discoverable and restorable.

## Documentation Changes Required

Update:

- `README.md`
- `QUICKSTART.md`
- `HOW_TO_INSTALL.md`
- `CONTRIBUTING.md`
- root `AGENTS.md`
- `oxidize-c/CUDA.md`
- Dockerfiles
- Any release/changelog docs

Docs must state:

- C is the primary runtime.
- Rust is legacy during deprecation.
- Supported model architectures.
- Supported hardware backends.
- Supported tools and deprecated tools.
- Migration examples from Rust commands to C commands.

## CI Changes Required

Add default C CI:

```bash
cd oxidize-c
make clean
make
make test
```

Add CLI smoke tests, for example:

```bash
./oxidize-c --inspect --model ../oxidize-core/tests/fixtures/valid-v3.gguf
```

Add optional CUDA CI only on CUDA runners:

```bash
cd oxidize-c
make cuda
```

Keep Rust CI until archival gates pass.

## Risks

- C port is currently model-narrow, so migration may take longer than expected.
- Server functionality could regress or disappear.
- Rust-only conversion/pruning/binding workflows may be underestimated.
- C memory-safety bugs can replace Rust compile-time guarantees.
- Hardware backend coverage may shrink.
- Shared tests currently live under `oxidize-core`, which complicates archiving.

## Risk Mitigations

- Add fuzzing for GGUF/tokenizer parsing.
- Use sanitizers in C CI:

```bash
CFLAGS='-std=c11 -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer -Wall -Wextra -pthread' make clean test
```

- Keep Rust until C has real integration tests.
- Keep a legacy branch/tag before moving files.
- Do staged documentation changes, not a one-shot rewrite.

## Rollback Plan

Before Phase 8:

- Revert docs/build defaults to Rust.
- Keep C as experimental runtime.

After Phase 8:

- Restore Rust from `archive/rust/` or legacy branch/tag.
- Re-enable Rust CI if a critical C blocker appears.

## Initial Checklist

- [ ] Approve `oxidize-c/` as the intended primary runtime.
- [ ] Decide whether C++ remains non-primary.
- [ ] Create Rust-crate-to-C-successor checklist.
- [ ] Add stable C API boundary.
- [ ] Add model dispatch layer instead of Gemma 4-only CLI coupling.
- [ ] Decide supported model architecture set.
- [ ] Add Llama support or formally narrow support.
- [ ] Add server replacement plan.
- [ ] Add CLI compatibility tests.
- [ ] Add sanitizer CI.
- [ ] Move shared fixtures out of Rust-owned tree.
- [ ] Benchmark C vs Rust.
- [ ] Update docs only after gates are met.
- [ ] Archive Rust only after deprecation window and maintainer sign-off.
