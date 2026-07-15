# Spec: Make C/C++ the Primary Oxidize Runtime and Archive Rust Safely

## Status

Draft — planning only. This spec defines the work required before the Rust implementation can be archived. It does **not** authorize immediate deletion of Rust crates.

## Problem

Oxidize currently treats Rust as the canonical implementation. The workspace README, quickstart, install docs, release narrative, and internal agent notes all point users and contributors to the Rust crates as the main product. The C and C++ ports exist, but they are not yet complete replacements:

- `oxidize-c/` is a narrow C implementation with GGUF, tensor, quantization, tokenizer, sampler, Gemma4-specific model code, CUDA experiments, tests, and a TUI.
- `oxidize-cpp/` is a larger Llama-family inference/performance port with CLI, server, CUDA/ROCm paths, autotune, kernels, quantization, tokenizer pieces, and training scaffolding.
- Rust still owns broad product functionality: core model infrastructure, CLI behavior, OpenAI-compatible server, Python bindings, FFI, SafeTensors-to-GGUF conversion, quantization tooling, merge/prune/training crates, distributed mesh, paged attention, WASM/WebGPU/Metal/Vulkan/MLX paths, and the primary test/documentation surface.

Deleting or archiving Rust now would remove product capabilities and contributor knowledge. To change that, the project needs an explicit migration plan, parity gates, and a staged archival process.

## Goal

Make `oxidize-cpp/` the primary production runtime for local LLM inference while preserving Rust until C/C++ reaches verified product parity. After parity is reached, move Rust crates into an archived/legacy state instead of deleting them outright.

## Non-goals

- Do not delete Rust crates in the first migration PR.
- Do not break existing Rust users without a deprecation window.
- Do not require `oxidize-c/` to become the primary runtime if `oxidize-cpp/` is selected as the replacement.
- Do not remove Go or Python ports as part of this migration.
- Do not claim parity without automated tests and benchmark evidence.

## Decision

Use `oxidize-cpp/` as the target primary implementation.

Rationale:

- It already has a CMake build, CLI, OpenAI-compatible server binary, CUDA/ROCm options, autotune, model loading, kernels, tokenizer support, and tests.
- It is closer to llama.cpp-style deployment and performance priorities.
- `oxidize-c/` should remain a small C experiment/reference unless separately expanded.

## Migration Principles

1. **Parity before archival**: Rust remains supported until C++ passes defined parity gates.
2. **Archive, do not delete**: Once replaced, Rust moves to `archive/rust/` or a dedicated legacy branch/tag with clear docs.
3. **One public `oxidize` UX**: The command users run should remain `oxidize`; implementation language should be an internal detail.
4. **Benchmark-backed claims**: Any replacement claim must include CPU and GPU benchmark results where applicable.
5. **No regression in server API**: OpenAI-compatible endpoints must remain behaviorally compatible.
6. **Preserve conversion/tooling workflows**: Model conversion, inspection, quantization, prune/merge workflows need replacements or documented external alternatives.

## Required Parity Matrix

### Core Inference

C++ must support:

- GGUF v2/v3 metadata and tensor loading.
- Llama-family dense models at minimum.
- Correct BOS/EOS handling.
- Prompt tokenization and detokenization from text, not only token-id benchmark paths.
- Streaming decode.
- Sampling parity:
  - greedy
  - temperature
  - top-k
  - top-p
  - min-p
  - frequency penalty
  - presence penalty
  - deterministic seed behavior
- KV cache lifecycle and reset behavior.
- Quantized inference for the currently supported Rust quantization set or a documented subset accepted by maintainers.

### Model Architecture Coverage

Before Rust archival, maintainers must choose one of these gates:

- **Strict gate**: C++ supports every Rust-supported model architecture.
- **Product gate**: C++ supports the architectures used by supported production models, and unsupported Rust-only architectures are formally deprecated.

At minimum, the spec recommends supporting:

- Llama-family GGUF models.
- Gemma-family models if currently marketed or used by `oxidize-c`.
- Architecture rejection with clear error messages for unsupported models.

### Hardware Backend Coverage

C++ must have documented behavior for:

- CPU inference.
- CUDA or ROCm, depending on build flags.
- NUMA autotune and thread planning.

For Rust-only backends, each must be replaced, deprecated, or explicitly moved to legacy status:

- Metal
- Vulkan
- MLX
- WebGPU/WASM

### CLI Parity

The C++ CLI must become the default `oxidize` binary and support the product-critical Rust CLI flows:

- `oxidize run <model>` behavior.
- `--model`, `--prompt`, `--max-tokens`.
- `--chat` or an equivalent chat mode.
- `--auto`, `--no-auto`, `--print-plan`.
- `--api-host`, `--api-port`, and server-by-default behavior if kept.
- Streaming output.
- JSON/timing output for benchmark automation.
- Clear errors for unsupported options.

### Server Parity

`oxidize-cpp-server` must match the OpenAI-compatible Rust server contract for supported features:

- `GET /health` or equivalent readiness endpoint.
- `/v1/models`.
- `/v1/chat/completions`.
- `/v1/completions` if currently supported by Rust.
- Streaming responses using SSE where Rust supports streaming.
- Compatible request/response JSON fields for common OpenAI clients.
- Placeholder/no-model behavior must be intentionally preserved or removed with a migration note.

If the Rust realtime WebSocket endpoint is still product-critical, add a C++ implementation or mark it deprecated before archival.

### Tooling Parity

Create C++ equivalents or replacement decisions for:

- `oxidize-quantize`
- `oxidize-convert`
- `oxidize-prune`
- `oxidize-merge`
- `oxidize-train`
- `oxidize-ffi`
- `oxidize-py`

Each tool must have one of:

1. C++ implementation.
2. Go/Python implementation selected as the supported replacement.
3. Explicit deprecation with docs and release notes.
4. Legacy-only status under archived Rust.

### Python and FFI Story

Before archiving Rust:

- Decide whether Python users should use:
  - pure `oxidize-python`,
  - new C++ bindings,
  - or archived `oxidize-py` with no new feature support.
- Document the supported path.
- If replacing `oxidize-py`, provide installation and smoke-test coverage.

### Tests and Validation

C++ must include automated tests for:

- GGUF parser fixtures currently used by Rust/Go/Python.
- Tokenizer encode/decode round trips.
- Sampling determinism.
- CLI smoke tests.
- Server JSON contract tests.
- Quantization decode/dequantization correctness.
- CPU kernel correctness.
- Autotune plan generation.
- At least one real tiny runnable GGUF integration test, if licensing allows.

Recommended CI commands:

```bash
cmake -S oxidize-cpp -B oxidize-cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build oxidize-cpp/build -j
ctest --test-dir oxidize-cpp/build --output-on-failure
```

Rust tests should continue running during migration until archival:

```bash
sfw cargo test -p oxidize-core -p oxidize-cli -p oxidize-server -p oxidize-kernels
```

### Benchmark Gates

Before claiming C++ as primary:

- Benchmark C++ against current Rust on the standard CPU host.
- Benchmark CUDA/ROCm paths where hardware is available.
- Publish prompt/decode throughput, memory usage, model, quantization, thread count, NUMA mode, and commit SHA.
- Define acceptable regression thresholds.

Suggested minimum gate:

- C++ CPU throughput is equal or better for the selected production model/quantization set.
- C++ server latency is not worse than Rust by more than an agreed threshold.
- C++ memory usage is equal or lower for large-model CPU deployments.

## Implementation Plan

### Phase 0 — Inventory

Deliverables:

- Add a tracked parity checklist under `plans/` or `docs/`.
- Map every Rust crate to one of: replace, keep temporarily, deprecate, archive.
- Map every user-facing command to the new C++ command or deprecation note.

Acceptance criteria:

- Maintainers can see exactly what Rust still owns.
- No code removal yet.

### Phase 1 — C++ Product Hardening

Deliverables:

- Complete text tokenization/detokenization in C++ CLI.
- Expand C++ server compatibility.
- Add server contract tests.
- Add CLI compatibility tests.
- Add fixture-based GGUF/parser/tokenizer tests.

Acceptance criteria:

- C++ binary can run normal prompt and chat flows without requiring pre-tokenized input.
- Common OpenAI clients work against the C++ server for supported endpoints.

### Phase 2 — Tooling Replacement

Deliverables:

- Replace or explicitly deprecate Rust quantize/convert/prune/merge/train/ffi/py workflows.
- Update docs to point to supported replacements.
- Add migration examples.

Acceptance criteria:

- Users can install, run inference, serve API, inspect/convert/quantize models, and use Python-supported paths without depending on active Rust development.

### Phase 3 — Default Binary Switch

Deliverables:

- Make build/install docs produce the C++ `oxidize` binary by default.
- Keep Rust binary available as `oxidize-rust` or through legacy instructions.
- Update Dockerfiles, quickstart, README, HOW_TO_INSTALL, and CI.
- Add release notes announcing Rust deprecation window.

Acceptance criteria:

- New users land on C++ by default.
- Existing Rust users have documented fallback instructions.

### Phase 4 — Deprecation Window

Deliverables:

- Keep Rust building in CI for a defined number of releases or days.
- Stop adding new features to Rust except critical fixes.
- Add warnings/docs that Rust is legacy.

Acceptance criteria:

- No major user-reported blocker remains for supported workflows.
- C++ passes parity, benchmark, and server compatibility gates.

### Phase 5 — Archive Rust

Deliverables:

- Move Rust workspace to `archive/rust/` or split it to a legacy branch/tag.
- Remove Rust from default build path.
- Keep license, README, and migration notes with archived code.
- Keep fixtures needed by Go/Python/C++ in a language-neutral location before moving Rust.

Acceptance criteria:

- Main repo default workflow no longer requires Rust.
- Archived Rust remains discoverable and restorable.
- CI is green for C++/Go/Python supported surfaces.

## Documentation Changes Required

Update:

- `README.md`
- `QUICKSTART.md`
- `HOW_TO_INSTALL.md`
- `CONTRIBUTING.md`
- root `AGENTS.md`
- Dockerfiles
- release notes/changelog if present

Docs must clearly state:

- C++ is primary.
- Rust is legacy during deprecation.
- Which commands changed.
- Which features are unsupported or deprecated.
- How to run old Rust code during the deprecation window.

## CI Changes Required

Add C++ default CI:

- Configure CMake.
- Build C++ CLI and server.
- Run C++ tests.
- Run CLI smoke test.
- Run server contract test.

Keep temporary Rust CI until archival gates pass.

After archival:

- Remove Rust from default CI.
- Optionally keep a manual legacy Rust workflow.

## Risks

- Loss of Rust-only functionality before users have replacements.
- Undetected OpenAI API incompatibilities.
- Tokenizer behavior drift causing generation differences.
- Python binding users losing functionality.
- Hardware backend regressions on Metal/Vulkan/MLX/WebGPU users.
- Conversion/pruning workflows becoming unavailable.

## Rollback Plan

Until Phase 5, rollback is simple:

- Restore Rust as default in docs/build scripts.
- Keep C++ as experimental/performance runtime.

After Phase 5:

- Restore from `archive/rust/` or the legacy branch/tag.
- Re-enable Rust CI workflow if needed.

## Initial Checklist

- [ ] Approve `oxidize-cpp/` as the primary target implementation.
- [ ] Create crate/tool parity checklist.
- [ ] Create OpenAI server contract test suite for C++.
- [ ] Add real text tokenization/detokenization parity tests.
- [ ] Benchmark C++ vs Rust on standard CPU host.
- [ ] Decide fate of Metal/Vulkan/MLX/WebGPU.
- [ ] Decide Python binding strategy.
- [ ] Switch default docs/build to C++ only after gates pass.
- [ ] Deprecate Rust with release notes.
- [ ] Archive Rust only after deprecation window and parity sign-off.
