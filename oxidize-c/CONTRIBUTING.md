# Contributing to oxidize-c

This document records the conventions every worker (human or agent) must
follow when adding to the C11 port in `oxidize-c/`.

## Build & test commands

| Command            | What it does                                                      |
|--------------------|-------------------------------------------------------------------|
| `make` / `make build` | Default CPU-only build (libc only) → `./oxidize-c`            |
| `make test`        | Build + run the test suite → `./test_runner`                      |
| `make lint`        | clang-tidy on `src/**/*.c`                                        |
| `make lib`         | Static library `liboxidize-c.a`                                  |
| `make clean`       | Remove all build artifacts                                        |
| `make avx512`      | AVX-512 BW+VNNI build (for `ai@192.168.1.121`)                    |
| `make cuda`        | CUDA backend build (`OC_CUDA`, for Modal L40S)                    |

The canonical build is `make` (no flags) — it must link only against libc,
libm, libpthread. Every optional backend (`OC_CUDA`, `OC_VULKAN`, `OC_METAL`,
`OC_WEBGPU`, `OC_ROCM`, `OC_AVX512`) is gated behind a build flag.

## C conventions

- **Standard:** C11 (`-std=c11`). No GNU extensions unless gated behind
  `#ifdef`.
- **Naming:** `OcPascalCase` for types, `oc_snake_case` for functions,
  `OC_SCREAMING_CASE` for macros/enums.
- **Headers:** public in `include/oxidize/`, private in `src/`. Use
  `#pragma once`.
- **Errors:** return `OcError` from all public functions. Never `exit()` on
  recoverable errors. `assert()` only in `#ifdef OC_DEBUG` blocks.
- **Memory:** `OcArena` for model-lifetime allocations; `mmap` for GGUF
  weights; `free()` only for short-lived allocations.
- **SIMD:** never call intrinsics directly outside `src/core/simd_*.c`.
  Use the dispatched function pointers from `core/simd.c`.
- **Logging:** `oc_log(level, fmt, ...)` with `OX_LOG_LEVEL` env filter. No
  `printf` in library code.

## Test conventions

The test framework is the in-repo, dependency-free `tests/framework.h`
plus `tests/framework_main.c`. It implements the exact subset of the
Criterion API this suite uses (`Test()`, `cr_assert*`/`cr_expect*`,
`.description`/`.disabled`, `cr_skip_test`, `--filter`, `--list`,
`--xml`, fork-per-test isolation) without any vendored headers or
prebuilt archives — the runner links only libc + libm + libpthread.

### File layout (one test file per source file)

Every `src/<module>.c` source file MUST have a paired
`tests/test_<module>.c` test file. This mirrors the Rust
`#[cfg(test)]` co-location convention. For example:

```
src/core/error.c       -> tests/test_error.c
src/core/arena.c       -> tests/test_arena.c
src/util/bytes.c       -> tests/test_bytes.c
src/format/gguf.c      -> tests/test_gguf.c   (added by gguf-parser feature)
src/compute/tensor.c   -> tests/test_tensor.c (added by compute feature)
```

When a source file has no testable units yet (e.g. a stub), still create the
paired `tests/test_<module>.c` with at least one smoke `Test(module, stub)`
case so `make test` reflects the file's existence.

`tests/test_smoke.c` is the permanent harness sanity check — it verifies
that the test framework itself is wired up correctly. Keep it; do not
delete it.

### Test authoring rules

1. **Use the framework API.** Each test is declared with the
   `Test(suite, case)` macro. The framework auto-registers it; you do not
   need a separate runner or registration boilerplate.

   ```c
   #include "framework.h"
   #include "oxidize/error.h"

   Test(error, msg_returns_ok)
   {
       cr_assert_str_eq(oc_error_msg(OC_OK), "ok", "expected 'ok'");
   }
   ```

2. **Use framework assertions, not `assert()`/`printf`.**
   - `cr_assert(cond, fmt, ...)` — abort the test on failure.
   - `cr_expect(cond, fmt, ...)` — record a failure but keep running.
   - `cr_assert_eq(a, b, fmt, ...)`, `cr_assert_neq`, `cr_assert_str_eq`,
     `cr_assert_null`, `cr_assert_not_null`, `cr_assert_lt/leq/gt/geq`, etc.
     (see `tests/framework.h` for the full set).

3. **No `main()` in test files.** `tests/framework_main.c` provides
   `main()`. The `make test` target compiles every `tests/test_*.c` into a
   single `test_runner` binary and links it with
   `tests/framework_main.o`.

4. **ASan + UBSan.** Tests are compiled with
   `-fsanitize=address,undefined` (valgrind is not installed locally, so
   ASan substitutes for leak/UB checking). Any new test must be ASan-clean.

5. **Test isolation.** The runner forks each test into its own process.
   Global state changes in one test do not leak into another. Do not rely
   on cross-test ordering.

6. **Naming.** `Test(<module>, <behavior>)` where `<module>` matches the
   source file's basename (e.g. `Test(arena, alloc_returns_aligned)` for
   `src/core/arena.c`). One logical assertion-group per `Test` case; split
   distinct behaviors into separate cases.

### Running tests

```bash
make test                # build + run all tests
./test_runner             # run already-built test_runner
./test_runner --help      # flag reference
./test_runner --filter "error/*"   # run only tests in the "error" suite
./test_runner --filter kv_cache_init  # suite/case via underscore
./test_runner --filter config_init    # exact case name; no config/init* glob
./test_runner --list --filter error   # list matching tests
./test_runner --list      # list all registered tests
./test_runner --verbose 1 # print PASS lines
./test_runner --xml out.xml  # JUnit (one <testsuite> per suite)
```

### Adding a new test file

1. Create `tests/test_<module>.c` (where `<module>` matches the source file
   basename, e.g. `test_gguf.c` for `src/format/gguf.c`).
2. `#include "framework.h"` first, then the project header under
   test (`#include "oxidize/<module>.h"`), then system headers.
3. Declare one `Test(<module>, <case>)` per behavior.
4. Run `make test`. The `tests/test_*.c` wildcard in the Makefile will pick
   it up automatically — no Makefile edits needed.

### Parity tests vs Rust

When porting a feature from Rust `oxidize-core`, prefer bit-exact parity
tests that compare C output against a captured Rust reference (golden
fixture). The convention is:

```c
Test(<module>, <behavior>_parity)
{
    /* load golden fixture from oxidize-core/tests/fixtures/<name>.bin */
    /* compute via C API */
    /* cr_assert_eq / cr_assert_arr_eq(actual, golden, size) against golden */
}
```

Golden fixtures are generated by the Rust reference (run on `ai@192.168.1.121`
or locally via `cargo run -p oxidize-cli`) and committed under
`oxidize-core/tests/fixtures/`. Tolerance: 0 for integer paths, ≤1e-4
relative for FP paths.

## Commit conventions

- **Branch:** `oxidize-c-v2` only. Never push to `master`.
- **Format:** `feat(oxidize-c): <description>` or `fix(oxidize-c): <description>`.
- **Staging:** `git add oxidize-c/<specific paths>` — never `git add -A`.
- **Trailer:** every commit must include
  ```
  Co-authored-by: factory-droid[bot] <138933559+factory-droid[bot]@users.noreply.github.com>
  ```
- **Scope:** stage only files related to your feature. Exclude unrelated
  workspace changes (other crates, target/, etc.).

## Coding guidelines (mirrors AGENTS.md)

- **No reference to old `oxidize-c/` git history.** Fresh implementation from
  Rust sources + architecture doc + SOTA research only.
- **Port, don't invent.** Rust `oxidize-core` is the source of truth for
  behavior, numerics, and naming. Novel optimization is permitted only at
  the kernel layer (SIMD/CUDA).
- **Bit-exact parity.** Dequantization and forward-pass logits must match
  Rust bit-exactly on tiny fixtures (integer: exact; FP: ≤1e-4 relative).
- **Config + Error + Trait trinity** per subsystem (mirrors Rust convention).
- **Flat module system** via `oc.h` umbrella header.
