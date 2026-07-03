# oxidize-cpp CPU Performance Spec

## Goal

Improve large-model CPU inference defaults for `oxidize-cpp` on dual-socket
NUMA hosts, especially the benchmark box `ai@192.168.1.132`:

- Small and medium dense models that fit inside one NUMA node should stay on one
  socket and avoid over-threading.
- Huge or near-RAM-limit models should interleave memory and avoid mmap
  whole-file prefetch.
- The CLI should make the chosen memory policy visible through `--print-plan`
  and apply it when loading GGUF weights.

## Source Research

Local research artifacts in `.firecrawl/` show that llama.cpp separates mmap
policy from NUMA policy: normal single-node loads can request sequential
prefetch, while NUMA or very large mappings avoid prefetch and use random access
advice. Linux `madvise(2)` documents `MADV_WILLNEED`, `MADV_SEQUENTIAL`, and
`MADV_RANDOM` as advisory policies, so oxidize-cpp can adopt this without
changing tensor formats or model math.

## Selected PR Scope

This PR intentionally avoids broad kernel work. Q6_K, IQ1, and NVFP4 fused
matmul paths need separate numeric parity fixtures and model-specific benchmark
fixtures. The focused change here is lower risk and directly affects every CPU
model load:

1. Add `TuningPlan::mmap_advice`.
2. Plan `mmap_advice="sequential_prefetch"` for dense/single-node loads.
3. Plan `mmap_advice="random"` for interleaved or near-RAM-limit loads.
4. Cap dense dual-socket single-node autotune at 16 threads on large Xeon-style
   hosts instead of using every logical CPU on the socket.
5. Apply the plan in `GgufModel::load()` so the CLI's `--auto` plan changes the
   actual mmap advice before weights are touched.
6. Fix the batched forward KV-cache layer stride to use the capped runtime KV
   context instead of the advertised model context, because large-context models
   can cap KV cache size at load time.

## Acceptance Criteria

- `autotune_test` proves the ai-box-shaped dense case selects
  `single/16/sequential_prefetch`.
- `autotune_test` proves huge and near-RAM-limit cases select
  `interleave/48/random`.
- `--print-plan --json` includes `mmap_advice`.
- CLI model loading passes the selected `mmap_advice` into GGUF mapping.
- Remote benchmark evidence attempts >5B, >35B, and >500B-class model surfaces
  on `ai@192.168.1.132`; unsupported architectures must fail cleanly without
  crash/OOM.
- A local regression check proves `forward_batched()` does not index KV-cache
  layers with the uncapped GGUF context size.

## Files

- `oxidize-cpp/include/oxidize/autotune.hpp`
- `oxidize-cpp/src/autotune.cpp`
- `oxidize-cpp/include/oxidize/gguf.hpp`
- `oxidize-cpp/src/gguf.cpp`
- `oxidize-cpp/include/oxidize/model_llama.hpp`
- `oxidize-cpp/src/model_llama.cpp`
- `oxidize-cpp/src/cli/main.cpp`
- `oxidize-cpp/src/tensor_cpu.cpp`
- `oxidize-cpp/tests/autotune_test.cpp`
- `oxidize-cpp/tests/gguf_mmap_advice_test.cpp`
- `scripts/bench-ai-box.sh`
- `scripts/test_bench_ai_box_matrix.sh`
- `scripts/test_cpp_kv_context_stride.sh`
