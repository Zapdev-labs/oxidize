# Oxidize — Engineering Roadmap

_Last updated: 2026-05-24._

This document is the plan-of-record for the four open performance items, the
vision-model question, and the end-user install path. It is intentionally
honest about scope, timeline, and risk; nothing here is decorative.

---

## 0. Where we are today

Measured on the dev box (Ryzen 7 PRO 6850H, 16 threads, AVX2, no AVX‑512, no
NVIDIA GPU, CPU-only, release build, `oxidize-bench`):

| Workload | Model | Throughput | Per-token |
| --- | --- | --- | --- |
| Decode | Qwen3-4B Q4_K_M | **14.6 tok/s** | 68.6 ms |
| Prefill (512 prompt tokens) | Qwen3-4B Q4_K_M | **30.1 tok/s** | 33.3 ms |

For reference, **llama.cpp** on the same hardware lands roughly at:

| Workload | Approx. llama.cpp number | Gap to oxidize |
| --- | --- | --- |
| Decode | ~20–25 tok/s | ~1.4–1.7× slower |
| Prefill | ~100+ tok/s | **~3.3× slower** |

These numbers are from our bench, not a head-to-head harness — see item **(5)**
below for replacing the guesswork with real apples-to-apples runs.

**Recent perf history** (last 5 commits): AVX2 Q4_K kernel rewrite, batched
Q4_K/Q8_K GEMM, batched prefill via `gemm_quantized_f32`, normal-model decode
correctness fixes, and mDNS removal from the mesh dependency graph to clear the
`hickory-proto` Dependabot advisory. The momentum is on decode; prefill is the
next obvious win.

---

## 1. Vision models — do we support them?

**No.** Confirmed by code search:

- `ModelArchitecture` only enumerates text-only architectures: Llama, Mistral,
  Mixtral, DeepSeek, Qwen, Gemma, Phi, Falcon, GPT‑2/J/NeoX.
- Zero references anywhere in `oxidize-core`, `oxidize-server`, or
  `oxidize-cli` to ViT, CLIP, LLaVA, Qwen-VL, Pixtral, Idefics, patch
  embeddings, image preprocessing, or multimodal token handling.
- The model loader is GGUF-only with no image pipeline (no resize, normalize,
  patchify, projector module).

### What adding vision actually requires

A minimum-viable LLaVA-style integration touches:

1. **Image preprocessor** — resize (typically 336²/448²/672²), center-crop,
   ImageNet-normalize, patchify into N×N tokens. CPU is fine; this isn't
   the hot path.
2. **ViT encoder** — separate transformer with 2D positional embeddings,
   runs once per image; output is a sequence of patch embeddings.
3. **Projector / connector** — usually a 2-layer GeLU MLP from ViT hidden
   dim → LM hidden dim. Some models (Qwen-VL) use cross-attention instead.
4. **Multimodal prompt assembly** — splice projected image tokens into the
   text token stream at `<image>` placeholders; needs token IDs reserved
   correctly per architecture.
5. **GGUF metadata + loader** — vision models bundle two networks; need
   parsing for `image_size`, `patch_size`, `vision_hidden_size`,
   `mm_projector_*`, etc. Some shipped as a single GGUF (LLaVA mmproj), some
   as two files.
6. **OpenAI-compatible image input** — `messages[].content` as an array
   with `type: "image_url"`, base64 decode or HTTPS fetch with size cap.
7. **Sampling-side care** — image placeholder tokens must be in the
   suppressed-token set so the model never tries to generate one.

### Vision: realistic plan

| Phase | Scope | Time |
| --- | --- | --- |
| **V0 — Spec** | ADR with: target model (recommend LLaVA-1.5 7B or Qwen2-VL-2B), GGUF format choice, API surface, sampling rules | 1–2 days |
| **V1 — Preprocessor + ViT** | Standalone library that takes an image file and produces patch embeddings. No LM integration. | 1 week |
| **V2 — Projector + token splice** | Inject projected embeddings into the text path. Validate against reference outputs from llama.cpp/LLaVA. | 1 week |
| **V3 — OpenAI image input** | Server accepts `image_url`/base64. End-to-end demo. | 3 days |
| **V4 — Second architecture** | Add Qwen-VL or Pixtral. Refactor any architecture-specific assumptions. | 1–2 weeks |

**Total: 4–6 weeks for one architecture, 6–10 weeks for two.** I do not
recommend starting this until at least items (1) and (2) below ship — the
ViT will run at the same CPU GEMM speed we have today, and 30 tok/s prefill
means a 576-patch ViT pass alone is ~20 s, which is unusable.

---

## 2. Performance roadmap

Five items, ordered by **value per session-day of work**:

### (1) Quantized KV cache — ✅ shipped 2026-05-24

`KvQuantization::TurboQuant` opt-in lands per-block (32-channel) symmetric
scales in place of the per-token (scale, min) layout. CLI: `--turboquant`.
Server: `--turboquant-kv`. Default unchanged.

- Memory profile (head_dim=128, GQA head_count=8 → token_size=1024):
  old Q4 ≈ 520 B/token, TurboQuant Q4 ≈ 640 B/token (+23 %), F32 ≈ 4096 B/token.
- Accuracy: dedicated test asserts per-block scales preserve a small-value
  channel region alongside a large-value region (asymmetric per-token fails
  this).
- 6 new tests; full suite still green (460 core + 43 server).

### (2) AVX2-tuned prefill GEMM — 🟡 next up (recommended)

**The single biggest win on the board.** Our prefill is ~3.3× behind
llama.cpp on the same CPU. That gap is almost entirely in `gemm_quantized_f32`.

| Sub-task | Notes |
| --- | --- |
| Block-tile the Q4_K × F32 prefill GEMM (8×4 or 8×8 register tiles) | Largest gain. |
| Vectorize K-block dot product with AVX2 FMA + PMADDUBSW | Already partially done for GEMV; needs to land on the batched path. |
| Fuse RMSNorm + first matmul (QKV) | Saves a full pass over the activation tensor at every layer. |
| Thread tiling: split on output rows, pin to cores via rayon | Current parallelism leaves cores idle on small batches. |
| Bench gate: write prefill regression test in `oxidize-bench` that fails if pp drops below current | Catch regressions in CI. |

**Estimated impact:** prefill 30 → 70–100 tok/s. **Effort:** 1 session
day for the GEMM tile + thread tiling, half a day for the bench gate.

### (3) Speculative decoding via DFlash — 🟡 highest user-perceived speedup

DFlash draft-model infrastructure exists (commits `9008812`, `9649e00`,
`1c3093b`). What's missing is the **verification path** in the main inference
loop and the **speculative sampling glue**.

| Sub-task | Notes |
| --- | --- |
| Wire draft model into the decode loop (gated by `--draft-model`) | Existing `DFlashDraftModel` already produces draft tokens. |
| Implement target-model batch verification (forward N tokens, accept prefix where draft logits agree) | This is the actual speculative-decoding algorithm. |
| Acceptance-rate metric + bench mode `--mode spec` | Need to measure speedup honestly. |
| Server flag `--draft-model PATH` | Otherwise this stays an internal feature. |

**Estimated impact:** decode 14 → 25–40 tok/s _if_ draft acceptance rate is
≥60 % for the workload. Lower acceptance = no gain. **Effort:** 2–3 session
days. The risk is acceptance-rate variance across prompts.

### (4) Continuous batching scheduler — 🔴 biggest scope

Today the server processes one generation request at a time even with the
PagedAttention scheduler. To compete with vLLM/sglang on multi-tenant serving
we need real continuous batching: pack N sequences into a single forward
pass and advance them independently.

| Sub-task | Notes |
| --- | --- |
| Multi-sequence batched `model.forward()` — currently single-sequence only | Requires per-sequence position tracking and per-sequence sampling. |
| Scheduler hand-off: pull N ready sequences per step, pad to common length, attention mask | The hard part. |
| KV cache: per-sequence views into the paged blocks | PagedAttention already supports this; just need plumbing. |
| Backpressure: rate limit if KV exhaustion is imminent | We already 503 on exhaustion; needs to happen earlier. |

**Estimated impact:** 4–8× aggregate throughput on multi-client workloads
at the cost of some single-stream latency. **Effort:** 3–5 session days,
invasive, touches both core and server.

### (5) Real load-test + bench harness — 🟡 do this alongside (2)

We are currently guessing at our gap to llama.cpp. We should not be.

| Sub-task | Notes |
| --- | --- |
| `scripts/bench_vs_llamacpp.sh` — runs identical prompts on both engines, same GGUF, reports tok/s + p50/p99 latency | Honest ground truth. |
| Multi-client load test (`hey` or `oha`) against `oxidize-server` | Measures saturation, not just single-stream. |
| Wire `axum-prometheus` for `/metrics` endpoint | Throughput, queue depth, KV utilization, scheduler stats. |
| Bench artifact in `results/` per commit, plot trend | Catch regressions before users do. |

**Estimated impact:** zero direct perf, but every other item on this list
becomes 2× easier to validate. **Effort:** 1 session day.

### Suggested order

1. **(2) AVX2 prefill** — ship next session. Real number to hit: 70 tok/s pp.
2. **(5) Bench harness** — same session if there's time, otherwise next.
3. **(3) Speculative decoding** — once (2) is in, the verification cost
   drops, making spec-decoding strictly more attractive.
4. **(4) Continuous batching** — only if the product needs multi-tenant
   serving. For local-single-user use cases, skip.
5. **Vision** — only after (2) lands. ViT throughput inherits prefill
   throughput.

---

## 3. End-user install path

The install script (`scripts/install.sh`) is shipped in this repo and is
designed to be served from a website with:

```
curl -fsSL https://<your-domain>/install.sh | bash
```

### What it does

- Detects OS (Linux / macOS) and CPU arch (x86_64 / aarch64).
- Downloads a release tarball from `OXIDIZE_RELEASE_URL` (overridable env;
  defaults to a placeholder GitHub-releases URL — **set this before going
  live**).
- Verifies the SHA-256 against `<tarball>.sha256` next to the artifact.
- Installs `oxidize-server`, `oxidize-cli`, and `oxidize-bench` into
  `INSTALL_DIR` (default: `$HOME/.local/bin`, falls back to `/usr/local/bin`
  if writable).
- Prints a clear PATH-export hint if the install dir isn't already on `$PATH`.
- Idempotent: re-running upgrades in place.
- Refuses to run as root with `--no-sudo` set; honours `OXIDIZE_VERSION` for
  pinning.

### What it explicitly does NOT do

- Does **not** edit shell rc files (a long-standing source of install-script
  rage). It prints the line; the user runs it.
- Does **not** require a package manager. No `brew`, no `apt`, no `cargo`.
- Does **not** download or install models. Model paths are the user's
  problem.

### Release-side prerequisites (must be done before the script is live)

1. **GitHub releases** (or equivalent) publishing tarballs named
   `oxidize-<version>-<os>-<arch>.tar.gz` containing the three binaries.
2. **Per-tarball `.sha256` files** alongside each artifact.
3. **CI workflow** that builds release binaries for `x86_64-unknown-linux-gnu`,
   `aarch64-apple-darwin`, `x86_64-apple-darwin`. (Windows is a separate
   problem — `axum` works there but cross-compile of the SIMD kernels needs
   testing.)
4. **Static linking** where reasonable (avoid glibc-version surprises). On
   Linux, build against `x86_64-unknown-linux-musl` for max portability or
   pin a known-old `manylinux` glibc.

### Smoke test for the install script

```
INSTALL_DIR=/tmp/oxidize-test bash scripts/install.sh --dry-run
INSTALL_DIR=/tmp/oxidize-test bash scripts/install.sh
/tmp/oxidize-test/oxidize-server --help
```

---

## 4. Open questions for the maintainer

1. **Target benchmark hardware** — what CPU/GPU is the "must-be-fast-on"
   machine? Tuning for AVX2 on Ryzen is very different from tuning for
   AVX-512 on Sapphire Rapids.
2. **Multi-tenant or single-user?** Decides whether item (4) is worth the
   3–5 session days.
3. **Vision: which architecture first?** LLaVA-1.5 is the easiest; Qwen2-VL
   is the most modern; Pixtral has the cleanest projector. Pick one.
4. **Release infra** — where do the install-script tarballs come from? If
   GitHub releases, fine; if a CDN-fronted bucket, the script's
   `OXIDIZE_RELEASE_URL` needs to point at that.
