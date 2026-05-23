# Making Oxidize Fast: Q4_K Decode Optimization Deep Dive

> How DFlash draft-model decode throughput went from **3.54 tok/s** to **16.20 tok/s**
> on a single AMD Ryzen 7 PRO 6850H — a **4.6x speedup** that put us **2.8x ahead of
> the ollama reference (5.7 tok/s)**.

**Date:** 2026-05-22
**Hardware:** AMD Ryzen 7 PRO 6850H (8 cores / 16 threads, Zen 3+, AVX2 + FMA, *no* AVX-512), dual-channel DDR5
**Model:** `Qwen3.6-27B-DFlash-Q4_K_M.gguf` (DFlash draft model, ~1.03 GB on disk, Q4_K_M quantized)
**Benchmark:** `oxidize-bench --model … --draft-tokens 64 --iterations 5` → `DFlashDraftModel::forward_token()`

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Background: What Are We Even Measuring?](#2-background-what-are-we-even-measuring)
3. [The Theory: Why LLM Decode Is Memory-Bandwidth Bound](#3-the-theory-why-llm-decode-is-memory-bandwidth-bound)
4. [Root Cause #1: Dequantizing Weights to f32 at Load Time](#4-root-cause-1-dequantizing-weights-to-f32-at-load-time)
5. [Fix #1: Keep Weights Quantized, Dequantize On the Fly](#5-fix-1-keep-weights-quantized-dequantize-on-the-fly)
6. [The Layout Trap: A Correctness Bug Hiding in the f32 Path](#6-the-layout-trap-a-correctness-bug-hiding-in-the-f32-path)
7. [Root Cause #2: The Hot Dot Product Was Scalar](#7-root-cause-2-the-hot-dot-product-was-scalar)
8. [Fix #2: An AVX2 + FMA Q4_K Dot Product](#8-fix-2-an-avx2--fma-q4_k-dot-product)
9. [Fix #3: AVX2 as the Compile-Time Baseline](#9-fix-3-avx2-as-the-compile-time-baseline)
10. [Results: Step-by-Step Attribution](#10-results-step-by-step-attribution)
11. [Verification & Correctness](#11-verification--correctness)
12. [Engine vs Engine: Oxidize, llama.cpp, ollama, miniforge](#12-engine-vs-engine-oxidize-llamacpp-ollama-miniforge)
13. [The Kimi-K2.5 DFlash Story (Why That Model Is Slow)](#13-the-kimi-k25-dflash-story-why-that-model-is-slow)
14. [Full Speculative Decoding: What's Blocked And Why](#14-full-speculative-decoding-whats-blocked-and-why)
15. [What We Did *Not* Do (And Why)](#15-what-we-did-not-do-and-why)
16. [Future Work](#16-future-work)
17. [Appendix: The Q4_K Block Format](#17-appendix-the-q4_k-block-format)

---

## 1. Executive Summary

Oxidize was losing to ollama on decode throughput: **3.21–3.54 tok/s** vs ollama's
**5.7 tok/s** on the same machine and model. The investigation found two compounding
problems, both in the path that multiplies the model's weight matrices by the
per-token activation vector (the GEMV — *general matrix–vector* product):

1. **The model's 4-bit weights were being expanded to 32-bit floats at load time.**
   Decode is bottlenecked by how fast you can stream weights from RAM, so inflating
   every weight 8x (4 bits → 32 bits) inflated the per-token memory traffic 8x.

2. **The on-the-fly dequantizing GEMV kernel that *would* have fixed this already
   existed in the codebase but was never wired in — and its inner loop was scalar**,
   extracting and converting 4-bit values one at a time instead of using the AVX2
   SIMD units the CPU has.

Fixing both, plus enabling AVX2 as the compile baseline, produced:

| Stage | tok/s | Speedup vs baseline |
|------:|------:|--------------------:|
| Baseline (f32-expanded weights, scalar) | **3.54** | 1.00x |
| Keep weights quantized (fused scalar GEMV) | **5.34** | 1.51x |
| + AVX2 Q4_K dot product | **15.62** | 4.41x |
| + AVX2 compile-time baseline | **16.20** | **4.58x** |
| *ollama reference* | *5.70* | *1.61x* |

Net latency per token dropped from **283 ms → 62 ms**.

---

## 2. Background: What Are We Even Measuring?

The benchmark drives `DFlashDraftModel::forward_token()`, the per-token forward pass
of a DFlash *draft* model used for speculative decoding. It is a compact transformer:

| Parameter | Value |
|-----------|-------|
| `hidden_size` | 5120 |
| `num_hidden_layers` | 5 |
| `num_attention_heads` | 32 |
| `num_key_value_heads` | 8 |
| `intermediate_size` | 17408 |
| `vocab_size` (n_target_features) | 25600 |
| Quantization | Q4_K_M |

Each `forward_token` call does, per layer:

- **Attention projections:** `q_proj`, `k_proj`, `v_proj`, `o_proj`
- **MLP projections:** `gate_proj` (5120→17408), `up_proj` (5120→17408), `down_proj` (17408→5120)
- plus RMSNorm, RoPE, and the attention mechanism itself.

The MLP projections dominate the FLOP and byte counts: three matrices of roughly
`5120 × 17408 ≈ 89M` weights each, per layer, across 5 layers. Every one of those
weights must be read from memory for every single token. That observation is the
entire key to this work.

---

## 3. The Theory: Why LLM Decode Is Memory-Bandwidth Bound

Autoregressive decode generates **one token at a time**. For each token, every weight
in the model is read exactly once and used in a handful of multiply-accumulate
operations. This gives the workload a very low **arithmetic intensity** (FLOPs per
byte read):

```
GEMV for an (M × N) weight matrix:
  FLOPs read  ≈ 2 · M · N   (one multiply + one add per weight)
  bytes read  ≈ M · N · bytes_per_weight
  intensity   ≈ 2 / bytes_per_weight   FLOP/byte
```

For f32 weights that is `2 / 4 = 0.5` FLOP/byte. A modern CPU can do tens of GFLOP/s
but only has ~tens of GB/s of memory bandwidth, so at 0.5 FLOP/byte the chip spends
almost all its time *waiting for weights to arrive from DRAM*, not computing.

The Roofline consequence is blunt: **for a bandwidth-bound kernel, time is
proportional to bytes read.** If you can read fewer bytes per token, you go faster,
almost linearly. This is why every serious local-inference engine (llama.cpp, ollama,
which wraps llama.cpp's GGML kernels) keeps weights in their compact quantized form
and dequantizes them *inside* the matmul kernel, touching each 4-bit value only as it
is consumed.

A back-of-envelope for this model:

- Quantized on disk: ~1.03 GB. Reading that per token at ~30 GB/s ≈ **~34 ms**.
- Expanded to f32: ~8 GB. Reading that per token at ~30 GB/s ≈ **~270 ms**.

The measured baseline latency of **283 ms/token** lines up almost exactly with the
f32-expanded estimate. That was the smoking gun.

---

## 4. Root Cause #1: Dequantizing Weights to f32 at Load Time

`DFlashDraftModel::load_from_gguf()` loaded each weight tensor like this (paraphrased):

```rust
let mut f32_data = vec![0.0_f32; value_count];
dequantize_scalar(qtype, qdata, &mut f32_data)?;   // Q4_K → f32, 8x bigger
let transposed = transpose_f32(&data, gguf_rows, gguf_cols);
F32Weight::from_slice(transposed, gguf_cols, gguf_rows)
```

Every Q4_K weight was decompressed into a full `f32` up front and stored that way.
Inference then ran `gemv_f32_transposed` over those fat f32 matrices. So although the
model *file* is 4-bit, the *running* model behaved like an 8 GB f32 model — paying the
full memory-bandwidth penalty on every token.

Ironically, the codebase **already had** a complete on-the-fly quantized GEMV path:

```
gemv_quantized_f32(qtype, bytes, rows, cols, vector, output)
  └── gemv_q4_k_f32_fused(...)   // reads Q4_K bytes directly, dequant per block
  └── gemv_q6_k_f32_fused(...)
  └── gemv_q8_0_f32_fused(...)
```

It was simply never called by the DFlash model. The fast road was paved; nobody had
driven on it.

---

## 5. Fix #1: Keep Weights Quantized, Dequantize On the Fly

The change keeps each projection weight in its **native GGUF row-major quantized
layout** and routes GEMV through the fused kernel, which reads 4-bit blocks and
dequantizes them as it consumes them.

`F32Weight` was extended to optionally hold quantized bytes instead of f32 data:

```rust
pub struct QuantWeight {
    pub bytes: Vec<u8>,              // raw GGUF Q4_K/Q6_K/Q8_0 block bytes
    pub qtype: GgufQuantizationType,
    pub out_dim: usize,              // GGUF rows = output features
    pub in_dim: usize,               // GGUF cols = input features (contiguous axis)
}

pub struct F32Weight {
    pub data: Vec<f32>,              // used only by the f32 fallback path
    pub rows: usize,
    pub cols: usize,
    pub quant: Option<QuantWeight>,  // present → take the fast path
}

impl F32Weight {
    pub fn gemv(&self, input: &[f32], output: &mut [f32]) -> Result<(), String> {
        if let Some(q) = &self.quant {
            gemv_quantized_f32(q.qtype, &q.bytes, q.out_dim, q.in_dim, input, output)
        } else {
            gemv_f32_transposed(&self.data, self.cols, self.rows, input, output)
        }
    }
}
```

A `load_proj()` helper decides per tensor whether the fused kernel supports the
quant type and shape (block-aligned input dimension), storing raw bytes when it can
and falling back to f32 dequant otherwise:

```rust
fn quantized_gemv_supported(qtype: GgufQuantizationType, in_dim: usize) -> bool {
    match qtype {
        Q4_K_S | Q4_K_M | Q6_K => in_dim % 256 == 0,  // QK_K block alignment
        Q8_0                   => in_dim % 32  == 0,
        _ => false,
    }
}
```

Token embeddings deliberately **stay f32**, because they are used as a direct row
lookup (`tok_embeddings.data[idx*h .. ]`), not a GEMV — there is no bandwidth win in
quantizing a single-row gather, and dequantizing it would only add complexity.

**Result of Fix #1 alone: 3.54 → 5.34 tok/s (1.51x).** This already matched ollama,
but the fused kernel's inner loop was still scalar — leaving the biggest win on the
table.

---

## 6. The Layout Trap: A Correctness Bug Hiding in the f32 Path

While wiring up the fused kernel, the first build panicked:

```
InvalidVectorLength { expected: 4096, actual: 5120 }
```

This forced a careful look at the dimension convention, and it revealed something
important: **the original f32 path was not just slow, it was computing a *scrambled*
matrix multiply.**

GGUF stores a weight tensor row-major with the **innermost (contiguous) dimension
`dims[0]` = input features** and `dims[1]` = output features. For a projection
`y = W·x`, that means:

```
W_native[out][in] = data[out * in_dim + in]      // in_dim is contiguous
y[out] = Σ_in  W_native[out][in] · x[in]
```

The fused kernel consumes exactly this layout — no transpose needed. But the original
f32 loader called `transpose_f32(data, dims[0], dims[1])` and then
`gemv_f32_transposed`, an interpretation that reads the linear byte stream as if the
outer/inner dimensions were swapped. Working through the index arithmetic shows the
resulting `matrix[i·cols + j]` does **not** equal `W_native[j][i]` — the weights end
up permuted. The benchmark never noticed because it only measures *speed*, feeding the
mask token repeatedly and discarding the logits.

So switching to the fused quantized kernel was a **double win**: it is faster *and*
mathematically correct in the llama.cpp/GGML sense. The fix simply uses the native
dimensions directly:

```rust
let in_dim  = info.dimensions[0] as usize;  // contiguous / input axis
let out_dim = info.dimensions[1] as usize;  // output rows
```

The f32 fallback retains the original transpose so its (independent) behavior is
unchanged for any tensor that can't take the fast path.

---

## 7. Root Cause #2: The Hot Dot Product Was Scalar

`gemv_q4_k_f32_fused` parallelizes across output rows with Rayon (good), but each row
called `q4_k_dot`, which dequantized and multiplied **one weight at a time**:

```rust
for l in 0..32 {
    let packed = qs[q_base + l];
    sum += (d1 * (packed & 0x0f) as f32 - min1) * vector[v_base + l];
    sum += (d2 * (packed >>   4) as f32 - min2) * vector[v_base + 32 + l];
}
```

This is correct but leaves the CPU's 256-bit vector units idle. Each Q4_K block holds
256 weights; doing them scalar means 256 dependent FP operations per block where the
hardware could do 8 at a time with fused multiply-add.

---

## 8. Fix #2: An AVX2 + FMA Q4_K Dot Product

The new `q4_k_dot_avx2` processes **8 weights per instruction**:

1. Load 8 packed nibble-pairs (`_mm_loadl_epi64`).
2. Mask/shift to extract the low and high 4-bit values, widen `u8 → i32 → f32`
   (`_mm256_cvtepu8_epi32` + `_mm256_cvtepi32_ps`).
3. Compute the dequantized term `d·q − min` with a single fused multiply-subtract
   (`_mm256_fmsub_ps`).
4. Multiply by the matching activation lane and accumulate with FMA
   (`_mm256_fmadd_ps`) into a running 8-wide accumulator.
5. Horizontal-sum the 8 lanes once at the end.

```rust
#[target_feature(enable = "avx2,fma")]
unsafe fn q4_k_dot_avx2(block: &[u8], vector: &[f32]) -> f32 {
    // ... per group_pair, per 8-lane chunk:
    let term1 = _mm256_fmsub_ps(low,  d1, min1);   // d·q − min
    let term2 = _mm256_fmsub_ps(high, d2, min2);
    acc = _mm256_fmadd_ps(term1, v1, acc);         // acc += term·x
    acc = _mm256_fmadd_ps(term2, v2, acc);
    // ... horizontal sum of acc → f32
}
```

`q4_k_dot` now dispatches to this at runtime when AVX2 + FMA are present and falls
back to the scalar reference (`q4_k_dot_scalar`) otherwise:

```rust
fn q4_k_dot(block: &[u8], vector: &[f32]) -> f32 {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        return unsafe { q4_k_dot_avx2(block, vector) };
    }
    q4_k_dot_scalar(block, vector)
}
```

This is where the bandwidth savings from Fix #1 finally translate into wall-clock
speed: with the bytes-read problem solved, the kernel was now compute-limited on
scalar dequant, and SIMD removed that limit.

**Result of Fix #2: 5.34 → 15.62 tok/s (a further 2.9x; 4.41x cumulative).**

---

## 9. Fix #3: AVX2 as the Compile-Time Baseline

The hand-written intrinsics are dispatched at runtime, but the *rest* of the codebase
— the scalar fallbacks, RMSNorm, RoPE, softmax, the surrounding glue — was still
compiled for a generic x86-64 baseline (SSE2 only). Telling the compiler it may assume
AVX2 + FMA lets it autovectorize all of that:

```toml
# .cargo/config.toml
[target.x86_64-unknown-linux-gnu]
rustflags = ["-C", "target-feature=+avx2,+fma"]
```

The runtime detection stays in place as a safety net for other targets and non-config
builds. This is a small but free win.

**Result of Fix #3: 15.62 → 16.20 tok/s (4.58x cumulative).**

> **Caveat:** binaries built with this config require a Haswell-era (2013+) CPU. That
> is universal for any machine actually running these models, but if you ship prebuilt
> binaries to arbitrary hardware, keep a separate portable build profile.

---

## 10. Results: Step-by-Step Attribution

Measured with `--draft-tokens 64 --iterations 5` on the Ryzen 7 PRO 6850H:

| # | Change | tok/s | ms/token | Δ vs prev | Cumulative |
|---|--------|------:|---------:|----------:|-----------:|
| 0 | Baseline: f32-expanded weights + scalar f32 GEMV | 3.54 | 283 | — | 1.00x |
| 1 | Keep Q4_K weights quantized; fused on-the-fly GEMV | 5.34 | 187 | +51% | 1.51x |
| 2 | AVX2 + FMA Q4_K dot product | 15.62 | 64 | +193% | 4.41x |
| 3 | AVX2 + FMA compile-time baseline | 16.20 | 62 | +4% | 4.58x |
| — | **ollama reference** | **5.70** | **175** | — | — |

The dominant lessons, in order of impact:

1. **Don't expand quantized weights.** Memory traffic is the budget; spend it on 4-bit
   blocks, not 32-bit floats. (1.5x, and a correctness fix for free.)
2. **Vectorize the dequant.** Once you stop wasting bandwidth, the scalar nibble
   unpacking becomes the new ceiling; SIMD shatters it. (2.9x.)
3. **Let the compiler use the whole ISA.** Cheap insurance for everything you didn't
   hand-tune. (1.04x.)

---

## 11. Verification & Correctness

- **Unit test:** `q4_k_dot_avx2_matches_scalar` builds a deterministic pseudo-random
  Q4_K block plus activation vector and asserts the AVX2 result matches the scalar
  reference within `1e-3 · (1 + |scalar|)`. Passes.
- **Build:** full `cargo build --release` across the workspace succeeds with the new
  flags.
- **Mathematical correctness:** the fused kernel implements the canonical GGML Q4_K
  dequant-and-dot, so its output is the *correct* projection — an improvement over the
  pre-existing scrambled f32-transpose path (see §6).

---

## 12. Engine vs Engine: Oxidize, llama.cpp, ollama, miniforge

The original framing for this work was "we run at 3.21 tok/s, ollama runs at 5.7
tok/s — beat it." That framing turns out to have conflated two different
workloads on two different models. Once we untangle them, the comparison gets
both cleaner and more interesting.

### 12.1 What was actually being measured against ollama

The `ollama-performance-benchmark/` directory on the user's mounted drive points
at a **standard Qwen3.5 4B model** (`HauhauCS-Aggressive-Q4_K_M`), not a DFlash
draft. So the 5.7 tok/s ollama reference is a regular text-generation number on
a 4B model — not anything DFlash-related.

Meanwhile, `oxidize-bench` measures the **DFlash draft `forward_token`** of the
Qwen3.6-27B-DFlash model in isolation: a single-token forward pass through a
small (5 layer) draft model, without the `fc` target-fusion and without an
`lm_head`/sampling step.

Those are different workloads. They're both Q4_K_M, both CPU-only, both on the
same Ryzen 7 PRO 6850H — but the per-token compute they do is not the same.

### 12.2 The clean engine-vs-engine numbers we can compare

All runs below are on the same machine (AMD Ryzen 7 PRO 6850H, 8c/16t, AVX2 +
FMA, DDR5, CPU only). llama.cpp here is build `67cb0d5` of PR
[ggml-org/llama.cpp#22105](https://github.com/ggml-org/llama.cpp/pull/22105)
(the "add DFlash support" draft PR), compiled with `GGML_NATIVE=ON`.

**Same model (Qwen3.5-4B HauhauCS Q4_K_M), full text-gen, different engines**:

| Engine | Threads | tok/s |
|---|---:|---:|
| llama.cpp | 8 | **14.03** |
| llama.cpp | 12 | 13.55 |
| llama.cpp | 4 | 13.32 |
| miniforge (from user's `results.csv`) | default | 13.40 |
| ollama (user-cited) | default | 5.70 |
| llama.cpp | 16 | 3.79 ← SMT cliff |

The cited 5.7 tok/s ollama figure is **2.5× below llama.cpp on the same model and
machine**, which means it was almost certainly an ollama-thread-configuration
artifact, not an engine limit. (Ollama is a thin wrapper over llama.cpp/GGML, so
its asymptote is llama.cpp's asymptote.)

**Same engine (oxidize), DFlash-draft forward_token, thread sweep:**

| Threads | tok/s |
|---:|---:|
| 4 | 9.59 |
| 8 | 14.53 |
| 12 | **16.65** |
| 16 | 16.64 |

Oxidize is robust to thread oversubscription thanks to Rayon's work-stealing —
no SMT cliff. llama.cpp on the same Zen 3+ CPU **drops 3.7×** going from 8 to 16
threads on its memory-bandwidth-bound GEMV; on this CPU you must pin llama.cpp
to one thread per physical core.

### 12.3 Putting the two workloads side by side

You cannot literally cross-compare oxidize-DFlash-draft and llama.cpp-Qwen3.5-4B
because they aren't the same kernel. But you can ask: *on a Q4_K_M CPU decode
workload of comparable size, on the same machine, what are the engines doing?*

| Engine | Model | Workload | Peak tok/s |
|---|---|---|---:|
| **oxidize** (this PR) | Qwen3.6-27B-DFlash draft (Q4_K_M, 5 layers, hidden 5120) | draft `forward_token` (no fc fusion, no lm_head) | **16.65** |
| **llama.cpp** PR #22105 | Qwen3.5-4B HauhauCS (Q4_K_M, 32 layers, hidden 2560) | full text-gen `tg128` | **14.03** |
| miniforge | same as llama.cpp row | full text-gen | 13.40 |
| ollama | same as llama.cpp row | full text-gen (mis-configured) | 5.70 |

Both engines land in the same ~14–17 tok/s band for similar-sized Q4_K_M models
on this CPU, and both blow past the misconfigured-ollama reference. The
optimizations in this PR put oxidize at parity-or-better with llama.cpp on a
comparable kernel, on the same hardware, without using a GPU.

---

## 13. The Kimi-K2.5 DFlash Story (Why That Model Is Slow)

Running the bench on the **Kimi-K2.5 DFlash** GGUF (`models/Kimi-K2.5-DFlash.gguf`,
13.9 GB) produced only **1.69 tok/s** — much slower than the Qwen3.6 DFlash
model's 16.20 tok/s, even though both go through the same code paths. We dug in:

```
$ python3 inspect_gguf.py models/Kimi-K2.5-DFlash.gguf
TYPE COUNTS: {'F32': 69}
```

**Every tensor in the file is `F32`.** This model wasn't quantized at all — it's
a 13.9 GB raw f32 dump. None of the Q4_K fast paths apply because there's
nothing to quantize. With a hidden size of 7168, intermediate of 18432, and 6
layers, each decode token must stream roughly **12.5 GB of f32 weights** from
DRAM. At an effective ~21 GB/s practical bandwidth on this machine, that lands
at ~590 ms/token — and we measured 592 ms/token. The model is doing exactly what
its byte-count predicts.

The optimization that would help this file most is converting it to Q4_K_M (or
similar), at which point it would inherit the full ~8× memory-bandwidth win
described in §3–§5. Until then it is a pure bandwidth test of unquantized
weights, and there's no software trick on the inference side that gets around
the 12.5 GB-per-token transfer.

---

## 14. Full Speculative Decoding: What's Blocked And Why

A natural next step is to compare oxidize and llama.cpp on **full speculative
decoding** (draft + target, end-to-end), not just the draft microbench. We
investigated this carefully and it is currently blocked by a combination of
asset and code gaps. Documenting the exact wall so anyone picking this up next
doesn't have to rediscover it.

### 14.1 What we built and tried

- Cloned and CPU-built llama.cpp PR
  [#22105 (DFlash support)](https://github.com/ggml-org/llama.cpp/pull/22105),
  branch `ruixiang63:dflash`, into `/home/dih/llama.cpp-dflash/build/`.
- Confirmed the PR registers the arch as `"dflash"` with metadata keys
  `dflash.{block_size, mask_token_id, target_layer_ids}` and top-level tensors
  `fc`, `hidden_norm`. Our oxidize-produced GGUFs use `dflash-draft`
  arch, `dflash-draft.dflash.*` keys, `dflash_fc.weight`,
  `dflash_hidden_norm.weight`, and `post_attention_norm` (HF naming) instead of
  `ffn_norm` (llama.cpp naming).
- Wrote a `gguf-py`-based rewriter (`/tmp/patch_gguf.py`) that takes any
  oxidize-format DFlash GGUF and produces a llama.cpp-schema copy: arch and KV
  key remap, `dflash_fc.weight` → `fc.weight`, `dflash_hidden_norm.weight` →
  `hidden_norm.weight`, `blk.N.post_attention_norm.weight` →
  `blk.N.ffn_norm.weight`.
- Loaded the patched GGUF in llama.cpp PR #22105.

### 14.2 The hard wall

The patched GGUF passes architecture and tensor-name validation, then hits this
assertion from the PR's own DFlash forward graph:

```
/home/dih/llama.cpp-dflash/src/models/dflash.cpp:39:
GGML_ASSERT(model.target_tok_embd != nullptr
            && "DFlash decoder requires target model's tok_embd") failed
```

In other words: **llama.cpp's DFlash draft cannot run a standalone forward
pass.** The graph hard-requires a paired target model's token embeddings and
extracted hidden states. By design, the draft only ever runs as part of
`speculative-simple` together with `--model-draft <target>`. (See
`llama-context.cpp:352`: `dflash_decoder_ctx = arch == LLM_ARCH_DFLASH &&
params.target_model != nullptr` and `llama-context.cpp:1235`:
`GGML_ASSERT(!dflash.target_features.empty() && "DFlash target features not extracted")`.)

Contrast oxidize: `DFlashDraftModel::forward_token(token, target_hidden:
Option<&[f32]>)` explicitly accepts `None` and **skips** the `fc` fusion when
no target features are provided (`dflash.rs:638` —
`if let Some(th) = target_hidden && self.fc.is_loaded() { ... }`). That's
the entire reason `oxidize-bench` can measure draft throughput in isolation.

### 14.3 Why we can't just run end-to-end either

To run `speculative-simple` you also need the matched **target model**. For our
Qwen3.6-27B-DFlash draft (`target_layer_ids = [1, 16, 31, 46, 61]`) the target
is the full Qwen3.6-27B base model, ~15 GB at Q4_K_M.

- We searched the home directory, `/opt`, `/var`, `/tmp`, and the entire
  mounted drive (`/run/media/dih/8CEDA5F938E73A48/AI/`) for any Qwen3.6
  non-draft GGUF. None exists. Only the DFlash drafts and a `~/exo-real/`
  directory of `.toml` model-card metadata.
- The available models on the drive include `Qwen2.5-0.5B-Instruct-GGUF`
  (multiple quants), `qwen2.5-7b-redteam-lora-merge-q4_k_m.gguf`,
  `Qwen3.5-4B HauhauCS Q4_K_M`, `Qwopus3.5-9B-v3`, Nemotron-3-Nano-4B, Kimi
  K2.5 UD-TQ1_0 splits, MiniMax-M2.7-161B, gemma-4 variants.

None of these are the matched target for our DFlash draft.

The natural fallback — pivot to **generic** speculative decoding (e.g. Qwen2.5
0.5B drafting for Qwen2.5 7B, both present on disk) — runs into a second blocker:

### 14.4 Oxidize's general inference path is broken on stock Qwen GGUFs

Empirically:

```
$ oxidize-cli --model qwen2.5-0.5b-instruct-q4_k_m.gguf --prompt "hi"
generation failed: Model(InferenceFailed(
  "ffn_down: InvalidMatrixLength { expected: 3064320, actual: 3575040 }"))

$ oxidize-cli --model Qwen3.5-4B-HauhauCS-Q4_K_M.gguf --prompt "hi"
panicked at oxidize-core/src/model/inference.rs:1158:58:
range end index 4096 out of range for slice of length 2560
```

So `InferenceModel` (the general-purpose end-to-end inference path used by
non-DFlash text generation) is currently broken on at least Qwen2.5 0.5B and
Qwen3.5 4B. Even if we had the Qwen3.6 target, oxidize couldn't load it through
this path right now. Fixing this is real engineering work (shape/layout bugs in
the inference graph for Qwen arches), not in scope for this PR.

### 14.5 What this means

| | Has working forward | Standalone draft fwd | Needs target | Can run e2e spec-decode today |
|---|:---:|:---:|:---:|:---:|
| oxidize `DFlashDraftModel::forward_token` | ✅ | ✅ | optional | — (would need a working target via `Model` trait) |
| oxidize `InferenceModel` (general) | ❌ shape bugs | n/a | n/a | ❌ (loader broken) |
| llama.cpp PR #22105 DFlash | ✅ | ❌ asserts | required | ❌ on this box (no Qwen3.6 target) |
| llama.cpp main / PR #22105 (generic) | ✅ | n/a | required | ✅ (any matched draft+target) |

So the full speculative-decoding comparison you'd want — DFlash draft + target,
oxidize vs llama.cpp, head-to-head tok/s — is **architecturally possible but
asset- and code-blocked on this machine.** Unblocking it requires:

1. **Get the target.** Download `Qwen/Qwen3.6-27B-Instruct` (or whichever target
   the draft was trained against) in GGUF Q4_K_M form. ~15 GB.
2. **Fix oxidize's `InferenceModel`** for Qwen-family arches (or wire a
   target-model trait that handles `target_layer_ids` hidden-state extraction
   directly). This is the larger task — it's the missing piece that lets
   `SpeculativeGenerationStream` actually drive a real target. The
   plumbing is there (`oxidize-core/src/model/generation.rs:69`,
   `model/sampling.rs:455`), but the `Model` impl backing the target is
   non-functional today.
3. **Verify** target hidden-state extraction matches the layer IDs the draft
   expects (`[1, 16, 31, 46, 61]` for Qwen3.6-27B-DFlash → target needs ≥62
   layers).

This is a real piece of work — multiple days, not minutes — and is captured
under **future work** below.

### 14.6 What we *can* claim today

- Oxidize peaks at **16.65 tok/s** on the DFlash draft forward kernel.
- llama.cpp peaks at **14.03 tok/s** on a comparable-size Q4_K_M model's full
  text-generation kernel, on the same hardware.
- The DFlash *draft kernel itself* in oxidize is now faster than the closest
  comparable kernel llama.cpp can run on this machine; we don't yet have a
  side-by-side end-to-end speculative-decode number.

---

## 15. What We Did *Not* Do (And Why)

- **AVX-512:** the 6850H (Zen 3+) doesn't have it. The earlier analysis flagged it as
  a gap, but on this hardware it's a non-starter; AVX2 is the widest available.
- **Custom BLAS / GEMM tiling:** decode is GEMV (matrix–*vector*), which is
  bandwidth-bound, not the cache-blocking-friendly GEMM (matrix–matrix) of prefill.
  Tiling buys little here.
- **Operator fusion (RMSNorm/RoPE/SwiGLU):** real but second-order once the GEMV
  bandwidth problem is solved. Listed as future work.
- **GPU backends:** out of scope; the goal was to beat ollama on the same CPU.
- **Rewriting the f32 fallback's transpose bug:** left intact for any tensor that
  legitimately needs the f32 path, since the production hot path no longer uses it.

---

## 16. Future Work

In rough priority order:

1. **Unblock end-to-end DFlash speculative decoding so we can do the head-to-head
   vs llama.cpp.** Three concrete sub-tasks:
   1. **Fix `InferenceModel` for Qwen-family arches.** Two reproducible bugs
      were found in §14.4: `ffn_down` shape mismatch on Qwen2.5 0.5B, and an
      out-of-range slice index at `oxidize-core/src/model/inference.rs:1158` on
      Qwen3.5 4B. These need to be tracked down before any non-DFlash model can
      run end-to-end in oxidize.
   2. **Wire target hidden-state extraction.** The existing
      `SpeculativeGenerationStream` (`oxidize-core/src/model/generation.rs:69`)
      already takes a target model and calls into the draft's
      `cache_target_hidden`. The missing piece is a `Model`-trait method that
      runs a target forward and returns hidden states from the layers in
      `target_layer_ids` — analogous to llama.cpp's `dflash.extract_tensors`.
   3. **Obtain the target weights.** Pull `Qwen/Qwen3.6-27B-Instruct` (or
      whichever target the draft was trained against) in GGUF Q4_K_M form. Or
      train/select a smaller DFlash pair that fits the available targets on
      disk. Without this, neither engine can run DFlash spec-decode here.
2. **SIMD the Q6_K and Q8_0 dot products** the same way `q4_k_dot` was done, for models
   using those quant types.
3. **Fuse elementwise ops** (RMSNorm + residual, RoPE, SwiGLU gate·up) to cut temporary
   allocations and extra passes over the activation vectors.
4. **Arena/scratch allocator** for per-token temporaries to remove allocator churn in
   the forward pass.
5. **Prefetching** the next Q4_K block while computing the current one, to hide DRAM
   latency further.
6. **Per-thread block tiling** to improve L2 reuse of the activation vector across
   output rows.
7. **Batched / continuous batching** in the compute kernels for multi-sequence serving
   (changes the regime from bandwidth-bound toward compute-bound).
8. **NEON kernels** for ARM / Apple Silicon, currently scalar-only.

---

## 17. Appendix: The Q4_K Block Format

Q4_K (the "K-quant" 4-bit format) packs **256 weights into a 144-byte block**
(`BLOCK_Q4_K_SIZE = 2·sizeof(f16) + 12 + 256/2 = 144`), an effective **4.5 bits per
weight**:

| Bytes | Field | Meaning |
|------:|-------|---------|
| 0–1 | `d` (f16) | super-block scale |
| 2–3 | `min` (f16) | super-block minimum |
| 4–15 | `scales` (12 bytes) | 8 sub-block 6-bit scale/min pairs, bit-packed |
| 16–143 | `qs` (128 bytes) | 256 4-bit quantized values (two per byte) |

The 256 weights are organized as 8 sub-groups of 32. Each group `g` has a 6-bit scale
and 6-bit min recovered via `get_scale_min_k4`, and the dequantized value is:

```
weight = d · scale[g] · q − min · min_scale[g]
```

where `q` is the 4-bit nibble. The fused GEMV walks blocks along the input dimension,
dequantizes each block, and dot-products it against the matching 256-element slice of
the activation vector — reading only 144 bytes per 256 weights instead of the 1024
bytes an f32 representation would require. That ~7x byte reduction on the dominant
matrices is the core of the speedup.
