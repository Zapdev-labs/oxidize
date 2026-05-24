# llama.cpp Features Missing from oxidize — Research Report

**Date:** 2026-05-22  
**Research method:** Web search, firecrawl scraping of llama.cpp source/docs, codebase grep of oxidize.

---

## Summary

llama.cpp has evolved significantly beyond oxidize's current feature set. The biggest gaps are in **GPU acceleration**, **advanced quantization (IQ quants, FP4)**, **multimodal/vision support**, **advanced sampling (XTC, DRY, dynamic temp)**, **structured output/grammar**, **function calling**, and **modern model architectures**. Below is a prioritized list of missing features.

---

## 1. 🔴 CRITICAL / HIGH IMPACT

### 1.1 GPU Acceleration Backends
**What llama.cpp has:** Full-featured CUDA, Metal (Apple Silicon), Vulkan, SYCL (Intel), HIP/ROCm (AMD) backends with optimized kernels. Multi-GPU tensor parallelism (added April 2026).  
**What oxidize has:** CPU-only inference with AVX2. Nascent GPU backend stubs (CUDA/Metal/WebGPU mentioned in perf reports but not competitive).  
**Impact:** GPU inference is 10–100× faster than CPU. This is the single biggest performance gap.

### 1.2 Importance-Matrix (I-Quants) Quantization
**What llama.cpp has:** IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ2_M, IQ3_XXS, IQ3_XS, IQ3_S, IQ3_M, IQ4_XS, IQ4_NL — all using importance-matrix (imatrix) aware quantization. Q1_0 (1-bit, April 2026). FP4/NVFP4 support (April 2026). Per-tensor quantization via regex.  
**What oxidize has:** Legacy quants (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K*, Q4_K, Q5_K, Q6_K). No IQ quants. No imatrix. No FP4.  
**Impact:** IQ quants achieve much better quality at lower bit-widths (e.g., IQ4_XS ~4.46 bpw outperforms Q4_K_M). Missing these means oxidize models are larger and/or lower quality for the same size.

### 1.3 Vision / Multimodal Model Support
**What llama.cpp has:** Full multimodal support via `libmtmd` (merged May 2025). Supports Llama 3.2 Vision, Qwen 2.5 VL, and other vision-language models in both server and CLI.  
**What oxidize has:** No vision/multimodal support.  
**Impact:** VLM support is now table-stakes for modern inference engines.

### 1.4 Advanced KV Cache Quantization & Management
**What llama.cpp has:** KV cache quant types: q4_0, q8_0, and more. TurboQuant research (under 3 bits, May 2026). FP8 KV cache. FlashAttention (April 2024). KV cache defragmentation. KV cache offloading.  
**What oxidize has:** KV cache in f32/f16/q8/q4. No FlashAttention. No defragmentation. No sub-4-bit KV cache.  
**Impact:** For long-context inference, KV cache memory dominates. FlashAttention and KV quant are essential for 32K+ contexts on consumer hardware.

### 1.5 Function Calling / Tool Use (OpenAI-compatible)
**What llama.cpp has:** Native OpenAI-style function calling in `llama-server` with `--jinja`. Supports native formats for Llama 3.x, Hermes 2/3, Qwen 2.5, Mistral Nemo, Firefunction v2, Command R7B, DeepSeek R1. Parallel tool calling. Generic fallback for unsupported templates.  
**What oxidize has:** Basic OpenAI-compatible HTTP server but no function calling / tool use support.  
**Impact:** Function calling is critical for agentic use cases and modern LLM applications.

---

## 2. 🟠 MEDIUM-HIGH IMPACT

### 2.1 Advanced Sampling Methods
**What llama.cpp has:**
- **XTC (Exclude Top Choices)** sampler — reduces repetition/GPT-isms without losing coherence
- **DRY (Don't Repeat Yourself)** sampler — configurable multiplier, base, allowed length, penalty window, sequence breakers
- **Dynamic Temperature (dynatemp)** — range + exponent parameters
- **Adaptive-p** sampling
- **Sampler chains** — fully configurable sampler pipeline (`--samplers` with `;` separated list)
- **Grammar-first sampling** option
- **Custom sampler sequences** via `--sampler-seq`

**What oxidize has:** Temperature, top-k, top-p, min-p, typical_p, tail_free_z, locally_typical_tau, mirostat, basic repetition penalties (frequency + presence + newline). No XTC, no DRY, no dynamic temp, no adaptive-p, no configurable sampler chains.

**Impact:** XTC and DRY are highly popular in the local-LLM community for creative writing and reducing AI-generated text patterns.

### 2.2 Grammar / Structured Output (GBNF)
**What llama.cpp has:** Full GBNF (GGML BNF) grammar engine with efficient state-machine implementation. JSON schema → grammar conversion (`json-schema-to-grammar.py`). `--grammar`, `--grammar-file`, `--json-schema` flags. Applied to sampled token first, then full vocab if needed (performance optimization).  
**What oxidize has:** Basic `GrammarConstraint` with BFS-based prefix acceptance check (very slow, does full search per token). No GBNF parser. No JSON-schema→grammar conversion.  
**Impact:** Oxidize's grammar implementation is functional but not efficient. GBNF + JSON schema is critical for reliable structured outputs.

### 2.3 Modern Model Architecture Support
**What llama.cpp has:** DeepSeek V2/V3/R1 architectures (with MLA attention), Llama 4 (Scout/Maverick), Qwen3 (including MoE), Gemma 4 (with audio support, April 2026), more MoE variants, vision encoders (CLIP, SigLIP), reranker models, embedding models.  
**What oxidize has:** Llama2, Llama3, Mistral, Mixtral, Qwen, Gemma, Phi, Falcon, GPT2, GPT-J, GPT-NeoX. No DeepSeek. No Llama 4. No Qwen3. No Gemma 4. No vision encoders. No rerankers. No embedding models.  
**Impact:** New models are adopting novel architectures (DeepSeek MLA, MoE) that can't be loaded without explicit support.

### 2.4 Embedding & Reranker Inference
**What llama.cpp has:** `--embedding` mode for llama-server. Reranker support (Qwen3 reranker merged). Reranking API endpoint.  
**What oxidize has:** No embedding endpoint. No reranker support.  
**Impact:** Essential for RAG pipelines and semantic search applications.

### 2.5 Batched / Continuous Batching in Server
**What llama.cpp has:** Continuous batching in `llama-server` for concurrent requests. Batched inference APIs.  
**What oxidize has:** Basic server but no continuous batching mentioned.  
**Impact:** Throughput for production serving is much higher with continuous batching.

---

## 3. 🟡 MEDIUM IMPACT

### 3.1 Jinja Chat Templates
**What llama.cpp has:** Full Jinja chat template support (`--jinja`, `--chat-template-file`). Template override. Tool-aware templates.  
**What oxidize has:** No Jinja template support for chat formatting.  
**Impact:** Chat templates are model-specific and critical for correct instruction-following behavior.

### 3.2 Importance Matrix Generation (imatrix)
**What llama.cpp has:** `llama-imatrix` tool generates importance matrices for quantization optimization.  
**What oxidize has:** No imatrix generation tool.  
**Impact:** Imatrix-aware quants are significantly higher quality; without the tool, users can't generate them.

### 3.3 Model Conversion Pipeline
**What llama.cpp has:** `convert_hf_to_gguf.py` converts HuggingFace models (PyTorch/Safetensors) to GGUF. Supports many architectures.  
**What oxidize has:** GGUF loader only; no conversion from HF formats.  
**Impact:** Users must rely on external tools to get models into oxidize-compatible GGUF format.

### 3.4 Advanced Speculative Decoding
**What llama.cpp has:** Draft model loading for speculative decoding. `common_sampler_sample_and_accept_n` for batched draft verification.  
**What oxidize has:** Basic speculative decode logic (single-draft-token verify) but no draft model loading infrastructure.  
**Impact:** Production speculative decoding needs a separate small draft model loaded alongside the target.

### 3.5 Prompt Caching / Context Shift
**What llama.cpp has:** Prompt caching for faster repeated queries. Context shift / rewind mechanisms.  
**What oxidize has:** KV cache exists but no explicit prompt caching or context shift features.  
**Impact:** Important for chat applications with long context windows.

### 3.6 Diffusion Model Support
**What llama.cpp has:** `llama-dream` / diffusion algorithm temperature support.  
**What oxidize has:** No diffusion model support.  
**Impact:** Niche — only relevant if oxidize wants to support image generation.

### 3.7 Layer Pruning
**What llama.cpp has:** `--prune-layers` in `llama-quantize` to remove layers during quantization.  
**What oxidize has:** No layer pruning.  
**Impact:** Useful for creating ultra-small models for edge deployment.

---

## 4. 🟢 LOWER IMPACT / NICE-TO-HAVE

### 4.1 Reasoning Budget Sampler
**What llama.cpp has:** Reasoning budget sampler for controlling thinking tokens (used with DeepSeek R1).  
**What oxidize has:** Not present.  

### 4.2 Windows Prebuilt Binaries with Multi-Backend
**What llama.cpp has:** Prebuilt Windows binaries with CUDA, Vulkan, HIP, SYCL variants.  
**What oxidize has:** Rust cargo build only.  

### 4.3 NPU Backend Support
**What llama.cpp has:** Experimental NPU backends (mentioned in Wikipedia).  
**What oxidize has:** Not present.  

### 4.4 Re-quantization Controls
**What llama.cpp has:** `--allow-requantize`, `--leave-output-tensor`, `--pure`, `--output-tensor-type`, `--token-embedding-type` for fine-grained quantization control.  
**What oxidize has:** Basic quantization but no per-tensor or re-quantization controls.  

---

## Priority Matrix

| Priority | Feature Area | Key Missing Items |
|----------|-------------|-------------------|
| 🔴 P0 | GPU Backends | CUDA, Metal, Vulkan, SYCL, HIP, Tensor Parallelism |
| 🔴 P0 | IQ Quantization | IQ1–IQ4 variants, imatrix, FP4/NVFP4, Q1_0 |
| 🔴 P0 | Vision Models | CLIP/vision tower, multimodal inference |
| 🔴 P0 | KV Cache | FlashAttention, KV q4_0/q8_0, defrag, TurboQuant |
| 🔴 P0 | Function Calling | OpenAI-style tools, parallel calls, Jinja templates |
| 🟠 P1 | Advanced Sampling | XTC, DRY, dynamic temp, adaptive-p, sampler chains |
| 🟠 P1 | Grammar/Structured Output | GBNF engine, JSON schema→grammar, efficient token filtering |
| 🟠 P1 | Modern Architectures | DeepSeek V2/V3/R1, Llama 4, Qwen3, Gemma 4, MoE variants |
| 🟠 P1 | Embedding/Reranker | Embedding endpoint, reranker models, reranking API |
| 🟠 P1 | Server Batching | Continuous batching for concurrent requests |
| 🟡 P2 | Chat Templates | Jinja template parsing, tool-aware templates |
| 🟡 P2 | Model Conversion | convert_hf_to_gguf equivalent |
| 🟡 P2 | Speculative Decode | Draft model loading, batched verification |
| 🟡 P2 | Prompt Caching | Cache/reuse KV for repeated prefixes |
| 🟢 P3 | Misc | Layer pruning, diffusion, reasoning budget, prebuilt binaries |

---

## Source References

- llama.cpp quantize README: https://github.com/ggml-org/llama.cpp/tree/master/tools/quantize
- llama.cpp function calling docs: https://github.com/ggml-org/llama.cpp/tree/master/docs/function-calling.md
- llama.cpp multimodal docs: https://github.com/ggml-org/llama.cpp/tree/master/docs/multimodal.md
- llama.cpp sampling.h / sampling.cpp (source)
- llama.cpp arg.cpp (CLI parameters)
- llama.cpp April 2026 releases (tensor parallelism, Q1_0, Gemma 4 audio): https://fazm.ai/blog/llama-cpp-release-april-2026
- TurboQuant discussion: https://github.com/ggml-org/llama.cpp/discussions/20969
- FP4 inference discussion: https://www.reddit.com/r/LocalLLaMA/comments/1svfjyv/fp4_inference_in_llamacpp
- Vision support PR: https://simonwillison.net/2025/May/10/llama-cpp-vision/
- Build docs (backends): https://github.com/ggml-org/llama.cpp/tree/master/docs/build.md
