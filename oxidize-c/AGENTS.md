# oxidize-c

**Domain:** Dependency-free C11 port of the oxidize LLM runtime

## OVERVIEW
Single-binary C11 implementation of the full inference stack: GGUF parsing, quant/dequant, tokenizer, model forward, generation, an OpenAI-compatible HTTP/WebSocket server, pruning, and LoRA finetuning, plus an optional CUDA fast path. Intentional standalone port (the workspace anti-pattern "don't use C for what Rust can do" does not apply to this deliberate C target). Only external link dependency is `-lm`; OpenMP is used when available.

## STRUCTURE
```
oxidize-c/
├── Makefile           # `make`, `make cuda`, `make test`, `make clean`
├── oc.h               # Single public header: full runtime surface
├── main.c             # CLI entry, one-shot gen loop, prune/finetune/--serve dispatch
├── gguf.c             # GGUF mmap loader + metadata/tensor lookup + writer
├── quant.c            # quant/dequant, block sizing, int8 activation quant, AL5/AL6/AL8/AL5_XS
├── kernels.c          # RMSNorm, RoPE (NeoX+YaRN), SwiGLU/GeGLU, softmax, gemv/gemm, attention
├── model.c            # load, oc_forward*, KV cache, MoE, Gated-DeltaNet, MTP draft
├── tokenizer.c        # GGUF-embedded tokenizer (tokenize/detokenize/EOG)
├── gen.c / gen.h      # sampling + generation driver, speculative decode (ngram/MTP)
├── server.c           # POSIX-socket OpenAI-compatible HTTP + WebSocket server
├── prune.c            # oc_prune_main (magnitude/Wanda/N:M masks)
├── finetune.c         # oc_finetune_main (LoRA + AdamW + CE-grad)
├── cuda.cu            # optional resident-CUDA forward path (OC_CUDA)
├── oc_iq_grids.h      # IQ dequant grids
└── test_oc.c          # unit test harness
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Add CLI flag | `main.c` | Arg parsing + generation loop |
| GGUF read/write | `gguf.c` | mmap loader, writer for pruned output |
| Add quant type | `quant.c` | AL-family types 240–243 live here |
| Kernel change | `kernels.c` | gemv/gemm, attention (f32 + int8 KV) |
| Forward / MoE / GDN | `model.c` | `oc_forward`, `oc_forward_all`, `oc_forward_train` |
| Server endpoint | `server.c` | `/v1/chat/completions`, `/v1/completions`, `/v1/realtime` |
| CUDA fast path | `cuda.cu` | Requires `OC_CUDA` define + `make cuda` |

## CLI
```text
oxidize-c --model <path.gguf> [--prompt --chat --max-tokens --ctx --kv-int8
          --draft --spec ngram|mtp|off --threads --temperature --top-k --top-p
          --seed --no-bos --stream --serve --port --host]
oxidize-c prune ...
oxidize-c finetune ...
```
Server: `POST /v1/chat/completions` and `POST /v1/completions` support SSE streaming; `GET /health` reports readiness; `ws://HOST:PORT/v1/realtime` is a WebSocket token-stream endpoint. Auth uses `OXIDIZE_API_KEY`/`OXIDIZE_API_KEYS`.

## BUILD / TEST / RUN
```bash
make            # -> ./oxidize-c (CFLAGS default: -O3 -march=native -flto -fopenmp)
make cuda       # -> ./oxidize-c-cuda (nvcc, -DOC_CUDA, cuBLAS/cudart)
make test       # compiles + runs ./test_oc
make clean
./oxidize-c --model model.gguf --prompt "hi" --stream
```

## NOTES
- Custom AL-family quant types (`AL5`, `AL5_XS`, `AL6`, `AL8`; ggml types 240–243) are shared with `oxidize-core` and `oxidize-cpp` — keep them bit-compatible.
- Int8 KV cache, MoE, Gated-DeltaNet linear-attention layers, and MTP speculative decode are all supported.
