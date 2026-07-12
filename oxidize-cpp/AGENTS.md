# oxidize-cpp

**Domain:** C++20 Llama-family inference engine (CPU + optional CUDA/ROCm)

## OVERVIEW
C++20 inference engine for Llama-family models with a benchmark CLI, an OpenAI-compatible HTTP server, and a from-scratch LoRA/full fine-tuning trainer. A core focus is **llama.cpp performance parity** for large-model CPU inference; the CUDA/ROCm backend adds a resident-forward fast path with graph replay on decode.

## STRUCTURE
```
oxidize-cpp/
├── CMakeLists.txt
├── include/oxidize/    # public headers
│   ├── model.hpp / model_llama.hpp   # Model interface + Llama impl
│   ├── gguf.hpp / tokenizer.hpp / tensor.hpp / quant.hpp / kernels.hpp
│   ├── sampler.hpp / autotune.hpp / numa_util.hpp / config.hpp
│   ├── cuda_backend.hpp / gpu_config.hpp
│   └── autograd.hpp / train_*.hpp / lora.hpp   # training
├── src/                # CPU oxidize_core library (model_llama, gguf, tokenizer, tensor_cpu, quant, kernels, sampler, vision, autotune, numa_util)
│   ├── cli/main.cpp            # benchmark CLI  -> oxidize-cpp
│   ├── cli/train_main.cpp      # training CLI   -> oxidize-cpp-train
│   ├── server/main.cpp + json  # HTTP server    -> oxidize-cpp-server
│   ├── cuda/*.cu               # GPU backend (backend, gemm, flash_attn, rope, rmsnorm, dequant, sampling, resident)
│   └── train/*.cpp             # autograd, forward, loss, optim, matmul, data, lora, grad_check
├── tests/              # parity_test, kernels_test, autotune_test
└── build/              # CMake artifacts (gitignored)
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Model forward / arch | `src/model_llama.cpp`, `include/oxidize/model_llama.hpp` | `LlamaModel`, `load_llama_gguf` |
| CPU kernels | `src/kernels.cpp`, `src/tensor_cpu.cpp` | GEMM/GEMV, attention |
| GPU backend | `src/cuda/*.cu` | Compiled as CUDA or reinterpreted as HIP |
| Autotune / NUMA | `src/autotune.cpp`, `src/numa_util.cpp` | `detect_hardware`, `plan_cpu`, `init_numa` |
| Server route | `src/server/main.cpp` | inference serialized behind a mutex |
| Training | `src/train/*.cpp` | autograd, LoRA/full SFT |
| Sampling | `src/sampler.cpp` | temperature/top-k/top-p/min-p/penalties |

## CLI
```text
oxidize-cpp --model <gguf> [--prompt --tokens "1,2,3" --max-tokens
            --cuda|--hip|--rocm|--gpu --seed --temperature --top-k --top-p --min-p
            --frequency-penalty --presence-penalty --no-cuda-graph --no-bos --stream
            --quantize q8_0 --numa single|interleave|all|replicate|<node>
            --threads --auto --no-auto --print-plan --json]
oxidize-cpp-train --model <gguf> --data <jsonl> [--mode lora|full --lr --steps
            --rank --alpha --seq-len --grad-accum --warmup --grad-clip --seed --overfit-one-batch]
oxidize-cpp-server --model <gguf>   # GET /health, GET /v1/models, POST /v1/chat/completions
oxidize-cpp-merge --a <model> --b <model> --output <path>   # SafeTensors merger
            [--method linear|slerp --t --preset kimi-k275 --attention-t --mlp-t
             --other-t --missing error|a|b --max-shard-gib --dry-run --self-test]
```

## BUILD / TEST / RUN
```bash
cmake -B build -S . && cmake --build build -j          # CPU
cmake -B build -DOXIDIZE_CUDA=ON  && cmake --build build -j   # CUDA
cmake -B build -DOXIDIZE_ROCM=ON  && cmake --build build -j   # ROCm
ctest --test-dir build                                  # parity/kernels/autotune/grad_check
./build/oxidize-cpp --model model.gguf --prompt "hi"
```

## NOTES
- `OXIDIZE_CUDA` (nvcc + cuBLAS/cublasLt, arch default `80;90`) and `OXIDIZE_ROCM` (hipcc + hipBLAS, arch default `gfx1100;gfx1030;gfx1012`) are mutually exclusive, both OFF by default.
- Benchmark CPU deployments on the NUMA box `ai@192.168.1.132`; dense models ≤192 GB use `--numa single --threads 16`, >192 GB use `--numa interleave --threads 48`.
- Shares the custom AL-family quant types with `oxidize-core` and `oxidize-c`.
- `oxidize-cpp-glm/` is a separate GLM fork (MLA/IQ1/MoE).
