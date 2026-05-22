# Qwen3.6-35B-A3B-DFlash

[**Paper**](https://arxiv.org/abs/2602.06036) \| [**GitHub**](https://github.com/z-lab/dflash) \| [**Blog**](https://z-lab.ai/projects/dflash/)

**DFlash** is a speculative decoding method that uses a lightweight **block diffusion** model to draft multiple tokens in parallel. This is the drafter model, which must be paired with [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B).

![DFlash Architecture](https://huggingface.co/z-lab/Qwen3.6-35B-A3B-DFlash/resolve/main/assets/dflash_system.png)

## Quick Start

### Installation

vLLM (We temporarily modify the installation through this PR to support interleaved SWA and ensure correct handling of target hidden states for optimal performance):

```bash
uv pip install vllm
uv pip install -U --torch-backend=auto "vllm @ git+https://github.com/vllm-project/vllm.git@refs/pull/40898/head"
```

SGLang:

```bash
uv pip install "git+https://github.com/sgl-project/sglang.git@refs/pull/20547/head#subdirectory=python"
```

### Launch Server

vLLM:

```bash
vllm serve Qwen/Qwen3.6-35B-A3B \
  --speculative-config '{"method": "dflash", "model": "z-lab/Qwen3.6-35B-A3B-DFlash", "num_speculative_tokens": 15}' \
  --attention-backend flash_attn \
  --max-num-batched-tokens 32768
```

SGLang:

```bash
# Optional: enable schedule overlapping (experimental, may not be stable)
# export SGLANG_ENABLE_SPEC_V2=1
# export SGLANG_ENABLE_DFLASH_SPEC_V2=1
# export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1

python -m sglang.launch_server \
    --model-path Qwen/Qwen3.6-35B-A3B \
    --speculative-algorithm DFLASH \
    --speculative-draft-model-path z-lab/Qwen3.6-35B-A3B-DFlash \
    --speculative-num-draft-tokens 16 \
    --tp-size 1 \
    --attention-backend fa3 \
    --mem-fraction-static 0.75 \
    --mamba-scheduler-strategy extra_buffer \
    --trust-remote-code
```

> **Tip:** For long-context or agentic workloads, add `--speculative-dflash-draft-window-size WINDOW_SIZE` to enable sliding-window attention for the drafter.

### Usage

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:30000/v1", api_key="EMPTY")

response = client.chat.completions.create(
    model="Qwen/Qwen3.6-35B-A3B",
    messages=[{"role": "user", "content": "Write a quicksort in Python."}],
    max_tokens=4096,
    temperature=0.0
)
print(response.choices[0].message.content)
```

## Benchmark Results

**Setup:** Single NVIDIA B200, SGLang, thinking enabled, max output length 4096. We report end-to-end throughput, including prefill time. See our [GitHub repository](https://github.com/z-lab/dflash) for reproduction scripts.

### Throughput and Speedup

DFlash achieves up to **2.9x** speedup at concurrency 1.

_Tokens/sec (speedup vs. autoregressive baseline)_

**Block Size = 16**

| Task | Concurrency | AR | **DFlash** |
| --- | --: | --: | --: |
| Math500 | 1 | 234 | **682 (2.9x)** |
|  | 8 | 1266 | **3138 (2.5x)** |
|  | 16 | 1954 | **4813 (2.5x)** |
|  | 32 | 2755 | **6520 (2.4x)** |
| GSM8K | 1 | 235 | **556 (2.4x)** |
|  | 8 | 1236 | **2564 (2.1x)** |
|  | 16 | 1886 | **3821 (2.0x)** |
|  | 32 | 2699 | **5239 (1.9x)** |
| HumanEval | 1 | 238 | **603 (2.5x)** |
|  | 8 | 1255 | **2800 (2.2x)** |
|  | 16 | 1944 | **4208 (2.2x)** |
|  | 32 | 2767 | **5782 (2.1x)** |
| MBPP | 1 | 235 | **559 (2.4x)** |
|  | 8 | 1224 | **2538 (2.1x)** |
|  | 16 | 1948 | **3816 (2.0x)** |
|  | 32 | 2780 | **5378 (1.9x)** |
| MT-Bench | 1 | 233 | **442 (1.9x)** |
|  | 8 | 1238 | **2028 (1.6x)** |
|  | 16 | 1885 | **2997 (1.6x)** |
|  | 32 | 2633 | **4034 (1.5x)** |
| Alpaca | 1 | 235 | **393 (1.7x)** |
|  | 8 | 1221 | **1782 (1.5x)** |
|  | 16 | 1844 | **2567 (1.4x)** |
|  | 32 | 2579 | **3689 (1.4x)** |

**Block Size = 8**

| Task | Concurrency | AR | **DFlash** |
| --- | --: | --: | --: |
| Math500 | 1 | 234 | **617 (2.6x)** |
|  | 8 | 1266 | **2839 (2.2x)** |
|  | 16 | 1954 | **4465 (2.3x)** |
|  | 32 | 2755 | **6614 (2.4x)** |
| GSM8K | 1 | 235 | **540 (2.3x)** |
|  | 8 | 1236 | **2466 (2.0x)** |
|  | 16 | 1886 | **3899 (2.1x)** |
|  | 32 | 2699 | **5713 (2.1x)** |
| HumanEval | 1 | 238 | **561 (2.4x)** |
|  | 8 | 1255 | **2655 (2.1x)** |
|  | 16 | 1944 | **4135 (2.1x)** |
|  | 32 | 2767 | **6059 (2.2x)** |
| MBPP | 1 | 235 | **497 (2.1x)** |
|  | 8 | 1224 | **2324 (1.9x)** |
|  | 16 | 1948 | **3636 (1.9x)** |
|  | 32 | 2780 | **4884 (1.8x)** |
| MT-Bench | 1 | 233 | **438 (1.9x)** |
|  | 8 | 1238 | **2060 (1.7x)** |
|  | 16 | 1885 | **3182 (1.7x)** |
|  | 32 | 2633 | **4720 (1.8x)** |
| Alpaca | 1 | 235 | **407 (1.7x)** |
|  | 8 | 1221 | **1880 (1.5x)** |
|  | 16 | 1844 | **2903 (1.6x)** |
|  | 32 | 2579 | **4115 (1.6x)** |

### Acceptance Length

| Task | B8 | B16 |
| --- | --: | --: |
| Math500 | 5.56 | 7.35 |
| GSM8K | 5.21 | 6.73 |
| HumanEval | 5.09 | 6.44 |
| MBPP | 4.78 | 5.83 |
| MT-Bench | 4.20 | 5.14 |
| Alpaca | 3.94 | 4.62 |

## Acknowledgements

Special thanks to [David Wang](https://davidwa.ng/) for his outstanding engineering support on this project. We are also grateful to [Modal](https://modal.com/), [InnoMatrix](https://innomatrix.ai/), and [Yotta Labs](https://www.yottalabs.ai/) for providing the compute resources used to train this draft model.

## Citation

If you find DFlash useful, please cite our work. To share feedback on DFlash or request new model support, please fill out this form: [DFlash Feedback](https://forms.gle/4YNwfqb4nJdqn6hq9).

```bibtex
@article{chen2026dflash,
  title   = {{DFlash: Block Diffusion for Flash Speculative Decoding}},
  author  = {Chen, Jian and Liang, Yesheng and Liu, Zhijian},
  journal = {arXiv preprint arXiv:2602.06036},
  year    = {2026}
}
```

Downloads last month70,189

Safetensors

Model size

0.5B params

Tensor type

BF16

·

Files info

Inference Providers [NEW](https://huggingface.co/docs/inference-providers)

[Text Generation](https://huggingface.co/tasks/text-generation "Learn more about text-generation")

This model isn't deployed by any Inference Provider. [🙋Ask for provider support](https://huggingface.co/spaces/huggingface/InferenceSupport/discussions/new?title=z-lab/Qwen3.6-35B-A3B-DFlash&description=React%20to%20this%20comment%20with%20an%20emoji%20to%20vote%20for%20%5Bz-lab%2FQwen3.6-35B-A3B-DFlash%5D(%2Fz-lab%2FQwen3.6-35B-A3B-DFlash)%20to%20be%20supported%20by%20Inference%20Providers.%0A%0A(optional)%20Which%20providers%20are%20you%20interested%20in%3F%20(Novita%2C%20Hyperbolic%2C%20Together%E2%80%A6)%0A)

## Model tree for z-lab/Qwen3.6-35B-A3B-DFlash

Finetunes

[2 models](https://huggingface.co/models?other=base_model:finetune:z-lab/Qwen3.6-35B-A3B-DFlash)

Quantizations

[Use with llama.cpp](https://huggingface.co/models?apps=llama.cpp&other=base_model:quantized:z-lab/Qwen3.6-35B-A3B-DFlash "Use with llama.cpp")[Use with LM Studio](https://huggingface.co/models?apps=lmstudio&other=base_model:quantized:z-lab/Qwen3.6-35B-A3B-DFlash "Use with LM Studio")[Use with Jan](https://huggingface.co/models?apps=jan&other=base_model:quantized:z-lab/Qwen3.6-35B-A3B-DFlash "Use with Jan")[Use with Ollama](https://huggingface.co/models?apps=ollama&other=base_model:quantized:z-lab/Qwen3.6-35B-A3B-DFlash "Use with Ollama")

[6 models](https://huggingface.co/models?other=base_model:quantized:z-lab/Qwen3.6-35B-A3B-DFlash)

## Space using z-lab/Qwen3.6-35B-A3B-DFlash1

## Collection including z-lab/Qwen3.6-35B-A3B-DFlash

[Block Diffusion for Flash Speculative Decoding•21 items•Updated 11 days ago• 117](https://huggingface.co/collections/z-lab/dflash)

## Paper for z-lab/Qwen3.6-35B-A3B-DFlash

[Paper • 2602.06036 •Published Feb 5• 80](https://huggingface.co/papers/2602.06036)