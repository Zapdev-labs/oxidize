# To run Qwen3.5 locally - [Read our Guide!](https://unsloth.ai/docs/models/qwen3.5)

_[Unsloth Dynamic 2.0](https://unsloth.ai/docs/basics/unsloth-dynamic-v2.0-gguf) achieves superior accuracy & outperforms other leading quants._

[![](https://github.com/unslothai/unsloth/raw/main/images/unsloth%20new%20logo.png)](https://github.com/unslothai/unsloth/)[![](https://github.com/unslothai/unsloth/raw/main/images/Discord%20button.png)](https://discord.gg/unsloth)[![](https://raw.githubusercontent.com/unslothai/unsloth/refs/heads/main/images/documentation%20green%20button.png)](https://unsloth.ai/docs/models/qwen3.5)

- You can now also fine-tune the model locally with [Unsloth](https://github.com/unslothai/unsloth).
- Read our [Qwen3.5 fine-tuning guide here](https://unsloth.ai/docs/models/qwen3.5/fine-tune).

* * *

# Qwen3.5-4B

![](https://qianwen-res.oss-accelerate.aliyuncs.com/logo_qwen3.5.png)

[![Qwen Chat](https://img.shields.io/badge/%F0%9F%92%9C%EF%B8%8F%20Qwen%20Chat%20-536af5)](https://chat.qwen.ai/)

> This repository contains model weights and configuration files for the post-trained model in the Hugging Face Transformers format.
>
> These artifacts are compatible with Hugging Face Transformers, vLLM, SGLang, KTransformers, etc.

Over recent months, we have intensified our focus on developing foundation models that deliver exceptional utility and performance. Qwen3.5 represents a significant leap forward, integrating breakthroughs in multimodal learning, architectural efficiency, reinforcement learning scale, and global accessibility to empower developers and enterprises with unprecedented capability and efficiency.

## Qwen3.5 Highlights

Qwen3.5 features the following enhancement:

- **Unified Vision-Language Foundation**: Early fusion training on multimodal tokens achieves cross-generational parity with Qwen3 and outperforms Qwen3-VL models across reasoning, coding, agents, and visual understanding benchmarks.

- **Efficient Hybrid Architecture**: Gated Delta Networks combined with sparse Mixture-of-Experts deliver high-throughput inference with minimal latency and cost overhead.

- **Scalable RL Generalization**: Reinforcement learning scaled across million-agent environments with progressively complex task distributions for robust real-world adaptability.

- **Global Linguistic Coverage**: Expanded support to 201 languages and dialects, enabling inclusive, worldwide deployment with nuanced cultural and regional understanding.

- **Next-Generation Training Infrastructure**: Near-100% multimodal training efficiency compared to text-only training and asynchronous RL frameworks supporting massive-scale agent scaffolds and environment orchestration.


![Benchmark Results](https://qianwen-res.oss-accelerate-overseas.aliyuncs.com/Qwen3.5/Figures/qwen3.5_small_size_score.png)

For more details, please refer to our blog post [Qwen3.5](https://qwen.ai/blog?id=qwen3.5).

## Model Overview

- Type: Causal Language Model with Vision Encoder
- Training Stage: Pre-training & Post-training
- Language Model
  - Number of Parameters: 4B
  - Hidden Dimension: 2560
  - Token Embedding: 248320 (Padded)
  - Number of Layers: 32
  - Hidden Layout: 8 × (3 × (Gated DeltaNet → FFN) → 1 × (Gated Attention → FFN))
  - Gated DeltaNet:
    - Number of Linear Attention Heads: 32 for V and 16 for QK
    - Head Dimension: 128
  - Gated Attention:
    - Number of Attention Heads: 16 for Q and 4 for KV
    - Head Dimension: 256
    - Rotary Position Embedding Dimension: 64
  - Feed Forward Network:
    - Intermediate Dimension: 9216
  - LM Output: 248320 (Tied to token embedding)
  - MTP: trained with multi-steps
- Context Length: 262,144 natively and extensible up to 1,010,000 tokens.

## Benchmark Results

### Language

|  | GPT-OSS-120B | GPT-OSS-20B | Qwen3-Next-80B-A3B-Thinking | Qwen3-30BA3B-Thinking-2507 | Qwen3.5-9B | Qwen3.5-4B |
| --- | --- | --- | --- | --- | --- | --- |
| Knowledge & STEM |
| MMLU-Pro | 80.8 | 74.8 | 82.7 | 80.9 | 82.5 | 79.1 |
| MMLU-Redux | 91.0 | 87.8 | 92.5 | 91.4 | 91.1 | 88.8 |
| C-Eval | 76.2 | 71.4 | 89.7 | 87.4 | 88.2 | 85.1 |
| SuperGPQA | 54.6 | 48.5 | 60.8 | 56.8 | 58.2 | 52.9 |
| GPQA Diamond | 80.1 | 71.5 | 77.2 | 73.4 | 81.7 | 76.2 |
| Instruction Following |
| IFEval | 88.9 | 88.2 | 88.9 | 88.9 | 91.5 | 89.8 |
| IFBench | 69.0 | 65.1 | 61.5 | 51.5 | 64.5 | 59.2 |
| MultiChallenge | 45.3 | 40.1 | 51.3 | 46.5 | 54.5 | 49.0 |
| Long Context |
| AA-LCR | 50.7 | 30.7 | 51.7 | 49.0 | 63.0 | 57.0 |
| LongBench v2 | 48.2 | 45.6 | 48.0 | 44.8 | 55.2 | 50.0 |
| Reasoning & Coding |
| HMMT Feb 25 | 90.0 | 76.7 | 73.7 | 63.1 | 83.2 | 74.0 |
| HMMT Nov 25 | 90.0 | 81.8 | 81.2 | 73.8 | 82.9 | 76.8 |
| LiveCodeBench v6 | 82.7 | 74.6 | 68.7 | 66.0 | 65.6 | 55.8 |
| OJBench | 41.5 | 36.3 | 29.7 | 25.1 | 29.2 | 24.1 |
| General Agent |
| BFCL-V4 | -- | -- | 49.7 | 42.4 | 66.1 | 50.3 |
| TAU2-Bench | -- | -- | 57.4 | 41.9 | 79.1 | 79.9 |
| VITA-Bench | -- | -- | 29.5 | 14.1 | 29.8 | 22.0 |
| DeepPlanning | -- | -- | 0.4 | 4.9 | 18.0 | 17.6 |
| Multilingualism |
| MMMLU | 78.2 | 69.7 | 81.3 | 78.4 | 81.2 | 76.1 |
| MMLU-ProX | 74.5 | 67.3 | 73.6 | 69.1 | 76.3 | 71.5 |
| NOVA-63 | 51.1 | 48.7 | 53.3 | 52.5 | 55.9 | 54.3 |
| INCLUDE | 74.0 | 65.3 | 78.3 | 74.4 | 75.6 | 71.0 |
| Global PIQA | 84.1 | 79.8 | 83.5 | 80.2 | 83.2 | 78.9 |
| PolyMATH | 54.0 | 30.9 | 62.4 | 52.6 | 57.3 | 51.1 |
| WMT24++ | 74.4 | 67.8 | 57.4 | 69.3 | 72.6 | 66.6 |
| MAXIFE | 83.7 | 80.1 | 79.9 | 77.4 | 83.4 | 78.0 |

\\* TAU2-Bench: we follow the official setup except for the airline domain, where all models are evaluated by applying the fixes proposed in the Claude Opus 4.5 system card.

\\* MMLU-ProX: we report the averaged accuracy on 29 languages.

\\* WMT24++: a harder subset of WMT24 after difficulty labeling and rebalancing; we report the averaged scores on 55 languages using XCOMET-XXL.

\\* MAXIFE: we report the accuracy on English + multilingual original prompts (totally 23 settings).

\\* Empty cells (--) indicate scores not yet available or not applicable.

### Vision Language

|  | GPT-5-Nano-2025-08-07 | Gemini-2.5-Flash-Lite | Qwen3-VL-30B-A3B | Qwen3.5-9B | Qwen3.5-4B |
| --- | --- | --- | --- | --- | --- |
| STEM and Puzzle |
| MMMU | 75.8 | 73.4 | 76.0 | 78.4 | 77.6 |
| MMMU-Pro | 57.2 | 59.7 | 63.0 | 70.1 | 66.3 |
| MathVision | 62.2 | 52.1 | 65.7 | 78.9 | 74.6 |
| Mathvista(mini) | 71.5 | 72.8 | 81.9 | 85.7 | 85.1 |
| We-Math | 62.5 | 32.1 | 70.0 | 75.2 | 75.4 |
| DynaMath | 78.0 | 69.9 | 80.1 | 83.6 | 83.3 |
| ZEROBench | 1.0 | 1.0 | 0.0 | 3.0 | 3.0 |
| ZEROBench\_sub | 22.2 | 19.2 | 23.7 | 31.1 | 26.3 |
| VlmsAreBlind | 66.7 | 68.4 | 72.5 | 93.7 | 92.6 |
| BabyVision | 14.4 | 17.5 | 18.6 | 28.6/25.8 | 16.0/19.1 |
| General VQA |
| RealWorldQA | 71.8 | 72.2 | 77.4 | 80.3 | 79.5 |
| MMStar | 68.6 | 69.1 | 75.5 | 79.7 | 78.3 |
| MMBenchEN-DEV-v1.1 | 80.3 | 82.7 | 88.9 | 90.1 | 89.4 |
| SimpleVQA | 46.0 | 54.1 | 54.3 | 51.2 | 43.4 |
| HallusionBench | 58.4 | 64.5 | 66.0 | 69.3 | 65.0 |
| Text Recognition and Document Understanding |
| OmniDocBench1.5 | 55.9 | 79.4 | 86.8 | 87.7 | 86.2 |
| CharXiv(RQ) | 50.1 | 56.1 | 56.6 | 73.0 | 70.8 |
| MMLongBench-Doc | 31.8 | 46.5 | 47.4 | 57.7 | 54.2 |
| CC-OCR | 58.9 | 72.9 | 77.8 | 79.3 | 76.7 |
| AI2D\_TEST | 81.9 | 85.7 | 86.9 | 90.2 | 89.6 |
| OCRBench | 75.3 | 82.5 | 83.9 | 89.2 | 85.0 |
| Spatial Intelligence |
| ERQA | 45.8 | 44.3 | 45.3 | 55.5 | 54.0 |
| CountBench | 80.0 | 79.2 | 90.0 | 97.2 | 96.3 |
| RefCOCO(avg) | -- | -- | 89.3 | 89.7 | 88.1 |
| EmbSpatialBench | 74.2 | 66.1 | 80.6 | 83.0 | 81.3 |
| RefSpatialBench | 12.6 | 11.2 | 54.2 | 58.5 | 54.6 |
| LingoQA | 57.0 | 17.8 | 62.0 | 80.4 | 74.4 |
| Hypersim | -- | -- | 11.4 | 13.5 | 12.5 |
| Nuscene | -- | -- | 10.3 | 11.8 | 9.9 |
| Video Understanding |
| VideoMME(w sub.) | 71.7 | 74.6 | 79.9 | 84.5 | 83.5 |
| VideoMME(w/o sub.) | 66.2 | 72.7 | 73.3 | 78.4 | 76.9 |
| VideoMMMU | 63.0 | 69.2 | 75.0 | 78.9 | 74.1 |
| MLVU | 69.2 | 78.5 | 78.9 | 84.4 | 82.8 |
| MVBench | -- | -- | 72.0 | 74.4 | 71.2 |
| LVBench | -- | 60.9 | 59.2 | 70.0 | 66.4 |
| MMVU | 63.1 | 65.3 | 66.1 | 67.8 | 64.9 |
| Visual Agent |
| ScreenSpot Pro | -- | -- | 60.5 | 65.2 | 60.3 |
| OSWorld-Verified | -- | -- | 30.6 | 41.8 | 35.6 |
| AndroidWorld | -- | -- | 55.0 | 57.8 | 58.6 |
| Tool Calling |
| TIR-Bench | 18.5 | 21.5 | 22.5 | 45.6/31.9 | 38.9/29.9 |
| V\* | 68.1 | 69.6 | 83.2 | 90.1/88.5 | 84.3/86.4 |
| Medical VQA |
| SLAKE | 57.0 | 65.0 | 68.8 | 79.0 | 76.1 |
| PMC-VQA | 37.8 | 48.8 | 51.5 | 57.9 | 55.5 |
| MedXpertQA-MM | 26.7 | 35.3 | 35.5 | 49.9 | 42.9 |

\\* MathVision: our model’s score is evaluated using a fixed prompt, e.g., “Please reason step by step, and put your final answer within \\boxed{}.” For other models, we report the higher score between runs with and without the \\boxed{} formatting.

\\* BabyVision: scores reported as "with CI / without CI".

\\* TIR-Bench and V\*: scores reported as "with CI / without CI".

\\* Empty cells (--) indicate scores not yet available or not applicable.

## Quickstart

> Qwen3.5 models operate in thinking mode by default, generating thinking content signified by `<think>\n...</think>\n\n` before producing the final responses.
> To disable thinking content and obtain direct response, refer to the examples [here](https://huggingface.co/unsloth/Qwen3.5-4B-GGUF#instruct-or-non-thinking-mode).

For streamlined integration, we recommend using Qwen3.5 via APIs. Below is a guide to use Qwen3.5 via OpenAI-compatible API.

### Serving Qwen3.5

Qwen3.5 can be served via APIs with popular inference frameworks.
In the following, we show example commands to launch OpenAI-Compatible API servers for Qwen3.5 models.

> Inference efficiency and throughput vary significantly across frameworks.
> We recommend using the latest framework versions to ensure optimal performance and compatibility.
> For production workloads or high-throughput scenarios, dedicated serving engines such as SGLang, KTransformers or vLLM are strongly recommended.

> The model has a default context length of 262,144 tokens.
> If you encounter out-of-memory (OOM) errors, consider reducing the context window.
> However, because Qwen3.5 leverages extended context for complex tasks, we advise maintaining a context length of at least 128K tokens to preserve thinking capabilities.

#### SGLang

[SGLang](https://github.com/sgl-project/sglang) is a fast serving framework for large language models and vision language models.
SGLang from the main branch of the open-source repository is required for Qwen3.5, which can be installed using the following command in a fresh environment:

```shell
uv pip install 'git+https://github.com/sgl-project/sglang.git#subdirectory=python&egg=sglang[all]'
```

See [its documentation](https://docs.sglang.ai/get_started/install.html) for more details.

The following will create API endpoints at `http://localhost:8000/v1`:

- **Standard Version**: The following command can be used to create an API endpoint with maximum context length 262,144 tokens using tensor parallel on 8 GPUs.


```shell
python -m sglang.launch_server --model-path Qwen/Qwen3.5-4B --port 8000 --tp-size 1 --mem-fraction-static 0.8 --context-length 262144 --reasoning-parser qwen3
```

- **Tool Use**: To support tool use, you can use the following command.


```shell
python -m sglang.launch_server --model-path Qwen/Qwen3.5-4B --port 8000 --tp-size 1 --mem-fraction-static 0.8 --context-length 262144 --reasoning-parser qwen3 --tool-call-parser qwen3_coder
```

- **Multi-Token Prediction (MTP)**: The following command is recommended for MTP:


```shell
python -m sglang.launch_server --model-path Qwen/Qwen3.5-4B --port 8000 --tp-size 1 --mem-fraction-static 0.8 --context-length 262144 --reasoning-parser qwen3 --speculative-algo NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4
```


#### vLLM

[vLLM](https://github.com/vllm-project/vllm) is a high-throughput and memory-efficient inference and serving engine for LLMs.
vLLM from the main branch of the open-source repository is required for Qwen3.5, which can be installed using the following command in a fresh environment:

```shell
uv pip install vllm --torch-backend=auto --extra-index-url https://wheels.vllm.ai/nightly
```

See [its documentation](https://docs.vllm.ai/en/stable/getting_started/installation/index.html) for more details.

For detailed Qwen3.5 usage guide, see the [vLLM Qwen3.5 recipe](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html).

The following will create API endpoints at `http://localhost:8000/v1`:

- **Standard Version**: The following command can be used to create an API endpoint with maximum context length 262,144 tokens using tensor parallel on 8 GPUs.


```shell
vllm serve Qwen/Qwen3.5-4B --port 8000 --tensor-parallel-size 1 --max-model-len 262144 --reasoning-parser qwen3
```

- **Tool Call**: To support tool use, you can use the following command.


```shell
vllm serve Qwen/Qwen3.5-4B --port 8000 --tensor-parallel-size 1 --max-model-len 262144 --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder
```

- **Multi-Token Prediction (MTP)**: The following command is recommended for MTP:


```shell
vllm serve Qwen/Qwen3.5-4B --port 8000 --tensor-parallel-size 1 --max-model-len 262144 --reasoning-parser qwen3 --speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}'
```

- **Text-Only**: The following command skips the vision encoder and multimodal profiling to free up memory for additional KV cache:


```shell
vllm serve Qwen/Qwen3.5-4B --port 8000 --tensor-parallel-size 1 --max-model-len 262144 --reasoning-parser qwen3 --language-model-only
```


#### KTransformers

[KTransformers](https://github.com/kvcache-ai/ktransformers) is a flexible framework for experiencing cutting-edge LLM inference optimizations with CPU-GPU heterogeneous computing.
For running Qwen3.5 with KTransformers, see the [KTransformers Deployment Guide](https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/Qwen3.5.md).

#### Hugging Face Transformers

Hugging Face Transformers contains a _lightweight_ server which can be used for quick testing and moderate load deployment.
The latest `transformers` is required for Qwen3.5:

```shell
pip install "transformers[serving] @ git+https://github.com/huggingface/transformers.git@main"
```

See [its documentation](https://huggingface.co/docs/transformers/main/serving) for more details. Please also make sure torchvision and pillow are installed.

Then, run `transformers serve` to launch a server with API endpoints at `http://localhost:8000/v1`; it will place the model on accelerators if available:

```shell
transformers serve --force-model Qwen/Qwen3.5-4B --port 8000 --continuous-batching
```

### Using Qwen3.5 via the Chat Completions API

The chat completions API is accessible via standard HTTP requests or OpenAI SDKs.
Here, we show examples using the OpenAI Python SDK.

Before starting, make sure it is installed and the API key and the API base URL is configured, e.g.:

```shell
pip install -U openai

# Set the following accordingly
export OPENAI_BASE_URL="http://localhost:8000/v1"
export OPENAI_API_KEY="EMPTY"
```

> We recommend using the following set of sampling parameters for generation
>
> - Thinking mode for general tasks: `temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=1.5, repetition_penalty=1.0`
> - Thinking mode for precise coding tasks (e.g. WebDev): `temperature=0.6, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=0.0, repetition_penalty=1.0`
> - Instruct (or non-thinking) mode for general tasks: `temperature=0.7, top_p=0.8, top_k=20, min_p=0.0, presence_penalty=1.5, repetition_penalty=1.0`
> - Instruct (or non-thinking) mode for reasoning tasks: `temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=1.5, repetition_penalty=1.0`
>
> Please note that the support for sampling parameters varies according to inference frameworks.

#### Text-Only Input

```python
from openai import OpenAI
# Configured by environment variables
client = OpenAI()

messages = [\
    {"role": "user", "content": "Type \"I love Qwen3.5\" backwards"},\
]

chat_response = client.chat.completions.create(
    model="Qwen/Qwen3.5-4B",
    messages=messages,
    max_tokens=81920,
    temperature=1.0,
    top_p=0.95,
    presence_penalty=1.5,
    extra_body={
        "top_k": 20,
    },
)
print("Chat response:", chat_response)
```

#### Image Input

```python
from openai import OpenAI
# Configured by environment variables
client = OpenAI()

messages = [\
    {\
        "role": "user",\
        "content": [\
            {\
                "type": "image_url",\
                "image_url": {\
                    "url": "https://qianwen-res.oss-accelerate.aliyuncs.com/Qwen3.5/demo/CI_Demo/mathv-1327.jpg"\
                }\
            },\
            {\
                "type": "text",\
                "text": "The centres of the four illustrated circles are in the corners of the square. The two big circles touch each other and also the two little circles. With which factor do you have to multiply the radii of the little circles to obtain the radius of the big circles?\nChoices:\n(A) $\\frac{2}{9}$\n(B) $\\sqrt{5}$\n(C) $0.8 \\cdot \\pi$\n(D) 2.5\n(E) $1+\\sqrt{2}$"\
            }\
        ]\
    }\
]

chat_response = client.chat.completions.create(
    model="Qwen/Qwen3.5-4B",
    messages=messages,
    max_tokens=81920,
    temperature=1.0,
    top_p=0.95,
    presence_penalty=1.5,
    extra_body={
        "top_k": 20,
    },
)
print("Chat response:", chat_response)
```

#### Video Input

```python
from openai import OpenAI
# Configured by environment variables
client = OpenAI()

messages = [\
    {\
        "role": "user",\
        "content": [\
            {\
                "type": "video_url",\
                "video_url": {\
                    "url": "https://qianwen-res.oss-accelerate.aliyuncs.com/Qwen3.5/demo/video/N1cdUjctpG8.mp4"\
                }\
            },\
            {\
                "type": "text",\
                "text": "Summarize the video content."\
            }\
        ]\
    }\
]

# When vLLM is launched with `--media-io-kwargs '{"video": {"num_frames": -1}}'`,
# video frame sampling can be configured via `extra_body` (e.g., by setting `fps`).
# This feature is currently supported only in vLLM.
#
# By default, `fps=2` and `do_sample_frames=True`.
# With `do_sample_frames=True`, you can customize the `fps` value to set your desired video sampling rate.
chat_response = client.chat.completions.create(
    model="Qwen/Qwen3.5-4B",
    messages=messages,
    max_tokens=81920,
    temperature=1.0,
    top_p=0.95,
    presence_penalty=1.5,
    extra_body={
        "top_k": 20,
        "mm_processor_kwargs": {"fps": 2, "do_sample_frames": True},
    },
)

print("Chat response:", chat_response)
```

#### Instruct (or Non-Thinking) Mode

> Qwen3.5 does not officially support the soft switch of Qwen3, i.e., `/think` and `/nothink`.

Qwen3.5 will think by default before response.
You can obtain direct response from the model without thinking by configuring the API parameters.
For example,

```python
from openai import OpenAI
# Configured by environment variables
client = OpenAI()

messages = [\
    {\
        "role": "user",\
        "content": [\
            {\
                "type": "image_url",\
                "image_url": {\
                    "url": "https://qianwen-res.oss-accelerate.aliyuncs.com/Qwen3.5/demo/RealWorld/RealWorld-04.png"\
                }\
            },\
            {\
                "type": "text",\
                "text": "Where is this?"\
            }\
        ]\
    }\
]

chat_response = client.chat.completions.create(
    model="Qwen/Qwen3.5-4B",
    messages=messages,
    max_tokens=32768,
    temperature=0.7,
    top_p=0.8,
    presence_penalty=1.5,
    extra_body={
        "top_k": 20,
        "chat_template_kwargs": {"enable_thinking": False},
    },
)
print("Chat response:", chat_response)
```

> If you are using APIs from Alibaba Cloud Model Studio, in addition to changing `model`, please use `"enable_thinking": False` instead of `"chat_template_kwargs": {"enable_thinking": False}`.

## Agentic Usage

Qwen3.5 excels in tool calling capabilities.

### Qwen-Agent

We recommend using [Qwen-Agent](https://github.com/QwenLM/Qwen-Agent) to quickly build Agent applications with Qwen3.5.

To define the available tools, you can use the MCP configuration file, use the integrated tool of Qwen-Agent, or integrate other tools by yourself.

```python
import os
from qwen_agent.agents import Assistant

# Define LLM
# Using Alibaba Cloud Model Studio
llm_cfg = {
    # Use the OpenAI-compatible model service provided by DashScope:
    'model': 'Qwen3.5-4B',
    'model_type': 'qwenvl_oai',
    'model_server': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
    'api_key': os.getenv('DASHSCOPE_API_KEY'),

    'generate_cfg': {
        'use_raw_api': True,
        # When using Dash Scope OAI API, pass the parameter of whether to enable thinking mode in this way
        'extra_body': {
            'enable_thinking': True
        },
    },
}

# Using OpenAI-compatible API endpoint.
# functionality of the deployment frameworks and let Qwen-Agent automate the related operations.
#
# llm_cfg = {
#     # Use your own model service compatible with OpenAI API by vLLM/SGLang:
#     'model': 'Qwen/Qwen3.5-4B',
#     'model_type': 'qwenvl_oai',
#     'model_server': 'http://localhost:8000/v1',  # api_base
#     'api_key': 'EMPTY',
#
#     'generate_cfg': {
#         'use_raw_api': True,
#         # When using vLLM/SGLang OAI API, pass the parameter of whether to enable thinking mode in this way
#         'extra_body': {
#             'chat_template_kwargs': {'enable_thinking': True}
#         },
#     },
# }

# Define Tools
tools = [\
    {'mcpServers': {  # You can specify the MCP configuration file\
            "filesystem": {\
                "command": "npx",\
                "args": ["-y", "@modelcontextprotocol/server-filesystem", "/Users/xxxx/Desktop"]\
            }\
        }\
    }\
]

# Define Agent
bot = Assistant(llm=llm_cfg, function_list=tools)

# Streaming generation
messages = [{'role': 'user', 'content': 'Help me organize my desktop.'}]
for responses in bot.run(messages=messages):
    pass
print(responses)

# Streaming generation
messages = [{'role': 'user', 'content': 'Develop a dog website and save it on the desktop'}]
for responses in bot.run(messages=messages):
    pass
print(responses)
```

### Qwen Code

[Qwen Code](https://github.com/QwenLM/qwen-code) is an open-source AI agent for the terminal, optimized for Qwen models. It helps you understand large codebases, automate tedious work, and ship faster.

For more information, please refer to [Qwen Code](https://qwenlm.github.io/qwen-code-docs/).

## Processing Ultra-Long Texts

Qwen3.5 natively supports context lengths of up to 262,144 tokens.
For long-horizon tasks where the total length (including both input and output) exceeds this limit, we recommend using RoPE scaling techniques to handle long texts effectively., e.g., YaRN.

YaRN is currently supported by several inference frameworks, e.g., `transformers`, `vllm`, `ktransformers` and `sglang`.
In general, there are two approaches to enabling YaRN for supported frameworks:

- Modifying the model configuration file:
In the `config.json` file, change the `rope_parameters` fields in `text_config` to:


```json
{
      "mrope_interleaved": true,
      "mrope_section": [\
          11,\
          11,\
          10\
      ],
      "rope_type": "yarn",
      "rope_theta": 10000000,
      "partial_rotary_factor": 0.25,
      "factor": 4.0,
      "original_max_position_embeddings": 262144,
}
```

- Passing command line arguments:

For `vllm`, you can use


```shell
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 vllm serve ... --hf-overrides '{"text_config": {"rope_parameters": {"mrope_interleaved": true, "mrope_section": [11, 11, 10], "rope_type": "yarn", "rope_theta": 10000000, "partial_rotary_factor": 0.25, "factor": 4.0, "original_max_position_embeddings": 262144}}}' --max-model-len 1010000
```


For `sglang` and `ktransformers`, you can use


```shell
SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 python -m sglang.launch_server ... --json-model-override-args '{"text_config": {"rope_parameters": {"mrope_interleaved": true, "mrope_section": [11, 11, 10], "rope_type": "yarn", "rope_theta": 10000000, "partial_rotary_factor": 0.25, "factor": 4.0, "original_max_position_embeddings": 262144}}}' --context-length 1010000
```


> All the notable open-source frameworks implement static YaRN, which means the scaling factor remains constant regardless of input length, **potentially impacting performance on shorter texts.**
> We advise modifying the `rope_parameters` configuration only when processing long contexts is required.
> It is also recommended to modify the `factor` as needed. For example, if the typical context length for your application is 524,288 tokens, it would be better to set `factor` as 2.0.

## Best Practices

To achieve optimal performance, we recommend the following settings:

1. **Sampling Parameters**:

   - We suggest using the following sets of sampling parameters depending on the mode and task type:
     - **Thinking mode for general tasks**:

       `temperature=1.0`, `top_p=0.95`, `top_k=20`, `min_p=0.0`, `presence_penalty=1.5`, `repetition_penalty=1.0`
     - **Thinking mode for precise coding tasks (e.g., WebDev)**:

       `temperature=0.6`, `top_p=0.95`, `top_k=20`, `min_p=0.0`, `presence_penalty=0.0`, `repetition_penalty=1.0`
     - **Instruct (or non-thinking) mode for general tasks**:

       `temperature=0.7`, `top_p=0.8`, `top_k=20`, `min_p=0.0`, `presence_penalty=1.5`, `repetition_penalty=1.0`
     - **Instruct (or non-thinking) mode for reasoning tasks**:

       `temperature=1.0`, `top_p=1.0`, `top_k=40`, `min_p=0.0`, `presence_penalty=2.0`, `repetition_penalty=1.0`
   - For supported frameworks, you can adjust the `presence_penalty` parameter between 0 and 2 to reduce endless repetitions. However, using a higher value may occasionally result in language mixing and a slight decrease in model performance.
2. **Adequate Output Length**: We recommend using an output length of 32,768 tokens for most queries. For benchmarking on highly complex problems, such as those found in math and programming competitions, we suggest setting the max output length to 81,920 tokens. This provides the model with sufficient space to generate detailed and comprehensive responses, thereby enhancing its overall performance.

3. **Standardize Output Format**: We recommend using prompts to standardize model outputs when benchmarking.

   - **Math Problems**: Include "Please reason step by step, and put your final answer within \\boxed{}." in the prompt.
   - **Multiple-Choice Questions**: Add the following JSON structure to the prompt to standardize responses: "Please show your choice in the `answer` field with only the choice letter, e.g., `"answer": "C"`."
4. **No Thinking Content in History**: In multi-turn conversations, the historical model output should only include the final output part and does not need to include the thinking content. It is implemented in the provided chat template in Jinja2. However, for frameworks that do not directly use the Jinja2 chat template, it is up to the developers to ensure that the best practice is followed.

5. **Long Video Understanding**: To optimize inference efficiency for plain text and images, the `size` parameter in the released `video_preprocessor_config.json` is conservatively configured. It is recommended to set the `longest_edge` parameter in the video\_preprocessor\_config file to 469,762,048 (corresponding to 224k video tokens) to enable higher frame-rate sampling for hour-scale videos and thereby achieve superior performance. For example,


```json
{"longest_edge": 469762048, "shortest_edge": 4096}
```


    Alternatively, override the default values via engine startup parameters. For implementation details, refer to: [vLLM](https://github.com/vllm-project/vllm/pull/34330) / [SGLang](https://github.com/sgl-project/sglang/pull/18467).


### Citation

If you find our work helpful, feel free to give us a cite.

```bibtex
@misc{qwen3.5,
    title  = {{Qwen3.5}: Towards Native Multimodal Agents},
    author = {{Qwen Team}},
    month  = {February},
    year   = {2026},
    url    = {https://qwen.ai/blog?id=qwen3.5}
}
```

Downloads last month650,957

GGUF

Model size

4B params

Architecture

qwen35

Chat template

Hardware compatibility

[Log In](https://huggingface.co/login) to add your hardware

2-bit

UD-IQ2\_XXS

1.52 GBUD-IQ2\_M

1.76 GBUD-Q2\_K\_XL

1.94 GB

3-bit

UD-IQ3\_XXS

1.95 GBQ3\_K\_S

2.11 GBQ3\_K\_M

2.29 GBUD-Q3\_K\_XL

2.44 GB

4-bit

IQ4\_XS

2.48 GBQ4\_K\_S

2.59 GBIQ4\_NL

2.58 GBQ4\_0

2.58 GBQ4\_1

2.78 GBQ4\_K\_M

2.74 GBUD-Q4\_K\_XL

2.91 GB

5-bit

Q5\_K\_S

3.02 GBQ5\_K\_M

3.14 GBUD-Q5\_K\_XL

3.25 GB

6-bit

Q6\_K

3.53 GBUD-Q6\_K\_XL

4.15 GB

8-bit

Q8\_0

4.48 GBUD-Q8\_K\_XL

5.95 GB

16-bit

BF16

8.42 GB

Inference Providers [NEW](https://huggingface.co/docs/inference-providers)

[Image-Text-to-Text](https://huggingface.co/tasks/image-text-to-text "Learn more about image-text-to-text")

This model isn't deployed by any Inference Provider. [🙋Ask for provider support](https://huggingface.co/spaces/huggingface/InferenceSupport/discussions/new?title=unsloth/Qwen3.5-4B-GGUF&description=React%20to%20this%20comment%20with%20an%20emoji%20to%20vote%20for%20%5Bunsloth%2FQwen3.5-4B-GGUF%5D(%2Funsloth%2FQwen3.5-4B-GGUF)%20to%20be%20supported%20by%20Inference%20Providers.%0A%0A(optional)%20Which%20providers%20are%20you%20interested%20in%3F%20(Novita%2C%20Hyperbolic%2C%20Together%E2%80%A6)%0A)

## Model tree for unsloth/Qwen3.5-4B-GGUF

Base model

[Qwen/Qwen3.5-4B-Base](https://huggingface.co/Qwen/Qwen3.5-4B-Base)

Finetuned

[Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B)

Quantized

( [192](https://huggingface.co/models?other=base_model:quantized:Qwen/Qwen3.5-4B))

this model

Quantizations

[Use with llama.cpp](https://huggingface.co/models?apps=llama.cpp&other=base_model:quantized:unsloth/Qwen3.5-4B-GGUF "Use with llama.cpp")[Use with LM Studio](https://huggingface.co/models?apps=lmstudio&other=base_model:quantized:unsloth/Qwen3.5-4B-GGUF "Use with LM Studio")[Use with Jan](https://huggingface.co/models?apps=jan&other=base_model:quantized:unsloth/Qwen3.5-4B-GGUF "Use with Jan")[Use with Ollama](https://huggingface.co/models?apps=ollama&other=base_model:quantized:unsloth/Qwen3.5-4B-GGUF "Use with Ollama")

[1 model](https://huggingface.co/models?other=base_model:quantized:unsloth/Qwen3.5-4B-GGUF)

## Spaces using unsloth/Qwen3.5-4B-GGUF5

## Collections including unsloth/Qwen3.5-4B-GGUF

[New 2.0 version of our Dynamic GGUF + Quants. Dynamic 2.0 achieves superior accuracy & SOTA quantization performance.•96 items•Updated 4 days ago• 626](https://huggingface.co/collections/unsloth/unsloth-dynamic-20-quants)

[Qwen3.5 is Qwen's new model family including Qwen3.5 Small: 0.8B, 2B, 4B, 9B and Qwen3.5 Medium: 35B-A3B, 27B, 122B-A10B and 397B-A17B.•25 items•Updated 29 days ago• 153](https://huggingface.co/collections/unsloth/qwen35)