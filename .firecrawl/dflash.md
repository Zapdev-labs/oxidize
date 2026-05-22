[Skip to content](https://github.com/z-lab/dflash#start-of-content)

You signed in with another tab or window. [Reload](https://github.com/z-lab/dflash) to refresh your session.You signed out in another tab or window. [Reload](https://github.com/z-lab/dflash) to refresh your session.You switched accounts on another tab or window. [Reload](https://github.com/z-lab/dflash) to refresh your session.Dismiss alert

{{ message }}

[z-lab](https://github.com/z-lab)/ **[dflash](https://github.com/z-lab/dflash)** Public

- [Notifications](https://github.com/login?return_to=%2Fz-lab%2Fdflash) You must be signed in to change notification settings
- [Fork\\
330](https://github.com/login?return_to=%2Fz-lab%2Fdflash)
- [Star\\
4.7k](https://github.com/login?return_to=%2Fz-lab%2Fdflash)


main

[**1** Branch](https://github.com/z-lab/dflash/branches) [**0** Tags](https://github.com/z-lab/dflash/tags)

[Go to Branches page](https://github.com/z-lab/dflash/branches)[Go to Tags page](https://github.com/z-lab/dflash/tags)

Go to file

Code

Open more actions menu

## Folders and files

| Name | Name | Last commit message | Last commit date |
| --- | --- | --- | --- |
| ## Latest commit<br>[![jianc99](https://avatars.githubusercontent.com/u/141193260?v=4&size=40)](https://github.com/jianc99)[jianc99](https://github.com/z-lab/dflash/commits?author=jianc99)<br>[update model list](https://github.com/z-lab/dflash/commit/94e4abc5e0c31b67bc1a9d30f1cc34ece28a8756)<br>2 weeks agoMay 10, 2026<br>[94e4abc](https://github.com/z-lab/dflash/commit/94e4abc5e0c31b67bc1a9d30f1cc34ece28a8756) · 2 weeks agoMay 10, 2026<br>## History<br>[90 Commits](https://github.com/z-lab/dflash/commits/main/) <br>Open commit details<br>[View commit history for this file.](https://github.com/z-lab/dflash/commits/main/) 90 Commits |
| [dflash](https://github.com/z-lab/dflash/tree/main/dflash "dflash") | [dflash](https://github.com/z-lab/dflash/tree/main/dflash "dflash") | [update interleaved SWA draft model](https://github.com/z-lab/dflash/commit/91c26e894ed4e9e72abc29a9de6307aef8007494 "update interleaved SWA draft model") | 2 weeks agoMay 5, 2026 |
| [.gitignore](https://github.com/z-lab/dflash/blob/main/.gitignore ".gitignore") | [.gitignore](https://github.com/z-lab/dflash/blob/main/.gitignore ".gitignore") | [modify vllm installation](https://github.com/z-lab/dflash/commit/c69a2bacbc6b4fc35cda87190db43ac60317f038 "modify vllm installation") | last monthApr 6, 2026 |
| [LICENSE](https://github.com/z-lab/dflash/blob/main/LICENSE "LICENSE") | [LICENSE](https://github.com/z-lab/dflash/blob/main/LICENSE "LICENSE") | [Add MIT License to the project](https://github.com/z-lab/dflash/commit/4571f6ac1281b5a5e27de0435ba36532644fdaa6 "Add MIT License to the project") | 4 months agoJan 5, 2026 |
| [README.md](https://github.com/z-lab/dflash/blob/main/README.md "README.md") | [README.md](https://github.com/z-lab/dflash/blob/main/README.md "README.md") | [update model list](https://github.com/z-lab/dflash/commit/94e4abc5e0c31b67bc1a9d30f1cc34ece28a8756 "update model list") | 2 weeks agoMay 10, 2026 |
| [pyproject.toml](https://github.com/z-lab/dflash/blob/main/pyproject.toml "pyproject.toml") | [pyproject.toml](https://github.com/z-lab/dflash/blob/main/pyproject.toml "pyproject.toml") | [update interleaved SWA draft model](https://github.com/z-lab/dflash/commit/91c26e894ed4e9e72abc29a9de6307aef8007494 "update interleaved SWA draft model") | 2 weeks agoMay 5, 2026 |
| View all files |

## Repository files navigation

# DFlash: Block Diffusion for Flash Speculative Decoding

[Permalink: DFlash: Block Diffusion for Flash Speculative Decoding](https://github.com/z-lab/dflash#dflash-block-diffusion-for-flash-speculative-decoding)

[**Paper**](https://arxiv.org/abs/2602.06036) \| [**Blog**](https://z-lab.ai/projects/dflash/) \| [**Models**](https://huggingface.co/collections/z-lab/dflash)

**DFlash** is a lightweight **block diffusion** model designed for speculative decoding. It enables efficient and high-quality parallel drafting.

![DFlash Architecture](https://raw.githubusercontent.com/jianc99/jianc99.github.io/master/images/dflash_system.png)

DFlash\_demo.mp4

## Supported Models

[Permalink: Supported Models](https://github.com/z-lab/dflash#supported-models)

| Model | DFlash Draft |
| --- | --- |
| gemma-4-31B-it | [z-lab/gemma-4-31B-it-DFlash](https://huggingface.co/z-lab/gemma-4-31B-it-DFlash) |
| gemma-4-26B-A4B-it | [z-lab/gemma-4-26B-A4B-it-DFlash](https://huggingface.co/z-lab/gemma-4-26B-A4B-it-DFlash) |
| MiniMax-M2.7 (Preview) | [z-lab/MiniMax-M2.7-DFlash](https://huggingface.co/z-lab/MiniMax-M2.7-DFlash) |
| MiniMax-M2.5 (Preview) | [z-lab/MiniMax-M2.5-DFlash](https://huggingface.co/z-lab/MiniMax-M2.5-DFlash) |
| Kimi-K2.6 (Preview) | [z-lab/Kimi-K2.6-DFlash](https://huggingface.co/z-lab/Kimi-K2.6-DFlash) |
| Kimi-K2.5 | [z-lab/Kimi-K2.5-DFlash](https://huggingface.co/z-lab/Kimi-K2.5-DFlash) |
| Qwen3.6-27B | [z-lab/Qwen3.6-27B-DFlash](https://huggingface.co/z-lab/Qwen3.6-27B-DFlash) |
| Qwen3.6-35B-A3B | [z-lab/Qwen3.6-35B-A3B-DFlash](https://huggingface.co/z-lab/Qwen3.6-35B-A3B-DFlash) |
| Qwen3.5-4B | [z-lab/Qwen3.5-4B-DFlash](https://huggingface.co/z-lab/Qwen3.5-4B-DFlash) |
| Qwen3.5-9B | [z-lab/Qwen3.5-9B-DFlash](https://huggingface.co/z-lab/Qwen3.5-9B-DFlash) |
| Qwen3.5-27B | [z-lab/Qwen3.5-27B-DFlash](https://huggingface.co/z-lab/Qwen3.5-27B-DFlash) |
| Qwen3.5-35B-A3B | [z-lab/Qwen3.5-35B-A3B-DFlash](https://huggingface.co/z-lab/Qwen3.5-35B-A3B-DFlash) |
| Qwen3.5-122B-A10B | [z-lab/Qwen3.5-122B-A10B-DFlash](https://huggingface.co/z-lab/Qwen3.5-122B-A10B-DFlash) |
| gpt-oss-20b | [z-lab/gpt-oss-20b-DFlash](https://huggingface.co/z-lab/gpt-oss-20b-DFlash) |
| gpt-oss-120b | [z-lab/gpt-oss-120b-DFlash](https://huggingface.co/z-lab/gpt-oss-120b-DFlash) |
| Qwen3-Coder-Next | [z-lab/Qwen3-Coder-Next-DFlash](https://huggingface.co/z-lab/Qwen3-Coder-Next-DFlash) |
| Qwen3-4B (non-thinking) | [z-lab/Qwen3-4B-DFlash-b16](https://huggingface.co/z-lab/Qwen3-4B-DFlash-b16) |
| Qwen3-8B (non-thinking) | [z-lab/Qwen3-8B-DFlash-b16](https://huggingface.co/z-lab/Qwen3-8B-DFlash-b16) |
| Qwen3-Coder-30B-A3B | [z-lab/Qwen3-Coder-30B-A3B-DFlash](https://huggingface.co/z-lab/Qwen3-Coder-30B-A3B-DFlash) |
| Llama-3.1-8B-Instruct | [z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat](https://huggingface.co/z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat) |
| DeepSeek-V4-Flash | Coming soon |
| DeepSeek-V4-Pro | Coming soon |
| GLM-5.1 | Coming soon |

> Feel free to open a GitHub issue to request support for additional models. We will also open-source the training recipe soon, so you can train your own DFlash draft model to accelerate any LLM.

## 📦 Installation

[Permalink: 📦 Installation](https://github.com/z-lab/dflash#-installation)

Use a separate virtual environment for each to avoid conflict.

| Backend | Install command |
| --- | --- |
| **Transformers** | `uv pip install -e ".[transformers]"` |
| **SGLang** | `uv pip install -e ".[sglang]"` |
| **vLLM** | See below |
| **MLX** (Apple Silicon) | `pip install -e ".[mlx]"` |

**vLLM:** vLLM v0.20.1+ includes core DFlash support. Use the standard install for most models:

```
uv pip install -e ".[vllm]"
```

Gemma4 DFlash currently needs our temporary vLLM Gemma4 build. Docker is recommended:

```
docker pull ghcr.io/z-lab/vllm-openai:gemma4-dflash-cu130
```

Source fallback for Gemma4:

```
uv pip install -U --torch-backend=auto \
  "vllm @ git+https://github.com/vllm-project/vllm.git@refs/pull/41703/head"
```

Newer non-Gemma4 SWA draft models use the SWA support branch:

```
uv pip install -U --torch-backend=auto \
  "vllm @ git+https://github.com/vllm-project/vllm.git@refs/pull/40898/head"
```

## 🚀 Quick Start

[Permalink: 🚀 Quick Start](https://github.com/z-lab/dflash#-quick-start)

### vLLM

[Permalink: vLLM](https://github.com/z-lab/dflash#vllm)

Gemma4 with Docker:

```
docker run --rm -it \
  --gpus all \
  --ipc=host \
  --shm-size=16g \
  -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  ghcr.io/z-lab/vllm-openai:gemma4-dflash-cu130 \
  google/gemma-4-26B-A4B-it \
  --host 0.0.0.0 \
  --port 8000 \
  --speculative-config '{"method": "dflash", "model": "z-lab/gemma-4-26B-A4B-it-DFlash", "num_speculative_tokens": 15, "attention_backend": "flash_attn"}' \
  --attention-backend triton_attn \
  --max-num-batched-tokens 32768 \
  --trust-remote-code
```

Non-Gemma4 models:

```
vllm serve Qwen/Qwen3.5-27B \
  --speculative-config '{"method": "dflash", "model": "z-lab/Qwen3.5-27B-DFlash", "num_speculative_tokens": 15}' \
  --attention-backend flash_attn \
  --max-num-batched-tokens 32768
```

### SGLang

[Permalink: SGLang](https://github.com/z-lab/dflash#sglang)

```
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1

# Optional: enable schedule overlapping (experimental, may not be stable)
# export SGLANG_ENABLE_SPEC_V2=1
# export SGLANG_ENABLE_DFLASH_SPEC_V2=1
# export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1

python -m sglang.launch_server \
    --model-path Qwen/Qwen3.5-35B-A3B \
    --speculative-algorithm DFLASH \
    --speculative-draft-model-path z-lab/Qwen3.5-35B-A3B-DFlash \
    --speculative-num-draft-tokens 16 \
    --tp-size 1 \
    --attention-backend trtllm_mha \
    --speculative-draft-attention-backend fa4 \
    --mem-fraction-static 0.75 \
    --mamba-scheduler-strategy extra_buffer \
    --trust-remote-code
```

### Transformers

[Permalink: Transformers](https://github.com/z-lab/dflash#transformers)

Only Qwen3 and LLaMA-3.1 models support the Transformers backend.

```
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

draft = AutoModel.from_pretrained("z-lab/Qwen3-8B-DFlash-b16", trust_remote_code=True, dtype="auto", device_map="cuda:0").eval()
target = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", dtype="auto", device_map="cuda:0").eval()
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

messages = [{"role": "user", "content": "How many positive whole-number divisors does 196 have?"}]
input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True, enable_thinking=False).to(draft.device)

output = draft.spec_generate(input_ids=input_ids, max_new_tokens=2048, temperature=0.0, target=target, stop_token_ids=[tokenizer.eos_token_id])
print(tokenizer.decode(output[0], skip_special_tokens=False))
```

### MLX (Apple Silicon)

[Permalink: MLX (Apple Silicon)](https://github.com/z-lab/dflash#mlx-apple-silicon)

There have been many great community DFlash implementations on MLX; we provide a simple and efficient one here, tested on an Apple M5 Pro with Qwen3, Qwen3.5 and Gemma-4 models.

```
from dflash.model_mlx import load, load_draft, stream_generate

model, tokenizer = load("Qwen/Qwen3.5-4B")
draft = load_draft("z-lab/Qwen3.5-4B-DFlash")

messages = [{"role": "user", "content": "How many positive whole-number divisors does 196 have?"}]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
tps = 0.0
for r in stream_generate(model, draft, tokenizer, prompt, block_size=16, max_tokens=2048, temperature=0.6):
    print(r.text, end="", flush=True)
    tps = r.generation_tps
print(f"\nThroughput: {tps:.2f} tok/s")
```

## 📊 Evaluation

[Permalink: 📊 Evaluation](https://github.com/z-lab/dflash#-evaluation)

All benchmarks share the same datasets (gsm8k, math500, humaneval, mbpp, mt-bench). Datasets are automatically downloaded and cached as JSONL in `cache/` on first run.

**vLLM**:

```
python -m dflash.benchmark --backend vllm \
    --base-url http://127.0.0.1:8000 --model Qwen/Qwen3.5-27B \
    --dataset gsm8k --num-prompts 128 --concurrency 1 --enable-thinking
```

**SGLang**:

```
python -m dflash.benchmark --backend sglang \
    --base-url http://127.0.0.1:30000 --model Qwen/Qwen3.5-35B-A3B \
    --dataset gsm8k --num-prompts 128 --concurrency 1 --enable-thinking
```

**Transformers** (Qwen3 and LLaMA only):

```
torchrun --nproc_per_node=8 -m dflash.benchmark --backend transformers \
    --model Qwen/Qwen3-8B --draft-model z-lab/Qwen3-8B-DFlash-b16 \
    --dataset gsm8k --max-samples 128
```

**MLX**:

```
python -m dflash.benchmark --backend mlx \
    --model mlx-community/gemma-4-31b-it-4bit --draft-model z-lab/gemma-4-31B-it-DFlash \
    --dataset gsm8k --max-samples 128 --enable-thinking
```

## Acknowledgement

[Permalink: Acknowledgement](https://github.com/z-lab/dflash#acknowledgement)

Huge thanks to [@dcw02](https://github.com/dcw02), [@gongy](https://github.com/gongy), and the team at [@modal-labs](https://github.com/modal-labs) for their fast, high-quality support in bringing DFlash to SGLang. And huge thanks as well to [@benchislett](https://github.com/benchislett) at NVIDIA for his work in bringing DFlash to vLLM and helping make it available to the broader serving community.

## Citation

[Permalink: Citation](https://github.com/z-lab/dflash#citation)

If you find DFlash useful, please cite our work. To share feedback on DFlash or request new model support, please fill out this form: [DFlash Feedback](https://forms.gle/4YNwfqb4nJdqn6hq9).

```
@article{chen2026dflash,
  title   = {{DFlash: Block Diffusion for Flash Speculative Decoding}},
  author  = {Chen, Jian and Liang, Yesheng and Liu, Zhijian},
  journal = {arXiv preprint arXiv:2602.06036},
  year    = {2026}
}
```

## About

DFlash: Block Diffusion for Flash Speculative Decoding


[dflash.z-lab.ai](https://dflash.z-lab.ai/ "https://dflash.z-lab.ai")

### Resources

[Readme](https://github.com/z-lab/dflash#readme-ov-file)

### License

[MIT license](https://github.com/z-lab/dflash#MIT-1-ov-file)

### Uh oh!

There was an error while loading. [Please reload this page](https://github.com/z-lab/dflash).

[Activity](https://github.com/z-lab/dflash/activity)

[Custom properties](https://github.com/z-lab/dflash/custom-properties)

### Stars

[**4.7k**\\
stars](https://github.com/z-lab/dflash/stargazers)

### Watchers

[**36**\\
watching](https://github.com/z-lab/dflash/watchers)

### Forks

[**330**\\
forks](https://github.com/z-lab/dflash/forks)

[Report repository](https://github.com/contact/report-content?content_url=https%3A%2F%2Fgithub.com%2Fz-lab%2Fdflash&report=z-lab+%28user%29)

## [Releases](https://github.com/z-lab/dflash/releases)

No releases published

## [Packages\  0](https://github.com/orgs/z-lab/packages?repo_name=dflash)

No packages published

## [Contributors\  4](https://github.com/z-lab/dflash/graphs/contributors)

- [![@jianc99](https://avatars.githubusercontent.com/u/141193260?s=64&v=4)](https://github.com/jianc99)[**jianc99** Jian Chen](https://github.com/jianc99)
- [![@shaun0927](https://avatars.githubusercontent.com/u/70629228?s=64&v=4)](https://github.com/shaun0927)[**shaun0927** Junghwan](https://github.com/shaun0927)
- [![@zhijian-liu](https://avatars.githubusercontent.com/u/5782437?s=64&v=4)](https://github.com/zhijian-liu)[**zhijian-liu** Zhijian Liu](https://github.com/zhijian-liu)
- [![@xiziqiao](https://avatars.githubusercontent.com/u/122404358?s=64&v=4)](https://github.com/xiziqiao)[**xiziqiao**](https://github.com/xiziqiao)

## Languages

- [Python100.0%](https://github.com/z-lab/dflash/search?l=python)

You can’t perform that action at this time.