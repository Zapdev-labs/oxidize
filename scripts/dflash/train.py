from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.dflash.config import DFlashTrainConfig, positional_loss_weights
from scripts.dflash.export_gguf import export_dflash_gguf
from scripts.dflash.model import DFlashDraftModel, block_attention_bias, sample_anchors


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cosine_lr(step: int, total: int, lr: float, warmup_ratio: float) -> float:
    warmup = max(int(total * warmup_ratio), 1)
    if step < warmup:
        return lr * (step + 1) / warmup
    progress = (step - warmup) / max(total - warmup, 1)
    return lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def resolve_language_model(model: nn.Module) -> nn.Module:
    candidates = [
        "model.language_model.model",
        "model.language_model",
        "language_model.model",
        "language_model",
        "model.model",
        "model",
    ]
    for path in candidates:
        obj: Any = model
        ok = True
        for part in path.split("."):
            if not hasattr(obj, part):
                ok = False
                break
            obj = getattr(obj, part)
        if not ok:
            continue
        layers = getattr(obj, "layers", None)
        if layers is not None:
            return obj
        inner = getattr(obj, "model", None)
        if inner is not None and getattr(inner, "layers", None) is not None:
            return inner
    raise RuntimeError("could not find decoder layers on the target model")


def resolve_embeddings(model: nn.Module, language_model: nn.Module) -> nn.Module:
    for obj in (language_model, model):
        for name in ("embed_tokens", "tok_embeddings"):
            if hasattr(obj, name):
                return getattr(obj, name)
        inner = getattr(obj, "model", None)
        if inner is not None and hasattr(inner, "embed_tokens"):
            return inner.embed_tokens
    raise RuntimeError("could not find token embeddings")


def resolve_lm_head(model: nn.Module, embeddings: nn.Module) -> nn.Module:
    for name in ("lm_head", "output"):
        if hasattr(model, name):
            return getattr(model, name)
        nested = getattr(model, "model", None)
        if nested is not None and hasattr(nested, name):
            return getattr(nested, name)
    return embeddings


class HiddenCatcher:
    def __init__(self, language_model: nn.Module, layer_ids: list[int]) -> None:
        self.layer_ids = list(layer_ids)
        self._cache: dict[int, torch.Tensor] = {}
        self._hooks = []
        layers = language_model.layers
        for idx in self.layer_ids:
            self._hooks.append(layers[idx].register_forward_hook(self._hook(idx)))

    def _hook(self, idx: int):
        def fn(_module, _inp, output):
            hidden = output[0] if isinstance(output, tuple) else output
            self._cache[idx] = hidden
            return output

        return fn

    def pop(self) -> torch.Tensor:
        missing = [i for i in self.layer_ids if i not in self._cache]
        if missing:
            raise RuntimeError(f"missing hidden states for layers {missing}")
        stacked = torch.cat([self._cache[i] for i in self.layer_ids], dim=-1)
        self._cache.clear()
        return stacked

    def close(self) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()


def load_target(cfg: DFlashTrainConfig, device: torch.device):
    from transformers import AutoConfig, AutoModelForImageTextToText, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(cfg.target_model, trust_remote_code=True)
    model_cfg = AutoConfig.from_pretrained(cfg.target_model, trust_remote_code=True)
    text_cfg = getattr(model_cfg, "text_config", model_cfg)
    cfg.vocab_size = int(getattr(text_cfg, "vocab_size", cfg.vocab_size))
    cfg.hidden_size = int(getattr(text_cfg, "hidden_size", cfg.hidden_size))
    cfg.target_n_layers = int(getattr(text_cfg, "num_hidden_layers", cfg.target_n_layers))
    cfg.rms_norm_eps = float(getattr(text_cfg, "rms_norm_eps", cfg.rms_norm_eps))
    rope = getattr(text_cfg, "rope_parameters", None) or {}
    if isinstance(rope, dict) and "rope_theta" in rope:
        cfg.rope_theta = float(rope["rope_theta"])
    elif hasattr(text_cfg, "rope_theta"):
        cfg.rope_theta = float(text_cfg.rope_theta)
    from scripts.dflash.config import build_target_layer_ids

    cfg.target_layer_ids = build_target_layer_ids(cfg.num_target_layers, cfg.target_n_layers)

    quant = None
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    if cfg.load_in_4bit and device.type == "cuda":
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        kwargs["quantization_config"] = quant
        kwargs["device_map"] = "auto"
    else:
        kwargs["torch_dtype"] = dtype
        kwargs["device_map"] = "auto" if device.type == "cuda" else None

    try:
        target = AutoModelForImageTextToText.from_pretrained(cfg.target_model, **kwargs)
    except Exception as exc:
        print(f"image-text load failed ({exc}); trying causal LM", flush=True)
        target = AutoModelForCausalLM.from_pretrained(cfg.target_model, **kwargs)
    target.eval()
    for p in target.parameters():
        p.requires_grad_(False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return target, tokenizer, dtype


def messages_from_row(row: dict[str, Any]) -> Optional[list[dict[str, str]]]:
    if "messages" in row and isinstance(row["messages"], list):
        out = []
        for msg in row["messages"]:
            role = str(msg.get("role", "user"))
            content = msg.get("content", "")
            if isinstance(content, list):
                text_parts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text_parts.append(str(part.get("text", "")))
                    elif isinstance(part, str):
                        text_parts.append(part)
                content = "\n".join(text_parts)
            if content:
                out.append({"role": role, "content": str(content)})
        return out or None
    if "conversation" in row:
        return messages_from_row({"messages": row["conversation"]})
    prompt = row.get("prompt") or row.get("instruction")
    response = row.get("response") or row.get("output") or row.get("completion")
    if prompt and response:
        return [
            {"role": "user", "content": str(prompt)},
            {"role": "assistant", "content": str(response)},
        ]
    return None


def _to_1d_token_ids(ids: Any) -> torch.Tensor:
    if isinstance(ids, torch.Tensor):
        t = ids
    elif isinstance(ids, dict) and "input_ids" in ids:
        t = torch.as_tensor(ids["input_ids"])
    elif hasattr(ids, "input_ids"):
        t = torch.as_tensor(ids.input_ids)
    elif hasattr(ids, "ids"):
        t = torch.as_tensor(ids.ids)
    else:
        t = torch.as_tensor(ids)
    if t.ndim == 2:
        t = t[0]
    return t.to(dtype=torch.long).reshape(-1)


def tokenize_messages(tokenizer, messages: list[dict[str, str]], max_seq_len: int) -> Optional[torch.Tensor]:
    try:
        ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_tensors="pt",
        )
    except Exception:
        text = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        ids = tokenizer(text, return_tensors="pt", add_special_tokens=True)
    ids = _to_1d_token_ids(ids)
    if ids.numel() < 32:
        return None
    return ids[:max_seq_len].contiguous()


def iter_tokenized(cfg: DFlashTrainConfig, tokenizer):
    from datasets import load_dataset

    ds = load_dataset(cfg.dataset_name, split=cfg.dataset_split, streaming=True)
    n = 0
    for row in ds:
        if n >= cfg.max_samples:
            break
        messages = messages_from_row(dict(row))
        if messages is None:
            continue
        ids = tokenize_messages(tokenizer, messages, cfg.max_seq_len)
        if ids is None:
            continue
        n += 1
        yield ids


@torch.no_grad()
def capture_target_hidden(
    target: nn.Module,
    language_model: nn.Module,
    catcher: HiddenCatcher,
    input_ids: torch.Tensor,
) -> torch.Tensor:
    device = next(language_model.parameters()).device
    tokens = input_ids.to(device).unsqueeze(0)
    attn = torch.ones_like(tokens)
    try:
        language_model(input_ids=tokens, attention_mask=attn, use_cache=False)
    except TypeError:
        language_model(tokens)
    hidden = catcher.pop()
    return hidden[0].to(dtype=torch.bfloat16 if hidden.dtype == torch.float32 else hidden.dtype)


def hf_token() -> str | None:
    for key in (
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "HUGGINGFACE_HUB_TOKEN",
        "HUGGINGFACE_TOKEN",
    ):
        value = os.environ.get(key)
        if value:
            return value
    return None


def push_to_hub(out_dir: Path, cfg: DFlashTrainConfig, extra: dict[str, Any]) -> str:
    token = hf_token()
    if not token:
        raise RuntimeError("no HF token in environment; attach Modal secret hf-token")
    from huggingface_hub import HfApi
    from huggingface_hub.errors import HfHubHTTPError

    api = HfApi(token=token)
    me = api.whoami()
    namespaces: list[str] = []
    if me.get("name"):
        namespaces.append(str(me["name"]))
    for org in me.get("orgs") or []:
        if isinstance(org, dict) and org.get("name"):
            namespaces.append(str(org["name"]))
        elif isinstance(org, str):
            namespaces.append(org)
    print(f"hf namespaces={namespaces} auth={me.get('auth', {}).get('type')}", flush=True)
    repo_ids = []
    if cfg.hub_repo:
        repo_ids.append(cfg.hub_repo)
    for ns in namespaces:
        repo_ids.append(f"{ns}/Qwen3.8-27B-ABLITERATED-DFlash")
    last_error: Exception | None = None
    repo_id = repo_ids[0] if repo_ids else "Qwen3.8-27B-ABLITERATED-DFlash"
    opened = False
    for repo_id in repo_ids:
        try:
            api.create_repo(repo_id, repo_type="model", private=cfg.hub_private, exist_ok=True)
            opened = True
            break
        except HfHubHTTPError as exc:
            last_error = exc
            print(f"create_repo {repo_id} failed: {exc}", flush=True)
            try:
                api.repo_info(repo_id, repo_type="model")
                opened = True
                print(f"using existing repo {repo_id}", flush=True)
                break
            except Exception:
                continue
    if not opened:
        raise RuntimeError(f"could not create or access a Hub repo: {last_error}")
    card = out_dir / "README.md"
    card.write_text(
        "\n".join(
            [
                "---",
                "license: apache-2.0",
                f"base_model: {cfg.target_model}",
                "tags:",
                "  - dflash",
                "  - speculative-decoding",
                "  - gguf",
                "  - qwen3.8",
                "library_name: gguf",
                "---",
                "",
                "# Qwen3.8-27B Abliterated DFlash draft",
                "",
                f"Custom DFlash block-diffusion draft trained against `{cfg.target_model}`.",
                "Use with oxidize-c / oxidize-core `--draft-model`.",
                "",
                f"- target layers: `{cfg.target_layer_ids}`",
                f"- block size: {cfg.block_size}",
                f"- draft layers: {cfg.num_hidden_layers}",
                f"- steps: {extra.get('steps')}",
                "",
            ]
        )
        + "\n"
    )
    api.upload_folder(
        repo_id=repo_id,
        folder_path=str(out_dir),
        repo_type="model",
        commit_message="Upload trained DFlash draft GGUF and checkpoints",
        ignore_patterns=["**/draft-step*.pt", "**/hidden-cache/**"],
    )
    url = f"https://huggingface.co/{repo_id}"
    print(f"uploaded private repo {url}", flush=True)
    return url


def save_embed_and_head_cpu(model_id: str, cache_dir: Path, token: str | None) -> None:
    import json

    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    cache_dir.mkdir(parents=True, exist_ok=True)
    index_path = hf_hub_download(model_id, "model.safetensors.index.json", token=token)
    weight_map = json.loads(Path(index_path).read_text())["weight_map"]
    embed_keys = [
        key
        for key in weight_map
        if key.endswith("embed_tokens.weight") and "visual" not in key and "vision" not in key
    ]
    head_keys = [key for key in weight_map if key.endswith("lm_head.weight")]
    if not embed_keys:
        raise RuntimeError("embed_tokens.weight not found in safetensors index")
    if not head_keys:
        head_keys = embed_keys
    pairs = [("embed.pt", embed_keys[0]), ("lm_head.pt", head_keys[0])]
    for out_name, tensor_name in pairs:
        shard = hf_hub_download(model_id, weight_map[tensor_name], token=token)
        with safe_open(shard, framework="pt", device="cpu") as handle:
            weight = handle.get_tensor(tensor_name).to(dtype=torch.float16).contiguous()
        torch.save({"weight": weight}, cache_dir / out_name)
        print(f"saved CPU {out_name} {tuple(weight.shape)} from {tensor_name}", flush=True)


def dump_hiddens(cfg: DFlashTrainConfig, cache_dir: Path, n_samples: int) -> int:
    cache_dir.mkdir(parents=True, exist_ok=True)
    save_embed_and_head_cpu(cfg.target_model, cache_dir, hf_token())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target, tokenizer, _dtype = load_target(cfg, device)
    language_model = resolve_language_model(target)
    catcher = HiddenCatcher(language_model, cfg.target_layer_ids)
    written = 0
    try:
        for ids in iter_tokenized(cfg, tokenizer):
            if written >= n_samples:
                break
            try:
                hidden = capture_target_hidden(target, language_model, catcher, ids)
            except torch.cuda.OutOfMemoryError as exc:
                torch.cuda.empty_cache()
                raise RuntimeError(f"target hidden dump OOM at sample {written}: {exc}") from None
            torch.save(
                {"input_ids": ids.detach().cpu().contiguous(), "hidden": hidden.detach().cpu().contiguous()},
                cache_dir / f"{written:06d}.pt",
            )
            written += 1
            if written % 10 == 0:
                print(f"dumped {written}/{n_samples} hidden sequences", flush=True)
            if device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        catcher.close()
        del catcher
        del language_model
        del target
        import gc

        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    print(f"hidden dump complete n={written}", flush=True)
    return written


def one_anchor_loss(
    draft: DFlashDraftModel,
    embeddings: nn.Module,
    lm_head: nn.Module,
    cfg: DFlashTrainConfig,
    seq: torch.Tensor,
    target_hidden: torch.Tensor,
    anchor: int,
) -> tuple[torch.Tensor, float]:
    draft_device = next(draft.parameters()).device
    dtype = next(draft.parameters()).dtype
    seq_len = int(seq.numel())
    blk = cfg.block_size
    noise_ids = torch.full((1, blk), cfg.mask_token_id, dtype=torch.long, device=draft_device)
    noise_ids[0, 0] = seq.to(draft_device)[anchor]
    labels = seq.to(draft_device)[anchor + 1 : anchor + blk].unsqueeze(0)
    context_keep = torch.zeros((1, seq_len), dtype=torch.bool, device=draft_device)
    context_keep[0, : anchor + 1] = True
    noise_pos = torch.arange(anchor, anchor + blk, device=draft_device).unsqueeze(0)
    emb_weight = embeddings.weight if hasattr(embeddings, "weight") else None
    emb_dev = emb_weight.device if emb_weight is not None else draft_device
    noise = embeddings(noise_ids.to(emb_dev)).to(device=draft_device, dtype=dtype)
    context = target_hidden.unsqueeze(0).to(device=draft_device, dtype=dtype)
    context_pos = torch.arange(seq_len, device=draft_device).unsqueeze(0)
    attn_bias = block_attention_bias(context_keep, blk, dtype)
    hidden = draft(noise, context, noise_pos, context_pos, attn_bias)
    pred = hidden[:, 1:, :]
    if isinstance(lm_head, nn.Embedding):
        w = lm_head.weight
        logits = F.linear(pred.to(device=w.device, dtype=torch.float32), w.float())
        labels = labels.to(w.device)
    else:
        head_dev = next(lm_head.parameters()).device
        logits = lm_head(pred.float().to(head_dev))
        labels = labels.to(head_dev)
    token_loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        reduction="none",
    ).view(1, cfg.block_predict)
    decay = torch.tensor(
        positional_loss_weights(cfg.block_predict, cfg.loss_decay_gamma),
        device=logits.device,
        dtype=token_loss.dtype,
    )
    loss = (token_loss * decay).mean()
    with torch.no_grad():
        acc = float((logits.argmax(dim=-1) == labels).float().mean().item())
    return loss, acc


def train(cfg: DFlashTrainConfig, out_dir: Path) -> dict[str, Any]:
    set_seed(cfg.seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} target={cfg.target_model}", flush=True)
    token = hf_token()
    if not token:
        raise RuntimeError("HF token missing; Modal secret hf-token is required to upload")
    from huggingface_hub import HfApi

    who = HfApi(token=token).whoami()
    print(f"hf user={who.get('name')} keys={[k for k in os.environ if 'HF' in k.upper() or 'HUGG' in k.upper()]}", flush=True)

    cache_dir = out_dir / "hidden-cache"
    samples = list(cache_dir.glob("[0-9]*.pt"))
    if len(samples) < 8:
        raise RuntimeError(
            f"hidden-cache has {len(samples)} sequences; run dump_hiddens in a separate GPU job first"
        )
    if not samples:
        raise RuntimeError("hidden-cache is empty")

    embed_w = torch.load(cache_dir / "embed.pt", map_location="cpu", weights_only=True)["weight"]
    head_w = torch.load(cache_dir / "lm_head.pt", map_location="cpu", weights_only=True)["weight"]
    embeddings = nn.Embedding.from_pretrained(embed_w.float(), freeze=True)
    lm_head = nn.Linear(head_w.shape[1], head_w.shape[0], bias=False)
    with torch.no_grad():
        lm_head.weight.copy_(head_w.float())
    for p in lm_head.parameters():
        p.requires_grad_(False)

    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    draft = DFlashDraftModel(cfg).to(device=device, dtype=dtype)
    draft.train()
    try:
        from bitsandbytes.optim import PagedAdamW8bit

        opt: Any = PagedAdamW8bit(
            (p for p in draft.parameters() if p.requires_grad),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            betas=(0.9, 0.95),
        )
        print("optimizer=PagedAdamW8bit", flush=True)
    except Exception as exc:
        print(f"PagedAdamW8bit unavailable ({exc}); using AdamW", flush=True)
        opt = AdamW(
            (p for p in draft.parameters() if p.requires_grad),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            betas=(0.9, 0.95),
        )
    gen = torch.Generator(device="cpu")
    gen.manual_seed(cfg.seed)
    (out_dir / "config.json").write_text(json.dumps(cfg.__dict__, indent=2) + "\n")
    step = 0
    opt.zero_grad(set_to_none=True)
    t0 = time.time()
    running = 0.0
    n_loss = 0
    history: list[dict[str, float]] = []
    epoch = 0
    while step < cfg.max_steps:
        epoch += 1
        order = torch.randperm(len(samples), generator=gen).tolist()
        for idx in order:
            if step >= cfg.max_steps:
                break
            packed = torch.load(samples[idx], map_location="cpu", weights_only=True)
            ids = packed["input_ids"]
            target_hidden = packed["hidden"].to(device=device, dtype=dtype)
            seq = ids.to(device)
            anchors = sample_anchors(int(seq.numel()), cfg.block_size, cfg.max_anchors, gen)
            if anchors.numel() == 0:
                continue
            n_a = int(anchors.numel())
            scale = float(cfg.grad_accum * n_a)
            acc_sum = 0.0
            loss_sum = 0.0
            try:
                for anchor in anchors.tolist():
                    loss, acc = one_anchor_loss(
                        draft, embeddings, lm_head, cfg, seq, target_hidden, int(anchor)
                    )
                    (loss / scale).backward()
                    acc_sum += acc
                    loss_sum += float(loss.detach())
                    del loss
            except torch.cuda.OutOfMemoryError as exc:
                opt.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                raise RuntimeError(f"draft step OOM: {exc}") from exc
            stats = {"n_anchors": float(n_a), "acc": acc_sum / n_a}
            running += loss_sum / n_a
            n_loss += 1
            if n_loss % cfg.grad_accum == 0:
                nn.utils.clip_grad_norm_(draft.parameters(), cfg.grad_clip)
                for pg in opt.param_groups:
                    pg["lr"] = cosine_lr(step, cfg.max_steps, cfg.lr, cfg.warmup_ratio)
                opt.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                avg = running / max(cfg.grad_accum, 1)
                row = {
                    "step": float(step),
                    "loss": avg,
                    "acc": float(stats["acc"]),
                    "lr": float(opt.param_groups[0]["lr"]),
                    "sec": time.time() - t0,
                }
                history.append(row)
                print(
                    f"step {step}/{cfg.max_steps} loss={avg:.4f} acc={stats['acc']:.3f} "
                    f"anchors={int(stats['n_anchors'])} lr={row['lr']:.2e}",
                    flush=True,
                )
                running = 0.0
                n_loss = 0
                if step % 100 == 0 or step == cfg.max_steps:
                    ckpt = out_dir / f"draft-step{step}.pt"
                    torch.save({"cfg": cfg.__dict__, "draft": draft.state_dict()}, ckpt)

    weights = out_dir / "dflash_draft.pt"
    torch.save({"cfg": cfg.__dict__, "draft": draft.state_dict()}, weights)
    gguf_path = out_dir / "Qwen3.8-27B-ABLITERATED-DFlash-F16.gguf"
    export_dflash_gguf(draft, cfg, gguf_path)
    summary = {
        "steps": step,
        "seconds": time.time() - t0,
        "gguf": str(gguf_path),
        "weights": str(weights),
        "target_layer_ids": cfg.target_layer_ids,
        "history_tail": history[-8:],
        "n_cache": len(samples),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    try:
        summary["hub_url"] = push_to_hub(out_dir, cfg, summary)
    except Exception as exc:
        summary["hub_error"] = f"{type(exc).__name__}: {exc}"
        print(f"hub upload failed: {summary['hub_error']}", flush=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    return summary

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a custom DFlash draft for Qwen3.8-27B")
    p.add_argument("--out-dir", default="/vol/dflash-out")
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--max-seq-len", type=int, default=None)
    p.add_argument("--max-anchors", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--no-4bit", action="store_true")
    p.add_argument("--target", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = DFlashTrainConfig()
    if args.target:
        cfg.target_model = args.target
    if args.max_steps is not None:
        cfg.max_steps = args.max_steps
    if args.max_samples is not None:
        cfg.max_samples = args.max_samples
    if args.max_seq_len is not None:
        cfg.max_seq_len = args.max_seq_len
    if args.max_anchors is not None:
        cfg.max_anchors = args.max_anchors
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.no_4bit:
        cfg.load_in_4bit = False
    if args.smoke:
        cfg.max_steps = 4
        cfg.max_samples = 16
        cfg.max_seq_len = 256
        cfg.max_anchors = 8
        cfg.grad_accum = 1
    train(cfg, Path(args.out_dir))


if __name__ == "__main__":
    main()
