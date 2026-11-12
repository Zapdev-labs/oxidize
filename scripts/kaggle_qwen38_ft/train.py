"""QLoRA fine-tune of Qwen3.8-27B on Kaggle 2x T4 inside a 10-minute wall clock.

Design (see README.md next to this file for the full spec):
  * frozen NF4 base (pre-quantized unsloth checkpoint, text tower only), fp16 compute;
  * LoRA r=16 (own fp16-compute / fp32-master module, no peft) on every projection
    (GDN in/out, attention q/k/v/o, MLP) of the layers that train;
  * two T4s as a 2-stage pipeline. Default "stream" mode: stage 0 (GPU0) holds a
    frozen prefix and runs forward-only in its own thread, stage 1 (GPU1) holds the
    LoRA layers + lm_head and does fwd+bwd; the split is cost-balanced under GPU0's
    memory, so neither GPU waits on the other (no pipeline bubble, any step size).
    "1f1b" mode trains all layers with a 1F1B schedule instead (slower, bubbles);
  * per-layer activation checkpointing; custom Turing Gated DeltaNet kernel (gdn_t4);
  * lm_head + CE computed only on assistant tokens, in vocab-sized chunks, with the
    gradient produced in the forward pass (no 248k-wide fp32 logits kept);
  * time-based LR schedule: warmup + cosine over the training budget, so the run
    adapts to whatever throughput the box delivers;
  * quality gates, run after the adapter is saved: held-out assistant NLL with LoRA
    off/on on both datasets, and top-k KL(base || tuned) (drift guard).

Run standalone (python train.py) or via push_kernel.py on Kaggle. All knobs are env
vars (OXFT_*), see `Cfg`.
"""
from __future__ import annotations

import json, math, os, random, re, sys, time, traceback, urllib.request
from dataclasses import dataclass, field, asdict

T_START = time.time()


def _env(name, default, cast=str):
    v = os.environ.get(name)
    return default if v is None or v == "" else cast(v)


@dataclass
class Cfg:
    model_dir: str = _env("OXFT_MODEL_DIR", "/root/qwen38-bnb4")
    out_dir: str = _env("OXFT_OUT", "/kaggle/working")
    devices: str = _env("OXFT_DEVICES", "cuda:0,cuda:1")
    mode: str = _env("OXFT_MODE", "stream")              # stream | 1f1b (see Pipe)
    split: int = _env("OXFT_SPLIT", -1, int)               # first layer of stage 1 (-1: cost-balanced)
    gpu0_gib: float = _env("OXFT_GPU0_GIB", 12.0, float)   # weight budget for stage 0 (T4: ~14.7 usable)
    # per-layer costs relative to a frozen forward (T4, 4096 tok, measured by bench_layer.py):
    # frozen fwd 220 ms, ckpt train 800 ms, no-ckpt train 560 ms, lm_head CE ~650 ms
    cost_train: float = _env("OXFT_COST_TRAIN", 3.6, float)
    cost_nockpt: float = _env("OXFT_COST_NOCKPT", 2.55, float)
    cost_ce: float = _env("OXFT_COST_CE", 3.0, float)
    nockpt: int = _env("OXFT_NOCKPT", 2, int)              # last N trainable layers skip checkpointing
    train_seconds: float = _env("OXFT_TRAIN_SECONDS", 1e9, float)   # capped by the deadline
    deadline_seconds: float = _env("OXFT_DEADLINE", 570.0, float)   # hard stop since process start
    max_len: int = _env("OXFT_MAX_LEN", 2048, int)
    mb_tokens: int = _env("OXFT_MB_TOKENS", 4096, int)    # padded tokens per micro-batch
    mbs_per_step: int = _env("OXFT_MBS_PER_STEP", 0, int)   # 0: 1 (stream) / 4 (1f1b, fills the pipe)
    lr: float = _env("OXFT_LR", 2e-4, float)
    warmup_frac: float = _env("OXFT_WARMUP", 0.08, float)
    rank: int = _env("OXFT_RANK", 16, int)
    alpha: int = _env("OXFT_ALPHA", 32, int)
    lora_from_layer: int = _env("OXFT_LORA_FROM", -1, int)   # -1: = split (stream) / 0 (1f1b)
    mix_tools: float = _env("OXFT_MIX_TOOLS", 0.35, float)  # token share of the tool-calling set
    eval_chat: int = _env("OXFT_EVAL_CHAT", 48, int)
    eval_tools: int = _env("OXFT_EVAL_TOOLS", 24, int)
    kl_topk: int = _env("OXFT_KL_TOPK", 32, int)
    ce_chunk: int = _env("OXFT_CE_CHUNK", 256, int)
    seed: int = _env("OXFT_SEED", 0, int)
    gdn_kernel: str = _env("OXFT_GDN", "gdn_t4")          # gdn_t4 | hf
    max_steps: int = _env("OXFT_MAX_STEPS", 0, int)
    trunc_layers: int = _env("OXFT_TRUNC_LAYERS", 0, int)  # debug: keep only the first N real layers
    save_adapter: bool = _env("OXFT_SAVE", 1, int) == 1
    webhook: str = os.environ.get("OXFT_WEBHOOK", "__WEBHOOK__")


C = Cfg()
if C.mbs_per_step <= 0:
    C.mbs_per_step = 1 if C.mode == "stream" else 4
LOG = []


def post(stage, **kw):
    kw = {"t": round(time.time() - T_START, 1), **kw}
    LOG.append({"stage": stage, **kw})
    print(f"[{stage}] {json.dumps(kw, default=str)[:1500]}", flush=True)
    if not C.webhook.startswith("http"):
        return
    try:
        req = urllib.request.Request(C.webhook, data=json.dumps({"stage": stage, **kw}, default=str).encode(),
                                     headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=5)
    except Exception as e:  # noqa: BLE001
        print("webhook failed:", e, flush=True)


import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.checkpoint import checkpoint  # noqa: E402

# --------------------------------------------------------------------------- data

TOOL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


def hermes_to_messages(row):
    """NousResearch/hermes-function-calling-v1 -> Qwen-native tools/tool_calls.

    The Hermes system prompt only restates the tool list in its own XML dialect, so it
    is dropped and the schema is passed through `tools=`, letting the Qwen3.8 template
    render its native <function=...><parameter=...> format (the format the base model
    was trained on)."""
    tools = row["tools"]
    if isinstance(tools, str):
        tools = json.loads(tools)
    msgs = []
    for m in row["conversations"]:
        role = {"system": "system", "human": "user", "gpt": "assistant", "tool": "tool"}[m["from"]]
        if role == "system":
            continue
        if role == "assistant":
            calls = []
            for js in TOOL_RE.findall(m["value"]):
                try:
                    c = json.loads(js)
                except json.JSONDecodeError:
                    return None
                args = c.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        return None
                calls.append({"type": "function", "function": {"name": c["name"], "arguments": args}})
            text = TOOL_RE.sub("", m["value"]).strip()
            msg = {"role": "assistant", "content": text}
            if calls:
                msg["tool_calls"] = calls
            msgs.append(msg)
        else:
            msgs.append({"role": role, "content": m["value"]})
    if not msgs or msgs[-1]["role"] != "assistant":
        return None
    return msgs, tools


def encode(tok, msgs, tools=None):
    """Token ids + labels for the final assistant turn (non-thinking mode).

    With enable_thinking=False the generation prompt ends in an empty think block that
    the template also renders for the assistant turn, so prompt is a strict prefix of
    the full conversation and only the answer (+<|im_end|>) is supervised."""
    kw = dict(tools=tools, enable_thinking=False, tokenize=False)
    prompt = tok.apply_chat_template(msgs[:-1], add_generation_prompt=True, **kw)
    full = tok.apply_chat_template(msgs, add_generation_prompt=False, **kw)
    if not full.startswith(prompt):
        return None
    full = full.rstrip("\n")                   # drop the trailing newline after <|im_end|>
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    ids = tok(full, add_special_tokens=False)["input_ids"]
    if ids[: len(p_ids)] != p_ids or len(ids) <= len(p_ids):
        return None
    labels = [-100] * len(p_ids) + ids[len(p_ids):]
    return ids, labels


def load_data(tok):
    from datasets import load_dataset
    rng = random.Random(C.seed)
    t = time.time()
    chat_tr = load_dataset("HuggingFaceH4/no_robots", split="train")
    chat_te = load_dataset("HuggingFaceH4/no_robots", split="test")
    tools_ds = load_dataset("NousResearch/hermes-function-calling-v1", "func_calling_singleturn", split="train")
    post("datasets_loaded", secs=round(time.time() - t, 1), chat=len(chat_tr), tools=len(tools_ds))

    def enc_chat(rows, n):
        out = []
        idx = list(range(len(rows)))
        rng.shuffle(idx)
        for i in idx:
            r = rows[i]
            msgs = [{"role": m["role"], "content": m["content"]} for m in r["messages"]]
            if msgs[-1]["role"] != "assistant":
                continue
            e = encode(tok, msgs)
            if e and len(e[0]) <= C.max_len:
                out.append(e)
            if len(out) >= n:
                break
        return out

    def enc_tools(rows, idx, n):
        out = []
        for i in idx:
            conv = hermes_to_messages(rows[i])
            if conv is None:
                continue
            e = encode(tok, conv[0], conv[1])
            if e and len(e[0]) <= C.max_len:
                out.append(e)
            if len(out) >= n:
                break
        return out

    t = time.time()
    tidx = list(range(len(tools_ds)))
    rng.shuffle(tidx)
    ev_tools = enc_tools(tools_ds, tidx[: C.eval_tools * 2], C.eval_tools)
    used = set(tidx[: C.eval_tools * 2])
    # enough examples for the whole budget at <= 1.5k tok/s (the stream cycles if it runs out)
    want_tokens = int(min(C.train_seconds, C.deadline_seconds) * 1500)
    tr_tools = enc_tools(tools_ds, [i for i in tidx if i not in used], max(64, int(want_tokens * C.mix_tools / 1500)))
    tr_chat = enc_chat(chat_tr, max(128, int(want_tokens * (1 - C.mix_tools) / 400)))
    ev_chat = enc_chat(chat_te, C.eval_chat)
    post("tokenized", secs=round(time.time() - t, 1), tr_chat=len(tr_chat), tr_tools=len(tr_tools),
         ev_chat=len(ev_chat), ev_tools=len(ev_tools),
         mean_len_chat=sum(len(e[0]) for e in tr_chat) // max(1, len(tr_chat)),
         mean_len_tools=sum(len(e[0]) for e in tr_tools) // max(1, len(tr_tools)))
    return tr_chat, tr_tools, ev_chat, ev_tools


def make_microbatches(examples, mb_tokens, pad_id, rng, chunk=64):
    """Length-bucketed right-padded micro-batches with B*Lpad <= mb_tokens.

    Right padding needs no attention mask: every layer is causal, so pads only see
    real tokens and nothing real sees a pad; pads carry label -100."""
    ex = sorted(examples, key=lambda e: len(e[0]))
    mbs, cur = [], []
    for e in ex:
        L = len(e[0])
        Lp = -(-max([len(x[0]) for x in cur] + [L]) // chunk) * chunk
        if cur and Lp * (len(cur) + 1) > mb_tokens:
            mbs.append(cur)
            cur = []
        cur.append(e)
    if cur:
        mbs.append(cur)
    out = []
    for mb in mbs:
        Lp = -(-max(len(e[0]) for e in mb) // chunk) * chunk
        ids = torch.full((len(mb), Lp), pad_id, dtype=torch.long)
        lab = torch.full((len(mb), Lp), -100, dtype=torch.long)
        for i, (a, b) in enumerate(mb):
            ids[i, : len(a)] = torch.tensor(a)
            lab[i, : len(b)] = torch.tensor(b)
        # shift: position t predicts token t+1
        lab = torch.cat([lab[:, 1:], torch.full((len(mb), 1), -100, dtype=torch.long)], 1)
        out.append((ids, lab))
    rng.shuffle(out)
    return out


def mixed_stream(tr_chat, tr_tools, rng):
    """Interleave the two sets by token share (mix_tools), in pools of ~32k tokens."""
    pool_tokens = 32768
    ic = it = 0
    rng.shuffle(tr_chat)
    rng.shuffle(tr_tools)
    while True:
        pool, nt, nc = [], 0, 0
        while nt + nc < pool_tokens:
            take_tools = (nt < C.mix_tools * (nt + nc + 1)) and tr_tools
            src = tr_tools if take_tools else tr_chat
            i = it if take_tools else ic
            e = src[i % len(src)]
            if take_tools:
                it += 1; nt += len(e[0])
            else:
                ic += 1; nc += len(e[0])
            pool.append(e)
        yield pool, (ic, it)


# --------------------------------------------------------------------------- model


def layer_gib(tc, kind):
    """NF4 weight bytes of one decoder layer (0.53 B/param incl. double-quant absmax;
    in_proj_qkv/a/b stay fp16 in the unsloth checkpoint)."""
    H, nf4, f16 = tc.hidden_size, 3 * tc.hidden_size * tc.intermediate_size, 0
    if kind == "linear_attention":
        dk, dv = tc.linear_key_head_dim, tc.linear_value_head_dim
        nk, nv = tc.linear_num_key_heads, tc.linear_num_value_heads
        f16 += H * (2 * nk * dk + nv * dv) + 2 * H * nv
        nf4 += 2 * H * nv * dv
    else:
        hd = getattr(tc, "head_dim", H // tc.num_attention_heads)
        nf4 += H * tc.num_attention_heads * hd * 3 + 2 * H * tc.num_key_value_heads * hd
    return (0.53 * nf4 + 2 * f16) / 2**30


def plan_split(tc, n_dev):
    """Stage boundary. stream: stage 0 = frozen prefix (fwd only), stage 1 = LoRA layers
    (fwd+bwd) + lm_head; pick the split that balances the two GPUs under stage 0's
    weight budget. 1f1b: halves."""
    n = tc.num_hidden_layers
    if C.split >= 0:
        return C.split
    if C.mode != "stream":
        return n // 2
    kinds = tc.layer_types
    best, best_cost = 0, float("inf")
    for s_ in range(1, n):
        if sum(layer_gib(tc, k) for k in kinds[:s_]) > C.gpu0_gib:
            break
        k = min(C.nockpt, n - s_)
        cost = max(float(s_), (n - s_ - k) * C.cost_train + k * C.cost_nockpt + C.cost_ce)
        if cost < best_cost:
            best, best_cost = s_, cost
    return best


def load_model():
    import transformers
    from transformers import AutoConfig, AutoTokenizer
    from transformers.models.qwen3_5 import modeling_qwen3_5 as mq

    if C.gdn_kernel == "gdn_t4":
        import gdn_t4
        mq.torch_chunk_gated_delta_rule = gdn_t4.chunk_gated_delta_rule
    t = time.time()
    tok = AutoTokenizer.from_pretrained(C.model_dir)
    top = AutoConfig.from_pretrained(C.model_dir)
    tcfg = top.text_config if hasattr(top, "text_config") else top
    if getattr(top, "quantization_config", None) is not None:
        tcfg.quantization_config = top.quantization_config   # pre-quantized: keep as-is
    tcfg.use_cache = False
    if C.trunc_layers:
        tcfg.num_hidden_layers = C.trunc_layers
        tcfg.layer_types = tcfg.layer_types[: C.trunc_layers]
    n = tcfg.num_hidden_layers
    devs = C.devices.split(",")
    split = plan_split(tcfg, len(set(devs)))
    dm = {"model.embed_tokens": "cpu", "model.rotary_emb": devs[0]}
    for i in range(n):
        dm[f"model.layers.{i}"] = devs[0] if i < split else devs[-1]
    dm["model.norm"] = devs[-1]
    dm["lm_head"] = devs[-1]
    if getattr(tcfg, "quantization_config", None) is not None:
        q = tcfg.quantization_config
        q = dict(q if isinstance(q, dict) else q.to_dict())
        q["llm_int8_enable_fp32_cpu_offload"] = True       # embed_tokens lives on CPU (lookup only)
        tcfg.quantization_config = q
    model = mq.Qwen3_5ForCausalLM.from_pretrained(C.model_dir, config=tcfg, device_map=dm, dtype=torch.float16,
                                                  attn_implementation="sdpa")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    post("model_loaded", secs=round(time.time() - t, 1), layers=n, split=split, mode=C.mode,
         tf=transformers.__version__,
         stage0_gib=round(sum(layer_gib(tcfg, k) for k in tcfg.layer_types[:split]), 2),
         mem=[round(torch.cuda.memory_allocated(d) / 2**30, 2) for d in sorted(set(devs))])
    return tok, model, devs, split


# --------------------------------------------------------------------------- LoRA


class LoRA(torch.nn.Module):
    """y = base(x) + s * (x A^T) B^T with fp32 master A/B, fp16 compute.

    peft's bnb LoRA runs the adapter in the adapter dtype: with fp32 adapters (needed
    for small updates) that casts every [T, 17408] activation to fp32 and uses SGEMM,
    which measured ~30% of a layer's train step on T4. Here the casts are on the
    rank-16 weights only, so the adapter is a pair of tiny fp16 tensor-core GEMMs."""

    enabled = True

    def __init__(self, base, r, alpha):
        super().__init__()
        self.base = base
        dev = base.weight.device
        fin, fout = base.in_features, base.out_features
        self.lora_A = torch.nn.Parameter(torch.empty(r, fin, device=dev, dtype=torch.float32))
        self.lora_B = torch.nn.Parameter(torch.zeros(fout, r, device=dev, dtype=torch.float32))
        torch.nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.scale = alpha / r

    def forward(self, x):
        y = self.base(x)
        if not LoRA.enabled:
            return y
        a = self.lora_A.to(x.dtype)
        b = (self.lora_B * self.scale).to(x.dtype)
        return y + F.linear(F.linear(x, a), b)


LORA_TARGETS = {"linear_attn": ("in_proj_qkv", "in_proj_z", "out_proj"),
                "self_attn": ("q_proj", "k_proj", "v_proj", "o_proj"),
                "mlp": ("gate_proj", "up_proj", "down_proj")}


def add_lora(model, lora_from):
    torch.manual_seed(C.seed)
    params, names = [], []
    for li, layer in enumerate(model.model.layers):
        if li < lora_from:
            continue
        for blk, projs in LORA_TARGETS.items():
            parent = getattr(layer, blk, None)
            if parent is None:
                continue
            for pn in projs:
                m = LoRA(getattr(parent, pn), C.rank, C.alpha)
                setattr(parent, pn, m)
                params += [m.lora_A, m.lora_B]
                names.append((f"model.language_model.layers.{li}.{blk}.{pn}", m))
    post("lora", trainable=sum(p.numel() for p in params), modules=len(names), from_layer=lora_from)
    return params, names


def save_adapter(names, tok, path, base_id="Qwen/Qwen3.8-27B"):
    """PEFT-format adapter keyed for the full Qwen3.8 checkpoint
    (Qwen3_5ForConditionalGeneration: model.language_model.layers.*)."""
    from safetensors.torch import save_file
    os.makedirs(path, exist_ok=True)
    sd = {}
    for n, m in names:
        sd[f"base_model.model.{n}.lora_A.weight"] = m.lora_A.detach().to("cpu", torch.float16).contiguous()
        sd[f"base_model.model.{n}.lora_B.weight"] = m.lora_B.detach().to("cpu", torch.float16).contiguous()
    save_file(sd, os.path.join(path, "adapter_model.safetensors"))
    layers = sorted({int(n.split(".")[3]) for n, _ in names})
    cfg = dict(peft_type="LORA", task_type="CAUSAL_LM", base_model_name_or_path=base_id, r=C.rank,
               lora_alpha=C.alpha, lora_dropout=0.0, bias="none", fan_in_fan_out=False, inference_mode=True,
               target_modules=sorted({p for v in LORA_TARGETS.values() for p in v}),
               layers_to_transform=layers, layers_pattern="layers", modules_to_save=None,
               init_lora_weights=True, use_rslora=False, use_dora=False)
    json.dump(cfg, open(os.path.join(path, "adapter_config.json"), "w"), indent=1)
    tok.save_pretrained(path)


# --------------------------------------------------------------------------- pipeline


class Pipe:
    """Two-stage layer pipeline over the HF decoder layers (no HF forward/mask machinery).

    stream: stage 0 holds only frozen layers and runs forward-only in its own thread,
    so it never waits on a backward; stage 1 holds the LoRA layers and the lm_head.
    1f1b: both stages train; micro-batches are issued 1F1B from one thread."""

    def __init__(self, model, devs, split, lora_from):
        m = model.model
        self.embed, self.rot, self.norm = m.embed_tokens, m.rotary_emb, m.norm
        self.layers = list(m.layers)
        self.lm_w = model.lm_head.weight          # frozen fp16 [V, D]
        self.d0, self.d1 = torch.device(devs[0]), torch.device(devs[-1])
        self.split, self.lora_from = split, lora_from
        n = len(self.layers)
        self.no_ckpt = set(range(max(lora_from, n - C.nockpt), n)) if C.nockpt > 0 else set()
        self._pos_cache = {}
        self._lock = __import__("threading").Lock()

    def _pos(self, B, T, dev):
        key = (B, T, str(dev))
        with self._lock:
            if key not in self._pos_cache:
                rdev = next(self.rot.buffers()).device
                pid = torch.arange(T, device=rdev).view(1, 1, T).expand(3, B, T)
                cos, sin = self.rot(torch.empty(1, 1, 1, device=rdev, dtype=torch.float16), pid)
                self._pos_cache[key] = (cos.to(dev), sin.to(dev), pid[0].to(dev))
                if len(self._pos_cache) > 128:
                    self._pos_cache.pop(next(iter(self._pos_cache)))
            return self._pos_cache[key]

    @staticmethod
    def _layer(layer, h, cos, sin, pid):
        h = layer(h, position_embeddings=(cos, sin), attention_mask=None, position_ids=pid, past_key_values=None)
        return h[0] if isinstance(h, tuple) else h

    def run_layers(self, lo, hi, h, train):
        B, T, _ = h.shape
        cos, sin, pid = self._pos(B, T, h.device)
        for li in range(lo, hi):
            layer = self.layers[li]
            if not train or li < self.lora_from:
                with torch.no_grad():
                    h = self._layer(layer, h, cos, sin, pid)
            elif li in self.no_ckpt:
                h = self._layer(layer, h, cos, sin, pid)
            else:
                h = checkpoint(self._layer, layer, h, cos, sin, pid, use_reentrant=False)
        return h

    def stage0(self, ids, train):
        h = self.embed(ids).to(self.d0, non_blocking=True)
        return self.run_layers(0, self.split, h, train)

    def stage1(self, h, train):
        h = self.run_layers(self.split, len(self.layers), h, train)
        return self.norm(h)


class ChunkedCE(torch.autograd.Function):
    """Sum of token CE over selected rows; gradient wrt h computed in forward,
    chunk by chunk, so no [n, 248k] fp32 logits survive. lm_head is frozen."""

    @staticmethod
    def forward(ctx, h, W, labels, chunk, scale):
        n = h.shape[0]
        dh = torch.empty_like(h)
        tot = torch.zeros((), device=h.device, dtype=torch.float32)
        for s in range(0, n, chunk):
            hc = h[s:s + chunk]
            lab = labels[s:s + chunk]
            logits = (hc @ W.t()).float()
            lse = torch.logsumexp(logits, -1)
            tot += (lse - logits.gather(1, lab[:, None]).squeeze(1)).sum()
            p = logits.sub_(lse[:, None]).exp_()
            p[torch.arange(p.shape[0], device=p.device), lab] -= 1.0
            dh[s:s + chunk] = (p.mul_(scale).to(W.dtype) @ W)
        ctx.save_for_backward(dh)
        return tot * scale

    @staticmethod
    def backward(ctx, g):
        (dh,) = ctx.saved_tensors
        return dh * g.to(dh.dtype), None, None, None, None


def lm_loss(pipe, hN, labels, scale):
    sel = labels.view(-1) != -100
    h = hN.reshape(-1, hN.shape[-1])[sel.to(hN.device)]
    lab = labels.view(-1)[sel].to(hN.device)
    return ChunkedCE.apply(h, pipe.lm_w, lab, C.ce_chunk, scale), int(sel.sum())


@torch.no_grad()
def logprobs_topk(pipe, hN, labels, k, ref=None):
    """Per assistant token: target NLL and either (top-k ids, logprobs) for the base
    model, or KL(base || tuned) restricted to (and renormalised on) the base top-k
    support when `ref` holds the base top-k."""
    sel = labels.view(-1) != -100
    h = hN.reshape(-1, hN.shape[-1])[sel.to(hN.device)]
    lab = labels.view(-1)[sel].to(hN.device)
    nll, ids_l, lp_l, kl = 0.0, [], [], 0.0
    for s in range(0, h.shape[0], C.ce_chunk):
        lg = (h[s:s + C.ce_chunk] @ pipe.lm_w.t()).float().log_softmax(-1)
        nll += -lg.gather(1, lab[s:s + C.ce_chunk, None]).sum().item()
        if ref is None:
            v, i = lg.topk(k, -1)
            ids_l.append(i.cpu()); lp_l.append(v.cpu())
        else:
            ri, rv = ref[0][s:s + C.ce_chunk].to(lg.device), ref[1][s:s + C.ce_chunk].to(lg.device)
            rb = rv - rv.logsumexp(-1, keepdim=True)          # renormalise both on the base top-k
            rt = lg.gather(1, ri)
            rt = rt - rt.logsumexp(-1, keepdim=True)
            kl += (rb.exp() * (rb - rt)).sum().item()
    n = int(sel.sum())
    if ref is None:
        return nll, n, (torch.cat(ids_l), torch.cat(lp_l))
    return nll, n, kl


def _producer(pipe, items, q, stop):
    """Stage-0 worker (stream mode): forward-only on GPU0, hands activations to GPU1."""
    try:
        for it in items:
            if stop.is_set():
                break
            ids, lab = it
            with torch.no_grad():
                h = pipe.stage0(ids, train=False).to(pipe.d1, non_blocking=True)
            q.put((h, lab))
    except Exception as ex:  # noqa: BLE001
        q.put(ex)
        return
    q.put(None)


def _start_producer(pipe, items, depth=3):
    import queue, threading
    q, stop = queue.Queue(maxsize=depth), threading.Event()
    th = threading.Thread(target=_producer, args=(pipe, items, q, stop), daemon=True)
    th.start()
    return q, stop, th


def _get(q):
    x = q.get()
    if isinstance(x, Exception):
        raise x
    return x


def evaluate_pair(pipe, mbs, tag):
    """Base (LoRA off) then tuned (LoRA on) held-out assistant NLL + top-k KL(base||tuned).
    Stream mode: stage 0 is frozen, so its activations are computed once (overlapped with
    the base pass on GPU1) and reused for the tuned pass."""
    stream = C.mode == "stream"
    out = {}
    cache, refs = [], []
    for phase in ("base", "tuned"):
        t = time.time()
        LoRA.enabled = phase == "tuned"
        nll, n, kl = 0.0, 0, 0.0
        if phase == "base" and stream:
            q, _, th = _start_producer(pipe, mbs)
            src = iter(lambda: _get(q), None)
        elif stream:
            src = iter(cache)
        else:
            src = ((pipe.stage0(ids, train=False), lab) for ids, lab in mbs)
        for j, (h, lab) in enumerate(src):
            if phase == "base" and stream:
                cache.append((h, lab))
            with torch.no_grad():
                hN = pipe.stage1(h.to(pipe.d1, non_blocking=True), train=False)
                r = logprobs_topk(pipe, hN, lab, C.kl_topk, None if phase == "base" else refs[j])
            nll += r[0]; n += r[1]
            if phase == "base":
                refs.append(r[2])
            else:
                kl += r[2]
        res = dict(nll=round(nll / max(1, n), 4), tokens=n, secs=round(time.time() - t, 1))
        if phase == "tuned":
            res["kl_topk"] = round(kl / max(1, n), 5)
        post(f"eval_{phase}_{tag}", **res)
        out[phase] = res
    LoRA.enabled = True
    return out["base"], out["tuned"]


def _optimizer(params):
    try:
        import bitsandbytes as bnb
        return bnb.optim.AdamW8bit(params, lr=C.lr, betas=(0.9, 0.99), weight_decay=0.0), "adamw8bit"
    except Exception:  # noqa: BLE001
        return torch.optim.AdamW(params, lr=C.lr, betas=(0.9, 0.99), weight_decay=0.0), "adamw"


def _mb_iter(stream, pad_id, rng):
    while True:
        pool, _ = next(stream)
        yield from make_microbatches(pool, C.mb_tokens, pad_id, rng)


def train(pipe, params, stream, pad_id, rng, t_train0, budget):
    opt, opt_name = _optimizer(params)
    step, tokens, sup_tokens, losses = 0, 0, 0, []
    stats = dict(opt=opt_name, budget=round(budget, 1), mode=C.mode)
    mbs_it = _mb_iter(stream, pad_id, rng)
    if C.mode == "stream":
        q, stop, th = _start_producer(pipe, mbs_it, depth=max(2, C.mbs_per_step + 1))

    def lr_at(frac):
        if frac < C.warmup_frac:
            return C.lr * max(frac, 0.02) / C.warmup_frac
        f = min(1.0, (frac - C.warmup_frac) / (1 - C.warmup_frac))
        return C.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * f)))

    while True:
        el = time.time() - t_train0
        if el >= budget or (C.max_steps and step >= C.max_steps):
            break
        for g in opt.param_groups:
            g["lr"] = lr_at(el / budget)
        step_loss = torch.zeros((), device=pipe.d1)
        if C.mode == "stream":
            items = [_get(q) for _ in range(C.mbs_per_step)]
            n_sup = sum(int((lab != -100).sum()) for _, lab in items)
            scale = 1.0 / max(1, n_sup)
            for h, lab in items:
                hN = pipe.stage1(h, train=True)
                loss, _ = lm_loss(pipe, hN, lab, scale)
                loss.backward()
                step_loss += loss.detach()
                tokens += h.shape[0] * h.shape[1]
            del items, h, hN, loss
        else:
            # ---- 1F1B over the micro-batches (issue order keeps both GPUs fed)
            mbs = [next(mbs_it) for _ in range(C.mbs_per_step)]
            n_sup = sum(int((lab != -100).sum()) for _, lab in mbs)
            scale = 1.0 / max(1, n_sup)
            fwd0 = []

            def F0(i):
                h = pipe.stage0(mbs[i][0], train=True)
                hb = h.detach().to(pipe.d1, non_blocking=True).requires_grad_(h.requires_grad)
                fwd0.append((h, hb))

            F0(0)
            for i in range(len(mbs)):
                if i + 1 < len(mbs):
                    F0(i + 1)
                h, hb = fwd0[i]
                hN = pipe.stage1(hb, train=True)
                loss, _ = lm_loss(pipe, hN, mbs[i][1], scale)
                loss.backward()
                step_loss += loss.detach()
                if h.requires_grad:
                    torch.autograd.backward(h, hb.grad.to(pipe.d0, non_blocking=True))
                fwd0[i] = None
            tokens += sum(ids.numel() for ids, _ in mbs)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        sup_tokens += n_sup
        lv = step_loss.item()
        if not math.isfinite(lv):
            post("nonfinite_loss", step=step)
            raise RuntimeError("non-finite loss")
        losses.append((step, round(lv, 4)))
        el = time.time() - t_train0
        post("step", step=step, loss=round(lv, 4), lr=f"{lr_at(el / budget):.2e}", el=round(el, 1),
             tok_s=round(tokens / el, 1), sup_tok=sup_tokens,
             mem=[round(torch.cuda.max_memory_allocated(d) / 2**30, 2) for d in sorted({pipe.d0, pipe.d1}, key=str)])
    el = time.time() - t_train0
    if C.mode == "stream":
        stop.set()
        stats["_producer"] = (q, th)
    stats.update(steps=step, tokens=tokens, sup_tokens=sup_tokens, secs=round(el, 1),
                 tok_s=round(tokens / max(el, 1e-6), 1), losses=losses)
    post("train_done", **{k: v for k, v in stats.items() if not k.startswith("_")})
    return stats


def _drain(q, th):
    while th.is_alive():
        try:
            q.get(timeout=0.5)
        except Exception:  # noqa: BLE001
            pass
    th.join()


def main():
    import threading
    random.seed(C.seed)
    torch.manual_seed(C.seed)
    rng = random.Random(C.seed)
    post("cfg", **asdict(C) | {"webhook": "set" if C.webhook.startswith("http") else "unset"})
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(C.model_dir)
    data = {}

    def _data():
        try:
            data["v"] = load_data(tok)
        except Exception as ex:  # noqa: BLE001
            data["e"] = ex
    dth = threading.Thread(target=_data, daemon=True)
    dth.start()                                # datasets + tokenization overlap the model load
    wth = None
    if C.gdn_kernel == "gdn_t4":
        import gdn_t4
        from transformers import AutoConfig
        tc0 = AutoConfig.from_pretrained(C.model_dir)
        tc0 = getattr(tc0, "text_config", tc0)

        def _warm():
            t = time.time()
            for d in sorted(set(C.devices.split(","))):
                gdn_t4.warmup(d, H=tc0.linear_num_value_heads, K=tc0.linear_key_head_dim,
                              V=tc0.linear_value_head_dim)
            post("gdn_warmup", secs=round(time.time() - t, 1))
        wth = threading.Thread(target=_warm, daemon=True)
        wth.start()                            # Triton JIT overlaps the model load too
    _, model, devs, split = load_model()
    lora_from = split if C.mode == "stream" else max(0, C.lora_from_layer)
    params, names = add_lora(model, lora_from)
    pipe = Pipe(model, devs, split, lora_from)
    dth.join()
    if wth is not None:
        wth.join()
    if "e" in data:
        raise data["e"]
    tr_chat, tr_tools, ev_chat, ev_tools = data["v"]
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    t0 = time.time()
    budget = min(C.train_seconds, C.deadline_seconds - (t0 - T_START) - 8.0)
    if budget < 10:
        raise RuntimeError(f"no time left to train (budget {budget:.1f}s)")
    stats = train(pipe, params, mixed_stream(tr_chat, tr_tools, rng), pad_id, rng, t0, budget)
    t_train_end = time.time()
    sd = os.path.join(C.out_dir, "qwen38-27b-oxft-lora")
    if C.save_adapter:
        save_adapter(names, tok, sd)
    t_saved = time.time()
    post("finetune_wallclock", start_to_adapter_saved=round(t_saved - T_START, 1),
         train=round(t_train_end - t0, 1), adapter=sd if C.save_adapter else None)
    prod = stats.pop("_producer", None)
    if prod:
        _drain(*prod)
    ev_c = make_microbatches(ev_chat, C.mb_tokens, pad_id, random.Random(1))
    ev_t = make_microbatches(ev_tools, C.mb_tokens, pad_id, random.Random(2))
    base_c, fin_c = evaluate_pair(pipe, ev_c, "chat")
    base_t, fin_t = evaluate_pair(pipe, ev_t, "tools")
    report = dict(cfg=asdict(C) | {"webhook": None}, split=split, lora_from=lora_from, train=stats,
                  base_chat=base_c, base_tools=base_t, tuned_chat=fin_c, tuned_tools=fin_t,
                  wallclock=dict(start_to_adapter_saved=round(t_saved - T_START, 1),
                                 total=round(time.time() - T_START, 1)),
                  gate=dict(chat_nll_delta=round(fin_c["nll"] - base_c["nll"], 4),
                            tools_nll_delta=round(fin_t["nll"] - base_t["nll"], 4),
                            chat_kl=fin_c.get("kl_topk"), tools_kl=fin_t.get("kl_topk")))
    report["gate"]["pass"] = (report["gate"]["chat_nll_delta"] <= 0.0 and report["gate"]["tools_nll_delta"] <= 0.0
                              and (fin_c.get("kl_topk") or 0) < 0.15
                              and report["wallclock"]["start_to_adapter_saved"] < 600)
    os.makedirs(C.out_dir, exist_ok=True)
    json.dump(report, open(os.path.join(C.out_dir, "oxft_report.json"), "w"), indent=1)
    json.dump(LOG, open(os.path.join(C.out_dir, "oxft_log.json"), "w"), indent=1, default=str)
    post("done", gate=report["gate"], wallclock=report["wallclock"])


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001
        post("error", err=repr(e), tb=traceback.format_exc()[-3000:])
        raise
