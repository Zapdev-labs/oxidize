#!/usr/bin/env python3
"""Train an oxidize EAGLE-4 drafter from an E4H1 hidden-state dump.

The drafter matches oxidize-c/src/model/eagle4.c: per-source RMSNorm, a
fusion projection, an embedding/hidden projection, one sliding-window
layer, a post-norm feedback state, and a low-rank candidate head.
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np


def load_dump(path: Path) -> tuple[np.ndarray, np.ndarray, list[int]]:
    data = path.read_bytes()
    if data[:4] != b"E4H1":
        raise SystemExit(f"{path} is not an E4H1 capture")
    n_tokens, n_layers, hidden = struct.unpack_from("<III", data, 4)
    off = 16
    layers = list(struct.unpack_from(f"<{n_layers}I", data, off))
    off += 4 * n_layers
    tokens = np.frombuffer(data, dtype="<u4", count=n_tokens, offset=off).copy()
    off += 4 * n_tokens
    hiddens = np.frombuffer(
        data, dtype="<f4", count=n_tokens * n_layers * hidden, offset=off
    ).copy()
    hiddens = hiddens.reshape(n_tokens, n_layers, hidden)
    return tokens, hiddens, layers


def rms(x: np.ndarray, weight: np.ndarray, eps: float) -> tuple[np.ndarray, np.ndarray]:
    var = np.mean(x * x, axis=-1, keepdims=True)
    inv = 1.0 / np.sqrt(var + eps)
    y = x * inv * weight
    return y, inv


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-np.clip(x, -40, 40)))


class Eagle4:
    """F32 drafter. Parameters are numpy arrays; grads are filled in-place."""

    def __init__(
        self,
        hidden: int,
        target: int,
        n_aux: int,
        vocab: int,
        intermediate: int,
        n_heads: int,
        head_dim: int,
        rank: int,
        window: int,
        eps: float,
        rng: np.random.Generator,
    ) -> None:
        self.hidden = hidden
        self.target = target
        self.n_aux = n_aux
        self.vocab = vocab
        self.intermediate = intermediate
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.rank = rank
        self.window = window
        self.eps = eps
        scale = 0.02

        def w(shape: tuple[int, ...]) -> np.ndarray:
            return rng.normal(0, scale, size=shape).astype(np.float32)

        self.p: dict[str, np.ndarray] = {
            "fc_norm.weight": np.ones(target, np.float32),
            "fc.weight": w((hidden, n_aux * target)),
            "embed_tokens.weight": w((vocab, hidden)),
            "eh_proj.weight": w((hidden, 2 * hidden)),
            "pre_norm.weight": np.ones(hidden, np.float32),
            "q_proj.weight": w((n_heads * head_dim, hidden)),
            "k_proj.weight": w((n_heads * head_dim, hidden)),
            "v_proj.weight": w((n_heads * head_dim, hidden)),
            "o_proj.weight": w((hidden, n_heads * head_dim)),
            "ffn_norm.weight": np.ones(hidden, np.float32),
            "gate_proj.weight": w((intermediate, hidden)),
            "up_proj.weight": w((intermediate, hidden)),
            "down_proj.weight": w((hidden, intermediate)),
            "post_norm.weight": np.ones(hidden, np.float32),
            "lm_head.weight": w((vocab, hidden)),
            "lr_down.weight": w((rank, hidden)),
            "lr_up.weight": w((vocab, rank)),
        }
        self.g = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.m = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.step_n = 0

    def zero_grad(self) -> None:
        for arr in self.g.values():
            arr.fill(0)

    def adam(self, lr: float) -> None:
        self.step_n += 1
        b1, b2, eps, wd = 0.9, 0.95, 1e-8, 0.01
        for name, p in self.p.items():
            grad = self.g[name]
            if name.endswith("weight") and p.ndim == 2:
                grad = grad + wd * p
            self.m[name] = b1 * self.m[name] + (1 - b1) * grad
            self.v[name] = b2 * self.v[name] + (1 - b2) * (grad * grad)
            mhat = self.m[name] / (1 - b1**self.step_n)
            vhat = self.v[name] / (1 - b2**self.step_n)
            p -= lr * mhat / (np.sqrt(vhat) + eps)

    def _trunk(
        self,
        token: int,
        aux: np.ndarray,
        g_override: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """Forward one draft step. g_override is the previous post-norm state
        used by later steps in a chain. The first step reads target features."""
        p = self.p
        eps = self.eps
        from_feedback = g_override is not None
        if from_feedback:
            aux_r = an = inv = cat = None
            g = g_override
        else:
            aux_r = aux.reshape(self.n_aux, self.target)
            an, inv = rms(aux_r, p["fc_norm.weight"], eps)
            cat = np.ascontiguousarray(an.reshape(-1))
            g = p["fc.weight"] @ cat
        emb = p["embed_tokens.weight"][token]
        cat2 = np.concatenate([emb, g])
        u = p["eh_proj.weight"] @ cat2
        un, uinv = rms(u[None, :], p["pre_norm.weight"], eps)
        un = un[0]
        v = p["v_proj.weight"] @ un
        proj = p["o_proj.weight"] @ v
        hidden = g + proj
        fn, finv = rms(hidden[None, :], p["ffn_norm.weight"], eps)
        fn = fn[0]
        gate = p["gate_proj.weight"] @ fn
        up = p["up_proj.weight"] @ fn
        act = silu(gate) * up
        hidden = hidden + p["down_proj.weight"] @ act
        post, pinv = rms(hidden[None, :], p["post_norm.weight"], eps)
        return {
            "aux": aux_r,
            "an": an,
            "inv": inv,
            "cat": cat,
            "g": g,
            "cat2": cat2,
            "u": u,
            "un": un,
            "uinv": uinv,
            "v": v,
            "fn": fn,
            "finv": finv,
            "gate": gate,
            "up": up,
            "act": act,
            "hidden": hidden,
            "pinv": pinv,
            "post": post[0],
            "from_feedback": from_feedback,
        }

    def _trunk_backward(
        self, token: int, cache: dict[str, np.ndarray], dpost: np.ndarray
    ) -> np.ndarray:
        p = self.p
        h = self.hidden
        hidden = cache["hidden"]
        inv_post = float(np.ravel(cache["pinv"])[0])
        dhidden = self._drms(hidden, p["post_norm.weight"], inv_post, dpost)
        self.g["post_norm.weight"] += dpost * hidden * inv_post
        act = cache["act"]
        dact = p["down_proj.weight"].T @ dhidden
        self.g["down_proj.weight"] += np.outer(dhidden, act)
        gate = cache["gate"]
        up = cache["up"]
        sig = 1.0 / (1.0 + np.exp(-np.clip(gate, -40, 40)))
        dgate = dact * up * (sig + gate * sig * (1.0 - sig))
        dup = dact * silu(gate)
        fn = cache["fn"]
        dfn = p["gate_proj.weight"].T @ dgate + p["up_proj.weight"].T @ dup
        self.g["gate_proj.weight"] += np.outer(dgate, fn)
        self.g["up_proj.weight"] += np.outer(dup, fn)
        inv_ffn = float(np.ravel(cache["finv"])[0])
        dhidden = dhidden + self._drms(hidden, p["ffn_norm.weight"], inv_ffn, dfn)
        self.g["ffn_norm.weight"] += dfn * hidden * inv_ffn
        dproj = dhidden
        v = cache["v"]
        un = cache["un"]
        self.g["o_proj.weight"] += np.outer(dproj, v)
        dattn = p["o_proj.weight"].T @ dproj
        self.g["v_proj.weight"] += np.outer(dattn, un)
        dun = p["v_proj.weight"].T @ dattn
        inv_pre = float(np.ravel(cache["uinv"])[0])
        du = self._drms(cache["u"], p["pre_norm.weight"], inv_pre, dun)
        self.g["pre_norm.weight"] += dun * cache["u"] * inv_pre
        cat2 = cache["cat2"]
        self.g["eh_proj.weight"] += np.outer(du, cat2)
        dcat2 = p["eh_proj.weight"].T @ du
        self.g["embed_tokens.weight"][token] += dcat2[:h]
        dg = dhidden + dcat2[h:]
        if cache["from_feedback"]:
            return dg
        cat = cache["cat"]
        self.g["fc.weight"] += np.outer(dg, cat)
        dan = (p["fc.weight"].T @ dg).reshape(self.n_aux, self.target)
        aux = cache["aux"]
        inv = cache["inv"]
        for i in range(self.n_aux):
            self.g["fc_norm.weight"] += dan[i] * aux[i] * float(inv[i, 0])
        return dg

    @staticmethod
    def _drms(x: np.ndarray, weight: np.ndarray, inv: float, dy: np.ndarray) -> np.ndarray:
        n = x.shape[0]
        dot = float(np.dot(dy * weight, x) / n)
        return dy * weight * inv - x * (inv ** 3) * dot

    def _head_grads(
        self,
        posts: np.ndarray,
        labels: np.ndarray,
        rng: np.random.Generator | None,
        n_sample: int,
    ) -> tuple[float, float, np.ndarray]:
        n = posts.shape[0]
        if n_sample >= self.vocab:
            cols = np.arange(self.vocab, dtype=np.int64)
        else:
            if rng is None:
                rng = np.random.default_rng(0)
            neg = rng.integers(0, self.vocab, size=n_sample, dtype=np.int64)
            cols = np.unique(np.concatenate([neg, labels]))
        label_at = np.searchsorted(cols, labels)
        head = self.p["lm_head.weight"][cols]
        logits = posts @ head.T
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        probs = exp / exp.sum(axis=1, keepdims=True)
        picked = probs[np.arange(n), label_at]
        loss = float(-np.log(np.maximum(picked, 1e-12)).mean())
        acc = float(np.mean(np.argmax(logits, axis=1) == label_at))
        dlogits = probs
        dlogits[np.arange(n), label_at] -= 1.0
        dlogits /= n
        self.g["lm_head.weight"][cols] += dlogits.T @ posts
        dposts = dlogits @ head
        low = posts @ self.p["lr_down.weight"].T
        up = self.p["lr_up.weight"][cols]
        approx = low @ up.T
        dapprox = (0.05 / max(int(cols.shape[0]), 1)) * (approx - logits) / n
        self.g["lr_up.weight"][cols] += dapprox.T @ low
        dlow = dapprox @ up
        self.g["lr_down.weight"] += dlow.T @ posts
        dposts += dlow @ self.p["lr_down.weight"]
        return loss, acc, dposts

    def sequence_loss(
        self,
        tokens: np.ndarray,
        aux: np.ndarray,
        rng: np.random.Generator | None = None,
        n_sample: int = 4096,
    ) -> tuple[float, float]:
        """Two-step teacher-forced loss. Step 0 reads target features. Step 1
        feeds the draft's own post-norm state back, matching chained drafting.
        Large vocabs use a sampled softmax that always includes the label."""
        self.zero_grad()
        n = int(tokens.shape[0]) - 1
        if n < 1:
            return 0.0, 0.0
        caches = [self._trunk(int(tokens[t]), aux[t]) for t in range(n)]
        posts = np.stack([c["post"] for c in caches])
        loss1, acc, dposts = self._head_grads(
            posts, tokens[1:].astype(np.int64), rng, n_sample
        )
        n2 = n - 1
        if n2 > 0:
            chain = [
                self._trunk(int(tokens[t + 1]), aux[t], g_override=caches[t]["post"])
                for t in range(n2)
            ]
            posts2 = np.stack([c["post"] for c in chain])
            loss2, _, dposts2 = self._head_grads(
                posts2, tokens[2 : 2 + n2].astype(np.int64), rng, n_sample
            )
            for t in range(n2):
                dposts[t] = dposts[t] + self._trunk_backward(
                    int(tokens[t + 1]), chain[t], dposts2[t]
                )
            loss = 0.5 * (loss1 + loss2)
        else:
            loss = loss1
        for t in range(n):
            self._trunk_backward(int(tokens[t]), caches[t], dposts[t])
        return loss, acc


def save_model(model: Eagle4, layers: list[int], out: Path, vocab_note: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    tensors = {name: arr for name, arr in model.p.items()}
    write_safetensors(out / "model.safetensors", tensors)
    cfg = {
        "model_type": "oxidize-eagle4",
        "architectures": ["OxidizeEagle4"],
        "hidden_size": model.hidden,
        "target_hidden_size": model.target,
        "num_aux_hidden_states": model.n_aux,
        "extract_layers": layers,
        "vocab_size": model.vocab,
        "num_attention_heads": model.n_heads,
        "num_key_value_heads": model.n_heads,
        "head_dim": model.head_dim,
        "intermediate_size": model.intermediate,
        "sliding_window": model.window,
        "rms_norm_eps": model.eps,
        "rope_theta": 10000.0,
        "use_rope": False,
        "lm_rank": model.rank,
        "n_candidates": min(128, model.vocab),
        "torch_dtype": "float32",
        "aux_hidden": "qwen4exp_hc_residual",
        "note": vocab_note,
    }
    (out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    card = f"""---
license: other
license_name: qwen-community-license-1.0
base_model: freakyskittle/CYBER-FROST-3.8-GGUF
tags:
- eagle4
- speculative-decoding
- draft-model
---

# CYBER-FROST-3.8 EAGLE-4 draft

Speculative-decoding drafter for the CYBER-FROST-3.8 Qwen4-Exp trunk.
It is not a standalone chat model.

The draft reads the hyper-connection residual (`hc_mult * n_embd` = {model.target})
after layers {layers}. Hidden size {model.hidden}, one layer, sliding window
{model.window}, low-rank candidate head rank {model.rank}.

{vocab_note}

Weights are float32 and load in oxidize-c with `--eagle4` pointing at this directory.
"""
    (out / "README.md").write_text(card)


def write_safetensors(path: Path, tensors: dict[str, np.ndarray]) -> None:
    header: dict[str, dict[str, object]] = {}
    blobs: list[bytes] = []
    offset = 0
    for name, arr in tensors.items():
        raw = np.ascontiguousarray(arr, dtype="<f4").tobytes()
        pad = (8 - (len(raw) % 8)) % 8
        header[name] = {
            "dtype": "F32",
            "shape": list(arr.shape),
            "data_offsets": [offset, offset + len(raw)],
        }
        blobs.append(raw + b"\0" * pad)
        offset += len(raw) + pad
    payload = json.dumps(header, separators=(",", ":")).encode()
    pad = (8 - (len(payload) % 8)) % 8
    payload += b" " * pad
    with path.open("wb") as f:
        f.write(struct.pack("<Q", len(payload)))
        f.write(payload)
        for blob in blobs:
            f.write(blob)


def self_test() -> None:
    rng = np.random.default_rng(0)
    hidden, target, vocab = 8, 4, 6
    model = Eagle4(hidden, target, 1, vocab, 8, 1, 8, 2, 8, 1e-6, rng)
    tokens = rng.integers(0, vocab, size=12, dtype=np.uint32)
    aux = rng.normal(size=(12, 1, target)).astype(np.float32)
    # Make the label a function of the aux so the net can fit it.
    tokens[1:] = (np.abs(aux[:-1, 0, 0]) * 10).astype(np.uint32) % vocab
    before, _ = model.sequence_loss(tokens, aux)
    for _ in range(40):
        model.sequence_loss(tokens, aux)
        model.adam(3e-3)
    after, acc = model.sequence_loss(tokens, aux)
    print(f"self-test loss {before:.3f} -> {after:.3f} acc {acc:.2f}")
    if not after < before:
        raise SystemExit("self-test did not reduce loss")


def load_sequences(path: Path) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[int]]:
    files = sorted(path.glob("*.e4h1")) if path.is_dir() else [path]
    if not files:
        raise SystemExit(f"no E4H1 captures in {path}")
    seqs: list[tuple[np.ndarray, np.ndarray]] = []
    layers: list[int] | None = None
    for f in files:
        tokens, hiddens, lay = load_dump(f)
        if tokens.shape[0] < 4:
            continue
        if layers is None:
            layers = lay
        elif lay != layers:
            raise SystemExit(f"{f} layers {lay} != {layers}")
        seqs.append((tokens, hiddens))
    if not seqs or layers is None:
        raise SystemExit("no usable sequences")
    return seqs, layers


def train(dump: Path, out: Path, steps: int, lr: float, seed: int) -> None:
    import os

    seqs, layers = load_sequences(dump)
    n_tok = sum(int(t.shape[0]) for t, _ in seqs)
    n_aux = seqs[0][1].shape[1]
    target = seqs[0][1].shape[2]
    vocab = int(os.environ.get("EAGLE4_VOCAB", max(int(t.max()) for t, _ in seqs) + 1))
    hold_n = max(1, len(seqs) // 5) if len(seqs) >= 5 else 0
    train_seqs = seqs[:-hold_n] if hold_n else seqs
    hold_seqs = seqs[-hold_n:] if hold_n else []
    rng = np.random.default_rng(seed)
    model = Eagle4(
        hidden=256,
        target=target,
        n_aux=n_aux,
        vocab=vocab,
        intermediate=1024,
        n_heads=4,
        head_dim=64,
        rank=64,
        window=4096,
        eps=1e-6,
        rng=rng,
    )
    print(
        f"sequences {len(seqs)} train {len(train_seqs)} hold {len(hold_seqs)} "
        f"tokens {n_tok} hidden {target}",
        flush=True,
    )
    history: list[dict[str, float]] = []
    best_hold = 1e9
    best_state = {k: v.copy() for k, v in model.p.items()}
    stale = 0
    epochs = max(steps, 1)
    for epoch in range(epochs):
        order = rng.permutation(len(train_seqs))
        losses = []
        accs = []
        for i in order:
            tokens, hiddens = train_seqs[int(i)]
            loss, acc = model.sequence_loss(tokens, hiddens, rng)
            model.adam(lr)
            losses.append(loss)
            accs.append(acc)
        row = {
            "epoch": float(epoch),
            "loss": float(np.mean(losses)),
            "acc": float(np.mean(accs)),
        }
        if hold_seqs:
            hloss = []
            hacc = []
            for tokens, hiddens in hold_seqs:
                loss, acc = model.sequence_loss(tokens, hiddens, rng)
                hloss.append(loss)
                hacc.append(acc)
            model.zero_grad()
            row["hold_loss"] = float(np.mean(hloss))
            row["hold_acc"] = float(np.mean(hacc))
            if row["hold_loss"] + 1e-4 < best_hold:
                best_hold = row["hold_loss"]
                best_state = {k: v.copy() for k, v in model.p.items()}
                stale = 0
            else:
                stale += 1
        else:
            best_state = {k: v.copy() for k, v in model.p.items()}
        history.append(row)
        extra = ""
        if "hold_loss" in row:
            extra = f"  hold {row['hold_loss']:.4f} acc {row['hold_acc']:.3f}"
        print(
            f"epoch {epoch:3d}  loss {row['loss']:.4f}  acc {row['acc']:.3f}{extra}",
            flush=True,
        )
        if stale >= 3:
            print("early stop", flush=True)
            break
    model.p = best_state
    final = history[-1]
    note = (
        f"Trained {len(history)} epochs on {n_tok} Q4_K_M hidden states "
        f"across {len(seqs)} sequences ({len(hold_seqs)} held out). "
        f"Two-step chain loss, 4096-way sampled softmax. "
        f"Last train loss {final['loss']:.4f}, acc {final['acc']:.3f}."
    )
    if "hold_loss" in final:
        note += f" Held-out loss {final.get('hold_loss', 0):.4f}, acc {final.get('hold_acc', 0):.3f}."
    save_model(model, layers, out, note)
    (out / "train_log.json").write_text(json.dumps({"history": history, "tokens": n_tok}) + "\n")
    print(note)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--dump", type=Path)
    ap.add_argument("--out", type=Path, default=Path("models/eagle4-cyber-frost"))
    ap.add_argument("--steps", type=int, default=12, help="epochs over the capture set")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return
    if args.dump is None:
        raise SystemExit("pass --dump or --self-test")
    train(args.dump, args.out, args.steps, args.lr, args.seed)


if __name__ == "__main__":
    main()
