"""PagedAttention generation path mirroring oxidize-golang/internal/generate/paged_run.go."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from oxidize_python.internal.generate.runtime import RunConfig

from oxidize_python.core.model.loader import LoaderConfig, load_gguf_model_from_path
from oxidize_python.core.model.model import Model, Session, Token
from oxidize_python.core.model.sampling import greedy
from oxidize_python.core.paged.paged import Scheduler, default_scheduler_config
from oxidize_python.core.tokenizer import from_gguf_metadata
from oxidize_python.core.tokenizer.bpe import BpeTokenizer
from oxidize_python.core.tokenizer.tokenizer import EncodeOptions, SpecialTokens
from oxidize_python.internal.gguf.parse import load_file


def _load_tokenizer(path: str):
    try:
        gguf = load_file(path)
        meta = {k: v.string for k, v in gguf.metadata.items() if v.string}
        return from_gguf_metadata(meta)
    except Exception:
        return BpeTokenizer([], [], SpecialTokens(bos=1, eos=2))


class _Stdout(Protocol):
    def write(self, s: str, /) -> object: ...


class PagedGenerateRuntime:
    def __init__(self, model_path: str) -> None:
        loaded = load_gguf_model_from_path(model_path, LoaderConfig())
        self.sched = Scheduler(default_scheduler_config())
        self.mdl: Model = loaded
        self.sessions: dict[int, Session] = {}
        self.prompts: dict[int, list[int]] = {}
        self.max_tok: dict[int, int] = {}
        self.generated: dict[int, int] = {}

    def enqueue(self, prompt_tokens: list[int], max_new: int) -> int:
        req = self.sched.add_request(prompt_tokens, max_new)
        rid = req.id
        self.prompts[rid] = list(prompt_tokens)
        self.max_tok[rid] = max_new
        self.generated[rid] = 0
        self.sessions[rid] = Session()
        return rid

    def step(self) -> dict[int, Token]:
        active = self.sched.step()
        if not active:
            return {}
        out: dict[int, Token] = {}
        for req in active:
            rid = req.id
            sess = self.sessions[rid]
            tokens = self.prompts[rid]
            if self.generated[rid] == 0 and len(tokens) > 1:
                batch = [Token(t) for t in tokens[:-1]]
                self.mdl.forward(batch, sess)
            last = tokens[-1]
            logits = self.mdl.forward([Token(last)], sess)
            next_tok = greedy(logits)
            out[rid] = next_tok
            self.prompts[rid] = [*tokens, int(next_tok)]
            self.generated[rid] += 1
            if self.generated[rid] >= self.max_tok[rid] or next_tok == 2:
                self.sched.finish(rid)
                self.sessions.pop(rid, None)
                self.prompts.pop(rid, None)
                self.max_tok.pop(rid, None)
                self.generated.pop(rid, None)
        return out


def run_paged_from_gguf(cfg: "RunConfig", stdout: _Stdout) -> None:
    path = cfg.model_path.strip()
    prompt = cfg.prompt.strip()
    if not path:
        raise ValueError("generate: empty model path")
    if not prompt:
        return

    tok = _load_tokenizer(path)
    prompt_tokens = tok.encode(prompt, EncodeOptions()) or [Token(1)]
    int_prompt = [int(t) for t in prompt_tokens]

    runtime = PagedGenerateRuntime(path)
    req_id = runtime.enqueue(int_prompt, cfg.max_new_tokens)

    start = time.monotonic()
    generated = 0
    stop = cfg.stop_token or Token(2)
    while generated < cfg.max_new_tokens:
        tokens = runtime.step()
        tok_out = tokens.get(req_id)
        if tok_out is None:
            if not tokens:
                break
            continue
        try:
            piece = tok.decode([tok_out])
        except Exception:
            piece = f"<{tok_out}>"
        stdout.write(piece)
        generated += 1
        if tok_out == stop or tok_out == 2:
            break

    elapsed = time.monotonic() - start
    if elapsed > 0 and generated > 0:
        speed = generated / elapsed
        stdout.write(f"\ngeneration stats: tokens={generated} speed={speed:.2f} tok/s (paged)\n")
