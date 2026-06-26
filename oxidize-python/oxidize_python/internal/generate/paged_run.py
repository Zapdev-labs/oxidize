"""PagedAttention generation path mirroring oxidize-golang/internal/generate/paged_run.go."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from oxidize_python.internal.generate.runtime import RunConfig

from oxidize_python.core.model.loader import LoaderConfig, load_gguf_model_from_path
from oxidize_python.core.model.model import Model, Session, Token
from oxidize_python.core.model.sampling import greedy
from oxidize_python.core.paged.paged import (
    Scheduler,
    SchedulerV2,
    SchedulerV2Config,
    SequenceStatus,
    default_scheduler_config,
    default_scheduler_v2_config,
)
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


class PagedGenerateRuntimeV2:
    """Budgeted three-phase paged runtime mirroring
    oxidize-golang/internal/server/paged_runtime.go::PagedRuntimeV2.

    It enforces a token budget per step, supports prefill chunking and prefix
    caching, and builds an InputBatch each step so multiple sequences can be
    processed together. The model forward is still driven per-sequence (one
    session each) because the pure-Python model backend does not yet expose a
    fused multi-sequence kernel, but scheduling, batching metadata, and block
    management mirror the Rust scheduler.
    """

    def __init__(
        self,
        model_path: str,
        config: SchedulerV2Config | None = None,
        total_blocks: int = 1024,
        block_size: int = 16,
    ) -> None:
        loaded = load_gguf_model_from_path(model_path, LoaderConfig())
        self.mdl: Model = loaded
        self.sched = SchedulerV2(
            config or default_scheduler_v2_config(), total_blocks, block_size
        )
        self.sessions: dict[int, Session] = {}
        self.model_prefilled: dict[int, int] = {}

    def enqueue(
        self,
        prompt_tokens: list[int],
        max_new: int,
        stop_token: int = 2,
        has_stop: bool = True,
    ) -> int:
        seq_id = self.sched.add_request(prompt_tokens, max_new, stop_token, has_stop)
        self.sessions[seq_id] = Session()
        self.model_prefilled[seq_id] = 0
        return seq_id

    def step(self) -> dict[int, Token]:
        res = self.sched.step()
        if not res.scheduled_seq_ids:
            return {}
        batch = self.sched.build_input_batch(res)

        sampled: dict[int, int] = {}
        out: dict[int, Token] = {}
        for i, seq_id in enumerate(batch.seq_ids):
            sess = self.sessions.get(seq_id)
            if sess is None:
                sess = Session()
                self.sessions[seq_id] = sess
            toks = [Token(t) for t in batch.token_ids[i]]
            if not toks:
                continue
            logits = self.mdl.forward(toks, sess)
            # Only sequences finishing their prefill (or decoding) produce a
            # sampled token this step. For a prefill chunk that does not yet
            # reach the end of the prompt, we keep accumulating context and skip
            # sampling so we do not emit mid-prompt tokens.
            if batch.is_prefill[i]:
                seq = self.sched.get_sequence(seq_id)
                if seq is None or seq.remaining_prefill_tokens() > 0:
                    continue
            next_tok = greedy(logits)
            sampled[seq_id] = int(next_tok)
            out[seq_id] = next_tok

        self.sched.postprocess_step(sampled)
        # Reap fully finished sequences' sessions.
        for seq_id in list(out.keys()):
            seq = self.sched.get_sequence(seq_id)
            if seq is not None and seq.status == SequenceStatus.FINISHED:
                self.sessions.pop(seq_id, None)
                self.model_prefilled.pop(seq_id, None)
        return out

    def stats(self) -> tuple[int, int, int]:
        return (
            self.sched.waiting_count(),
            self.sched.running_count(),
            self.sched.pool.free_count(),
        )


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
