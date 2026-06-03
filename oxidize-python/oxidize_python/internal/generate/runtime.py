"""GGUF-backed generation runtime mirroring oxidize-golang/internal/generate/runtime.go."""

from __future__ import annotations

import time
from dataclasses import dataclass

from oxidize_python.core.model.generation import GenerationStream, default_generation_config
from oxidize_python.core.model.inference import InferenceModel
from oxidize_python.core.model.loader import LoaderConfig, load_gguf_model_from_path
from oxidize_python.core.model.model import Session, Token
from oxidize_python.core.tokenizer import from_gguf_metadata
from oxidize_python.core.tokenizer.bpe import BpeTokenizer
from oxidize_python.core.tokenizer.tokenizer import EncodeOptions, SpecialTokens
from oxidize_python.internal.gguf.parse import load_file


@dataclass
class RunConfig:
    model_path: str = ""
    prompt: str = ""
    max_new_tokens: int = 64
    temperature: float = 0.8
    top_p: float = 0.9
    stop_token: Token = 2


def default_run_config() -> RunConfig:
    return RunConfig()


def run_from_gguf(cfg: RunConfig, stdout: object) -> None:
    path = cfg.model_path.strip()
    prompt = cfg.prompt.strip()
    if not path:
        raise ValueError("generate: empty model path")
    if not prompt:
        return

    loaded = load_gguf_model_from_path(path, LoaderConfig())
    if not isinstance(loaded, InferenceModel):
        raise TypeError(f"generate: expected InferenceModel, got {type(loaded)}")
    if loaded.stack is None or not loaded.stack.loaded():
        raise ValueError(f"generate: model {path!r} has no loadable transformer weights")

    try:
        gguf = load_file(path)
        meta = {k: v.string for k, v in gguf.metadata.items() if v.string}
        tok = from_gguf_metadata(meta)
    except Exception:
        tok = BpeTokenizer([], [], SpecialTokens(bos=1, eos=2))

    prompt_tokens = tok.encode(prompt, EncodeOptions())
    if not prompt_tokens:
        prompt_tokens = [1]

    session = Session()
    gen_cfg = default_generation_config()
    if cfg.max_new_tokens > 0:
        gen_cfg.max_new_tokens = cfg.max_new_tokens
    if cfg.stop_token:
        gen_cfg.stop_token = cfg.stop_token
    gen_cfg.sampling.temperature = cfg.temperature
    gen_cfg.sampling.top_p = cfg.top_p

    stream = GenerationStream(loaded, session, gen_cfg)
    stream.seed(prompt_tokens)

    start = time.monotonic()
    for _ in range(gen_cfg.max_new_tokens):
        token, done, err = stream.next()
        if err is not None:
            raise err
        if done:
            break
        try:
            piece = tok.decode([token])
        except Exception:
            piece = f"<{token}>"
        stdout.write(piece)

    elapsed = time.monotonic() - start
    tokens = session.consumed_tokens()
    speed = tokens / elapsed if elapsed > 0 and tokens > 0 else 0.0
    stdout.write(f"\ngeneration stats: tokens={tokens} speed={speed:.2f} tok/s\n")
