"""Token streaming for server SSE mirroring oxidize-golang/internal/generate/stream.go."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from oxidize_python.core.model.generation import GenerationStream, default_generation_config
from oxidize_python.core.model.inference import InferenceModel
from oxidize_python.core.model.loader import LoaderConfig
from oxidize_python.core.model.model import Session, Token
from oxidize_python.core.tokenizer import from_gguf_metadata
from oxidize_python.core.tokenizer.bpe import BpeTokenizer
from oxidize_python.core.tokenizer.tokenizer import EncodeOptions, SpecialTokens
from oxidize_python.internal.generate import PlaceholderSpec, placeholder_text
from oxidize_python.internal.generate.cache import inference_from_cache
from oxidize_python.internal.gguf.parse import load_file


@dataclass
class CompletionParams:
    max_tokens: int = 64
    temperature: float = 0.8
    top_p: float = 0.9
    top_k: int = 0
    loader: LoaderConfig | None = None


TokenHandler = Callable[[str], None]


def _load_tokenizer(model_path: str):
    try:
        gguf = load_file(model_path)
        meta = {k: v.string for k, v in gguf.metadata.items() if v.string}
        return from_gguf_metadata(meta)
    except Exception:
        return BpeTokenizer([], [], SpecialTokens(bos=1, eos=2))


def stream_completion(
    model_path: str,
    prompt: str,
    params: CompletionParams,
    on_token: TokenHandler | None = None,
) -> str:
    prompt = prompt.strip()
    if not prompt:
        return ""
    if params.max_tokens <= 0:
        params.max_tokens = 64
    loader = params.loader or LoaderConfig()

    try:
        inference, _entry = inference_from_cache(model_path, loader)
    except Exception:
        text = placeholder_text(PlaceholderSpec()) or ""
        if on_token and text:
            on_token(text)
        return text

    if (
        not isinstance(inference, InferenceModel)
        or inference.stack is None
        or not inference.stack.loaded()
    ):
        text = placeholder_text(PlaceholderSpec()) or ""
        if on_token and text:
            on_token(text)
        return text

    tok = _load_tokenizer(model_path)
    prompt_tokens = tok.encode(prompt, EncodeOptions()) or [Token(1)]

    session = Session()
    gen_cfg = default_generation_config()
    gen_cfg.max_new_tokens = params.max_tokens
    gen_cfg.sampling.temperature = params.temperature
    gen_cfg.sampling.top_p = params.top_p
    if params.top_k > 0:
        gen_cfg.sampling.top_k = params.top_k
    if params.temperature <= 0:
        gen_cfg.sampling.temperature = 0.0

    stream = GenerationStream(inference, session, gen_cfg)
    stream.seed(prompt_tokens)

    parts: list[str] = []
    for _ in range(params.max_tokens):
        token, done, err = stream.next()
        if err is not None:
            break
        if done:
            break
        try:
            piece = tok.decode([token])
        except Exception:
            piece = f"<{token}>"
        parts.append(piece)
        if on_token:
            on_token(piece)
    return "".join(parts)
