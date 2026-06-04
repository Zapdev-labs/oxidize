"""GGUF-backed generation runtime mirroring oxidize-golang/internal/generate/runtime.go."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from io import StringIO

from oxidize_python.core.backends.factory import BackendConfig, create_compute_backend
from oxidize_python.core.model.dflash import DFlashConfig
from oxidize_python.core.model.dflash_heuristic import HeuristicDFlashDraft
from oxidize_python.core.model.generation import (
    GenerationStream,
    SpeculativeGenerationStream,
    default_generation_config,
    default_speculative_generation_config,
)
from oxidize_python.core.model.inference import InferenceModel
from oxidize_python.core.model.loader import LoaderConfig, load_gguf_model_from_path
from oxidize_python.core.model.model import Model, Session, Token
from oxidize_python.core.model.speculative import (
    SpeculativeConfig,
    SpeculativeDecoder,
    load_draft_model_for_speculative,
)
from oxidize_python.core.tokenizer import from_gguf_metadata
from oxidize_python.core.tokenizer.bpe import BpeTokenizer
from oxidize_python.core.tokenizer.tokenizer import EncodeOptions, SpecialTokens
from oxidize_python.core.vision.vision import Modality, StubPreprocessor, default_config
from oxidize_python.internal.generate.cache import inference_from_cache
from oxidize_python.internal.generate.paged_run import run_paged_from_gguf
from oxidize_python.internal.gguf.parse import load_file


@dataclass
class RunConfig:
    model_path: str = ""
    prompt: str = ""
    max_new_tokens: int = 64
    temperature: float = 0.8
    top_p: float = 0.9
    top_k: int = 0
    stop_token: Token = 2
    draft_model_path: str = ""
    draft_tokens_per_step: int = 4
    loader: LoaderConfig = field(default_factory=LoaderConfig)
    use_paged: bool = False
    use_dflash_fusion: bool = False
    vision: bool = False
    image_path: str = ""
    stop_token: Token = 2


def default_run_config() -> RunConfig:
    return RunConfig()


def _load_tokenizer(path: str):
    try:
        gguf = load_file(path)
        meta = {k: v.string for k, v in gguf.metadata.items() if v.string}
        return from_gguf_metadata(meta)
    except Exception:
        return BpeTokenizer([], [], SpecialTokens(bos=1, eos=2))


def _apply_loader_hints(model: Model, cfg: LoaderConfig) -> None:
    if not isinstance(model, InferenceModel):
        return
    if cfg.ctx_size > 0:
        model.config.context_size = cfg.ctx_size
    if cfg.threads > 0:
        pass
    be_cfg = BackendConfig(
        name=cfg.backend,
        n_gpu_layers=cfg.n_gpu_layers,
        gpus=cfg.gpus,
    )
    _, effective, warning = create_compute_backend(be_cfg)
    if warning:
        print(warning)


def _generation_stream(
    target: Model,
    cfg: RunConfig,
    session: Session,
) -> GenerationStream | SpeculativeGenerationStream:
    gen_cfg = default_generation_config()
    if cfg.max_new_tokens > 0:
        gen_cfg.max_new_tokens = cfg.max_new_tokens
    if cfg.stop_token:
        gen_cfg.stop_token = cfg.stop_token
    gen_cfg.sampling.temperature = cfg.temperature
    gen_cfg.sampling.top_p = cfg.top_p
    if cfg.top_k > 0:
        gen_cfg.sampling.top_k = cfg.top_k

    draft_path = cfg.draft_model_path.strip() or cfg.loader.draft_model.strip()
    draft_tokens = cfg.draft_tokens_per_step or cfg.loader.draft_tokens
    if draft_path:
        draft_loaded = load_gguf_model_from_path(draft_path, cfg.loader)
        spec_cfg = default_speculative_generation_config()
        spec_cfg.generation = gen_cfg
        spec_cfg.draft_tokens_per_step = max(1, draft_tokens)
        return SpeculativeGenerationStream(draft_loaded, target, session, spec_cfg)
    if cfg.loader.draft_model:
        draft = load_draft_model_for_speculative(target)
        spec_cfg = default_speculative_generation_config()
        spec_cfg.generation = gen_cfg
        spec_cfg.draft_tokens_per_step = max(1, draft_tokens)
        return SpeculativeGenerationStream(draft, target, session, spec_cfg)
    return GenerationStream(target, session, gen_cfg)


def _emit_token(tok, token: Token, stdout: object) -> None:
    try:
        piece = tok.decode([token])
    except Exception:
        piece = f"<{token}>"
    stdout.write(piece)


def run_from_gguf(cfg: RunConfig, stdout: object) -> None:
    path = cfg.model_path.strip()
    prompt = cfg.prompt.strip()
    if not path:
        raise ValueError("generate: empty model path")
    if not prompt:
        return

    if cfg.use_paged:
        run_paged_from_gguf(cfg, stdout)
        return

    if cfg.vision and cfg.image_path.strip():
        try:
            raw = _read_image_bytes(cfg.image_path.strip())
            pre = StubPreprocessor(default_config())
            enc = pre.process(raw, Modality.IMAGE)
            stdout.write(f"# vision: preprocessed image ({enc!r})\n")
        except OSError:
            pass

    inference, entry = inference_from_cache(path, cfg.loader)
    _apply_loader_hints(inference, cfg.loader)
    if entry.warning:
        stdout.write(f"# backend: {entry.backend} ({entry.warning})\n")
    if inference.stack is None or not inference.stack.loaded():
        raise ValueError(f"generate: model {path!r} has no loadable transformer weights")

    tok = _load_tokenizer(path)
    prompt_tokens = tok.encode(prompt, EncodeOptions()) or [Token(1)]

    session = Session()
    start = time.monotonic()
    draft_path = cfg.draft_model_path.strip() or cfg.loader.draft_model.strip()

    if draft_path or cfg.use_dflash_fusion:
        draft: Model
        if draft_path:
            draft = load_gguf_model_from_path(draft_path, cfg.loader)
        else:
            draft = HeuristicDFlashDraft(inference, DFlashConfig())
        if cfg.use_dflash_fusion:
            dec = SpeculativeDecoder(
                draft,
                inference,
                session,
                SpeculativeConfig(
                    draft_tokens_per_step=max(1, cfg.draft_tokens_per_step),
                    max_new_tokens=cfg.max_new_tokens,
                ),
            )
            inference.forward(prompt_tokens, session)
            for _ in range(cfg.max_new_tokens):
                accepted = dec.step()
                if not accepted:
                    break
                for token in accepted:
                    _emit_token(tok, token, stdout)
            elapsed = time.monotonic() - start
            tokens = session.consumed_tokens()
            speed = tokens / elapsed if elapsed > 0 and tokens > 0 else 0.0
            stdout.write(f"\ngeneration stats: tokens={tokens} speed={speed:.2f} tok/s (dflash)\n")
            return

        stream = _generation_stream(inference, cfg, session)
        stream.seed(prompt_tokens)
        for _ in range(cfg.max_new_tokens):
            token, done, err = stream.next()
            if err is not None:
                raise err
            if done:
                break
            _emit_token(tok, token, stdout)
    else:
        stream = _generation_stream(inference, cfg, session)
        stream.seed(prompt_tokens)
        for _ in range(cfg.max_new_tokens):
            token, done, err = stream.next()
            if err is not None:
                raise err
            if done:
                break
            _emit_token(tok, token, stdout)

    elapsed = time.monotonic() - start
    tokens = session.consumed_tokens()
    speed = tokens / elapsed if elapsed > 0 and tokens > 0 else 0.0
    stdout.write(f"\ngeneration stats: tokens={tokens} speed={speed:.2f} tok/s\n")


def _read_image_bytes(image_path: str) -> bytes:
    with open(image_path, "rb") as f:
        return f.read()


def completion_text(
    model_path: str,
    prompt: str,
    max_tokens: int = 64,
    *,
    temperature: float | None = None,
    top_p: float | None = None,
) -> str:
    prompt = prompt.strip()
    if not prompt:
        return ""
    if max_tokens <= 0:
        max_tokens = 64
    buf = StringIO()
    cfg = RunConfig(
        model_path=model_path,
        prompt=prompt,
        max_new_tokens=max_tokens,
        temperature=0.0 if temperature is None else temperature,
        top_p=0.9 if top_p is None else top_p,
    )
    run_from_gguf(cfg, buf)
    text = buf.getvalue()
    if "\ngeneration stats:" in text:
        text = text.split("\ngeneration stats:")[0]
    return text.strip()
