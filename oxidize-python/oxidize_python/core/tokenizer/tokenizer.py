"""Tokenizer core mirroring oxidize-golang/core/tokenizer/tokenizer.go."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from oxidize_python.internal.gguf.parse import load_file


class Error(Exception):
    def __init__(self, message: str, kind: str = "", token: int = 0) -> None:
        if kind == "unknown_token":
            super().__init__(f"tokenizer: unknown token {token}")
        else:
            super().__init__(f"tokenizer: {message}")


class LoadError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(f"tokenizer load: {message}")


@dataclass
class ChatMessage:
    role: str
    content: str


@dataclass
class EncodeOptions:
    add_bos: bool = False
    add_eos: bool = False
    pad_to: int = 0


@dataclass
class SpecialTokens:
    unknown: int = 0
    bos: int = 1
    eos: int = 2
    pad: int = 0
    separator: int = 0
    cls: int = 0
    mask: int | None = None


class Tokenizer(Protocol):
    def name(self) -> str: ...
    def encode(self, text: str, opts: EncodeOptions) -> list[int]: ...
    def decode(self, tokens: list[int]) -> str: ...
    def decode_skip_special(self, tokens: list[int]) -> str: ...
    def special_tokens(self) -> SpecialTokens: ...
    def vocab_size(self) -> int: ...


class StreamingDetokenizer:
    def __init__(self, tok: Tokenizer) -> None:
        self._tok = tok
        self._buffer = bytearray()

    def push(self, token: int) -> str:
        chunk = self._tok.decode([token])
        self._buffer.extend(chunk.encode())
        out = bytearray()
        i = 0
        while i < len(self._buffer):
            r, size = _decode_rune(self._buffer[i:])
            if r == "\ufffd" and size == 1:
                break
            out.extend(self._buffer[i : i + size])
            i += size
        self._buffer = self._buffer[i:]
        return out.decode()

    def flush(self) -> str:
        s = self._buffer.decode(errors="replace")
        self._buffer.clear()
        return s


def _decode_rune(b: bytes) -> tuple[str, int]:
    if not b:
        return "\ufffd", 0
    if b[0] < 0x80:
        return chr(b[0]), 1
    if b[0] < 0xC0:
        return "\ufffd", 1
    if b[0] < 0xE0:
        if len(b) < 2:
            return "\ufffd", 1
        return chr((b[0] & 0x1F) << 6 | (b[1] & 0x3F)), 2
    if b[0] < 0xF0:
        if len(b) < 3:
            return "\ufffd", 1
        return chr((b[0] & 0x0F) << 12 | (b[1] & 0x3F) << 6 | (b[2] & 0x3F)), 3
    if len(b) < 4:
        return "\ufffd", 1
    return chr((b[0] & 0x07) << 18 | (b[1] & 0x3F) << 12 | (b[2] & 0x3F) << 6 | (b[3] & 0x3F)), 4


def load_from_gguf_file(path: str | Path) -> Tokenizer:
    gguf = load_file(path)
    meta = {k: v.string for k, v in gguf.metadata.items() if v.string}
    return from_gguf_metadata(meta)


def from_gguf_metadata(metadata: dict[str, str]) -> Tokenizer:
    from oxidize_python.core.tokenizer.bpe import BpeTokenizer
    from oxidize_python.core.tokenizer.sp import SentencePieceUnigramTokenizer
    from oxidize_python.core.tokenizer.tiktoken import TiktokenTokenizer
    from oxidize_python.core.tokenizer.wordpiece import WordPieceTokenizer

    model = metadata.get("tokenizer.ggml.model", "").strip().lower()
    match model:
        case "llama" | "gpt2" | "bpe":
            return BpeTokenizer(
                vocab=_build_vocab(metadata),
                merges=[],
                special=SpecialTokens(
                    unknown=_uint32_from(metadata, "tokenizer.ggml.unknown_token_id", 0),
                    bos=_uint32_from(metadata, "tokenizer.ggml.bos_token_id", 1),
                    eos=_uint32_from(metadata, "tokenizer.ggml.eos_token_id", 2),
                ),
            )
        case "llama-spm" | "sentencepiece" | "unigram":
            return SentencePieceUnigramTokenizer(
                pieces=_default_sentence_pieces(),
                scores=_default_scores(),
                special=SpecialTokens(
                    unknown=_uint32_from(metadata, "tokenizer.ggml.unknown_token_id", 0),
                    bos=_uint32_from(metadata, "tokenizer.ggml.bos_token_id", 1),
                    eos=_uint32_from(metadata, "tokenizer.ggml.eos_token_id", 2),
                ),
            )
        case "wordpiece" | "bert":
            return WordPieceTokenizer(
                vocab=_build_vocab(metadata),
                special=SpecialTokens(
                    unknown=_uint32_from(metadata, "tokenizer.ggml.unknown_token_id", 100),
                    cls=_uint32_from(metadata, "tokenizer.ggml.cls_token_id", 101),
                    separator=_uint32_from(metadata, "tokenizer.ggml.separator_token_id", 102),
                    pad=_uint32_from(metadata, "tokenizer.ggml.padding_token_id", 0),
                    mask=_uint32_from(metadata, "tokenizer.ggml.mask_token_id", 103),
                ),
            )
        case "tiktoken" | "cl100k_base" | "o200k_base":
            return TiktokenTokenizer(
                pattern="",
                ranks=_default_tiktoken_ranks(),
                special=SpecialTokens(
                    bos=_uint32_from(metadata, "tokenizer.ggml.bos_token_id", 1),
                    eos=_uint32_from(metadata, "tokenizer.ggml.eos_token_id", 2),
                ),
            )
        case _:
            raise LoadError(f"unsupported tokenizer model: {model}")


def _build_vocab(metadata: dict[str, str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for k, v in metadata.items():
        if k.startswith("tokenizer.ggml.tokens."):
            idx = k.removeprefix("tokenizer.ggml.tokens.")
            out[v] = _parse_uint(idx)
    return out


def _default_sentence_pieces() -> list[str]:
    return ["<unk>", "<s>", "</s>", "▁", "▁the", "▁a", "▁an", "▁of", "▁to", "▁is", "▁and"]


def _default_scores() -> list[float]:
    return [0.0, 0.0, 0.0, -1.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0]


def _default_tiktoken_ranks() -> dict[str, float]:
    return {}


def _uint32_from(m: dict[str, str], key: str, default: int) -> int:
    return _parse_uint(m[key]) if key in m else default


def _parse_uint(s: str) -> int:
    n = 0
    for c in s:
        if not c.isdigit():
            return 0
        n = n * 10 + (ord(c) - 48)
    return n


def encode_with_special_tokens(tok: Tokenizer, text: str, opts: EncodeOptions) -> list[int]:
    tokens = tok.encode(text, opts)
    if opts.add_bos:
        tokens = [tok.special_tokens().bos] + tokens
    if opts.add_eos:
        tokens = tokens + [tok.special_tokens().eos]
    if opts.pad_to > 0 and len(tokens) < opts.pad_to:
        pad = tok.special_tokens().pad
        while len(tokens) < opts.pad_to:
            tokens.append(pad)
    return tokens


def process_chat_template(
    template: str, messages: list[ChatMessage], add_generation_prompt: bool
) -> str:
    out = _expand_for_loops(template, messages)
    out = _expand_conditionals(out, add_generation_prompt)
    return _substitute_vars(out, messages, add_generation_prompt)


def _expand_for_loops(template: str, messages: list[ChatMessage]) -> str:
    while "{% for " in template:
        start = template.index("{% for ")
        end_loop = template.index("{% endfor %}", start)
        header_end = start + len("{% for ")
        header = template[header_end : template.index("%}", start)]
        parts = header.split(" in ", 1)
        if len(parts) != 2:
            return template[:start] + template[start:]
        var_name = parts[0].strip()
        body_start = template.index("%}", start) + 2
        body = template[body_start:end_loop]
        rendered = ""
        for m in messages:
            r = body
            for src, dst in (
                (f'{{{{ {var_name}["role"] }}}}', m.role),
                (f'{{{{ {var_name}["content"] }}}}', m.content),
                (f"{{{{ {var_name}['role'] }}}}", m.role),
                (f"{{{{ {var_name}['content'] }}}}", m.content),
                (f"{{{{ {var_name}.role }}}}", m.role),
                (f"{{{{ {var_name}.content }}}}", m.content),
            ):
                r = r.replace(src, dst)
            rendered += r
        template = template[:start] + rendered + template[end_loop + len("{% endfor %}") :]
    return template


def _expand_conditionals(template: str, add_gen: bool) -> str:
    while "{% if " in template:
        start = template.index("{% if ")
        end_loop = template.index("{% endif %}", start)
        cond = template[start + len("{% if ") : template.index("%}", start)].strip()
        body_start = template.index("%}", start) + 2
        body = template[body_start:end_loop]
        keep = add_gen if cond == "add_generation_prompt" else True
        head, tail = template[:start], template[end_loop + len("{% endif %}") :]
        template = head + (body if keep else "") + tail
    return template


def _substitute_vars(template: str, messages: list[ChatMessage], add_gen: bool) -> str:
    while "{{" in template:
        start = template.index("{{")
        end = template.index("}}", start)
        expr = template[start + 2 : end].strip()
        val = _resolve_var(expr, messages, add_gen)
        template = template[:start] + val + template[end + 2 :]
    return template


def _resolve_var(expr: str, messages: list[ChatMessage], add_gen: bool) -> str:
    if expr.startswith("messages["):
        idx_end = expr.index("]")
        idx = int(expr[len("messages[") : idx_end])
        if 0 <= idx < len(messages):
            rest = expr[idx_end + 1 :].strip("[]'\"")
            if rest == "role":
                return messages[idx].role
            if rest == "content":
                return messages[idx].content
    if expr == "add_generation_prompt":
        return "true" if add_gen else "false"
    if expr == "bos_token":
        return "<s>"
    if expr == "eos_token":
        return "</s>"
    return ""


def process_chat_template_from_gguf_metadata(
    metadata: dict[str, str], messages: list[ChatMessage], add_gen: bool
) -> str:
    template = metadata.get("tokenizer.chat_template")
    if not template:
        template = "{{ bos_token }}" + _render_messages(messages) + "{{ eos_token }}"
        if add_gen:
            template += "<|assistant|>"
    return process_chat_template(template, messages, add_gen)


def _render_messages(messages: list[ChatMessage]) -> str:
    return "".join(f"{m.role}: {m.content}\n" for m in messages)
