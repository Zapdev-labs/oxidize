from __future__ import annotations

from dataclasses import dataclass, field

from oxidize_python.core.tokenizer.tokenizer import EncodeOptions, SpecialTokens, _decode_rune


@dataclass
class SentencePieceUnigramTokenizer:
    pieces: list[str] = field(default_factory=list)
    reverse: dict[int, str] = field(default_factory=dict)
    scores: list[float] = field(default_factory=list)
    special: SpecialTokens = field(default_factory=SpecialTokens)

    def __post_init__(self) -> None:
        if not self.reverse and self.pieces:
            self.reverse = {i: p for i, p in enumerate(self.pieces)}
        if not self.scores:
            self.scores = [0.0] * len(self.pieces)

    def name(self) -> str:
        return "sentencepiece"

    def special_tokens(self) -> SpecialTokens:
        return self.special

    def vocab_size(self) -> int:
        return len(self.pieces)

    def encode(self, text: str, opts: EncodeOptions) -> list[int]:
        text = text.replace(" ", "▁")
        if not text:
            return []
        out: list[int] = []
        rest = text
        while rest:
            best = self.special.unknown
            best_score = -1e30
            best_len = 0
            for i, p in enumerate(self.pieces):
                if len(p) > len(rest) or rest[: len(p)] != p:
                    continue
                if len(p) > best_len or (len(p) == best_len and self.scores[i] > best_score):
                    best_len, best_score, best = len(p), self.scores[i], i
            if best == self.special.unknown and best_len == 0:
                _, size = _decode_rune(rest.encode())
                if size <= 0:
                    size = 1
                out.append(self.special.unknown)
                rest = rest[size:]
                continue
            out.append(best)
            rest = rest[best_len:]
        return out

    def decode(self, tokens: list[int]) -> str:
        sb = []
        for tid in tokens:
            if tid in (self.special.bos, self.special.eos, self.special.pad):
                continue
            sb.append(self.reverse.get(tid, ""))
        return "".join(sb).replace("▁", " ")

    def decode_skip_special(self, tokens: list[int]) -> str:
        return self.decode(tokens)
