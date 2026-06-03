from __future__ import annotations

import re
from dataclasses import dataclass, field

from oxidize_python.core.tokenizer.tokenizer import EncodeOptions, SpecialTokens


@dataclass
class TiktokenTokenizer:
    pattern: str = ""
    ranks: dict[str, float] = field(default_factory=dict)
    reverse: dict[int, str] = field(default_factory=dict)
    special: SpecialTokens = field(default_factory=SpecialTokens)
    _compiled: re.Pattern[str] | None = None

    def __post_init__(self) -> None:
        if not self.pattern:
            self.pattern = r"'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+"
        self._compiled = re.compile(self.pattern)

    def name(self) -> str:
        return "tiktoken"

    def special_tokens(self) -> SpecialTokens:
        return self.special

    def vocab_size(self) -> int:
        return len(self.ranks)

    def encode(self, text: str, opts: EncodeOptions) -> list[int]:
        if not text:
            return []
        assert self._compiled is not None
        out: list[int] = []
        for chunk in self._compiled.findall(text):
            out.extend(self._bpe(chunk.encode()))
        return out

    def _bpe(self, word: bytes) -> list[int]:
        if len(word) == 1:
            return [self._lookup(word.decode())]
        parts = [word.decode()]
        while True:
            pairs = [(parts[i], parts[i + 1]) for i in range(len(parts) - 1)]
            best_idx = -1
            best_rank = 1e30
            for i, p in enumerate(pairs):
                key = p[0] + p[1]
                if key in self.ranks:
                    rank = self.ranks[key]
                    if best_idx < 0 or rank < best_rank:
                        best_idx, best_rank = i, rank
            if best_idx < 0:
                break
            parts = (
                parts[:best_idx]
                + [parts[best_idx] + parts[best_idx + 1]]
                + parts[best_idx + 2 :]
            )
        return [self._lookup(p) for p in parts]

    def _lookup(self, p: str) -> int:
        return int(self.ranks.get(p, 0))

    def decode(self, tokens: list[int]) -> str:
        sb = []
        for tid in tokens:
            if tid in (self.special.bos, self.special.eos, self.special.pad):
                continue
            piece = self.reverse.get(tid)
            if piece:
                sb.append(piece)
            else:
                sb.append(chr(tid))
        return "".join(sb)

    def decode_skip_special(self, tokens: list[int]) -> str:
        return self.decode(tokens)
