from __future__ import annotations

from dataclasses import dataclass, field

from oxidize_python.core.tokenizer.tokenizer import EncodeOptions, SpecialTokens


@dataclass
class BpeTokenizer:
    vocab: dict[str, int] = field(default_factory=dict)
    reverse: dict[int, str] = field(default_factory=dict)
    merges: list[str] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)
    special: SpecialTokens = field(default_factory=SpecialTokens)

    def __post_init__(self) -> None:
        if not self.reverse and self.vocab:
            self.reverse = {v: k for k, v in self.vocab.items()}

    def name(self) -> str:
        return "bpe"

    def special_tokens(self) -> SpecialTokens:
        return self.special

    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode(self, text: str, opts: EncodeOptions) -> list[int]:
        if not text:
            return []
        out: list[int] = []
        for w in text.split():
            runes = ("▁" + w.lstrip()).encode()
            out.extend(self._bpe(runes))
        return out

    def _bpe(self, word: bytes) -> list[int]:
        if not word:
            return []
        parts = [chr(c) for c in word]
        while True:
            pairs = [(parts[i], parts[i + 1]) for i in range(len(parts) - 1)]
            best_idx = -1
            best_rank = -1
            for i, p in enumerate(pairs):
                merge = p[0] + p[1]
                rank, ok = self._score_for(merge)
                if ok and (best_idx < 0 or rank < best_rank):
                    best_idx, best_rank = i, rank
            if best_idx < 0:
                break
            merged = (
                parts[:best_idx] + [parts[best_idx] + parts[best_idx + 1]] + parts[best_idx + 2 :]
            )
            parts = merged
        return [self.vocab.get(p, self.special.unknown) for p in parts]

    def _score_for(self, merge: str) -> tuple[int, bool]:
        for i, m in enumerate(self.merges):
            if m == merge:
                return i, True
        if merge in self.scores:
            return int(self.scores[merge]), True
        return 0, False

    def decode(self, tokens: list[int]) -> str:
        sb = []
        for tid in tokens:
            if tid in (self.special.bos, self.special.eos, self.special.pad):
                continue
            sb.append(self.reverse.get(tid, ""))
        return "".join(sb).replace("▁", " ")

    def decode_skip_special(self, tokens: list[int]) -> str:
        return self.decode(tokens)
