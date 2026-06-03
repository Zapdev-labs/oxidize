from __future__ import annotations

from dataclasses import dataclass, field

from oxidize_python.core.tokenizer.tokenizer import EncodeOptions, SpecialTokens


@dataclass
class WordPieceTokenizer:
    vocab: dict[str, int] = field(default_factory=dict)
    reverse: dict[int, str] = field(default_factory=dict)
    special: SpecialTokens = field(default_factory=SpecialTokens)
    unk_token: str = "[UNK]"
    max_input_chars: int = 100

    def __post_init__(self) -> None:
        if not self.reverse and self.vocab:
            self.reverse = {v: k for k, v in self.vocab.items()}
        if self.unk_token in self.vocab:
            self.special.unknown = self.vocab[self.unk_token]
        elif self.vocab:
            self.unk_token = next(iter(self.vocab))

    def name(self) -> str:
        return "wordpiece"

    def special_tokens(self) -> SpecialTokens:
        return self.special

    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode(self, text: str, opts: EncodeOptions) -> list[int]:
        if not text:
            return []
        out: list[int] = []
        for word in _tokenize_basic(text):
            out.extend(self._tokenize_word(word))
        return out

    def _tokenize_word(self, word: str) -> list[int]:
        if len(word) > self.max_input_chars:
            return [self.special.unknown]
        pieces: list[str] = []
        start = 0
        while start < len(word):
            end = len(word)
            cur = ""
            ok = False
            while end > start:
                sub = word[start:end]
                if start > 0:
                    sub = "##" + sub
                if sub in self.vocab:
                    cur, ok = sub, True
                    break
                end -= 1
            if not ok:
                return [self.special.unknown]
            pieces.append(cur)
            start = end
        return [self.vocab.get(p, self.special.unknown) for p in pieces]

    def decode(self, tokens: list[int]) -> str:
        sb = []
        for tid in tokens:
            if tid in (self.special.cls, self.special.separator, self.special.pad):
                continue
            piece = self.reverse.get(tid, "")
            sb.append(piece.removeprefix("##"))
        return "".join(sb)

    def decode_skip_special(self, tokens: list[int]) -> str:
        return self.decode(tokens)


def _tokenize_basic(text: str) -> list[str]:
    words: list[str] = []
    current: list[str] = []
    for r in text:
        if r.isalnum():
            current.append(r)
        else:
            if current:
                words.append("".join(current))
                current = []
    if current:
        words.append("".join(current))
    return words
