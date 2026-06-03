from oxidize_python.core.tokenizer.bpe import BpeTokenizer
from oxidize_python.core.tokenizer.sp import SentencePieceUnigramTokenizer
from oxidize_python.core.tokenizer.tokenizer import EncodeOptions, SpecialTokens
from oxidize_python.core.tokenizer.wordpiece import WordPieceTokenizer


def test_bpe_roundtrip() -> None:
    tok = BpeTokenizer(
        vocab={"<unk>": 0, "<s>": 1, "</s>": 2, "a": 3, "b": 4, "c": 5, "ab": 6, "abc": 7, "▁": 8},
        merges=["ab", "abc"],
        special=SpecialTokens(unknown=0, bos=1, eos=2, pad=0),
    )
    tokens = tok.encode("abc", EncodeOptions())
    assert tokens
    assert "abc" in tok.decode(tokens)


def test_sentencepiece_encode() -> None:
    tok = SentencePieceUnigramTokenizer(
        pieces=["<unk>", "▁a", "▁ab", "a", "b"],
        scores=[0, -1, -2, -2, -2],
        special=SpecialTokens(unknown=0, bos=0, eos=0, pad=0),
    )
    tokens = tok.encode("ab", EncodeOptions())
    assert tokens
    tok.decode(tokens)


def test_wordpiece_encode() -> None:
    tok = WordPieceTokenizer(
        vocab={"[UNK]": 0, "##a": 1, "ab": 2},
        special=SpecialTokens(unknown=0),
    )
    tokens = tok.encode("ab", EncodeOptions())
    assert tokens
