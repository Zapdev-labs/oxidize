#![no_main]

use libfuzzer_sys::fuzz_target;
use oxidize_core::tokenizer::{
    BpeTokenizer, LoadedTokenizer, SentencePieceUnigramTokenizer, TiktokenTokenizer,
    WordPieceTokenizer,
};

fuzz_target!(|data: &[u8]| {
    let text = String::from_utf8_lossy(data);

    let bpe = LoadedTokenizer::Bpe(BpeTokenizer::train(&["hello world", "fuzz input"], 16));
    let sentencepiece = LoadedTokenizer::SentencePiece(
        SentencePieceUnigramTokenizer::new(&[
            ("hello", -0.2),
            (" ", -0.1),
            ("world", -0.2),
            ("fuzz", -0.3),
            ("input", -0.3),
        ])
        .with_unknown_token("<unk>"),
    );
    let wordpiece = LoadedTokenizer::WordPiece(
        WordPieceTokenizer::new(&["hello", "world", "fuzz", "input", " ", "<unk>"])
            .with_unknown_token("<unk>"),
    );
    let tiktoken = LoadedTokenizer::Tiktoken(TiktokenTokenizer::new(
        &[b"h", b"e", b"l", b"o", b" ", b"w", b"r", b"d", b"f", b"u", b"z", b"i", b"n", b"p"],
        &[],
    ));

    for tokenizer in [&bpe, &sentencepiece, &wordpiece, &tiktoken] {
        let encoded = tokenizer.encode(&text);
        let _ = tokenizer.decode(&encoded);
        let _ = tokenizer.decode_without_special_tokens(&encoded);
        let _ = tokenizer.heal_tokens(&encoded);
    }
});
