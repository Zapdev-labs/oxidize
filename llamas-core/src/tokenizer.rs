use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenizerError {
    UnknownToken(u32),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BpeTokenizer {
    vocab: HashMap<String, u32>,
    id_to_token: HashMap<u32, String>,
    merges: HashMap<(u32, u32), usize>,
    merged_token_ids: HashMap<(u32, u32), u32>,
    unknown_token: Option<u32>,
}

impl BpeTokenizer {
    pub fn train(corpus: &[&str], merge_limit: usize) -> Self {
        let mut vocab = HashMap::new();
        let mut id_to_token = HashMap::new();
        let mut next_id = 0_u32;

        for sample in corpus {
            for ch in sample.chars() {
                let token = ch.to_string();
                if !vocab.contains_key(&token) {
                    vocab.insert(token.clone(), next_id);
                    id_to_token.insert(next_id, token);
                    next_id = next_id.saturating_add(1);
                }
            }
        }

        let mut sequences: Vec<Vec<u32>> = corpus
            .iter()
            .map(|sample| {
                sample
                    .chars()
                    .map(|ch| {
                        let key = ch.to_string();
                        *vocab
                            .get(&key)
                            .expect("character must exist in base vocab during BPE training")
                    })
                    .collect()
            })
            .collect();

        let mut merges = HashMap::new();
        let mut merged_token_ids = HashMap::new();
        for rank in 0..merge_limit {
            let pair_counts = count_adjacent_pairs(&sequences);
            let Some((best_pair, _)) = pair_counts.into_iter().max_by_key(|(_, count)| *count)
            else {
                break;
            };

            let merged = format!(
                "{}{}",
                id_to_token
                    .get(&best_pair.0)
                    .expect("left token must exist in vocab during merge"),
                id_to_token
                    .get(&best_pair.1)
                    .expect("right token must exist in vocab during merge")
            );

            if vocab.contains_key(&merged) {
                continue;
            }

            let merged_id = next_id;
            next_id = next_id.saturating_add(1);
            vocab.insert(merged.clone(), merged_id);
            id_to_token.insert(merged_id, merged);
            merges.insert(best_pair, rank);
            merged_token_ids.insert(best_pair, merged_id);

            for sequence in &mut sequences {
                *sequence = apply_merge(sequence, best_pair, merged_id);
            }
        }

        Self {
            vocab,
            id_to_token,
            merges,
            merged_token_ids,
            unknown_token: None,
        }
    }

    pub fn with_unknown_token(mut self, token: &str) -> Self {
        let token_id = if let Some(id) = self.vocab.get(token).copied() {
            id
        } else {
            let id = self
                .id_to_token
                .keys()
                .copied()
                .max()
                .map_or(0, |max_id| max_id.saturating_add(1));
            self.vocab.insert(token.to_owned(), id);
            self.id_to_token.insert(id, token.to_owned());
            id
        };
        self.unknown_token = Some(token_id);
        self
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut sequence: Vec<u32> = text
            .chars()
            .filter_map(|ch| {
                let key = ch.to_string();
                self.vocab.get(&key).copied().or(self.unknown_token)
            })
            .collect();

        if sequence.len() < 2 {
            return sequence;
        }

        while let Some((pair, merged_id)) = self.best_merge_for_sequence(&sequence) {
            sequence = apply_merge(&sequence, pair, merged_id);
        }

        sequence
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String, TokenizerError> {
        let mut out = String::new();
        for id in ids {
            let Some(piece) = self.id_to_token.get(id) else {
                return Err(TokenizerError::UnknownToken(*id));
            };
            out.push_str(piece);
        }
        Ok(out)
    }

    pub fn merges_len(&self) -> usize {
        self.merges.len()
    }

    fn best_merge_for_sequence(&self, sequence: &[u32]) -> Option<((u32, u32), u32)> {
        let present_pairs: HashSet<(u32, u32)> =
            sequence.windows(2).map(|w| (w[0], w[1])).collect();
        self.merges
            .iter()
            .filter(|(pair, _)| present_pairs.contains(pair))
            .min_by_key(|(_, rank)| *rank)
            .and_then(|(pair, _)| {
                self.merged_token_ids
                    .get(pair)
                    .copied()
                    .map(|id| (*pair, id))
            })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SentencePieceUnigramTokenizer {
    vocab: HashMap<String, u32>,
    id_to_token: HashMap<u32, String>,
    piece_scores: HashMap<u32, f32>,
    unknown_token: Option<u32>,
}

impl SentencePieceUnigramTokenizer {
    pub fn new(pieces: &[(&str, f32)]) -> Self {
        let mut vocab = HashMap::new();
        let mut id_to_token = HashMap::new();
        let mut piece_scores = HashMap::new();

        for (idx, (piece, score)) in pieces.iter().enumerate() {
            let id = idx as u32;
            let token = (*piece).to_owned();
            vocab.insert(token.clone(), id);
            id_to_token.insert(id, token);
            piece_scores.insert(id, *score);
        }

        Self {
            vocab,
            id_to_token,
            piece_scores,
            unknown_token: None,
        }
    }

    pub fn with_unknown_token(mut self, token: &str) -> Self {
        let token_id = if let Some(id) = self.vocab.get(token).copied() {
            id
        } else {
            let id = self
                .id_to_token
                .keys()
                .copied()
                .max()
                .map_or(0, |max_id| max_id.saturating_add(1));
            self.vocab.insert(token.to_owned(), id);
            self.id_to_token.insert(id, token.to_owned());
            self.piece_scores.insert(id, f32::NEG_INFINITY);
            id
        };
        self.unknown_token = Some(token_id);
        self
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        if text.is_empty() {
            return Vec::new();
        }

        let boundaries = char_boundaries(text);
        let mut encoded = Vec::new();
        let mut boundary_idx = 0;

        while boundary_idx + 1 < boundaries.len() {
            let start = boundaries[boundary_idx];
            let segment = &text[start..];

            if let Some((ids, consumed)) = self.best_segmentation(segment) {
                encoded.extend(ids);
                boundary_idx += consumed;
                continue;
            }

            if let Some(unk) = self.unknown_token {
                encoded.push(unk);
            }
            boundary_idx += 1;
        }

        encoded
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String, TokenizerError> {
        let mut out = String::new();
        for id in ids {
            let Some(piece) = self.id_to_token.get(id) else {
                return Err(TokenizerError::UnknownToken(*id));
            };
            out.push_str(piece);
        }
        Ok(out)
    }

    fn best_segmentation(&self, text: &str) -> Option<(Vec<u32>, usize)> {
        let boundaries = char_boundaries(text);
        let token_count = boundaries.len().saturating_sub(1);
        if token_count == 0 {
            return Some((Vec::new(), 0));
        }

        let mut best_scores = vec![f32::NEG_INFINITY; token_count + 1];
        let mut backtrack: Vec<Option<(usize, u32)>> = vec![None; token_count + 1];
        best_scores[0] = 0.0;

        for i in 0..token_count {
            if !best_scores[i].is_finite() {
                continue;
            }

            for j in (i + 1)..=token_count {
                let piece = &text[boundaries[i]..boundaries[j]];
                let Some(id) = self.vocab.get(piece).copied() else {
                    continue;
                };
                let score = *self.piece_scores.get(&id).unwrap_or(&0.0);
                let candidate = best_scores[i] + score;
                if candidate > best_scores[j] {
                    best_scores[j] = candidate;
                    backtrack[j] = Some((i, id));
                }
            }
        }

        let end = (1..=token_count)
            .rev()
            .find(|idx| best_scores[*idx].is_finite())?;

        let mut ids = Vec::new();
        let mut cursor = end;
        while cursor > 0 {
            let (prev, id) = backtrack[cursor]?;
            ids.push(id);
            cursor = prev;
        }
        ids.reverse();
        Some((ids, end))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WordPieceTokenizer {
    vocab: HashMap<String, u32>,
    id_to_token: HashMap<u32, String>,
    unknown_token: Option<u32>,
}

impl WordPieceTokenizer {
    pub fn new(vocab_tokens: &[&str]) -> Self {
        let mut vocab = HashMap::new();
        let mut id_to_token = HashMap::new();

        for (idx, token) in vocab_tokens.iter().enumerate() {
            let id = idx as u32;
            let token = (*token).to_owned();
            vocab.insert(token.clone(), id);
            id_to_token.insert(id, token);
        }

        Self {
            vocab,
            id_to_token,
            unknown_token: None,
        }
    }

    pub fn with_unknown_token(mut self, token: &str) -> Self {
        let token_id = if let Some(id) = self.vocab.get(token).copied() {
            id
        } else {
            let id = self
                .id_to_token
                .keys()
                .copied()
                .max()
                .map_or(0, |max_id| max_id.saturating_add(1));
            self.vocab.insert(token.to_owned(), id);
            self.id_to_token.insert(id, token.to_owned());
            id
        };
        self.unknown_token = Some(token_id);
        self
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        if text.is_empty() {
            return Vec::new();
        }

        let mut encoded = Vec::new();
        let mut current_word = String::new();

        for ch in text.chars() {
            if ch.is_whitespace() {
                self.encode_word_into(&current_word, &mut encoded);
                current_word.clear();

                let ws = ch.to_string();
                if let Some(id) = self.vocab.get(&ws).copied() {
                    encoded.push(id);
                } else if let Some(unk) = self.unknown_token {
                    encoded.push(unk);
                }
            } else {
                current_word.push(ch);
            }
        }
        self.encode_word_into(&current_word, &mut encoded);

        encoded
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String, TokenizerError> {
        let mut out = String::new();
        for id in ids {
            let Some(piece) = self.id_to_token.get(id) else {
                return Err(TokenizerError::UnknownToken(*id));
            };

            if let Some(stripped) = piece.strip_prefix("##") {
                out.push_str(stripped);
            } else {
                out.push_str(piece);
            }
        }
        Ok(out)
    }

    fn encode_word_into(&self, word: &str, encoded: &mut Vec<u32>) {
        if word.is_empty() {
            return;
        }

        let boundaries = char_boundaries(word);
        let mut start_idx = 0;
        let token_count = boundaries.len().saturating_sub(1);

        while start_idx < token_count {
            let mut found: Option<(u32, usize)> = None;

            for end_idx in ((start_idx + 1)..=token_count).rev() {
                let mut candidate = word[boundaries[start_idx]..boundaries[end_idx]].to_owned();
                if start_idx > 0 {
                    candidate = format!("##{candidate}");
                }
                if let Some(id) = self.vocab.get(&candidate).copied() {
                    found = Some((id, end_idx));
                    break;
                }
            }

            if let Some((id, next_idx)) = found {
                encoded.push(id);
                start_idx = next_idx;
            } else {
                if let Some(unk) = self.unknown_token {
                    encoded.push(unk);
                }
                return;
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TiktokenTokenizer {
    vocab: HashMap<Vec<u8>, u32>,
    id_to_token: HashMap<u32, Vec<u8>>,
    merges: HashMap<(u32, u32), usize>,
    merged_token_ids: HashMap<(u32, u32), u32>,
    unknown_token: Option<u32>,
}

impl TiktokenTokenizer {
    pub fn new(vocab_tokens: &[&[u8]], merge_pairs: &[(&[u8], &[u8])]) -> Self {
        let mut vocab = HashMap::new();
        let mut id_to_token = HashMap::new();

        for (idx, token) in vocab_tokens.iter().enumerate() {
            let id = idx as u32;
            let token = (*token).to_vec();
            vocab.insert(token.clone(), id);
            id_to_token.insert(id, token);
        }

        let mut merges = HashMap::new();
        let mut merged_token_ids = HashMap::new();
        for (rank, (left, right)) in merge_pairs.iter().enumerate() {
            let Some(left_id) = vocab.get(*left).copied() else {
                continue;
            };
            let Some(right_id) = vocab.get(*right).copied() else {
                continue;
            };

            let mut merged = Vec::with_capacity(left.len() + right.len());
            merged.extend_from_slice(left);
            merged.extend_from_slice(right);
            let Some(merged_id) = vocab.get(&merged).copied() else {
                continue;
            };

            let pair = (left_id, right_id);
            merges.insert(pair, rank);
            merged_token_ids.insert(pair, merged_id);
        }

        Self {
            vocab,
            id_to_token,
            merges,
            merged_token_ids,
            unknown_token: None,
        }
    }

    pub fn with_unknown_token(mut self, token: &[u8]) -> Self {
        let token_id = if let Some(id) = self.vocab.get(token).copied() {
            id
        } else {
            let id = self
                .id_to_token
                .keys()
                .copied()
                .max()
                .map_or(0, |max_id| max_id.saturating_add(1));
            self.vocab.insert(token.to_vec(), id);
            self.id_to_token.insert(id, token.to_vec());
            id
        };
        self.unknown_token = Some(token_id);
        self
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut sequence: Vec<u32> = text
            .as_bytes()
            .iter()
            .filter_map(|byte| {
                let single = [*byte];
                self.vocab.get(single.as_slice()).copied().or(self.unknown_token)
            })
            .collect();

        if sequence.len() < 2 {
            return sequence;
        }

        while let Some((pair, merged_id)) = self.best_merge_for_sequence(&sequence) {
            sequence = apply_merge(&sequence, pair, merged_id);
        }

        sequence
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String, TokenizerError> {
        let mut bytes = Vec::new();
        for id in ids {
            let Some(piece) = self.id_to_token.get(id) else {
                return Err(TokenizerError::UnknownToken(*id));
            };
            bytes.extend_from_slice(piece);
        }
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    fn best_merge_for_sequence(&self, sequence: &[u32]) -> Option<((u32, u32), u32)> {
        let present_pairs: HashSet<(u32, u32)> =
            sequence.windows(2).map(|w| (w[0], w[1])).collect();
        self.merges
            .iter()
            .filter(|(pair, _)| present_pairs.contains(pair))
            .min_by_key(|(_, rank)| *rank)
            .and_then(|(pair, _)| {
                self.merged_token_ids
                    .get(pair)
                    .copied()
                    .map(|id| (*pair, id))
            })
    }
}

fn count_adjacent_pairs(sequences: &[Vec<u32>]) -> HashMap<(u32, u32), usize> {
    let mut pair_counts = HashMap::new();
    for sequence in sequences {
        for pair in sequence.windows(2) {
            *pair_counts.entry((pair[0], pair[1])).or_insert(0) += 1;
        }
    }
    pair_counts
}

fn apply_merge(sequence: &[u32], pair: (u32, u32), merged_id: u32) -> Vec<u32> {
    let mut merged = Vec::with_capacity(sequence.len());
    let mut idx = 0;
    while idx < sequence.len() {
        if idx + 1 < sequence.len() && sequence[idx] == pair.0 && sequence[idx + 1] == pair.1 {
            merged.push(merged_id);
            idx += 2;
            continue;
        }
        merged.push(sequence[idx]);
        idx += 1;
    }
    merged
}

fn char_boundaries(text: &str) -> Vec<usize> {
    let mut boundaries: Vec<usize> = text.char_indices().map(|(idx, _)| idx).collect();
    boundaries.push(text.len());
    boundaries
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trains_and_merges_common_pairs() {
        let tokenizer = BpeTokenizer::train(&["banana", "bandana"], 4);
        assert!(tokenizer.merges_len() > 0);

        let encoded = tokenizer.encode("banana");
        assert!(encoded.len() < "banana".chars().count());
        assert_eq!(
            tokenizer.decode(&encoded).expect("decode should succeed"),
            "banana"
        );
    }

    #[test]
    fn supports_unknown_token_fallback() {
        let tokenizer = BpeTokenizer::train(&["abc"], 2).with_unknown_token("<unk>");
        let encoded = tokenizer.encode("abcz");
        assert!(!encoded.is_empty());

        let decoded = tokenizer.decode(&encoded).expect("decode should succeed");
        assert_eq!(decoded, "abc<unk>");
    }

    #[test]
    fn decode_errors_on_unknown_id() {
        let tokenizer = BpeTokenizer::train(&["abc"], 1);
        let err = tokenizer
            .decode(&[999])
            .expect_err("unknown token id should fail");
        assert_eq!(err, TokenizerError::UnknownToken(999));
    }

    #[test]
    fn sentencepiece_prefers_higher_probability_path() {
        let tokenizer = SentencePieceUnigramTokenizer::new(&[
            ("a", -3.0),
            ("b", -3.0),
            ("ab", -0.5),
            ("aba", -0.1),
        ]);

        let encoded = tokenizer.encode("aba");
        assert_eq!(encoded.len(), 1);
        assert_eq!(
            tokenizer.decode(&encoded).expect("decode should succeed"),
            "aba"
        );
    }

    #[test]
    fn sentencepiece_uses_unknown_for_unmatched_text() {
        let tokenizer = SentencePieceUnigramTokenizer::new(&[("a", -0.5), ("b", -0.5)])
            .with_unknown_token("<unk>");

        let encoded = tokenizer.encode("abz");
        assert_eq!(
            tokenizer.decode(&encoded).expect("decode should succeed"),
            "ab<unk>"
        );
    }

    #[test]
    fn sentencepiece_round_trips_known_pieces() {
        let tokenizer = SentencePieceUnigramTokenizer::new(&[
            ("hello", -0.2),
            (" ", -0.1),
            ("world", -0.2),
            ("hell", -1.5),
            ("o", -1.0),
        ]);

        let encoded = tokenizer.encode("hello world");
        assert_eq!(
            tokenizer.decode(&encoded).expect("decode should succeed"),
            "hello world"
        );
    }

    #[test]
    fn wordpiece_encodes_with_greedy_longest_match() {
        let tokenizer = WordPieceTokenizer::new(&["play", "##ing", "##er", " "]);
        let encoded = tokenizer.encode("player playing");

        assert_eq!(
            tokenizer.decode(&encoded).expect("decode should succeed"),
            "player playing"
        );
        assert_eq!(encoded.len(), 5);
    }

    #[test]
    fn wordpiece_uses_unknown_for_unmatched_word() {
        let tokenizer =
            WordPieceTokenizer::new(&["hello", "world", " "]).with_unknown_token("<unk>");
        let encoded = tokenizer.encode("hello mars");

        assert_eq!(
            tokenizer.decode(&encoded).expect("decode should succeed"),
            "hello <unk>"
        );
    }

    #[test]
    fn wordpiece_decode_errors_on_unknown_id() {
        let tokenizer = WordPieceTokenizer::new(&["a"]);
        let err = tokenizer
            .decode(&[42])
            .expect_err("unknown token id should fail");
        assert_eq!(err, TokenizerError::UnknownToken(42));
    }

    #[test]
    fn tiktoken_merges_by_rank_and_round_trips() {
        let tokenizer = TiktokenTokenizer::new(
            &[b"h", b"e", b"l", b"o", b"he", b"ll", b"hell", b"hello"],
            &[(b"h", b"e"), (b"l", b"l"), (b"he", b"ll"), (b"hell", b"o")],
        );

        let encoded = tokenizer.encode("hello");
        assert_eq!(encoded.len(), 1);
        assert_eq!(
            tokenizer.decode(&encoded).expect("decode should succeed"),
            "hello"
        );
    }

    #[test]
    fn tiktoken_supports_utf8_bytes() {
        let tokenizer = TiktokenTokenizer::new(&[b"h", b"i", b" ", &[0xc3], &[0xa9], b"\xc3\xa9"], &[]);
        let encoded = tokenizer.encode("hi \u{00e9}");
        assert_eq!(
            tokenizer.decode(&encoded).expect("decode should succeed"),
            "hi \u{00e9}"
        );
    }

    #[test]
    fn tiktoken_uses_unknown_token_for_missing_bytes() {
        let tokenizer = TiktokenTokenizer::new(&[b"a"], &[]).with_unknown_token(b"<unk>");
        let encoded = tokenizer.encode("ab");
        assert_eq!(
            tokenizer.decode(&encoded).expect("decode should succeed"),
            "a<unk>"
        );
    }
}
