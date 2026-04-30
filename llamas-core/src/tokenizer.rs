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
                self.vocab
                    .get(&key)
                    .copied()
                    .or(self.unknown_token)
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
            .and_then(|(pair, _)| self.merged_token_ids.get(pair).copied().map(|id| (*pair, id)))
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
}
