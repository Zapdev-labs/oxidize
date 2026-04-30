use std::collections::{BTreeMap, HashMap, HashSet};

use crate::gguf::{GgufMetadataValue, GgufParseError};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenizerError {
    UnknownToken(u32),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenizerLoadError {
    MissingMetadata(&'static str),
    InvalidMetadataType(&'static str),
    UnsupportedTokenizerModel(String),
    InvalidMergeEntry(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChatMessage<'a> {
    pub role: &'a str,
    pub content: &'a str,
}

impl From<GgufParseError> for TokenizerLoadError {
    fn from(_: GgufParseError) -> Self {
        Self::InvalidMetadataType("gguf")
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum LoadedTokenizer {
    Bpe(BpeTokenizer),
    SentencePiece(SentencePieceUnigramTokenizer),
    WordPiece(WordPieceTokenizer),
    Tiktoken(TiktokenTokenizer),
}

impl LoadedTokenizer {
    pub fn encode(&self, text: &str) -> Vec<u32> {
        match self {
            Self::Bpe(tokenizer) => tokenizer.encode(text),
            Self::SentencePiece(tokenizer) => tokenizer.encode(text),
            Self::WordPiece(tokenizer) => tokenizer.encode(text),
            Self::Tiktoken(tokenizer) => tokenizer.encode(text),
        }
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String, TokenizerError> {
        match self {
            Self::Bpe(tokenizer) => tokenizer.decode(ids),
            Self::SentencePiece(tokenizer) => tokenizer.decode(ids),
            Self::WordPiece(tokenizer) => tokenizer.decode(ids),
            Self::Tiktoken(tokenizer) => tokenizer.decode(ids),
        }
    }

    pub fn special_tokens(&self) -> &SpecialTokens {
        match self {
            Self::Bpe(tokenizer) => &tokenizer.special_tokens,
            Self::SentencePiece(tokenizer) => &tokenizer.special_tokens,
            Self::WordPiece(tokenizer) => &tokenizer.special_tokens,
            Self::Tiktoken(tokenizer) => &tokenizer.special_tokens,
        }
    }

    pub fn encode_with_special_tokens(&self, text: &str, options: EncodeOptions) -> Vec<u32> {
        let mut encoded = self.encode(text);
        self.special_tokens()
            .apply_encode_options(&mut encoded, options);
        encoded
    }

    pub fn decode_without_special_tokens(&self, ids: &[u32]) -> Result<String, TokenizerError> {
        let filtered: Vec<u32> = ids
            .iter()
            .copied()
            .filter(|id| !self.special_tokens().is_special(*id))
            .collect();
        self.decode(&filtered)
    }

    pub fn streaming_detokenizer(&self) -> StreamingDetokenizer<'_> {
        StreamingDetokenizer::new(self)
    }
}

#[derive(Debug, Clone)]
pub struct StreamingDetokenizer<'a> {
    tokenizer: &'a LoadedTokenizer,
    pending_bytes: Vec<u8>,
}

impl<'a> StreamingDetokenizer<'a> {
    pub fn new(tokenizer: &'a LoadedTokenizer) -> Self {
        Self {
            tokenizer,
            pending_bytes: Vec::new(),
        }
    }

    pub fn push(&mut self, id: u32) -> Result<String, TokenizerError> {
        match self.tokenizer {
            LoadedTokenizer::Bpe(tokenizer) => tokenizer
                .id_to_token
                .get(&id)
                .cloned()
                .ok_or(TokenizerError::UnknownToken(id)),
            LoadedTokenizer::SentencePiece(tokenizer) => tokenizer
                .id_to_token
                .get(&id)
                .cloned()
                .ok_or(TokenizerError::UnknownToken(id)),
            LoadedTokenizer::WordPiece(tokenizer) => tokenizer
                .id_to_token
                .get(&id)
                .map(|piece| piece.strip_prefix("##").unwrap_or(piece).to_owned())
                .ok_or(TokenizerError::UnknownToken(id)),
            LoadedTokenizer::Tiktoken(tokenizer) => {
                let Some(piece) = tokenizer.id_to_token.get(&id) else {
                    return Err(TokenizerError::UnknownToken(id));
                };
                self.pending_bytes.extend_from_slice(piece);
                Ok(consume_pending_utf8(&mut self.pending_bytes))
            }
        }
    }

    pub fn finish(&mut self) -> String {
        if self.pending_bytes.is_empty() {
            return String::new();
        }
        let out = String::from_utf8_lossy(&self.pending_bytes).into_owned();
        self.pending_bytes.clear();
        out
    }
}

fn consume_pending_utf8(pending_bytes: &mut Vec<u8>) -> String {
    if pending_bytes.is_empty() {
        return String::new();
    }

    let mut out = String::new();
    loop {
        match std::str::from_utf8(pending_bytes) {
            Ok(valid) => {
                out.push_str(valid);
                pending_bytes.clear();
                break;
            }
            Err(error) => {
                let valid_up_to = error.valid_up_to();
                if valid_up_to > 0 {
                    let valid = std::str::from_utf8(&pending_bytes[..valid_up_to])
                        .expect("valid prefix from utf8_error.valid_up_to");
                    out.push_str(valid);
                    pending_bytes.drain(..valid_up_to);
                    continue;
                }

                let Some(error_len) = error.error_len() else {
                    break;
                };
                out.push_str(&String::from_utf8_lossy(&pending_bytes[..error_len]));
                pending_bytes.drain(..error_len);
            }
        }
    }
    out
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct EncodeOptions {
    pub add_bos: bool,
    pub add_eos: bool,
    pub pad_to: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SpecialTokens {
    pub unknown: Option<u32>,
    pub bos: Option<u32>,
    pub eos: Option<u32>,
    pub pad: Option<u32>,
    pub separator: Option<u32>,
    pub cls: Option<u32>,
    pub mask: Option<u32>,
}

impl SpecialTokens {
    fn from_metadata(metadata: &BTreeMap<String, GgufMetadataValue>) -> Self {
        Self {
            unknown: metadata_u32(metadata, "tokenizer.ggml.unknown_token_id"),
            bos: metadata_u32(metadata, "tokenizer.ggml.bos_token_id"),
            eos: metadata_u32(metadata, "tokenizer.ggml.eos_token_id"),
            pad: metadata_u32(metadata, "tokenizer.ggml.padding_token_id")
                .or_else(|| metadata_u32(metadata, "tokenizer.ggml.pad_token_id")),
            separator: metadata_u32(metadata, "tokenizer.ggml.separator_token_id")
                .or_else(|| metadata_u32(metadata, "tokenizer.ggml.sep_token_id")),
            cls: metadata_u32(metadata, "tokenizer.ggml.cls_token_id"),
            mask: metadata_u32(metadata, "tokenizer.ggml.mask_token_id"),
        }
    }

    fn is_special(&self, id: u32) -> bool {
        self.unknown == Some(id)
            || self.bos == Some(id)
            || self.eos == Some(id)
            || self.pad == Some(id)
            || self.separator == Some(id)
            || self.cls == Some(id)
            || self.mask == Some(id)
    }

    fn apply_encode_options(&self, encoded: &mut Vec<u32>, options: EncodeOptions) {
        if options.add_bos && let Some(bos) = self.bos {
            encoded.insert(0, bos);
        }
        if options.add_eos && let Some(eos) = self.eos {
            encoded.push(eos);
        }
        if let Some(target_len) = options.pad_to
            && let Some(pad) = self.pad
            && encoded.len() < target_len
        {
            encoded.resize(target_len, pad);
        }
    }
}

pub fn load_tokenizer_from_gguf_metadata(
    metadata: &BTreeMap<String, GgufMetadataValue>,
) -> Result<LoadedTokenizer, TokenizerLoadError> {
    let model = metadata_string(metadata, "tokenizer.ggml.model")?;
    match model.as_str() {
        "llama" => Ok(LoadedTokenizer::SentencePiece(load_sentencepiece(metadata)?)),
        "bert" => Ok(LoadedTokenizer::WordPiece(load_wordpiece(metadata)?)),
        "gpt2" => Ok(LoadedTokenizer::Bpe(load_bpe(metadata)?)),
        "tiktoken" => Ok(LoadedTokenizer::Tiktoken(load_tiktoken(metadata)?)),
        _ => Err(TokenizerLoadError::UnsupportedTokenizerModel(model)),
    }
}

pub fn process_chat_template(
    template: &str,
    messages: &[ChatMessage<'_>],
    add_generation_prompt: bool,
) -> String {
    let mut rendered = template.to_owned();

    rendered = render_for_messages_block(&rendered, messages);
    rendered = render_if_generation_prompt(&rendered, add_generation_prompt);

    rendered
}

pub fn process_chat_template_from_gguf_metadata(
    metadata: &BTreeMap<String, GgufMetadataValue>,
    messages: &[ChatMessage<'_>],
    add_generation_prompt: bool,
) -> Result<Option<String>, TokenizerLoadError> {
    let template = metadata
        .get("tokenizer.chat_template")
        .or_else(|| metadata.get("tokenizer.ggml.chat_template"));
    match template {
        Some(GgufMetadataValue::String(template)) => Ok(Some(process_chat_template(
            template,
            messages,
            add_generation_prompt,
        ))),
        Some(_) => Err(TokenizerLoadError::InvalidMetadataType(
            "tokenizer.chat_template",
        )),
        None => Ok(None),
    }
}

fn render_for_messages_block(template: &str, messages: &[ChatMessage<'_>]) -> String {
    let starts = ["{% for message in messages %}", "{%- for message in messages %}"];
    let ends = ["{% endfor %}", "{%- endfor %}"];

    let Some((start_idx, start_token)) = starts
        .iter()
        .find_map(|token| template.find(token).map(|idx| (idx, *token)))
    else {
        return template.to_owned();
    };

    let loop_start = start_idx + start_token.len();
    let Some((end_idx, end_token)) = ends
        .iter()
        .find_map(|token| template[loop_start..].find(token).map(|idx| (loop_start + idx, *token)))
    else {
        return template.to_owned();
    };

    let loop_body = &template[loop_start..end_idx];
    let mut expanded = String::new();
    for message in messages {
        expanded.push_str(&render_message_template(loop_body, message));
    }

    let mut out = String::with_capacity(template.len() + expanded.len());
    out.push_str(&template[..start_idx]);
    out.push_str(&expanded);
    out.push_str(&template[end_idx + end_token.len()..]);
    out
}

fn render_if_generation_prompt(template: &str, add_generation_prompt: bool) -> String {
    let starts = [
        "{% if add_generation_prompt %}",
        "{%- if add_generation_prompt %}",
    ];
    let ends = ["{% endif %}", "{%- endif %}"];

    let Some((start_idx, start_token)) = starts
        .iter()
        .find_map(|token| template.find(token).map(|idx| (idx, *token)))
    else {
        return template.to_owned();
    };

    let if_start = start_idx + start_token.len();
    let Some((end_idx, end_token)) = ends
        .iter()
        .find_map(|token| template[if_start..].find(token).map(|idx| (if_start + idx, *token)))
    else {
        return template.to_owned();
    };

    let condition_body = if add_generation_prompt {
        &template[if_start..end_idx]
    } else {
        ""
    };

    let mut out = String::with_capacity(template.len());
    out.push_str(&template[..start_idx]);
    out.push_str(condition_body);
    out.push_str(&template[end_idx + end_token.len()..]);
    out
}

fn render_message_template(template: &str, message: &ChatMessage<'_>) -> String {
    template
        .replace("{{ message['role'] }}", message.role)
        .replace("{{message['role']}}", message.role)
        .replace("{{ message.role }}", message.role)
        .replace("{{message.role}}", message.role)
        .replace("{{ message['content'] }}", message.content)
        .replace("{{message['content']}}", message.content)
        .replace("{{ message.content }}", message.content)
        .replace("{{message.content}}", message.content)
}

fn load_sentencepiece(
    metadata: &BTreeMap<String, GgufMetadataValue>,
) -> Result<SentencePieceUnigramTokenizer, TokenizerLoadError> {
    let tokens = metadata_string_array(metadata, "tokenizer.ggml.tokens")?;
    let scores = metadata_f32_array(metadata, "tokenizer.ggml.scores")?;
    if tokens.len() != scores.len() {
        return Err(TokenizerLoadError::InvalidMetadataType(
            "tokenizer.ggml.scores",
        ));
    }

    let mut vocab = HashMap::new();
    let mut id_to_token = HashMap::new();
    let mut piece_scores = HashMap::new();
    for (id, (token, score)) in tokens.into_iter().zip(scores).enumerate() {
        let id = id as u32;
        vocab.insert(token.clone(), id);
        id_to_token.insert(id, token);
        piece_scores.insert(id, score);
    }

    Ok(SentencePieceUnigramTokenizer {
        vocab,
        id_to_token,
        piece_scores,
        special_tokens: SpecialTokens::from_metadata(metadata),
    })
}

fn load_wordpiece(
    metadata: &BTreeMap<String, GgufMetadataValue>,
) -> Result<WordPieceTokenizer, TokenizerLoadError> {
    let tokens = metadata_string_array(metadata, "tokenizer.ggml.tokens")?;
    let mut vocab = HashMap::new();
    let mut id_to_token = HashMap::new();
    for (id, token) in tokens.into_iter().enumerate() {
        let id = id as u32;
        vocab.insert(token.clone(), id);
        id_to_token.insert(id, token);
    }
    Ok(WordPieceTokenizer {
        vocab,
        id_to_token,
        special_tokens: SpecialTokens::from_metadata(metadata),
    })
}

fn load_bpe(metadata: &BTreeMap<String, GgufMetadataValue>) -> Result<BpeTokenizer, TokenizerLoadError> {
    let tokens = metadata_string_array(metadata, "tokenizer.ggml.tokens")?;
    let merges = metadata_string_array(metadata, "tokenizer.ggml.merges").unwrap_or_default();

    let mut vocab = HashMap::new();
    let mut id_to_token = HashMap::new();
    for (id, token) in tokens.into_iter().enumerate() {
        let id = id as u32;
        vocab.insert(token.clone(), id);
        id_to_token.insert(id, token);
    }

    let mut merge_ranks = HashMap::new();
    let mut merged_token_ids = HashMap::new();
    for (rank, merge) in merges.into_iter().enumerate() {
        let (left, right) = merge
            .split_once(' ')
            .ok_or_else(|| TokenizerLoadError::InvalidMergeEntry(merge.clone()))?;
        let Some(left_id) = vocab.get(left).copied() else {
            continue;
        };
        let Some(right_id) = vocab.get(right).copied() else {
            continue;
        };
        let merged = format!("{left}{right}");
        let Some(merged_id) = vocab.get(&merged).copied() else {
            continue;
        };
        let pair = (left_id, right_id);
        merge_ranks.insert(pair, rank);
        merged_token_ids.insert(pair, merged_id);
    }

    Ok(BpeTokenizer {
        vocab,
        id_to_token,
        merges: merge_ranks,
        merged_token_ids,
        special_tokens: SpecialTokens::from_metadata(metadata),
    })
}

fn load_tiktoken(
    metadata: &BTreeMap<String, GgufMetadataValue>,
) -> Result<TiktokenTokenizer, TokenizerLoadError> {
    let tokens = metadata_string_array(metadata, "tokenizer.ggml.tokens")?;
    let merges = metadata_string_array(metadata, "tokenizer.ggml.merges").unwrap_or_default();

    let mut vocab = HashMap::new();
    let mut id_to_token = HashMap::new();
    for (id, token) in tokens.into_iter().enumerate() {
        let id = id as u32;
        let bytes = token.into_bytes();
        vocab.insert(bytes.clone(), id);
        id_to_token.insert(id, bytes);
    }

    let mut merge_ranks = HashMap::new();
    let mut merged_token_ids = HashMap::new();
    for (rank, merge) in merges.into_iter().enumerate() {
        let (left, right) = merge
            .split_once(' ')
            .ok_or_else(|| TokenizerLoadError::InvalidMergeEntry(merge.clone()))?;
        let Some(left_id) = vocab.get(left.as_bytes()).copied() else {
            continue;
        };
        let Some(right_id) = vocab.get(right.as_bytes()).copied() else {
            continue;
        };
        let mut merged = left.as_bytes().to_vec();
        merged.extend_from_slice(right.as_bytes());
        let Some(merged_id) = vocab.get(&merged).copied() else {
            continue;
        };
        let pair = (left_id, right_id);
        merge_ranks.insert(pair, rank);
        merged_token_ids.insert(pair, merged_id);
    }

    Ok(TiktokenTokenizer {
        vocab,
        id_to_token,
        merges: merge_ranks,
        merged_token_ids,
        special_tokens: SpecialTokens::from_metadata(metadata),
    })
}

fn metadata_string(
    metadata: &BTreeMap<String, GgufMetadataValue>,
    key: &'static str,
) -> Result<String, TokenizerLoadError> {
    match metadata.get(key) {
        Some(GgufMetadataValue::String(value)) => Ok(value.clone()),
        Some(_) => Err(TokenizerLoadError::InvalidMetadataType(key)),
        None => Err(TokenizerLoadError::MissingMetadata(key)),
    }
}

fn metadata_string_array(
    metadata: &BTreeMap<String, GgufMetadataValue>,
    key: &'static str,
) -> Result<Vec<String>, TokenizerLoadError> {
    match metadata.get(key) {
        Some(GgufMetadataValue::Array(values)) => values
            .values
            .iter()
            .map(|value| match value {
                GgufMetadataValue::String(token) => Ok(token.clone()),
                _ => Err(TokenizerLoadError::InvalidMetadataType(key)),
            })
            .collect(),
        Some(_) => Err(TokenizerLoadError::InvalidMetadataType(key)),
        None => Err(TokenizerLoadError::MissingMetadata(key)),
    }
}

fn metadata_f32_array(
    metadata: &BTreeMap<String, GgufMetadataValue>,
    key: &'static str,
) -> Result<Vec<f32>, TokenizerLoadError> {
    match metadata.get(key) {
        Some(GgufMetadataValue::Array(values)) => values
            .values
            .iter()
            .map(|value| match value {
                GgufMetadataValue::Float32(score) => Ok(*score),
                _ => Err(TokenizerLoadError::InvalidMetadataType(key)),
            })
            .collect(),
        Some(_) => Err(TokenizerLoadError::InvalidMetadataType(key)),
        None => Err(TokenizerLoadError::MissingMetadata(key)),
    }
}

fn metadata_u32(metadata: &BTreeMap<String, GgufMetadataValue>, key: &'static str) -> Option<u32> {
    match metadata.get(key) {
        Some(GgufMetadataValue::Uint8(value)) => Some((*value).into()),
        Some(GgufMetadataValue::Uint16(value)) => Some((*value).into()),
        Some(GgufMetadataValue::Uint32(value)) => Some(*value),
        Some(GgufMetadataValue::Uint64(value)) => (*value).try_into().ok(),
        Some(GgufMetadataValue::Int8(value)) if *value >= 0 => Some((*value as u8).into()),
        Some(GgufMetadataValue::Int16(value)) if *value >= 0 => Some((*value as u16).into()),
        Some(GgufMetadataValue::Int32(value)) if *value >= 0 => (*value).try_into().ok(),
        Some(GgufMetadataValue::Int64(value)) if *value >= 0 => (*value).try_into().ok(),
        _ => None,
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BpeTokenizer {
    vocab: HashMap<String, u32>,
    id_to_token: HashMap<u32, String>,
    merges: HashMap<(u32, u32), usize>,
    merged_token_ids: HashMap<(u32, u32), u32>,
    special_tokens: SpecialTokens,
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
            special_tokens: SpecialTokens::default(),
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
        self.special_tokens.unknown = Some(token_id);
        self
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut sequence: Vec<u32> = text
            .chars()
            .filter_map(|ch| {
                let key = ch.to_string();
                self.vocab.get(&key).copied().or(self.special_tokens.unknown)
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
    special_tokens: SpecialTokens,
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
            special_tokens: SpecialTokens::default(),
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
        self.special_tokens.unknown = Some(token_id);
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

            if let Some(unk) = self.special_tokens.unknown {
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
    special_tokens: SpecialTokens,
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
            special_tokens: SpecialTokens::default(),
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
        self.special_tokens.unknown = Some(token_id);
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
                } else if let Some(unk) = self.special_tokens.unknown {
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
                if let Some(unk) = self.special_tokens.unknown {
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
    special_tokens: SpecialTokens,
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
            special_tokens: SpecialTokens::default(),
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
        self.special_tokens.unknown = Some(token_id);
        self
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut sequence: Vec<u32> = text
            .as_bytes()
            .iter()
            .filter_map(|byte| {
                let single = [*byte];
                self.vocab.get(single.as_slice()).copied().or(self.special_tokens.unknown)
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
    use crate::gguf::{GgufMetadataArray, GgufMetadataType};

    fn metadata_strings(values: &[&str]) -> GgufMetadataValue {
        GgufMetadataValue::Array(GgufMetadataArray {
            element_type: GgufMetadataType::String,
            values: values
                .iter()
                .map(|value| GgufMetadataValue::String((*value).to_owned()))
                .collect(),
        })
    }

    fn metadata_scores(values: &[f32]) -> GgufMetadataValue {
        GgufMetadataValue::Array(GgufMetadataArray {
            element_type: GgufMetadataType::Float32,
            values: values
                .iter()
                .map(|value| GgufMetadataValue::Float32(*value))
                .collect(),
        })
    }

    #[test]
    fn processes_chat_template_for_messages_and_generation_prompt() {
        let template = "{% for message in messages %}<|{{ message['role'] }}|>\n{{ message['content'] }}\n{% endfor %}{% if add_generation_prompt %}<|assistant|>\n{% endif %}";
        let messages = [
            ChatMessage {
                role: "system",
                content: "You are helpful.",
            },
            ChatMessage {
                role: "user",
                content: "Hi",
            },
        ];

        let rendered = process_chat_template(template, &messages, true);
        assert_eq!(
            rendered,
            "<|system|>\nYou are helpful.\n<|user|>\nHi\n<|assistant|>\n"
        );
    }

    #[test]
    fn processes_chat_template_from_gguf_metadata() {
        let metadata = BTreeMap::from([(
            "tokenizer.chat_template".to_owned(),
            GgufMetadataValue::String(
                "{% for message in messages %}{{message.role}}: {{message.content}}\n{% endfor %}"
                    .to_owned(),
            ),
        )]);
        let messages = [ChatMessage {
            role: "user",
            content: "hello",
        }];

        let rendered = process_chat_template_from_gguf_metadata(&metadata, &messages, false)
            .expect("chat template metadata should parse")
            .expect("template should exist");
        assert_eq!(rendered, "user: hello\n");
    }

    #[test]
    fn chat_template_metadata_returns_none_when_missing() {
        let metadata = BTreeMap::new();
        let rendered = process_chat_template_from_gguf_metadata(&metadata, &[], false)
            .expect("missing template should not error");
        assert_eq!(rendered, None);
    }

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

    #[test]
    fn loads_sentencepiece_tokenizer_from_gguf_metadata() {
        let metadata = BTreeMap::from([
            (
                "tokenizer.ggml.model".to_owned(),
                GgufMetadataValue::String("llama".to_owned()),
            ),
            (
                "tokenizer.ggml.tokens".to_owned(),
                metadata_strings(&["he", "llo", "hello", "<unk>"]),
            ),
            (
                "tokenizer.ggml.scores".to_owned(),
                metadata_scores(&[-1.0, -1.0, -0.1, -99.0]),
            ),
            (
                "tokenizer.ggml.unknown_token_id".to_owned(),
                GgufMetadataValue::Uint32(3),
            ),
        ]);

        let tokenizer =
            load_tokenizer_from_gguf_metadata(&metadata).expect("metadata should load tokenizer");
        assert_eq!(tokenizer.decode(&tokenizer.encode("hello")), Ok("hello".to_owned()));
        assert_eq!(tokenizer.decode(&tokenizer.encode("x")), Ok("<unk>".to_owned()));
    }

    #[test]
    fn loads_gpt2_bpe_tokenizer_from_gguf_metadata() {
        let metadata = BTreeMap::from([
            (
                "tokenizer.ggml.model".to_owned(),
                GgufMetadataValue::String("gpt2".to_owned()),
            ),
            (
                "tokenizer.ggml.tokens".to_owned(),
                metadata_strings(&["h", "e", "l", "o", "he", "ll", "hell", "hello"]),
            ),
            ("tokenizer.ggml.merges".to_owned(), metadata_strings(&["h e", "l l", "he ll", "hell o"])),
        ]);

        let tokenizer =
            load_tokenizer_from_gguf_metadata(&metadata).expect("metadata should load tokenizer");
        assert_eq!(tokenizer.decode(&tokenizer.encode("hello")), Ok("hello".to_owned()));
    }



    #[test]
    fn loads_special_token_ids_from_gguf_metadata() {
        let metadata = BTreeMap::from([
            (
                "tokenizer.ggml.model".to_owned(),
                GgufMetadataValue::String("llama".to_owned()),
            ),
            (
                "tokenizer.ggml.tokens".to_owned(),
                metadata_strings(&["a", "b", "<unk>", "<s>", "</s>", "<pad>"]),
            ),
            (
                "tokenizer.ggml.scores".to_owned(),
                metadata_scores(&[-1.0, -1.0, -99.0, -99.0, -99.0, -99.0]),
            ),
            (
                "tokenizer.ggml.unknown_token_id".to_owned(),
                GgufMetadataValue::Uint32(2),
            ),
            (
                "tokenizer.ggml.bos_token_id".to_owned(),
                GgufMetadataValue::Uint32(3),
            ),
            (
                "tokenizer.ggml.eos_token_id".to_owned(),
                GgufMetadataValue::Uint32(4),
            ),
            (
                "tokenizer.ggml.padding_token_id".to_owned(),
                GgufMetadataValue::Uint32(5),
            ),
        ]);

        let tokenizer =
            load_tokenizer_from_gguf_metadata(&metadata).expect("metadata should load tokenizer");
        assert_eq!(
            tokenizer.special_tokens(),
            &SpecialTokens {
                unknown: Some(2),
                bos: Some(3),
                eos: Some(4),
                pad: Some(5),
                separator: None,
                cls: None,
                mask: None,
            }
        );
    }

    #[test]
    fn encode_with_special_tokens_adds_bos_eos_and_padding() {
        let mut tokenizer = WordPieceTokenizer::new(&["h", "##i", "<unk>", "<s>", "</s>", "<pad>"]);
        tokenizer.special_tokens = SpecialTokens {
            unknown: Some(2),
            bos: Some(3),
            eos: Some(4),
            pad: Some(5),
            separator: None,
            cls: None,
            mask: None,
        };

        let tokenizer = LoadedTokenizer::WordPiece(tokenizer);
        let encoded = tokenizer.encode_with_special_tokens(
            "hi",
            EncodeOptions {
                add_bos: true,
                add_eos: true,
                pad_to: Some(6),
            },
        );
        assert_eq!(encoded, vec![3, 0, 1, 4, 5, 5]);
        assert_eq!(
            tokenizer
                .decode_without_special_tokens(&encoded)
                .expect("decode should succeed"),
            "hi"
        );
    }

    #[test]
    fn fails_for_unsupported_gguf_tokenizer_model() {
        let metadata = BTreeMap::from([(
            "tokenizer.ggml.model".to_owned(),
            GgufMetadataValue::String("unknown".to_owned()),
        )]);

        let err = load_tokenizer_from_gguf_metadata(&metadata)
            .expect_err("unsupported model should fail");
        assert_eq!(
            err,
            TokenizerLoadError::UnsupportedTokenizerModel("unknown".to_owned())
        );
    }

    #[test]
    fn streaming_detokenizer_decodes_incrementally_for_wordpiece() {
        let tokenizer = LoadedTokenizer::WordPiece(
            WordPieceTokenizer::new(&["play", "##ing", " "]).with_unknown_token("<unk>"),
        );
        let mut stream = tokenizer.streaming_detokenizer();

        assert_eq!(stream.push(0).expect("token should decode"), "play");
        assert_eq!(stream.push(1).expect("token should decode"), "ing");
        assert_eq!(stream.push(2).expect("token should decode"), " ");
        assert_eq!(stream.finish(), "");
    }

    #[test]
    fn streaming_detokenizer_buffers_partial_utf8_for_tiktoken() {
        let tokenizer = LoadedTokenizer::Tiktoken(TiktokenTokenizer::new(
            &[&[0xc3], &[0xa9], b"!"],
            &[],
        ));
        let mut stream = tokenizer.streaming_detokenizer();

        assert_eq!(stream.push(0).expect("first byte token should decode"), "");
        assert_eq!(stream.push(1).expect("second byte token should decode"), "é");
        assert_eq!(stream.push(2).expect("ascii token should decode"), "!");
        assert_eq!(stream.finish(), "");
    }

    #[test]
    fn streaming_detokenizer_reports_unknown_token() {
        let tokenizer = LoadedTokenizer::Bpe(BpeTokenizer::train(&["ab"], 1));
        let mut stream = tokenizer.streaming_detokenizer();

        let err = stream
            .push(999)
            .expect_err("unknown token id should fail in stream");
        assert_eq!(err, TokenizerError::UnknownToken(999));
    }
}
