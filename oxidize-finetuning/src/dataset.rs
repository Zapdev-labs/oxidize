use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use serde::Deserialize;

use crate::error::{FinetuneError, Result};

#[derive(Debug, Clone)]
pub struct SftExample {
    pub text: String,
    pub token_ids: Vec<u32>,
}

#[derive(Debug, Deserialize)]
struct JsonlRow {
    #[serde(default)]
    instruction: String,
    #[serde(default)]
    input: String,
    #[serde(default)]
    output: String,
    #[serde(default)]
    text: String,
    #[serde(default)]
    messages: Vec<JsonlMessage>,
}

#[derive(Debug, Deserialize)]
struct JsonlMessage {
    role: String,
    content: String,
}

pub fn load_jsonl_sft(path: impl AsRef<Path>) -> Result<Vec<SftExample>> {
    let file = File::open(path.as_ref()).map_err(|e| FinetuneError::Model(e.to_string()))?;
    let reader = BufReader::new(file);
    let mut out = Vec::new();
    for (line_no, line) in reader.lines().enumerate() {
        let line = line.map_err(|e| FinetuneError::Model(e.to_string()))?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let row: JsonlRow = serde_json::from_str(trimmed)
            .map_err(|e| FinetuneError::Model(format!("jsonl line {}: {e}", line_no + 1)))?;
        let text = row_to_text(&row);
        if !text.is_empty() {
            out.push(SftExample {
                text,
                token_ids: Vec::new(),
            });
        }
    }
    if out.is_empty() {
        return Err(FinetuneError::EmptyDataset);
    }
    Ok(out)
}

/// Pack tokenized examples into training chunks.
///
/// With `pack = true`, examples are concatenated (separated by `eos`) into
/// chunks of `max_seq_len` tokens so batched forward windows are full — the
/// same throughput trick unsloth/llama.cpp use. The trailing chunk may be
/// shorter than `max_seq_len` (it is kept when it holds at least 2 tokens).
/// With `pack = false`, each example becomes its own chunk (truncated to
/// `max_seq_len`).
pub fn pack_chunks(
    examples: &[SftExample],
    max_seq_len: usize,
    eos: u32,
    pack: bool,
) -> Vec<Vec<u32>> {
    let max_seq_len = max_seq_len.max(2);
    let mut chunks = Vec::new();
    if !pack {
        for ex in examples {
            if ex.token_ids.len() >= 2 {
                // Copy only the kept prefix rather than cloning the full vector
                // and truncating (avoids O(n) work on long, truncated examples).
                let take = max_seq_len.min(ex.token_ids.len());
                chunks.push(ex.token_ids[..take].to_vec());
            }
        }
        return chunks;
    }
    let mut current: Vec<u32> = Vec::with_capacity(max_seq_len);
    for ex in examples {
        if ex.token_ids.is_empty() {
            continue;
        }
        let mut remaining = &ex.token_ids[..];
        while !remaining.is_empty() {
            if !current.is_empty() {
                current.push(eos);
                if current.len() >= max_seq_len {
                    chunks.push(std::mem::replace(
                        &mut current,
                        Vec::with_capacity(max_seq_len),
                    ));
                    continue;
                }
            }
            let room = max_seq_len - current.len();
            let take = room.min(remaining.len());
            current.extend_from_slice(&remaining[..take]);
            remaining = &remaining[take..];
            if current.len() >= max_seq_len {
                chunks.push(std::mem::replace(
                    &mut current,
                    Vec::with_capacity(max_seq_len),
                ));
            }
        }
    }
    if current.len() >= 2 {
        chunks.push(current);
    }
    chunks
}

fn row_to_text(row: &JsonlRow) -> String {
    if !row.text.is_empty() {
        return row.text.clone();
    }
    if !row.messages.is_empty() {
        let mut s = String::new();
        for m in &row.messages {
            s.push_str("<|im_start|>");
            s.push_str(&m.role);
            s.push('\n');
            s.push_str(&m.content);
            s.push_str("<|");
            s.push_str("im_end");
            s.push_str("|>\n");
        }
        return s;
    }
    if row.input.is_empty() {
        format!(
            "<|im_start|>user\n{}\n<|im_start|>assistant\n{}\n",
            row.instruction, row.output
        )
    } else {
        format!(
            "<|im_start|>user\n{}\n{}\n<|im_start|>assistant\n{}\n",
            row.instruction, row.input, row.output
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ex(ids: &[u32]) -> SftExample {
        SftExample {
            text: String::new(),
            token_ids: ids.to_vec(),
        }
    }

    #[test]
    fn packing_fills_chunks_and_separates_with_eos() {
        let examples = vec![ex(&[1, 2, 3]), ex(&[4, 5]), ex(&[6, 7, 8, 9])];
        let chunks = pack_chunks(&examples, 6, 0, true);
        // Examples within a chunk are EOS-separated; a chunk boundary is
        // already a separator, so no EOS opens the next chunk.
        assert_eq!(chunks, vec![vec![1, 2, 3, 0, 4, 5], vec![6, 7, 8, 9]]);
        assert_eq!(chunks[0].len(), 6);
        for c in &chunks {
            assert!(c.len() >= 2 && c.len() <= 6);
        }
    }

    #[test]
    fn packing_terminates_when_eos_fills_chunk_exactly() {
        // 5-token example into len-6 chunks: eos after it lands at index 5,
        // exactly filling the chunk — must not loop forever.
        let examples = vec![ex(&[1, 2, 3, 4, 5]), ex(&[6, 7, 8])];
        let chunks = pack_chunks(&examples, 6, 0, true);
        let flat: Vec<u32> = chunks.iter().flatten().copied().collect();
        assert_eq!(flat, vec![1, 2, 3, 4, 5, 0, 6, 7, 8]);
    }

    #[test]
    fn no_pack_truncates_per_example() {
        let examples = vec![ex(&[1, 2, 3, 4, 5]), ex(&[9])];
        let chunks = pack_chunks(&examples, 4, 0, false);
        assert_eq!(chunks, vec![vec![1, 2, 3, 4]]);
    }
}
