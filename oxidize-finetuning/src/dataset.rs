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
