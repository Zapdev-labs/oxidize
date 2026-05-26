//! OpenAI-compatible request/response schemas.

use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessageInput>,
    #[serde(default)]
    pub response_format: Option<ResponseFormat>,
    #[serde(default)]
    pub guided_json: Option<Value>,
    #[serde(default)]
    pub json_schema: Option<Value>,
    #[serde(default)]
    pub guided_regex: Option<String>,
    #[serde(default)]
    pub guided_choice: Option<Vec<String>>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub max_completion_tokens: Option<usize>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub min_p: Option<f32>,
    #[serde(default)]
    pub typical_p: Option<f32>,
    #[serde(default)]
    pub tail_free_z: Option<f32>,
    #[serde(default)]
    pub stop: Option<StopSequences>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub n: Option<usize>,
    #[serde(default)]
    pub best_of: Option<usize>,
}

#[derive(Debug, Deserialize)]
pub struct ChatMessageInput {
    pub role: String,
    pub content: String,
    #[serde(default)]
    pub images: Option<Vec<String>>,
}

#[derive(Debug, Deserialize)]
pub struct CompletionRequest {
    pub model: String,
    pub prompt: String,
    #[serde(default)]
    pub response_format: Option<ResponseFormat>,
    #[serde(default)]
    pub guided_json: Option<Value>,
    #[serde(default)]
    pub json_schema: Option<Value>,
    #[serde(default)]
    pub guided_regex: Option<String>,
    #[serde(default)]
    pub guided_choice: Option<Vec<String>>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub min_p: Option<f32>,
    #[serde(default)]
    pub typical_p: Option<f32>,
    #[serde(default)]
    pub tail_free_z: Option<f32>,
    #[serde(default)]
    pub stop: Option<StopSequences>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub echo: bool,
    #[serde(default)]
    pub n: Option<usize>,
    #[serde(default)]
    pub best_of: Option<usize>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum StopSequences {
    One(String),
    Many(Vec<String>),
}

impl StopSequences {
    pub fn into_vec(self) -> Vec<String> {
        match self {
            Self::One(value) => vec![value],
            Self::Many(values) => values,
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseFormat {
    Text,
    JsonObject,
    JsonSchema { json_schema: Value },
}

impl ResponseFormat {
    pub fn output_text(&self) -> &'static str {
        match self {
            Self::Text => "",
            Self::JsonObject => "{}",
            Self::JsonSchema { json_schema } => {
                let _ = json_schema;
                "{}"
            }
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct EmbeddingsRequest {
    pub model: String,
    pub input: Value,
}

#[derive(Debug, Serialize)]
pub struct ModelsResponse {
    pub object: &'static str,
    pub data: Vec<ModelData>,
}

#[derive(Debug, Serialize)]
pub struct ModelData {
    pub id: String,
    pub object: &'static str,
    pub owned_by: &'static str,
}
