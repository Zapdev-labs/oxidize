//! Wire types for the Realtime WebSocket protocol. Names are chosen to be
//! wire-compatible with OpenAI's Realtime text events so existing clients work.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// A tool/function the model may call. Wire-compatible subset.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct RealtimeTool {
    #[serde(rename = "type", default = "default_tool_type")]
    pub tool_type: String,
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub parameters: Option<Value>,
}

fn default_tool_type() -> String {
    "function".to_owned()
}

/// `tool_choice`: "auto" | "none" | "required". Defaults to auto.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum ToolChoice {
    #[default]
    Auto,
    None,
    Required,
}

/// Partial session config sent by `session.update` (all fields optional).
#[derive(Debug, Clone, Default, Deserialize)]
pub struct SessionUpdate {
    #[serde(default)]
    pub instructions: Option<String>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub max_response_output_tokens: Option<usize>,
    #[serde(default)]
    pub tools: Option<Vec<RealtimeTool>>,
    #[serde(default)]
    pub tool_choice: Option<ToolChoice>,
}

/// A single conversation item created by `conversation.item.create`.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct ConversationItemInput {
    #[serde(rename = "type")]
    pub item_type: String, // "message" | "function_call_output"
    #[serde(default)]
    pub role: Option<String>,
    #[serde(default)]
    pub content: Option<Vec<ContentPart>>,
    #[serde(default)]
    pub call_id: Option<String>,
    #[serde(default)]
    pub output: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct ContentPart {
    #[serde(rename = "type")]
    pub part_type: String, // "input_text" | "text"
    #[serde(default)]
    pub text: Option<String>,
}

/// Client → server events.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type")]
pub enum ClientEvent {
    #[serde(rename = "session.update")]
    SessionUpdate { session: SessionUpdate },
    #[serde(rename = "conversation.item.create")]
    ConversationItemCreate { item: ConversationItemInput },
    #[serde(rename = "response.create")]
    ResponseCreate,
    #[serde(rename = "response.cancel")]
    ResponseCancel,
}

/// Server → client events. Serialized with a `type` tag.
#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(tag = "type")]
pub enum ServerEvent {
    #[serde(rename = "session.created")]
    SessionCreated { session: Value },
    #[serde(rename = "session.updated")]
    SessionUpdated { session: Value },
    #[serde(rename = "conversation.item.created")]
    ConversationItemCreated { item: Value },
    #[serde(rename = "response.created")]
    ResponseCreated { response: Value },
    #[serde(rename = "response.output_item.added")]
    ResponseOutputItemAdded { item: Value },
    #[serde(rename = "response.text.delta")]
    ResponseTextDelta { delta: String },
    #[serde(rename = "response.text.done")]
    ResponseTextDone { text: String },
    #[serde(rename = "response.function_call_arguments.delta")]
    ResponseFunctionCallArgumentsDelta { call_id: String, delta: String },
    #[serde(rename = "response.function_call_arguments.done")]
    ResponseFunctionCallArgumentsDone { call_id: String, name: String, arguments: String },
    #[serde(rename = "response.done")]
    ResponseDone { response: Value },
    #[serde(rename = "error")]
    Error { error: Value },
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn deserializes_session_update_event() {
        let raw = json!({
            "type": "session.update",
            "session": { "instructions": "be terse", "temperature": 0.2 }
        });
        let event: ClientEvent = serde_json::from_value(raw).expect("valid event");
        match event {
            ClientEvent::SessionUpdate { session } => {
                assert_eq!(session.instructions.as_deref(), Some("be terse"));
                assert_eq!(session.temperature, Some(0.2));
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn deserializes_response_create_event() {
        let raw = json!({ "type": "response.create" });
        let event: ClientEvent = serde_json::from_value(raw).expect("valid event");
        assert!(matches!(event, ClientEvent::ResponseCreate));
    }

    #[test]
    fn serializes_text_delta_event_with_type_tag() {
        let event = ServerEvent::ResponseTextDelta { delta: "hi".to_owned() };
        let value = serde_json::to_value(&event).expect("serializable");
        assert_eq!(value["type"], "response.text.delta");
        assert_eq!(value["delta"], "hi");
    }

    #[test]
    fn tool_choice_defaults_to_auto() {
        assert_eq!(ToolChoice::default(), ToolChoice::Auto);
    }
}
