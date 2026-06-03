//! Socket-independent realtime session state. Unit-tested directly.

use crate::realtime::protocol::{
    ConversationItemInput, RealtimeTool, SessionUpdate, ToolChoice,
};

/// Mutable per-connection config, seeded from server defaults.
#[derive(Debug, Clone, Default)]
pub struct SessionConfig {
    pub instructions: Option<String>,
    pub temperature: Option<f32>,
    pub max_tokens: Option<usize>,
    pub tools: Vec<RealtimeTool>,
    pub tool_choice: ToolChoice,
}

/// A turn in the running transcript.
#[derive(Debug, Clone, PartialEq)]
pub enum ConversationItem {
    Message { role: String, text: String },
    FunctionCallOutput { call_id: String, output: String },
}

/// The full session: config + transcript. No socket dependency.
#[derive(Debug, Default)]
pub struct RealtimeSession {
    pub config: SessionConfig,
    pub items: Vec<ConversationItem>,
}

impl RealtimeSession {
    pub fn new() -> Self {
        Self::default()
    }

    /// Merge a partial `session.update` into the config. Only present fields
    /// override; absent fields are left unchanged.
    pub fn apply_session_update(&mut self, update: SessionUpdate) {
        if let Some(instructions) = update.instructions {
            self.config.instructions = Some(instructions);
        }
        if let Some(temperature) = update.temperature {
            self.config.temperature = Some(temperature);
        }
        if let Some(max_tokens) = update.max_response_output_tokens {
            self.config.max_tokens = Some(max_tokens);
        }
        if let Some(tools) = update.tools {
            self.config.tools = tools;
        }
        if let Some(tool_choice) = update.tool_choice {
            self.config.tool_choice = tool_choice;
        }
    }

    /// Append a conversation item from a client `conversation.item.create`.
    /// Returns false if the item could not be interpreted.
    pub fn add_item(&mut self, item: ConversationItemInput) -> bool {
        match item.item_type.as_str() {
            "message" => {
                let role = item.role.unwrap_or_else(|| "user".to_owned());
                let text = item
                    .content
                    .unwrap_or_default()
                    .into_iter()
                    .filter_map(|part| part.text)
                    .collect::<Vec<_>>()
                    .join("");
                self.items.push(ConversationItem::Message { role, text });
                true
            }
            "function_call_output" => {
                let Some(call_id) = item.call_id else {
                    return false;
                };
                self.items.push(ConversationItem::FunctionCallOutput {
                    call_id,
                    output: item.output.unwrap_or_default(),
                });
                true
            }
            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::realtime::protocol::ContentPart;

    #[test]
    fn session_update_merges_only_present_fields() {
        let mut session = RealtimeSession::new();
        session.config.temperature = Some(0.9);
        session.apply_session_update(SessionUpdate {
            instructions: Some("be brief".to_owned()),
            temperature: None,
            max_response_output_tokens: Some(64),
            tools: None,
            tool_choice: None,
        });
        assert_eq!(session.config.instructions.as_deref(), Some("be brief"));
        assert_eq!(session.config.temperature, Some(0.9)); // unchanged
        assert_eq!(session.config.max_tokens, Some(64));
    }

    #[test]
    fn add_message_item_appends_text() {
        let mut session = RealtimeSession::new();
        let ok = session.add_item(ConversationItemInput {
            item_type: "message".to_owned(),
            role: Some("user".to_owned()),
            content: Some(vec![ContentPart {
                part_type: "input_text".to_owned(),
                text: Some("hello".to_owned()),
            }]),
            call_id: None,
            output: None,
        });
        assert!(ok);
        assert_eq!(
            session.items,
            vec![ConversationItem::Message {
                role: "user".to_owned(),
                text: "hello".to_owned()
            }]
        );
    }

    #[test]
    fn add_function_call_output_requires_call_id() {
        let mut session = RealtimeSession::new();
        let ok = session.add_item(ConversationItemInput {
            item_type: "function_call_output".to_owned(),
            role: None,
            content: None,
            call_id: None,
            output: Some("{}".to_owned()),
        });
        assert!(!ok);
        assert!(session.items.is_empty());
    }
}
