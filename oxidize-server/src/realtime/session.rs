//! Socket-independent realtime session state. Unit-tested directly.
//!
//! The transcript is bounded ([`MAX_SESSION_ITEMS`], [`MAX_TRANSCRIPT_BYTES`]):
//! a long-lived WebSocket is not covered by the HTTP body limit, so the oldest
//! items are evicted (and counted in `evicted_items`) instead of growing forever.
//!
//! Trust model: `session.update` instructions and tool schemas are rendered
//! verbatim into the system prompt ([`RealtimeSession::build_messages`]). That
//! is by design for local single-tenant inference, where the client already
//! controls every message. A multi-tenant proxy in front of this server must
//! not let one tenant supply instructions/tools for another tenant's session.

use crate::realtime::protocol::{ConversationItemInput, RealtimeTool, SessionUpdate, ToolChoice};

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

/// Maximum transcript items kept per session; older items are evicted.
pub const MAX_SESSION_ITEMS: usize = 1024;
/// Maximum transcript text bytes kept per session; older items are evicted.
pub const MAX_TRANSCRIPT_BYTES: usize = 8 * 1024 * 1024;

impl ConversationItem {
    fn byte_len(&self) -> usize {
        match self {
            Self::Message { role, text } => role.len() + text.len(),
            Self::FunctionCallOutput { call_id, output } => call_id.len() + output.len(),
        }
    }
}

/// The full session: config + transcript. No socket dependency.
#[derive(Debug)]
pub struct RealtimeSession {
    pub config: SessionConfig,
    pub items: Vec<ConversationItem>,
    /// Items dropped from the front of the transcript to honor the caps.
    pub evicted_items: usize,
    transcript_bytes: usize,
    max_items: usize,
    max_bytes: usize,
}

impl Default for RealtimeSession {
    fn default() -> Self {
        Self::with_limits(MAX_SESSION_ITEMS, MAX_TRANSCRIPT_BYTES)
    }
}

impl RealtimeSession {
    pub fn new() -> Self {
        Self::default()
    }

    /// Session with custom transcript caps (each clamped to at least 1).
    pub fn with_limits(max_items: usize, max_bytes: usize) -> Self {
        Self {
            config: SessionConfig::default(),
            items: Vec::new(),
            evicted_items: 0,
            transcript_bytes: 0,
            max_items: max_items.max(1),
            max_bytes: max_bytes.max(1),
        }
    }

    /// Total text bytes currently held in the transcript.
    pub fn transcript_bytes(&self) -> usize {
        self.transcript_bytes
    }

    /// Append an item, evicting the oldest until both caps hold. The newest
    /// item is always kept (truncation of a single oversized item is left to
    /// the generation path's context limit).
    pub fn push_item(&mut self, item: ConversationItem) {
        self.transcript_bytes += item.byte_len();
        self.items.push(item);
        let mut evict = 0;
        let mut bytes = self.transcript_bytes;
        while self.items.len() - evict > 1
            && (self.items.len() - evict > self.max_items || bytes > self.max_bytes)
        {
            bytes -= self.items[evict].byte_len();
            evict += 1;
        }
        if evict > 0 {
            self.items.drain(..evict);
            self.transcript_bytes = bytes;
            self.evicted_items += evict;
        }
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
                self.push_item(ConversationItem::Message { role, text });
                true
            }
            "function_call_output" => {
                let Some(call_id) = item.call_id else {
                    return false;
                };
                self.push_item(ConversationItem::FunctionCallOutput {
                    call_id,
                    output: item.output.unwrap_or_default(),
                });
                true
            }
            _ => false,
        }
    }
}

/// A rendered chat message ready for `render_chat_prompt`.
#[derive(Debug, Clone, PartialEq)]
pub struct RenderedMessage {
    pub role: String,
    pub content: String,
}

/// A successfully parsed tool call extracted from generated text.
#[derive(Debug, Clone, PartialEq)]
pub struct ParsedToolCall {
    pub name: String,
    pub arguments: String,
}

const TOOL_CALL_FENCE_OPEN: &str = "```tool_call";
const TOOL_CALL_FENCE_CLOSE: &str = "```";

impl RealtimeSession {
    /// Render the transcript into chat messages, injecting a system message that
    /// combines user instructions with a tool-call preamble (when tools exist).
    pub fn build_messages(&self) -> Vec<RenderedMessage> {
        let mut messages = Vec::new();
        let mut system = self.config.instructions.clone().unwrap_or_default();
        if !self.config.tools.is_empty() {
            if !system.is_empty() {
                system.push_str("\n\n");
            }
            system.push_str(&tool_preamble(&self.config.tools));
        }
        if !system.is_empty() {
            messages.push(RenderedMessage {
                role: "system".to_owned(),
                content: system,
            });
        }
        for item in &self.items {
            match item {
                ConversationItem::Message { role, text } => messages.push(RenderedMessage {
                    role: role.clone(),
                    content: text.clone(),
                }),
                ConversationItem::FunctionCallOutput { call_id, output } => {
                    messages.push(RenderedMessage {
                        role: "tool".to_owned(),
                        content: format!("[tool result for {call_id}]: {output}"),
                    });
                }
            }
        }
        messages
    }
}

fn tool_preamble(tools: &[RealtimeTool]) -> String {
    let schemas = serde_json::to_string_pretty(tools).unwrap_or_else(|_| "[]".to_owned());
    format!(
        "You can call the following tools. When you decide to call a tool, emit \
         EXACTLY one fenced block and nothing else:\n\
         ```tool_call\n{{\"name\": \"<tool name>\", \"arguments\": {{ ... }}}}\n```\n\
         Available tools (JSON schemas):\n{schemas}"
    )
}

/// Scan generated text for a `tool_call` fenced block. Returns the call on hit.
pub fn parse_tool_call(text: &str) -> Option<ParsedToolCall> {
    let start = text.find(TOOL_CALL_FENCE_OPEN)? + TOOL_CALL_FENCE_OPEN.len();
    let rest = &text[start..];
    let end = rest.find(TOOL_CALL_FENCE_CLOSE)?;
    let body = rest[..end].trim();
    let value: serde_json::Value = serde_json::from_str(body).ok()?;
    let name = value.get("name")?.as_str()?.to_owned();
    let arguments = value
        .get("arguments")
        .map(|args| {
            if args.is_string() {
                args.as_str().unwrap_or_default().to_owned()
            } else {
                args.to_string()
            }
        })
        .unwrap_or_else(|| "{}".to_owned());
    Some(ParsedToolCall { name, arguments })
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

    #[test]
    fn build_prompt_includes_tool_preamble_when_tools_present() {
        let mut session = RealtimeSession::new();
        session.config.instructions = Some("be helpful".to_owned());
        session.config.tools = vec![RealtimeTool {
            tool_type: "function".to_owned(),
            name: "get_weather".to_owned(),
            description: Some("Get weather".to_owned()),
            parameters: Some(serde_json::json!({"type": "object"})),
        }];
        session.push_item(ConversationItem::Message {
            role: "user".to_owned(),
            text: "weather?".to_owned(),
        });
        let messages = session.build_messages();
        let system = &messages[0];
        assert_eq!(system.role, "system");
        assert!(system.content.contains("be helpful"));
        assert!(system.content.contains("get_weather"));
        assert!(system.content.contains("tool_call"));
        // user turn preserved
        assert!(
            messages
                .iter()
                .any(|m| m.role == "user" && m.content == "weather?")
        );
    }

    fn user_message(text: &str) -> ConversationItemInput {
        ConversationItemInput {
            item_type: "message".to_owned(),
            role: Some("user".to_owned()),
            content: Some(vec![ContentPart {
                part_type: "input_text".to_owned(),
                text: Some(text.to_owned()),
            }]),
            call_id: None,
            output: None,
        }
    }

    /// server-006: N > cap appends keep item count and bytes bounded.
    #[test]
    fn transcript_is_bounded_by_item_cap() {
        let mut session = RealtimeSession::with_limits(8, usize::MAX);
        for i in 0..1000 {
            assert!(session.add_item(user_message(&format!("m{i}"))));
            assert!(session.items.len() <= 8);
        }
        assert_eq!(session.items.len(), 8);
        assert_eq!(session.evicted_items, 992);
        assert_eq!(
            session.items.last(),
            Some(&ConversationItem::Message {
                role: "user".to_owned(),
                text: "m999".to_owned()
            })
        );
    }

    #[test]
    fn transcript_is_bounded_by_byte_cap() {
        let mut session = RealtimeSession::with_limits(usize::MAX, 1000);
        let chunk = "x".repeat(96); // 100 bytes with role "user"
        for _ in 0..500 {
            session.add_item(user_message(&chunk));
            assert!(session.transcript_bytes() <= 1000);
            let actual: usize = session.items.iter().map(ConversationItem::byte_len).sum();
            assert_eq!(actual, session.transcript_bytes());
        }
        assert_eq!(session.items.len(), 10);
        assert_eq!(session.evicted_items, 490);
    }

    #[test]
    fn oversized_single_item_is_kept_alone() {
        let mut session = RealtimeSession::with_limits(16, 10);
        session.add_item(user_message("short"));
        session.add_item(user_message(&"y".repeat(64)));
        assert_eq!(session.items.len(), 1);
        assert_eq!(session.evicted_items, 1);
    }

    #[test]
    fn parse_tool_call_hit() {
        let text =
            "sure\n```tool_call\n{\"name\":\"get_weather\",\"arguments\":{\"city\":\"SF\"}}\n```";
        let parsed = parse_tool_call(text).expect("should parse");
        assert_eq!(parsed.name, "get_weather");
        assert_eq!(parsed.arguments, "{\"city\":\"SF\"}");
    }

    #[test]
    fn parse_tool_call_miss_returns_none() {
        assert!(parse_tool_call("just plain text").is_none());
    }
}
