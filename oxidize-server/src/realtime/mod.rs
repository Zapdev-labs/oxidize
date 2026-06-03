//! OpenAI Realtime-compatible WebSocket API (text + best-effort tool calls).

pub mod protocol;
pub mod session;

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use axum::{
    extract::{
        State,
        ws::{Message, WebSocket, WebSocketUpgrade},
    },
    response::Response,
};
use futures_util::{SinkExt, StreamExt};
use serde_json::json;

use crate::app::AppState;
use crate::realtime::protocol::{ClientEvent, ServerEvent};
use crate::realtime::session::{RealtimeSession, RenderedMessage, parse_tool_call};
use crate::runtime::generate::{
    GenerationError, GenerationRequest, generate_text_streaming_blocking,
    generate_with_scheduler_streaming_blocking, render_chat_prompt,
};
use crate::schema::ChatMessageInput;

/// `GET /v1/realtime` — upgrade to a Realtime websocket session.
pub async fn realtime_handler(State(state): State<AppState>, ws: WebSocketUpgrade) -> Response {
    ws.on_upgrade(move |socket| handle_socket(socket, state))
}

async fn handle_socket(socket: WebSocket, state: AppState) {
    state.metrics.realtime_connections.inc();
    let (mut sink, mut stream) = socket.split();

    // Writer task: serialize ServerEvents to the socket.
    let (tx, mut rx) = tokio::sync::mpsc::channel::<ServerEvent>(256);
    let writer = tokio::spawn(async move {
        while let Some(event) = rx.recv().await {
            let text = serde_json::to_string(&event).unwrap_or_default();
            if sink.send(Message::Text(text.into())).await.is_err() {
                break;
            }
        }
    });

    let mut session = RealtimeSession::new();
    // Seed defaults from the loaded runtime, if any.
    if let Some(paged) = state.paged.as_ref() {
        session.config.temperature = Some(paged.runtime.defaults.temperature);
    } else if let Some(model) = state.model.as_ref() {
        session.config.temperature = Some(model.defaults.temperature);
    }

    let _ = tx
        .send(ServerEvent::SessionCreated {
            session: json!({ "model": runtime_id(&state) }),
        })
        .await;

    let mut in_flight: Option<(Arc<AtomicBool>, tokio::task::JoinHandle<()>)> = None;

    while let Some(Ok(message)) = stream.next().await {
        let text = match message {
            Message::Text(text) => text,
            Message::Close(_) => break,
            _ => continue,
        };
        let event: ClientEvent = match serde_json::from_str(&text) {
            Ok(event) => event,
            Err(error) => {
                let _ = tx
                    .send(ServerEvent::Error {
                        error: json!({ "type": "invalid_request_error", "message": error.to_string() }),
                    })
                    .await;
                continue;
            }
        };

        match event {
            ClientEvent::SessionUpdate { session: update } => {
                session.apply_session_update(update);
                let _ = tx
                    .send(ServerEvent::SessionUpdated {
                        session: json!({ "model": runtime_id(&state) }),
                    })
                    .await;
            }
            ClientEvent::ConversationItemCreate { item } => {
                if session.add_item(item) {
                    let _ = tx
                        .send(ServerEvent::ConversationItemCreated { item: json!({}) })
                        .await;
                } else {
                    let _ = tx
                        .send(ServerEvent::Error {
                            error: json!({ "type": "invalid_request_error", "message": "unsupported item" }),
                        })
                        .await;
                }
            }
            ClientEvent::ResponseCreate => {
                if in_flight.as_ref().is_some_and(|(_, handle)| !handle.is_finished()) {
                    let _ = tx
                        .send(ServerEvent::Error {
                            error: json!({ "type": "invalid_request_error", "message": "a response is already in progress" }),
                        })
                        .await;
                    continue;
                }
                let cancel = Arc::new(AtomicBool::new(false));
                let handle = spawn_response(&state, &session, tx.clone(), Arc::clone(&cancel));
                in_flight = Some((cancel, handle));
            }
            ClientEvent::ResponseCancel => {
                if let Some((cancel, _)) = in_flight.as_ref() {
                    cancel.store(true, Ordering::Relaxed);
                }
            }
        }
    }

    // Client disconnected: trip cancel and drop tasks.
    if let Some((cancel, _)) = in_flight.as_ref() {
        cancel.store(true, Ordering::Relaxed);
    }
    drop(tx);
    let _ = writer.await;
    state.metrics.realtime_connections.dec();
}

fn runtime_id(state: &AppState) -> String {
    state
        .paged
        .as_ref()
        .map(|paged| paged.runtime.id.clone())
        .or_else(|| state.model.as_ref().map(|model| model.id.clone()))
        .unwrap_or_else(|| "oxidize-default".to_owned())
}

/// Build a `GenerationRequest` and spawn the streaming generation, forwarding
/// deltas as `ServerEvent`s. Returns the join handle for the driving task.
fn spawn_response(
    state: &AppState,
    session: &RealtimeSession,
    tx: tokio::sync::mpsc::Sender<ServerEvent>,
    cancel: Arc<AtomicBool>,
) -> tokio::task::JoinHandle<()> {
    let messages = session.build_messages();
    let temperature = session.config.temperature;
    let max_tokens = session.config.max_tokens;
    let has_tools = !session.config.tools.is_empty();
    let state = state.clone();

    tokio::spawn(async move {
        let _ = tx
            .send(ServerEvent::ResponseCreated { response: json!({ "status": "in_progress" }) })
            .await;
        state.metrics.realtime_responses_total.inc();

        let prompt = render_prompt(&state, &messages);
        let request = GenerationRequest {
            prompt,
            max_tokens,
            temperature,
            top_p: None,
            top_k: None,
            min_p: None,
            typical_p: None,
            tail_free_z: None,
            stop: Vec::new(),
            seed: None,
            echo: false,
        };

        let (gen_tx, mut gen_rx) =
            tokio::sync::mpsc::channel::<Result<String, GenerationError>>(128);

        if let Some(paged) = state.paged.clone() {
            let cancel = Arc::clone(&cancel);
            tokio::task::spawn_blocking(move || {
                generate_with_scheduler_streaming_blocking(paged, request, gen_tx, cancel);
            });
        } else if let Some(model) = state.model.clone() {
            let cancel = Arc::clone(&cancel);
            tokio::task::spawn_blocking(move || {
                generate_text_streaming_blocking(model, request, gen_tx, cancel);
            });
        } else {
            let _ = tx
                .send(ServerEvent::Error {
                    error: json!({ "type": "server_error", "message": "no model loaded" }),
                })
                .await;
            return;
        }

        // Accumulate text; deltas are emitted live. Tool-call detection happens
        // on the full text at completion (best-effort).
        let mut full = String::new();
        while let Some(item) = gen_rx.recv().await {
            match item {
                Ok(piece) if piece.is_empty() => {} // terminal marker
                Ok(piece) => {
                    full.push_str(&piece);
                    let _ = tx
                        .send(ServerEvent::ResponseTextDelta { delta: piece })
                        .await;
                }
                Err(GenerationError::KvCacheExhausted) => {
                    let _ = tx
                        .send(ServerEvent::Error {
                            error: json!({ "type": "server_error", "code": "kv_cache_exhausted", "message": "KV cache exhausted" }),
                        })
                        .await;
                    return;
                }
                Err(error) => {
                    let _ = tx
                        .send(ServerEvent::Error {
                            error: json!({ "type": "server_error", "message": error.to_string() }),
                        })
                        .await;
                    return;
                }
            }
        }

        if has_tools && let Some(call) = parse_tool_call(&full) {
            let call_id = format!("call_{}", uuid::Uuid::new_v4().simple());
            let _ = tx
                .send(ServerEvent::ResponseFunctionCallArgumentsDelta {
                    call_id: call_id.clone(),
                    delta: call.arguments.clone(),
                })
                .await;
            let _ = tx
                .send(ServerEvent::ResponseFunctionCallArgumentsDone {
                    call_id,
                    name: call.name,
                    arguments: call.arguments,
                })
                .await;
        } else {
            let _ = tx
                .send(ServerEvent::ResponseTextDone { text: full })
                .await;
        }

        let _ = tx
            .send(ServerEvent::ResponseDone { response: json!({ "status": "completed" }) })
            .await;
    })
}

/// Render messages through the active runtime's chat template.
fn render_prompt(state: &AppState, messages: &[RenderedMessage]) -> String {
    let inputs: Vec<ChatMessageInput> = messages
        .iter()
        .map(|message| ChatMessageInput {
            role: message.role.clone(),
            content: message.content.clone(),
            images: None,
        })
        .collect();
    if let Some(paged) = state.paged.as_ref() {
        render_chat_prompt(&paged.runtime, &inputs)
    } else if let Some(model) = state.model.as_ref() {
        render_chat_prompt(model, &inputs)
    } else {
        inputs
            .iter()
            .map(|message| format!("{}: {}", message.role, message.content))
            .collect::<Vec<_>>()
            .join("\n")
    }
}
