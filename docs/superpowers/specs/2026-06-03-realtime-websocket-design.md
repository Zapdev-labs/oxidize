# Realtime WebSocket API — Design

**Date:** 2026-06-03
**Crate:** `oxidize-server`
**Status:** Approved for planning

## Goal

Add an OpenAI Realtime-style WebSocket endpoint to `oxidize-server`, supporting
**text streaming and function calling**. Audio (STT/TTS/VAD) is explicitly out of
scope — the engine is text-only — but the event protocol uses wire-compatible
names so existing Realtime text clients work without modification.

## Constraints (from the existing codebase)

- Server is **axum 0.8**, OpenAI-compatible REST, currently streaming over **SSE**.
- Streaming today is built on `mpsc::Sender<Result<String, GenerationError>>` plus
  an `Arc<AtomicBool>` cancel flag, driven by
  `generate_with_scheduler_streaming_blocking` (paged path only).
- **No tool/function support** exists anywhere (`schema.rs`, generation path).
- **No streaming sequential path** — only `generate_text` (blocking) exists for the
  non-paged runtime.
- Auth, request limits, audit, and metrics are axum middleware layers; a WS upgrade
  route slots into the existing router behind them.

## Architecture (Approach A: stateful per-connection actor)

New route `GET /v1/realtime` using axum's `WebSocketUpgrade` (requires enabling
axum's `ws` feature). It sits behind the same auth/limits/audit/metrics middleware
as the REST routes.

- **Auth:** API key via header, with `?api_key=` query-param fallback (browsers
  cannot set custom headers on a WebSocket). Auth failure closes the socket with a
  policy-violation close code.
- **On upgrade:** spawn one connection task. Split the socket into:
  - a **reader** loop deserializing client events into the session, and
  - a **writer** half fed by an `mpsc::Sender<ServerEvent>`, so generation deltas
    and event acks never block each other.

### `RealtimeSession` (socket-independent, unit-testable)

Holds:
- `config`: model, instructions, temperature, max_tokens, tools, tool_choice.
- `items: Vec<ConversationItem>` — the running transcript.
- `in_flight: Option<InFlight>` — current response's cancel `AtomicBool` + join handle.

Pure methods (no socket dependency, tested directly):
- `apply_session_update(...)` — merge partial session config.
- `add_item(item)` — append a conversation item.
- `build_prompt()` — render transcript + tool-instruction preamble through the
  existing chat template via `render_chat_prompt`.
- `take_cancel()` — hand out / trip the cancel flag.

### Generation runtime selection

- Prefer the **paged** scheduler path (`generate_with_scheduler_streaming_blocking`).
- **Fall back to sequential** when paged runtime is absent. This requires a **new
  streaming variant of the sequential path** (`generate_text_streaming_blocking`)
  mirroring the paged channel/cancel contract, so `RealtimeSession` drives either
  runtime through one `mpsc` + `AtomicBool` interface.

## Event Protocol (subset, wire-compatible names)

**Client → server:**
- `session.update`
- `conversation.item.create`
- `response.create`
- `response.cancel`

**Server → client:**
- `session.created`, `session.updated`
- `conversation.item.created`
- `response.created`, `response.output_item.added`
- `response.text.delta`, `response.text.done`
- `response.function_call_arguments.delta`, `response.function_call_arguments.done`
- `response.done`
- `error`

Flow: each `response.create` builds a prompt and runs the selected streaming
generation. Each token → `response.text.delta`; completion → `response.text.done`
+ `response.done`. `response.cancel` trips the in-flight cancel `AtomicBool`.

## Function Calling

- Add `tools` + `tool_choice` to a realtime-local request type (not the shared REST
  `ChatCompletionRequest`).
- `build_prompt()` injects a tool-instruction preamble (the tool JSON schemas +
  an instruction to emit a tool call as a fenced JSON block / sentinel) into the
  system prompt.
- Generated output is scanned for the tool-call marker:
  - **Hit:** emit `response.function_call_arguments.delta/done` and a
    `conversation.item` of type `function_call` instead of text.
  - **Miss:** emit as plain text deltas.
- The tool result returns from the client as `conversation.item.create` with type
  `function_call_output`, appended to the transcript for the next turn.
- **Limitation (explicit):** this is best-effort, model-dependent output parsing.
  Models not trained for tool calling may not emit the marker reliably. No
  constrained decoding is wired in for v1.

## Errors & Lifecycle

- Failures map to a Realtime `error` event (`{type, code, message}`) and keep the
  socket open — except auth failure, which closes with a policy code.
- KV-cache exhaustion → `error` with code `kv_cache_exhausted`.
- Client disconnect trips the cancel flag and drops the session task.
- Metrics: active realtime connections gauge + responses counter.

## Testing

- **Unit** (`#[cfg(test)]` co-located, per project convention): `session.update`
  merge, prompt building with/without tools, tool-call parse hit/miss, cancel
  mid-flight, runtime fallback selection.
- **Integration:** a real WS client (`tokio-tungstenite`, dev-dependency) driving
  connect → `response.create` → deltas → `response.done`, plus a function-call
  round trip.

## Out of Scope

- Audio: `input_audio_buffer`, audio deltas, VAD/turn detection.
- Constrained/guided decoding for tool calls.
- Multi-response concurrency on a single connection (one in-flight response at a time).
