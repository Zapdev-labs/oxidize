//! `POST /v1/chat/completions` handler.

use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use axum::{
    Json,
    extract::State,
    http::StatusCode,
    response::{
        IntoResponse, Response,
        sse::{Event, KeepAlive, Sse},
    },
};
use futures_util::stream;
use serde_json::json;

use crate::app::AppState;
use crate::audit::AuditEvent;
use crate::routes::responses::{
    chat_completion_response, chat_completion_stream_response,
    chat_completion_stream_response_paged, generation_error_response, model_not_found,
    validate_candidate_count,
};
use crate::runtime::batched_engine::EngineSubmit;
use crate::runtime::generate::{
    GenerationError, GenerationRequest, GenerationResult, generate_text,
    generate_with_scheduler_blocking, generate_with_scheduler_streaming_blocking,
    render_chat_prompt, trim_stop_text,
};
use crate::schema::{ChatCompletionRequest, ResponseFormat, StopSequences};
use oxidize_core::tokenizer::EncodeOptions;

pub async fn chat_completions(
    State(state): State<AppState>,
    Json(payload): Json<ChatCompletionRequest>,
) -> Response {
    let start_time = std::time::Instant::now();
    let request_id = uuid::Uuid::new_v4().to_string();

    if let Some(response) = validate_candidate_count(payload.n, payload.best_of) {
        return response;
    }

    let model_id = payload.model.clone();
    let stream = payload.stream;

    // --- PagedAttention path ---
    if let Some(paged) = state.paged.clone() {
        if payload.model != paged.runtime.id {
            return model_not_found(&payload.model);
        }
        let prompt = render_chat_prompt(&paged.runtime, &payload.messages);
        let req = GenerationRequest {
            prompt,
            max_tokens: payload.max_tokens.or(payload.max_completion_tokens),
            temperature: payload.temperature,
            top_p: payload.top_p,
            top_k: payload.top_k,
            min_p: payload.min_p,
            typical_p: payload.typical_p,
            tail_free_z: payload.tail_free_z,
            stop: payload
                .stop
                .map(StopSequences::into_vec)
                .unwrap_or_default(),
            seed: payload.seed,
            echo: false,
        };

        if stream {
            let (tx, rx) = tokio::sync::mpsc::channel::<Result<String, GenerationError>>(128);
            let cancel = Arc::new(AtomicBool::new(false));
            // Continuous-batching engine: submit and let the shared engine thread
            // batch this request's decode with all other in-flight sequences.
            if let Some(engine) = paged.engine.clone() {
                engine.submit(EngineSubmit {
                    request: req,
                    out_tx: tx,
                });
                return chat_completion_stream_response_paged(model_id, rx, cancel);
            }
            let cancel_for_task = Arc::clone(&cancel);
            let paged_for_task = Arc::clone(&paged);
            tokio::task::spawn_blocking(move || {
                generate_with_scheduler_streaming_blocking(
                    paged_for_task,
                    req,
                    tx,
                    cancel_for_task,
                );
            });
            return chat_completion_stream_response_paged(model_id, rx, cancel);
        }

        let generated = if let Some(engine) = paged.engine.clone() {
            // Non-streaming over the batched engine: submit, then drain the token
            // channel to a full completion. One `Ok(piece)` per decoded token.
            let (tx, mut rx) =
                tokio::sync::mpsc::channel::<Result<String, GenerationError>>(128);
            let prompt = req.prompt.clone();
            let echo = req.echo;
            let stops = req.stop.clone();
            engine.submit(EngineSubmit {
                request: req,
                out_tx: tx,
            });
            let mut text = String::new();
            let mut completion = 0usize;
            let mut stream_err: Option<GenerationError> = None;
            while let Some(msg) = rx.recv().await {
                match msg {
                    Ok(piece) => {
                        text.push_str(&piece);
                        completion += 1;
                    }
                    Err(err) => {
                        stream_err = Some(err);
                        break;
                    }
                }
            }
            let result = match stream_err {
                Some(err) => Err(err),
                None => {
                    let prompt_tokens = paged
                        .runtime
                        .tokenizer
                        .encode_with_special_tokens(
                            &prompt,
                            EncodeOptions {
                                add_bos: paged.runtime.tokenizer.add_bos_default(),
                                add_eos: false,
                                pad_to: None,
                            },
                        )
                        .len();
                    let text = trim_stop_text(&text, &stops);
                    let text = if echo { format!("{prompt}{text}") } else { text };
                    Ok(GenerationResult {
                        text,
                        prompt_tokens,
                        completion_tokens: completion,
                    })
                }
            };
            Ok(result)
        } else {
            tokio::task::spawn_blocking(move || generate_with_scheduler_blocking(&paged, req))
                .await
                .map_err(|e| GenerationError::Other(format!("generation task failed: {e}")))
        };

        let _result = match generated {
            Ok(Ok(result)) => {
                let duration = start_time.elapsed();
                let event = AuditEvent::new(request_id.clone(), "generation_complete")
                    .with_model(&model_id)
                    .with_tokens(result.prompt_tokens, result.completion_tokens)
                    .with_duration(duration)
                    .with_temperature(payload.temperature)
                    .with_stop_reason("stop")
                    .with_streamed(stream);
                state.audit.log(event);
                state.metrics.record_tokens(
                    result.prompt_tokens,
                    result.completion_tokens,
                    duration.as_secs_f64(),
                );
                return chat_completion_response(model_id, result);
            }
            Ok(Err(err)) => {
                let duration = start_time.elapsed();
                let event = AuditEvent::new(request_id.clone(), "generation_error")
                    .with_model(&model_id)
                    .with_duration(duration)
                    .with_error(&err.to_string());
                state.audit.log(event);
                state.metrics.record_error("generation");
                return generation_error_response(err);
            }
            Err(err) => {
                let duration = start_time.elapsed();
                let event = AuditEvent::new(request_id.clone(), "generation_error")
                    .with_model(&model_id)
                    .with_duration(duration)
                    .with_error(&err.to_string());
                state.audit.log(event);
                state.metrics.record_error("task_panic");
                return generation_error_response(err);
            }
        };
    }

    // --- Sequential fallback path ---
    if let Some(runtime) = state.model.clone() {
        if payload.model != runtime.id {
            return model_not_found(&payload.model);
        }
        let prompt = render_chat_prompt(&runtime, &payload.messages);
        let generated = generate_text(
            runtime,
            GenerationRequest {
                prompt,
                max_tokens: payload.max_tokens.or(payload.max_completion_tokens),
                temperature: payload.temperature,
                top_p: payload.top_p,
                top_k: payload.top_k,
                min_p: payload.min_p,
                typical_p: payload.typical_p,
                tail_free_z: payload.tail_free_z,
                stop: payload
                    .stop
                    .map(StopSequences::into_vec)
                    .unwrap_or_default(),
                seed: payload.seed,
                echo: false,
            },
        )
        .await;
        return match generated {
            Ok(result) if stream => {
                let duration = start_time.elapsed();
                let event = AuditEvent::new(request_id.clone(), "generation_complete")
                    .with_model(&model_id)
                    .with_tokens(result.prompt_tokens, result.completion_tokens)
                    .with_duration(duration)
                    .with_temperature(payload.temperature)
                    .with_stop_reason("stop")
                    .with_streamed(true);
                state.audit.log(event);
                state.metrics.record_tokens(
                    result.prompt_tokens,
                    result.completion_tokens,
                    duration.as_secs_f64(),
                );
                chat_completion_stream_response(model_id, result.text)
            }
            Ok(result) => {
                let duration = start_time.elapsed();
                let event = AuditEvent::new(request_id.clone(), "generation_complete")
                    .with_model(&model_id)
                    .with_tokens(result.prompt_tokens, result.completion_tokens)
                    .with_duration(duration)
                    .with_temperature(payload.temperature)
                    .with_stop_reason("stop")
                    .with_streamed(false);
                state.audit.log(event);
                state.metrics.record_tokens(
                    result.prompt_tokens,
                    result.completion_tokens,
                    duration.as_secs_f64(),
                );
                chat_completion_response(model_id, result)
            }
            Err(error) => {
                let duration = start_time.elapsed();
                let event = AuditEvent::new(request_id.clone(), "generation_error")
                    .with_model(&model_id)
                    .with_duration(duration)
                    .with_error(&error.to_string());
                state.audit.log(event);
                state.metrics.record_error("generation");
                generation_error_response(error)
            }
        };
    }

    // --- No-model placeholder fallback ---
    let response_content = payload
        .response_format
        .as_ref()
        .map_or("", ResponseFormat::output_text);
    let response_content = if let Some(choices) = payload.guided_choice.as_ref()
        && let Some(choice) = choices.first()
    {
        choice.as_str()
    } else if payload.guided_json.is_some() || payload.json_schema.is_some() {
        "{}"
    } else if let Some(regex) = payload.guided_regex.as_ref() {
        regex.as_str()
    } else {
        response_content
    };

    if stream {
        let model = payload.model;
        let model_for_end = model.clone();
        let stream = stream::iter(vec![
            Ok::<Event, std::convert::Infallible>(
                Event::default().data(
                    json!({
                        "id": "chatcmpl-placeholder",
                        "object": "chat.completion.chunk",
                        "created": 0,
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": { "role": "assistant", "content": response_content },
                                "finish_reason": null
                            }
                        ]
                    })
                    .to_string(),
                ),
            ),
            Ok(Event::default().data(
                json!({
                    "id": "chatcmpl-placeholder",
                    "object": "chat.completion.chunk",
                    "created": 0,
                    "model": model_for_end,
                    "choices": [{ "index": 0, "delta": {}, "finish_reason": "stop" }]
                })
                .to_string(),
            )),
            Ok(Event::default().data("[DONE]")),
        ]);

        return Sse::new(stream)
            .keep_alive(KeepAlive::default())
            .into_response();
    }

    (
        StatusCode::OK,
        Json(json!({
            "id": "chatcmpl-placeholder",
            "object": "chat.completion",
            "created": 0,
            "model": payload.model,
            "choices": [
                {
                    "index": 0,
                    "message": { "role": "assistant", "content": response_content },
                    "finish_reason": "stop"
                }
            ],
            "usage": { "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0 }
        })),
    )
        .into_response()
}
