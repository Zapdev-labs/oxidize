//! Shared continuous-batching engine for the server.
//!
//! The legacy paged path ([`generate_with_scheduler_blocking`] and its streaming
//! sibling) spawns one blocking task per request and holds the model lock for
//! that request's ENTIRE generation — so concurrent requests serialize and the
//! GPU/CPU runs one decode token at a time (a bandwidth-bound GEMV). This module
//! replaces that with a single long-lived engine thread: every request submits
//! into a shared queue, and each engine step fans ALL in-flight sequences'
//! decode tokens into ONE [`ContinuousBatchEngine::step`] (i.e. one
//! `forward_batch`), so the weights stream from memory once per step and the
//! matmuls become amortized GEMMs. That is the structural throughput lever.
//!
//! Output reuses the existing paged SSE contract: each request gets a
//! [`tokio::sync::mpsc`] sender; the engine sends one `Ok(piece)` per decoded
//! token and drops the sender on completion (the responder reads channel-close as
//! end-of-stream). On send failure (client disconnect / full channel) the engine
//! cancels that sequence, freeing its slot for others.
//!
//! Enabled when `OX_BATCHED_DECODE` is set AND the loaded model is the
//! [`InferenceModel`] backend (the only one with a batched `forward_batch`).
//! `OX_BATCH_MAX` overrides the maximum batch width (default 32).

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::mpsc;

use oxidize_core::inference::{BatchConfig, ContinuousBatchEngine, SeqId};
use oxidize_core::model::Token;
use oxidize_core::sampling::{SamplingConfig, sample};
use oxidize_core::tokenizer::EncodeOptions;
use rand::{Rng, SeedableRng, rngs::StdRng};

use crate::runtime::generate::{GenerationError, GenerationRequest};
use crate::runtime::model::ModelRuntime;

/// A request handed to the engine thread plus the channel to stream tokens back.
pub struct EngineSubmit {
    pub request: GenerationRequest,
    pub out_tx: tokio::sync::mpsc::Sender<Result<String, GenerationError>>,
}

/// Handle to the shared engine thread. Cloneable senders let any number of
/// request handlers submit concurrently.
#[derive(Clone)]
pub struct BatchedEngineHandle {
    submit_tx: mpsc::SyncSender<EngineSubmit>,
}

impl BatchedEngineHandle {
    /// Submit a request; returns `false` if the engine thread has stopped.
    pub fn submit(&self, req: EngineSubmit) -> bool {
        self.submit_tx.try_send(req).is_ok()
    }

    /// Spawn the engine thread for `runtime`. Returns `None` when batched decode
    /// is disabled or the model backend does not support it.
    pub fn spawn_if_enabled(runtime: Arc<ModelRuntime>) -> Option<Self> {
        if std::env::var("OX_BATCHED_DECODE").is_err() {
            return None;
        }
        // Only the Inference backend exposes forward_batch.
        let (supported, ctx) = {
            let mut guard = runtime.model.blocking_lock();
            match guard.as_inference_mut() {
                Some(inf) => (true, inf.config().context_size),
                None => (false, 0),
            }
        };
        if !supported {
            tracing::warn!(
                "OX_BATCHED_DECODE set but model backend has no batched forward; \
                 continuous batching disabled"
            );
            return None;
        }

        let max_batch = std::env::var("OX_BATCH_MAX")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&v| v > 0)
            .unwrap_or(32);
        let cfg = BatchConfig {
            max_batch,
            default_capacity_tokens: ctx.max(256),
        };

        let (submit_tx, submit_rx) = mpsc::sync_channel::<EngineSubmit>(256);
        let thread_runtime = Arc::clone(&runtime);
        let spawned = std::thread::Builder::new()
            .name("oxidize-batch-engine".to_string())
            .spawn(move || engine_loop(thread_runtime, cfg, &submit_rx));
        match spawned {
            Ok(_join) => {
                tracing::info!(max_batch, "continuous-batching engine started");
                Some(Self { submit_tx })
            }
            Err(error) => {
                tracing::error!(%error, "failed to start batched-decode engine thread");
                None
            }
        }
    }
}

/// Per-sequence decode state held by the engine thread.
struct SeqState {
    out_tx: tokio::sync::mpsc::Sender<Result<String, GenerationError>>,
    sampling: SamplingConfig,
    rng: StdRng,
}

fn argmax(v: &[f32]) -> Token {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > best_v {
            best_v = x;
            best = i;
        }
    }
    best as Token
}

/// Sample one token for sequence `sid` using its own sampler + RNG. Falls back to
/// greedy argmax if the sequence state is missing or sampling errors.
fn select_token(states: &mut BTreeMap<SeqId, SeqState>, sid: SeqId, logits: &[f32]) -> Token {
    if let Some(st) = states.get_mut(&sid) {
        let r = st.rng.r#gen::<f32>();
        match sample(logits, st.sampling, r) {
            Ok(t) => t,
            Err(_) => argmax(logits),
        }
    } else {
        argmax(logits)
    }
}

/// Resolve a submission against runtime defaults and admit it into the engine.
fn admit(
    runtime: &ModelRuntime,
    engine: &mut ContinuousBatchEngine,
    states: &mut BTreeMap<SeqId, SeqState>,
    submit: EngineSubmit,
) {
    let request = submit.request;
    let prompt_tokens = runtime.tokenizer.encode_with_special_tokens(
        &request.prompt,
        EncodeOptions {
            add_bos: runtime.tokenizer.add_bos_default(),
            add_eos: false,
            pad_to: None,
        },
    );
    if prompt_tokens.is_empty() {
        let _ = submit
            .out_tx
            .blocking_send(Err(GenerationError::Other("empty prompt".to_string())));
        return;
    }

    let max_new = request
        .max_tokens
        .unwrap_or(runtime.defaults.max_tokens)
        .max(1);
    let temperature = request.temperature.unwrap_or(runtime.defaults.temperature);
    let sampling = SamplingConfig {
        temperature,
        top_p: request.top_p.or(runtime.defaults.top_p),
        top_k: request.top_k.or(runtime.defaults.top_k),
        min_p: request.min_p,
        typical_p: request.typical_p,
        tail_free_z: request.tail_free_z,
        ..SamplingConfig::default()
    };
    let stop_token = runtime.tokenizer.special_tokens().eos;
    let rng = match request.seed {
        Some(seed) => StdRng::seed_from_u64(seed),
        None => StdRng::from_entropy(),
    };

    let id = engine.submit(prompt_tokens, max_new, stop_token);
    states.insert(
        id,
        SeqState {
            out_tx: submit.out_tx,
            sampling,
            rng,
        },
    );
}

/// Deliver one produced token to its sequence's stream. Removes the sequence on
/// completion or on a dead output channel (cancelling it in the engine).
fn route_output(
    engine: &mut ContinuousBatchEngine,
    states: &mut BTreeMap<SeqId, SeqState>,
    runtime: &ModelRuntime,
    out: oxidize_core::inference::StepOutput,
) {
    let Some(st) = states.get(&out.seq_id) else {
        return;
    };
    let piece = runtime.tokenizer.decode(&[out.token]).unwrap_or_default();
    let send_ok = match st.out_tx.try_send(Ok(piece)) {
        Ok(()) => true,
        Err(tokio::sync::mpsc::error::TrySendError::Full(_))
        | Err(tokio::sync::mpsc::error::TrySendError::Closed(_)) => false,
    };
    if !send_ok {
        // Client gone — reclaim the slot so other sequences keep the batch full.
        engine.cancel(out.seq_id);
        states.remove(&out.seq_id);
    } else if out.finished {
        // Dropping the sender closes the channel → responder emits the stop chunk.
        states.remove(&out.seq_id);
    }
}

/// Engine thread main loop: admit queued requests, then issue one batched decode
/// step over all active sequences, repeating until idle (then block for work).
fn engine_loop(
    runtime: Arc<ModelRuntime>,
    cfg: BatchConfig,
    submit_rx: &mpsc::Receiver<EngineSubmit>,
) {
    // The engine is sized from the model's KV geometry, so build it under the
    // lock once (the backend was already verified to be Inference).
    let mut engine = {
        let mut guard = runtime.model.blocking_lock();
        match guard.as_inference_mut() {
            Some(inf) => ContinuousBatchEngine::new(inf, cfg),
            None => return,
        }
    };
    let mut states: BTreeMap<SeqId, SeqState> = BTreeMap::new();

    loop {
        // When idle, block until a request arrives; exit if all senders dropped.
        if !engine.has_work() {
            match submit_rx.recv() {
                Ok(req) => admit(&runtime, &mut engine, &mut states, req),
                Err(_) => return,
            }
        }
        // Pull in any other queued requests so they join this batch.
        while let Ok(req) = submit_rx.try_recv() {
            admit(&runtime, &mut engine, &mut states, req);
        }
        if !engine.has_work() {
            continue;
        }

        // One batched forward over every active sequence, holding the lock only
        // for the step itself.
        let step = {
            let mut guard = runtime.model.blocking_lock();
            match guard.as_inference_mut() {
                Some(inf) => engine.step(inf, |sid, logits| select_token(&mut states, sid, logits)),
                None => return,
            }
        };
        match step {
            Ok(outputs) => {
                for out in outputs {
                    route_output(&mut engine, &mut states, &runtime, out);
                }
            }
            Err(error) => {
                let msg = format!("batched forward failed: {error:?}");
                for st in states.values() {
                    let _ = st
                        .out_tx
                        .blocking_send(Err(GenerationError::Other(msg.clone())));
                }
                states.clear();
                // Active sequences are now invalid; drop them and resume serving.
                engine = {
                    let mut guard = runtime.model.blocking_lock();
                    match guard.as_inference_mut() {
                        Some(inf) => ContinuousBatchEngine::new(inf, cfg),
                        None => return,
                    }
                };
            }
        }
    }
}
