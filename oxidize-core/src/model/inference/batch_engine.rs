//! Continuous-batching decode engine.
//!
//! Single-stream decode is memory-bound: each step streams every weight from
//! HBM to produce ONE token, so the projection matmuls run as bandwidth-bound
//! GEMVs with the tensor cores idle. The fix is to keep many in-flight sequences
//! and fan all of their per-step decode tokens into ONE
//! [`InferenceModel::forward_batch`] call — the weights are read once and reused
//! across `B` rows, turning each GEMV into a GEMM (tile-able, compute-amortized).
//! That is the structural lever behind vLLM/TensorRT-LLM throughput on a GPU,
//! and it amortizes weight I/O on the CPU path too.
//!
//! This engine is model-agnostic bookkeeping: it owns per-sequence KV
//! ([`SeqKv`]) and decode state, admits new requests up to a batch cap, prefills
//! each one, then issues a single batched decode per [`step`](ContinuousBatchEngine::step).
//! It never owns the model — the caller passes `&mut InferenceModel` per step, so
//! it slots behind the server's model lock without changing ownership.
//!
//! Token selection is injected as a closure (`select`), so greedy/argmax decode
//! (used in tests for bit-exact equivalence to standalone runs) and sampler-based
//! decode (used by the server) share the same loop.
//!
//! KV correctness is inherited from [`InferenceModel::forward_batch`], which
//! keeps each sequence's KV in its own contiguous [`SeqKv`] and uses that
//! sequence's own position for RoPE — so a sequence decoded inside a batch is
//! bit-identical to the same sequence decoded alone (proven in tests below and in
//! `inference.rs`). Prefill is currently per-sequence (one [`forward_batch`] call
//! per prompt token); only DECODE is batched. Batched/chunked prefill is a
//! follow-up — prefill is compute-bound, decode is where the memory-bound win is.

use crate::model::{ModelError, Token};

use super::{InferenceModel, SeqKv};

/// Opaque identifier handed back from [`ContinuousBatchEngine::submit`].
pub type SeqId = u64;

/// Configuration for a [`ContinuousBatchEngine`].
#[derive(Debug, Clone, Copy)]
pub struct BatchConfig {
    /// Maximum number of sequences decoded together in one batched step. This is
    /// the batch width `B` of the amortizing GEMM; larger `B` raises aggregate
    /// throughput until the matmul becomes compute-bound.
    pub max_batch: usize,
    /// Default KV capacity (positions) reserved per sequence. A submission whose
    /// `prompt + max_new` exceeds this is given a larger buffer automatically.
    pub default_capacity_tokens: usize,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            max_batch: 32,
            default_capacity_tokens: 2048,
        }
    }
}

/// One token produced for one sequence in a [`ContinuousBatchEngine::step`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StepOutput {
    pub seq_id: SeqId,
    pub token: Token,
    /// `true` if this token completed the sequence (hit `max_new` or emitted the
    /// stop token). The sequence is removed from the active set after this.
    pub finished: bool,
}

/// A request waiting to be admitted (prefilled) into the active decode set.
struct Pending {
    id: SeqId,
    prompt: Vec<Token>,
    max_new: usize,
    stop: Option<Token>,
}

/// Decode-time state for one admitted (already-prefilled) sequence. The KV lives
/// in the parallel `active_kv` vector at the same index so the active set can be
/// sliced directly into [`InferenceModel::forward_batch`].
struct ActiveMeta {
    id: SeqId,
    /// Absolute position of the NEXT token to decode (== KV length written).
    pos: usize,
    /// Most recently produced token, fed as the input of the next decode step.
    last: Token,
    /// Number of tokens generated so far (counts the prefill's first token).
    generated: usize,
    max_new: usize,
    stop: Option<Token>,
}

/// Continuous-batching decode driver around [`InferenceModel::forward_batch`].
///
/// Lifecycle: [`submit`](Self::submit) requests, then call [`step`](Self::step)
/// repeatedly while [`has_work`](Self::has_work) is true. Each step admits and
/// prefills waiting requests up to `max_batch`, then decodes every active
/// sequence in a single batched forward.
pub struct ContinuousBatchEngine {
    cfg: BatchConfig,
    kv_layers: usize,
    kv_len: usize,
    pending: std::collections::VecDeque<Pending>,
    active_meta: Vec<ActiveMeta>,
    active_kv: Vec<SeqKv>,
    next_id: SeqId,
}

impl ContinuousBatchEngine {
    /// Build an engine sized for `model`'s KV geometry. Reads only shape
    /// (`kv_layer_count`, `kv_row_len`); it does not borrow the model afterwards.
    pub fn new(model: &InferenceModel, cfg: BatchConfig) -> Self {
        Self {
            cfg,
            kv_layers: model.kv_layer_count(),
            kv_len: model.kv_row_len(),
            pending: std::collections::VecDeque::new(),
            active_meta: Vec::new(),
            active_kv: Vec::new(),
            next_id: 1,
        }
    }

    /// Queue a request. `prompt` is the full prompt token sequence, `max_new` the
    /// maximum number of tokens to generate, `stop` an optional stop token. The
    /// returned [`SeqId`] tags this sequence's [`StepOutput`]s.
    pub fn submit(&mut self, prompt: Vec<Token>, max_new: usize, stop: Option<Token>) -> SeqId {
        let id = self.next_id;
        self.next_id += 1;
        self.pending.push_back(Pending {
            id,
            prompt,
            max_new,
            stop,
        });
        id
    }

    /// Number of sequences currently being decoded (the live batch width).
    pub fn active_len(&self) -> usize {
        self.active_meta.len()
    }

    /// Number of requests admitted but is waiting in the queue.
    pub fn pending_len(&self) -> usize {
        self.pending.len()
    }

    /// `true` while any sequence is pending or actively decoding.
    pub fn has_work(&self) -> bool {
        !self.pending.is_empty() || !self.active_meta.is_empty()
    }

    /// Remove a sequence (pending or active), e.g. on client disconnect, freeing
    /// its batch slot for other work. Returns `true` if it was present.
    pub fn cancel(&mut self, id: SeqId) -> bool {
        if let Some(p) = self.pending.iter().position(|p| p.id == id) {
            self.pending.remove(p);
            return true;
        }
        if let Some(a) = self.active_meta.iter().position(|m| m.id == id) {
            self.active_meta.swap_remove(a);
            self.active_kv.swap_remove(a);
            return true;
        }
        false
    }

    /// Advance every in-flight sequence by one token.
    ///
    /// Order per step: (1) batched-decode the sequences already active (one
    /// [`forward_batch`](InferenceModel::forward_batch) over the whole active
    /// set), retiring any that finish; (2) admit waiting requests up to
    /// `max_batch`, prefilling each and emitting its first token. `select` maps a
    /// sequence's logits to its next token (e.g. argmax, or a sampler).
    ///
    /// Returns one [`StepOutput`] per token produced this step (decode tokens
    /// first, then newly-admitted sequences' first tokens).
    pub fn step<F>(
        &mut self,
        model: &mut InferenceModel,
        mut select: F,
    ) -> Result<Vec<StepOutput>, ModelError>
    where
        F: FnMut(SeqId, &[f32]) -> Token,
    {
        let mut outputs = Vec::new();
        self.decode_active(model, &mut select, &mut outputs)?;
        self.admit_and_prefill(model, &mut select, &mut outputs)?;
        Ok(outputs)
    }

    /// One batched decode over the whole active set, then retire finished rows.
    fn decode_active<F>(
        &mut self,
        model: &mut InferenceModel,
        select: &mut F,
        outputs: &mut Vec<StepOutput>,
    ) -> Result<(), ModelError>
    where
        F: FnMut(SeqId, &[f32]) -> Token,
    {
        if self.active_meta.is_empty() {
            return Ok(());
        }

        // The whole point: ALL active sequences' decode tokens go through ONE
        // forward_batch, so the weights stream from memory once for the batch.
        let rows: Vec<(Token, usize)> = self
            .active_meta
            .iter()
            .map(|m| (m.last, m.pos))
            .collect();
        // The CPU `forward_batch` is the engine's decode path. The device batched
        // forward (`forward_batch_gpu`) is intentionally NOT used here: it is
        // lockstep-only (every row must share one position/length) and keeps KV
        // device-side, whereas this engine multiplexes variable-length sequences
        // with host-resident `SeqKv` and prefills on CPU. Mixing the two against a
        // shared `SeqKv` would read an unwritten device KV prefix (silent garbage).
        // The GPU batched path is exercised by the lockstep throughput bench
        // (`batched_decode_bench` / `gpu-batched-tps`) instead.
        let logits = model.forward_batch(&rows, &mut self.active_kv, true)?;

        let mut finished_idx: Vec<usize> = Vec::new();
        for (i, meta) in self.active_meta.iter_mut().enumerate() {
            let token = select(meta.id, &logits[i]);
            meta.last = token;
            meta.pos += 1;
            meta.generated += 1;
            let finished =
                meta.generated >= meta.max_new || meta.stop.is_some_and(|s| s == token);
            outputs.push(StepOutput {
                seq_id: meta.id,
                token,
                finished,
            });
            if finished {
                finished_idx.push(i);
            }
        }

        // Retire finished rows. swap_remove keeps active_meta/active_kv aligned;
        // iterate high→low so earlier indices stay valid.
        for &i in finished_idx.iter().rev() {
            self.active_meta.swap_remove(i);
            self.active_kv.swap_remove(i);
        }
        Ok(())
    }

    /// Admit waiting requests up to the batch cap, prefilling each and emitting
    /// its first generated token.
    fn admit_and_prefill<F>(
        &mut self,
        model: &mut InferenceModel,
        select: &mut F,
        outputs: &mut Vec<StepOutput>,
    ) -> Result<(), ModelError>
    where
        F: FnMut(SeqId, &[f32]) -> Token,
    {
        while self.active_meta.len() < self.cfg.max_batch {
            let Some(req) = self.pending.pop_front() else {
                break;
            };
            if req.prompt.is_empty() || req.max_new == 0 {
                continue;
            }

            let cap = self
                .cfg
                .default_capacity_tokens
                .max(req.prompt.len() + req.max_new + 1);
            let mut kv = SeqKv::new(self.kv_layers, cap, self.kv_len);

            // Per-sequence prefill: feed each prompt token in order so the KV is
            // populated and the final token's logits seed the first output. Only
            // the last position needs logits.
            let mut pos = 0usize;
            let last_idx = req.prompt.len() - 1;
            let mut first_logits: Vec<f32> = Vec::new();
            for (i, &tok) in req.prompt.iter().enumerate() {
                let need_logits = i == last_idx;
                let out =
                    model.forward_batch(&[(tok, pos)], std::slice::from_mut(&mut kv), need_logits)?;
                if need_logits {
                    first_logits = out.into_iter().next().unwrap_or_default();
                }
                pos += 1;
            }

            let token = select(req.id, &first_logits);
            let generated = 1usize;
            let finished = generated >= req.max_new || req.stop.is_some_and(|s| s == token);
            outputs.push(StepOutput {
                seq_id: req.id,
                token,
                finished,
            });
            if finished {
                continue;
            }
            self.active_meta.push(ActiveMeta {
                id: req.id,
                pos,
                last: token,
                generated,
                max_new: req.max_new,
                stop: req.stop,
            });
            self.active_kv.push(kv);
        }
        Ok(())
    }
}
