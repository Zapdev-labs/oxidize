//! C-compatible FFI over oxidize-core.
//!
//! Exposes:
//!   - oxidize_gemv_quantized  — fused AVX2+FMA quantized GEMV
//!   - oxidize_model_load      — load a GGUF model from disk
//!   - oxidize_model_forward   — run one decode step, return logits
//!   - oxidize_model_free      — free the model handle
//!   - oxidize_session_new / oxidize_session_free
//!   - oxidize_ffi_version

use oxidize_core::gguf::{GgufQuantizationType, load_mapped_gguf};
use oxidize_core::inference::{InferenceConfig, InferenceModel};
use oxidize_core::model::{Model, Session};
use oxidize_core::tensor;
use std::ffi::{CStr, c_char};
use std::sync::Once;

static RAYON_INIT: Once = Once::new();

/// Configure the Rayon global thread pool once.
/// On hyperthreaded machines, using physical-core count beats logical-core count
/// for memory-bandwidth-bound GEMV (measured: 15 vs 14 tok/s on Ryzen 7 PRO 6850H).
fn init_thread_pool() {
    RAYON_INIT.call_once(|| {
        // Allow runtime override for benchmarking different thread counts.
        let threads = if let Ok(s) = std::env::var("OXIDIZE_THREADS") {
            s.parse::<usize>().unwrap_or(0)
        } else {
            0
        };
        let threads = if threads > 0 {
            threads
        } else {
            let logical = std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(8);
            // Heuristic: halve on HT machines (logical divisible by 2 and > 4),
            // clamp to [4, 12] for memory-bandwidth efficiency.
            if logical > 4 && logical.is_multiple_of(2) {
                (logical / 2).clamp(4, 12)
            } else {
                logical.clamp(4, 12)
            }
        };
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
            .ok();
    });
}

static VERSION: &[u8] = b"0.1.0\0";

#[unsafe(no_mangle)]
pub extern "C" fn oxidize_ffi_version() -> *const c_char {
    VERSION.as_ptr().cast()
}

fn to_gguf_type(t: u32) -> Option<GgufQuantizationType> {
    match t {
        0 => Some(GgufQuantizationType::F32),
        1 => Some(GgufQuantizationType::F16),
        2 => Some(GgufQuantizationType::Q4_0),
        3 => Some(GgufQuantizationType::Q4_1),
        6 => Some(GgufQuantizationType::Q8_0),
        7 => Some(GgufQuantizationType::Q2_K),
        8 => Some(GgufQuantizationType::Q3_K_S),
        9 => Some(GgufQuantizationType::Q3_K_M),
        10 => Some(GgufQuantizationType::Q3_K_L),
        11 => Some(GgufQuantizationType::Q4_K_S),
        12 => Some(GgufQuantizationType::Q4_K_M),
        13 => Some(GgufQuantizationType::Q5_K_S),
        14 => Some(GgufQuantizationType::Q5_K_M),
        15 => Some(GgufQuantizationType::Q6_K),
        _ => None,
    }
}

/// output[i] = dot(row_i_of_W, vector) for all rows, using AVX2+FMA when available.
/// Returns 0 on success, -1 on error.
///
/// # Safety
/// All pointers must be valid for their stated lengths and not aliased.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oxidize_gemv_quantized(
    quant_type: u32,
    qbytes: *const u8,
    qbytes_len: usize,
    rows: usize,
    cols: usize,
    vector: *const f32,
    output: *mut f32,
) -> i32 {
    let Some(qt) = to_gguf_type(quant_type) else {
        return -1;
    };
    let qb = unsafe { std::slice::from_raw_parts(qbytes, qbytes_len) };
    let v = unsafe { std::slice::from_raw_parts(vector, cols) };
    let o = unsafe { std::slice::from_raw_parts_mut(output, rows) };
    match tensor::gemv_quantized_f32(qt, qb, rows, cols, v, o) {
        Ok(()) => 0,
        Err(_) => -1,
    }
}

struct ModelHandle {
    model: InferenceModel,
}

/// Load a GGUF model from `path`. Returns an opaque handle or NULL on error.
///
/// # Safety
/// `path` must be a valid nul-terminated UTF-8 string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oxidize_model_load(path: *const c_char) -> *mut std::ffi::c_void {
    init_thread_pool();
    let Ok(path_str) = unsafe { CStr::from_ptr(path) }.to_str() else {
        return std::ptr::null_mut();
    };
    let mapped = match load_mapped_gguf(path_str) {
        Ok(m) => m,
        Err(_) => return std::ptr::null_mut(),
    };
    // Request THP for weight pages to reduce TLB pressure during inference.
    // Only activates when RAM headroom is sufficient (see advise_huge_pages docs).
    #[cfg(target_os = "linux")]
    let _ = mapped.advise_huge_pages();
    let config = InferenceConfig::from_gguf(&mapped);
    let model = match InferenceModel::load_from_gguf(&mapped, config, true) {
        Ok(m) => m,
        Err(_) => return std::ptr::null_mut(),
    };
    let handle = Box::new(ModelHandle { model });
    Box::into_raw(handle) as *mut std::ffi::c_void
}

/// Free a model handle returned by oxidize_model_load.
///
/// # Safety
/// `handle` must be a pointer returned by `oxidize_model_load` and not yet freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oxidize_model_free(handle: *mut std::ffi::c_void) {
    if !handle.is_null() {
        unsafe { drop(Box::from_raw(handle as *mut ModelHandle)) };
    }
}

/// Returns the vocab size of the model, or 0 on error.
///
/// # Safety
/// `handle` must be a valid model handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oxidize_model_vocab_size(handle: *mut std::ffi::c_void) -> u32 {
    if handle.is_null() {
        return 0;
    }
    let h = unsafe { &*(handle as *mut ModelHandle) };
    h.model.config().vocab_size as u32
}

/// Create a new inference session. Must be freed with oxidize_session_free.
#[unsafe(no_mangle)]
pub extern "C" fn oxidize_session_new() -> *mut std::ffi::c_void {
    Box::into_raw(Box::new(Session::new())) as *mut std::ffi::c_void
}

/// Reset a session (clears KV cache position counter).
///
/// # Safety
/// `session` must be a valid session handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oxidize_session_reset(session: *mut std::ffi::c_void) {
    if !session.is_null() {
        let s = unsafe { &mut *(session as *mut Session) };
        *s = Session::new();
    }
}

/// Free a session handle.
///
/// # Safety
/// `session` must be a pointer returned by `oxidize_session_new` and not yet freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oxidize_session_free(session: *mut std::ffi::c_void) {
    if !session.is_null() {
        unsafe { drop(Box::from_raw(session as *mut Session)) };
    }
}

/// Run one decode step.
///
/// `tokens` / `n_tokens`: the prompt tokens (for prefill) or a single new token (decode).
/// `logits_out`: caller-allocated f32 buffer of length `vocab_size`.
///
/// Returns 0 on success, -1 on error.
///
/// # Safety
/// All pointers must be valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oxidize_model_forward(
    handle: *mut std::ffi::c_void,
    session: *mut std::ffi::c_void,
    tokens: *const u32,
    n_tokens: usize,
    logits_out: *mut f32,
    vocab_size: usize,
) -> i32 {
    if handle.is_null() || session.is_null() || tokens.is_null() || logits_out.is_null() {
        return -1;
    }
    let h = unsafe { &mut *(handle as *mut ModelHandle) };
    let s = unsafe { &mut *(session as *mut Session) };
    let toks: Vec<u32> = unsafe { std::slice::from_raw_parts(tokens, n_tokens) }.to_vec();
    match h.model.forward(&toks, s) {
        Ok(logits) => {
            // Refuse to copy a partial/truncated result: the caller-provided buffer
            // length must exactly match the produced logits length.
            if logits.len() != vocab_size {
                return -1;
            }
            let out = unsafe { std::slice::from_raw_parts_mut(logits_out, vocab_size) };
            out.copy_from_slice(&logits);
            0
        }
        Err(_) => -1,
    }
}

/// Sample greedily (argmax) from a logits buffer.
///
/// # Safety
/// `logits` must be valid for `vocab_size` f32 values.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oxidize_sample_argmax(logits: *const f32, vocab_size: usize) -> u32 {
    let l = unsafe { std::slice::from_raw_parts(logits, vocab_size) };
    l.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}
