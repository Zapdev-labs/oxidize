//! Env-gated attention debug dump (`OX_ATTN_DUMP`).
//!
//! When the environment variable `OX_ATTN_DUMP` names a file path, the very FIRST
//! decode token's FIRST `gpu_native`-eligible attention layer writes a small set
//! of labeled float vectors to that file — once per process. The SAME labels,
//! order, and sizes are emitted by BOTH the CPU attention island
//! (`model/inference/layers.rs`) and the GPU fused path
//! (`backends/cuda/gpu_native_forward.rs`), so the operator can run the two
//! configurations (default vs. `OX_GPU_ATTN`) into two files and `diff` them to
//! localize where the fused on-device attention diverges from the CPU reference.
//!
//! This module is COMPLETELY inert unless `OX_ATTN_DUMP` is set: the only cost on
//! the default path is one relaxed atomic-bool load (`should_dump`) per attention
//! layer. It is architecture-independent (no CUDA types) so it compiles on every
//! target; the GPU side merely copies a few small device vectors to host once and
//! calls the same `write_block` entry point.
//!
//! Emitted block (labels are exact; counts are clamped to the available length):
//! ```text
//! # OX_ATTN_DUMP path=<cpu|gpu> pos=<P> layer=<L> kv_layer=<K>
//! norm_in <N floats>      (post-RMSNorm input to QKV; full hidden vector)
//! q_rope <256 floats>     (Q after RoPE, first 256)
//! k_cur <128 floats>      (K after RoPE for the current position, first 128)
//! v_cur <128 floats>      (V for the current position, first 128)
//! attn_out <256 floats>   (attention output before Wo, first 256)
//! ```

use std::io::Write;
use std::sync::atomic::{AtomicBool, Ordering};

/// `Some(path)` when `OX_ATTN_DUMP` is set to a non-empty file path, else `None`.
/// Evaluated once and cached.
fn dump_path() -> Option<&'static std::path::Path> {
    static PATH: std::sync::OnceLock<Option<std::path::PathBuf>> = std::sync::OnceLock::new();
    PATH.get_or_init(|| match std::env::var("OX_ATTN_DUMP") {
        Ok(p) if !p.is_empty() => Some(std::path::PathBuf::from(p)),
        _ => None,
    })
    .as_deref()
}

/// One-shot guard: the dump fires for the FIRST eligible attention layer of the
/// FIRST decode token only, then never again for the lifetime of the process.
static FIRED: AtomicBool = AtomicBool::new(false);

/// Cheap predicate for the hot path: `true` only when `OX_ATTN_DUMP` is set AND
/// the one-shot dump has not yet been written. Does NOT consume the one-shot
/// token (so a caller can decide to dump and let [`write_block`] claim it).
#[inline]
pub(crate) fn should_dump() -> bool {
    dump_path().is_some() && !FIRED.load(Ordering::Relaxed)
}

/// Append the labeled debug block to the `OX_ATTN_DUMP` file. Claims the one-shot
/// token: the first successful call wins; concurrent/subsequent calls return
/// without writing. `tag` is `"cpu"` or `"gpu"` (the path that produced the
/// vectors). The four vectors are written with the fixed labels documented in the
/// module header; each is truncated to the documented size if longer (and written
/// in full if shorter — the operator sees the true available length).
///
/// No-op (and does NOT claim the token) unless `OX_ATTN_DUMP` is set.
#[cold]
pub(crate) fn write_block(
    tag: &str,
    pos: usize,
    layer: usize,
    kv_layer: usize,
    norm_in: &[f32],
    q_rope: &[f32],
    k_cur: &[f32],
    v_cur: &[f32],
    attn_out: &[f32],
) {
    let Some(path) = dump_path() else { return };
    // Claim the one-shot token exactly once.
    if FIRED.swap(true, Ordering::SeqCst) {
        return;
    }

    let mut buf = String::new();
    buf.push_str(&format!(
        "# OX_ATTN_DUMP path={tag} pos={pos} layer={layer} kv_layer={kv_layer}\n"
    ));
    write_labeled(&mut buf, "norm_in", norm_in, norm_in.len());
    write_labeled(&mut buf, "q_rope", q_rope, 256);
    write_labeled(&mut buf, "k_cur", k_cur, 128);
    write_labeled(&mut buf, "v_cur", v_cur, 128);
    write_labeled(&mut buf, "attn_out", attn_out, 256);

    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
    {
        let _ = f.write_all(buf.as_bytes());
        let _ = f.flush();
    }
}

/// Write `label` followed by up to `max` whitespace-separated floats on one line.
/// Floats use a fixed high-precision format so the two runs are byte-comparable.
fn write_labeled(buf: &mut String, label: &str, data: &[f32], max: usize) {
    let n = data.len().min(max);
    buf.push_str(label);
    for &v in &data[..n] {
        buf.push(' ');
        buf.push_str(&format!("{v:.6e}"));
    }
    buf.push('\n');
}
