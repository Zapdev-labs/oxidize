//! Two-node pipeline-parallel decode driver.
//!
//! Stage 0 ("head") owns the prompt, tokenizer, embedding table, and runs
//! layers `[0, split)`. It sends hidden state + position to stage 1 over TCP.
//!
//! Stage 1 ("tail") runs layers `[split, L)`, applies the final RMS norm and
//! lm_head, samples (argmax for now), and sends the chosen token back to head
//! which decides whether to print it (post-prompt) and feeds it to the next
//! forward step.
//!
//! Wire protocol v2 (length-prefixed framing, all integers little-endian):
//!   Head → Tail : tag=0x01 HIDDEN   { pos: u32, wants_token: u8,
//!                                    hidden_f16: [u16; h] }
//!                 tag=0xFE BYE
//!   Tail → Head : tag=0x10 TOKEN    { token: u32 }   only when wants_token=1
//!
//! f16 transport halves bytes-on-wire vs f32. `wants_token=0` lets the head
//! stream all prompt-prefill positions to the tail without per-step recv,
//! so head's pos=N+1 forward can run while tail is still processing pos=N
//! (real pipeline overlap for prefill). Decode is still synchronous since
//! every step depends on the previous token.
//!
//! Both nodes mmap the full GGUF (true per-shard loading is a follow-up).

use oxidize_core::gguf::MappedGgufFile;
use oxidize_core::inference::{InferenceConfig, InferenceModel};
use oxidize_core::model::{Model, Session};
use oxidize_core::model_loader::{GgufModelLoader, ModelLoader};
use oxidize_core::tokenizer::{EncodeOptions, load_tokenizer_from_gguf_metadata};

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::Path;
use std::time::Instant;

const TAG_HIDDEN: u8 = 0x01;
const TAG_BYE: u8 = 0xFE;
const TAG_TOKEN: u8 = 0x10;

/// Inclusive log helper.
fn log(stage: &str, msg: impl AsRef<str>) {
    eprintln!("[pipe/{stage}] {}", msg.as_ref());
}

fn load_model(model_path: &Path, use_mmap: bool) -> Result<InferenceModel, String> {
    let loader = GgufModelLoader;
    let mapped = loader
        .load(model_path)
        .map_err(|e| format!("load gguf: {e}"))?;
    let config = config_from_metadata(&mapped);
    InferenceModel::load_from_gguf(&mapped, config, use_mmap)
}

fn config_from_metadata(mapped: &MappedGgufFile) -> InferenceConfig {
    use oxidize_core::gguf::GgufMetadataValue;
    let meta = &mapped.parsed().metadata;
    let arch = match meta.get("general.architecture") {
        Some(GgufMetadataValue::String(s)) => s.clone(),
        _ => "llama".to_string(),
    };
    let key = |suffix: &str| format!("{arch}.{suffix}");
    let u32_of = |k: &str| -> Option<usize> {
        match meta.get(k)? {
            GgufMetadataValue::Uint32(v) => Some(*v as usize),
            GgufMetadataValue::Int32(v) if *v >= 0 => Some(*v as usize),
            GgufMetadataValue::Uint64(v) => Some(*v as usize),
            GgufMetadataValue::Int64(v) if *v >= 0 => Some(*v as usize),
            _ => None,
        }
    };
    let f32_of = |k: &str| -> Option<f32> {
        match meta.get(k)? {
            GgufMetadataValue::Float32(v) => Some(*v),
            GgufMetadataValue::Float64(v) => Some(*v as f32),
            GgufMetadataValue::Uint32(v) => Some(*v as f32),
            GgufMetadataValue::Int32(v) => Some(*v as f32),
            _ => None,
        }
    };
    let hidden_size = u32_of(&key("embedding_length")).unwrap_or(2048);
    let layer_count = u32_of(&key("block_count")).unwrap_or(22);
    let num_attention_heads = u32_of(&key("attention.head_count")).unwrap_or(16);
    let num_key_value_heads =
        u32_of(&key("attention.head_count_kv")).unwrap_or(num_attention_heads);
    let intermediate_size = u32_of(&key("feed_forward_length")).unwrap_or(hidden_size * 4);
    let context_size = u32_of(&key("context_length")).unwrap_or(4096);
    let vocab_size = u32_of(&key("vocab_size"))
        .or_else(|| match meta.get("tokenizer.ggml.tokens") {
            Some(GgufMetadataValue::Array(a)) => Some(a.values.len()),
            _ => None,
        })
        .unwrap_or(32000);
    let rope_theta = f32_of(&key("rope.freq_base")).unwrap_or(10000.0);
    let rms_norm_eps = f32_of(&key("attention.layer_norm_rms_epsilon")).unwrap_or(1e-5);
    let key_value_head_dim = u32_of(&key("attention.key_length")).unwrap_or_else(|| {
        hidden_size
            .checked_div(num_attention_heads)
            .unwrap_or(hidden_size)
    });
    InferenceConfig {
        vocab_size,
        context_size,
        layer_count,
        hidden_size,
        intermediate_size,
        num_attention_heads,
        num_key_value_heads,
        key_value_head_dim,
        rms_norm_eps,
        rope_theta,
        ..Default::default()
    }
}

fn argmax_f32(logits: &[f32]) -> u32 {
    let mut best_idx = 0_usize;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best_idx = i;
        }
    }
    best_idx as u32
}

fn write_all(stream: &mut TcpStream, buf: &[u8]) -> std::io::Result<()> {
    stream.write_all(buf)
}

fn read_exact(stream: &mut TcpStream, buf: &mut [u8]) -> std::io::Result<()> {
    stream.read_exact(buf)
}

/// IEEE-754 f32 → f16 with round-to-nearest-even. Out-of-range values clamp
/// to ±inf. Subnormals flush to zero (hidden state never hits them in practice).
#[inline]
fn f32_to_f16_bits(f: f32) -> u16 {
    let b = f.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp_unbiased = ((b >> 23) & 0xff) as i32 - 127;
    let mant = b & 0x7fffff;
    if exp_unbiased > 15 {
        // Overflow or NaN passthrough.
        if exp_unbiased == 128 && mant != 0 {
            return sign | 0x7e00; // NaN
        }
        return sign | 0x7c00; // ±inf
    }
    if exp_unbiased < -14 {
        return sign; // flush to zero
    }
    let e16 = (exp_unbiased + 15) as u32;
    // Round-to-nearest-even on the low 13 mantissa bits.
    let round = (mant & 0x1000) >> 12;
    let sticky = (mant & 0x0fff != 0) as u32;
    let lsb = (mant & 0x2000) >> 13;
    let m16 = (mant >> 13) + (round & (sticky | lsb));
    if m16 > 0x3ff {
        let e16 = e16 + 1;
        if e16 >= 31 {
            return sign | 0x7c00;
        }
        return sign | ((e16 as u16) << 10);
    }
    sign | ((e16 as u16) << 10) | (m16 as u16)
}

#[inline]
fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let frac = (bits & 0x03FF) as u32;
    let f32_bits = if exp == 0 {
        if frac == 0 {
            sign << 31
        } else {
            let mut fnorm = frac;
            let mut e = -14_i32;
            while (fnorm & 0x0400) == 0 {
                fnorm <<= 1;
                e -= 1;
            }
            fnorm &= 0x03FF;
            (sign << 31) | (((e + 127) as u32) << 23) | (fnorm << 13)
        }
    } else if exp == 0x1F {
        (sign << 31) | 0x7F80_0000 | (frac << 13)
    } else {
        let e = exp as i32 - 15 + 127;
        (sign << 31) | ((e as u32) << 23) | (frac << 13)
    };
    f32::from_bits(f32_bits)
}

fn send_hidden(
    stream: &mut TcpStream,
    pos: u32,
    wants_token: bool,
    hidden: &[f32],
    scratch: &mut Vec<u16>,
) -> std::io::Result<()> {
    scratch.clear();
    scratch.reserve(hidden.len());
    for &v in hidden {
        scratch.push(f32_to_f16_bits(v));
    }
    // [tag u8][pos u32][wants u8][len u32][hidden_f16 bytes]
    let payload_bytes = (scratch.len() * 2) as u32;
    let mut header = [0u8; 1 + 4 + 1 + 4];
    header[0] = TAG_HIDDEN;
    header[1..5].copy_from_slice(&pos.to_le_bytes());
    header[5] = wants_token as u8;
    header[6..10].copy_from_slice(&payload_bytes.to_le_bytes());
    write_all(stream, &header)?;
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(scratch.as_ptr() as *const u8, scratch.len() * 2) };
    write_all(stream, bytes)
}

fn send_bye(stream: &mut TcpStream) -> std::io::Result<()> {
    write_all(stream, &[TAG_BYE])
}

fn send_token(stream: &mut TcpStream, token: u32) -> std::io::Result<()> {
    let mut buf = [0u8; 1 + 4];
    buf[0] = TAG_TOKEN;
    buf[1..5].copy_from_slice(&token.to_le_bytes());
    write_all(stream, &buf)
}

fn recv_tag(stream: &mut TcpStream) -> std::io::Result<u8> {
    let mut tag = [0u8; 1];
    read_exact(stream, &mut tag)?;
    Ok(tag[0])
}

/// Receives a HIDDEN frame body. Returns (pos, wants_token). Decodes f16
/// payload into `into` as f32.
fn recv_hidden_payload(
    stream: &mut TcpStream,
    into: &mut Vec<f32>,
    f16_scratch: &mut Vec<u16>,
) -> std::io::Result<(u32, bool)> {
    let mut buf = [0u8; 4 + 1 + 4];
    read_exact(stream, &mut buf)?;
    let pos = u32::from_le_bytes(buf[..4].try_into().unwrap());
    let wants_token = buf[4] != 0;
    let nbytes = u32::from_le_bytes(buf[5..9].try_into().unwrap()) as usize;
    if !nbytes.is_multiple_of(2) {
        return Err(std::io::Error::other("hidden payload not f16-aligned"));
    }
    let n = nbytes / 2;
    f16_scratch.resize(n, 0);
    let bytes: &mut [u8] =
        unsafe { std::slice::from_raw_parts_mut(f16_scratch.as_mut_ptr() as *mut u8, nbytes) };
    read_exact(stream, bytes)?;
    into.resize(n, 0.0);
    for (dst, &src) in into.iter_mut().zip(f16_scratch.iter()) {
        *dst = f16_bits_to_f32(src);
    }
    Ok((pos, wants_token))
}

fn recv_token_payload(stream: &mut TcpStream) -> std::io::Result<u32> {
    let mut buf = [0u8; 4];
    read_exact(stream, &mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

/// Stage 0 / head. Connects to tail at `peer_addr`, tokenizes `prompt`,
/// feeds prompt tokens through the half-pipeline, then generates
/// `max_new_tokens` more by feeding tail's chosen token back into itself.
pub fn run_head(
    model_path: &Path,
    peer_addr: &str,
    prompt: &str,
    max_new_tokens: usize,
    use_mmap: bool,
) -> std::io::Result<()> {
    log("head", format!("loading model {}", model_path.display()));
    let mapped = GgufModelLoader
        .load(model_path)
        .map_err(|e| std::io::Error::other(format!("load gguf: {e}")))?;
    let tokenizer = load_tokenizer_from_gguf_metadata(&mapped.parsed().metadata)
        .map_err(|e| std::io::Error::other(format!("tokenizer: {e:?}")))?;
    let mut model = load_model(model_path, use_mmap)
        .map_err(|e| std::io::Error::other(format!("model: {e}")))?;
    let layer_count = model.layer_count();
    let split = layer_count / 2;
    let head_range = 0..split;
    log(
        "head",
        format!(
            "loaded: layers={layer_count}, head_layers={:?}, hidden={}",
            head_range,
            model.config_hidden_size(),
        ),
    );

    log("head", format!("connecting to tail at {peer_addr}"));
    let mut stream = TcpStream::connect(peer_addr)?;
    stream.set_nodelay(true)?;
    let _ = stream
        .set_write_timeout(Some(std::time::Duration::from_secs(60)))
        .and(stream.set_read_timeout(Some(std::time::Duration::from_secs(120))));
    // Bigger socket buffers help latency on long activation streams.
    {
        use std::os::fd::AsRawFd;
        let fd = stream.as_raw_fd();
        unsafe {
            let bufsz: libc::c_int = 4 * 1024 * 1024;
            libc::setsockopt(
                fd,
                libc::SOL_SOCKET,
                libc::SO_SNDBUF,
                &bufsz as *const _ as *const _,
                std::mem::size_of_val(&bufsz) as _,
            );
            libc::setsockopt(
                fd,
                libc::SOL_SOCKET,
                libc::SO_RCVBUF,
                &bufsz as *const _ as *const _,
                std::mem::size_of_val(&bufsz) as _,
            );
        }
    }

    // Send the split point so tail knows its own layer range.
    let mut hello = [0u8; 8];
    hello[..4].copy_from_slice(&(layer_count as u32).to_le_bytes());
    hello[4..8].copy_from_slice(&(split as u32).to_le_bytes());
    stream.write_all(&hello)?;

    let prompt_ids = tokenizer.encode_with_special_tokens(
        prompt,
        EncodeOptions {
            add_bos: true,
            add_eos: false,
            pad_to: None,
        },
    );
    log("head", format!("prompt tokens: {}", prompt_ids.len()));

    let eos = tokenizer.special_tokens().eos;
    let mut emitted: Vec<u32> = Vec::with_capacity(max_new_tokens);
    let mut f16_scratch: Vec<u16> = Vec::new();
    let start = Instant::now();

    // --- Prefill phase: stream all prompt positions to tail without
    // waiting per step. Only the final prompt position asks for a token
    // (= the first generated token). Head's compute for pos+1 overlaps
    // with tail's compute for pos.
    let n_prompt = prompt_ids.len();
    for (pos, &tok) in prompt_ids.iter().enumerate() {
        model.embed_token_into_workspace(tok);
        model
            .run_layer_range_in_workspace(pos, head_range.clone())
            .map_err(|e| std::io::Error::other(format!("head forward: {e:?}")))?;
        let wants_token = pos + 1 == n_prompt;
        let hidden = model.hidden_state().to_vec();
        send_hidden(
            &mut stream,
            pos as u32,
            wants_token,
            &hidden,
            &mut f16_scratch,
        )?;
    }
    let prefill_done = Instant::now();
    log(
        "head",
        format!(
            "prefill streamed {} positions in {:.2}s (head-side)",
            n_prompt,
            (prefill_done - start).as_secs_f64()
        ),
    );

    let first_token = match recv_tag(&mut stream)? {
        TAG_TOKEN => recv_token_payload(&mut stream)?,
        other => {
            return Err(std::io::Error::other(format!(
                "head: unexpected tag {:#x} (expected TOKEN after prefill)",
                other
            )));
        }
    };
    let prefill_full_done = Instant::now();
    log(
        "head",
        format!(
            "first token after {:.2}s total (incl. tail catch-up)",
            (prefill_full_done - start).as_secs_f64()
        ),
    );
    emitted.push(first_token);

    // --- Decode phase: each step is autoregressive; sync round-trip per token.
    let mut next_input = first_token;
    let mut decode_pos = n_prompt;
    let decode_start = Instant::now();
    while emitted.len() < max_new_tokens {
        if let Some(e) = eos
            && next_input == e
        {
            break;
        }
        model.embed_token_into_workspace(next_input);
        model
            .run_layer_range_in_workspace(decode_pos, head_range.clone())
            .map_err(|e| std::io::Error::other(format!("head forward: {e:?}")))?;
        let hidden = model.hidden_state().to_vec();
        send_hidden(
            &mut stream,
            decode_pos as u32,
            true,
            &hidden,
            &mut f16_scratch,
        )?;
        let tok = match recv_tag(&mut stream)? {
            TAG_TOKEN => recv_token_payload(&mut stream)?,
            other => {
                return Err(std::io::Error::other(format!(
                    "head: unexpected tag {:#x} in decode",
                    other
                )));
            }
        };
        emitted.push(tok);
        next_input = tok;
        decode_pos += 1;
    }
    let decode_elapsed = decode_start.elapsed();
    let elapsed = start.elapsed();
    send_bye(&mut stream)?;

    let decoded = tokenizer
        .decode_without_special_tokens(&emitted)
        .unwrap_or_else(|e| format!("<decode error: {e:?}>"));
    println!("\n=== Pipeline output ===");
    println!("{decoded}");
    let decode_n = emitted.len().saturating_sub(1);
    println!(
        "\n[pipeline] prompt={} tokens, decode={} tokens, total {:.2}s",
        n_prompt,
        decode_n,
        elapsed.as_secs_f64()
    );
    if decode_n > 0 {
        println!(
            "[pipeline] decode-only: {:.2} tok/s ({:.1} ms/tok)",
            decode_n as f64 / decode_elapsed.as_secs_f64().max(1e-9),
            decode_elapsed.as_secs_f64() * 1000.0 / decode_n as f64
        );
    }
    Ok(())
}

/// Stage 1 / tail. Listens at `listen_addr`, accepts head, then loops:
///   recv hidden → set hidden → run tail layers → final_head → argmax → send token
pub fn run_tail(model_path: &Path, listen_addr: &str, use_mmap: bool) -> std::io::Result<()> {
    log("tail", format!("loading model {}", model_path.display()));
    let mut model = load_model(model_path, use_mmap)
        .map_err(|e| std::io::Error::other(format!("model: {e}")))?;
    let layer_count = model.layer_count();
    log(
        "tail",
        format!(
            "loaded: layers={layer_count}, hidden={}",
            model.config_hidden_size()
        ),
    );

    let listener = TcpListener::bind(listen_addr)?;
    log("tail", format!("listening on {listen_addr}"));
    let (mut stream, peer) = listener.accept()?;
    stream.set_nodelay(true)?;
    {
        use std::os::fd::AsRawFd;
        let fd = stream.as_raw_fd();
        unsafe {
            let bufsz: libc::c_int = 4 * 1024 * 1024;
            libc::setsockopt(
                fd,
                libc::SOL_SOCKET,
                libc::SO_SNDBUF,
                &bufsz as *const _ as *const _,
                std::mem::size_of_val(&bufsz) as _,
            );
            libc::setsockopt(
                fd,
                libc::SOL_SOCKET,
                libc::SO_RCVBUF,
                &bufsz as *const _ as *const _,
                std::mem::size_of_val(&bufsz) as _,
            );
        }
    }
    log("tail", format!("accepted {peer}"));

    let mut hello = [0u8; 8];
    read_exact(&mut stream, &mut hello)?;
    let head_layer_count = u32::from_le_bytes(hello[..4].try_into().unwrap()) as usize;
    let split = u32::from_le_bytes(hello[4..8].try_into().unwrap()) as usize;
    if head_layer_count != layer_count {
        return Err(std::io::Error::other(format!(
            "layer-count mismatch: head={head_layer_count} tail={layer_count}"
        )));
    }
    let tail_range = split..layer_count;
    log("tail", format!("tail layer range = {:?}", tail_range));

    let mut hidden_buf: Vec<f32> = Vec::new();
    let mut f16_scratch: Vec<u16> = Vec::new();
    let mut step_count = 0_usize;
    let mut total_secs = 0.0_f64;

    loop {
        match recv_tag(&mut stream)? {
            TAG_HIDDEN => {
                let t0 = Instant::now();
                let (pos, wants_token) =
                    recv_hidden_payload(&mut stream, &mut hidden_buf, &mut f16_scratch)?;
                let pos = pos as usize;
                model
                    .set_hidden_state(&hidden_buf)
                    .map_err(|e| std::io::Error::other(format!("set hidden: {e:?}")))?;
                model
                    .run_layer_range_in_workspace(pos, tail_range.clone())
                    .map_err(|e| std::io::Error::other(format!("tail forward: {e:?}")))?;
                if wants_token {
                    let logits = model
                        .final_head_from_workspace()
                        .map_err(|e| std::io::Error::other(format!("final head: {e:?}")))?;
                    let tok = argmax_f32(&logits);
                    send_token(&mut stream, tok)?;
                }
                step_count += 1;
                total_secs += t0.elapsed().as_secs_f64();
                if step_count.is_multiple_of(16) {
                    log(
                        "tail",
                        format!(
                            "{step_count} steps, avg {:.2} ms/step",
                            (total_secs * 1000.0) / step_count as f64
                        ),
                    );
                }
            }
            TAG_BYE => {
                log("tail", "received BYE, exiting");
                break;
            }
            other => {
                return Err(std::io::Error::other(format!(
                    "tail: unexpected tag {:#x} from head",
                    other
                )));
            }
        }
    }

    // Keep `session` from being optimized away if unused (signaling clean exit).
    let _ = Session::new();
    Ok(())
}
