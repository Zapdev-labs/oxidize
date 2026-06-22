use super::*;

fn dump_emitted_tokens(tokens: &[u32]) -> io::Result<()> {
    let Ok(path) = std::env::var("OX_GOLDEN_TOKENS") else {
        return Ok(());
    };
    let encoded = tokens
        .iter()
        .map(u32::to_string)
        .collect::<Vec<_>>()
        .join(",");
    std::fs::write(path, encoded)
}

pub(super) fn run_single_shot_mode<W: Write>(prompt: &str, writer: &mut W) -> io::Result<()> {
    let prompt = prompt.trim();
    if prompt.is_empty() {
        return Ok(());
    }
    write_generated_response(prompt, writer).map(|_| ())
}

pub(super) fn write_generated_response<W: Write>(
    prompt: &str,
    writer: &mut W,
) -> io::Result<String> {
    write_generated_response_with_clock(prompt, writer, Instant::now)
}

pub(super) fn write_generated_response_cached<W: Write>(
    prompt: &str,
    prompt_cache: &mut PromptCache,
    writer: &mut W,
) -> io::Result<String> {
    if let Some(cached_response) = prompt_cache.get(prompt) {
        writeln!(writer, "{cached_response}")?;
        writeln!(
            writer,
            "generation stats: tokens=0 speed=0.00 tok/s (cache hit)"
        )?;
        return Ok(cached_response.to_owned());
    }

    let response = write_generated_response(prompt, writer)?;
    prompt_cache.insert(prompt, &response);
    Ok(response)
}

pub(super) fn format_generation_stats(tokens: usize, elapsed: Duration) -> String {
    let elapsed_seconds = elapsed.as_secs_f64();
    let speed = if elapsed_seconds > 0.0 {
        tokens as f64 / elapsed_seconds
    } else {
        0.0
    };
    format!(
        "generation stats: tokens={} speed={:.2} tok/s",
        tokens, speed
    )
}

pub(super) fn suppressed_generation_tokens(
    tokenizer: &LoadedTokenizer,
    vocab_size: usize,
) -> Vec<u32> {
    let special_tokens = tokenizer.special_tokens();
    let mut suppressed = Vec::new();
    let mut seen = HashSet::new();
    for token in [
        special_tokens.unknown,
        special_tokens.bos,
        special_tokens.pad,
        special_tokens.separator,
        special_tokens.cls,
        special_tokens.mask,
    ]
    .into_iter()
    .flatten()
    {
        if seen.insert(token) {
            suppressed.push(token);
        }
    }

    for token in 0..vocab_size {
        let token = token as u32;
        if seen.contains(&token) || special_tokens.eos == Some(token) {
            continue;
        }
        if let Ok(piece) = tokenizer.decode(&[token])
            && (piece.starts_with("[PAD") || (piece.starts_with('<') && piece.ends_with('>')))
            && seen.insert(token)
        {
            suppressed.push(token);
        }
    }
    suppressed
}

pub(super) fn write_generated_response_with_clock<W: Write, F: FnMut() -> Instant>(
    prompt: &str,
    writer: &mut W,
    mut now: F,
) -> io::Result<String> {
    let started_at = now();
    let response = greeting(prompt);
    let response_tokens = response.split_whitespace().count();
    if response_tokens == 0 {
        writeln!(writer, "{response}")?;
        writeln!(
            writer,
            "{}",
            format_generation_stats(response_tokens, now().saturating_duration_since(started_at))
        )?;
        return Ok(response);
    }
    for generated in 1..=response_tokens {
        writeln!(
            writer,
            "generation progress: {generated}/{response_tokens} tokens"
        )?;
    }
    writeln!(writer, "{response}")?;
    writeln!(
        writer,
        "{}",
        format_generation_stats(response_tokens, now().saturating_duration_since(started_at))
    )?;
    Ok(response)
}

pub(super) fn dflash_gguf_has_io_tensors(mapped: &MappedGgufFile) -> bool {
    let infos = mapped.mapped_tensor_infos();
    let has_output = infos
        .iter()
        .any(|tensor| tensor.name == "lm_head.weight" || tensor.name == "output.weight");
    let has_embed = infos.iter().any(|tensor| {
        tensor.name == "model.embed_tokens.weight" || tensor.name == "tok_embeddings.weight"
    });
    has_output && has_embed
}

pub(super) fn dflash_byte_smoke_tokenizer() -> LoadedTokenizer {
    static TOK: std::sync::OnceLock<LoadedTokenizer> = std::sync::OnceLock::new();
    TOK.get_or_init(|| {
        let bytes: &'static [[u8; 1]] = Box::leak(
            (0u8..=255)
                .map(|byte| [byte])
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        );
        let vocab: Vec<&[u8]> = bytes.iter().map(|entry| entry.as_slice()).collect();
        LoadedTokenizer::Tiktoken(TiktokenTokenizer::new(&vocab, &[]))
    })
    .clone()
}

#[allow(clippy::too_many_arguments)]
pub(super) fn generate_with_model<W: Write, M: Model + ?Sized>(
    prompt: &str,
    model: &mut M,
    tokenizer: &LoadedTokenizer,
    max_tokens: usize,
    temperature: f32,
    top_p: Option<f32>,
    top_k: Option<usize>,
    writer: &mut W,
) -> io::Result<String> {
    use futures_core::Stream;
    use std::pin::Pin;
    use std::sync::Arc;
    use std::task::{Context, Poll, Waker};

    let started_at = Instant::now();
    let mut session = Session::new();

    // Encode prompt using the model's tokenizer (add BOS for generation)
    let prompt_tokens = tokenizer.encode_with_special_tokens(
        prompt,
        EncodeOptions {
            add_bos: tokenizer.add_bos_default(),
            add_eos: false,
            pad_to: None,
        },
    );

    let eos_token = tokenizer.special_tokens().eos;
    let suppressed_tokens = suppressed_generation_tokens(tokenizer, model.vocab_size());

    let config = GenerationConfig {
        max_new_tokens: max_tokens,
        stop_token: eos_token,
        suppressed_tokens,
        sampling: SamplingConfig {
            temperature,
            top_p,
            top_k,
            ..SamplingConfig::default()
        },
        ..GenerationConfig::default()
    };

    let mut rng = rand::thread_rng();
    let mut stream = GenerationStream::new(model, &mut session, &prompt_tokens, config, || {
        rand::Rng::r#gen::<f32>(&mut rng)
    });

    let waker = Waker::from(Arc::new(NoopWaker));
    let mut cx = Context::from_waker(&waker);
    let mut pinned = Pin::new(&mut stream);

    let mut generated_tokens: Vec<u32> = Vec::new();

    loop {
        match Stream::poll_next(pinned.as_mut(), &mut cx) {
            Poll::Ready(Some(Ok(token))) => {
                generated_tokens.push(token);
            }
            Poll::Ready(Some(Err(e))) => {
                return Err(io::Error::other(format!("generation error: {:?}", e)));
            }
            Poll::Ready(None) => break,
            Poll::Pending => break,
        }
    }

    dump_emitted_tokens(&generated_tokens)?;
    let response = tokenizer
        .decode_without_special_tokens(&generated_tokens)
        .unwrap_or_default();
    if !response.is_empty() {
        write!(writer, "{response}")?;
    } else if !generated_tokens.is_empty() {
        write!(writer, "[generated token ids: {generated_tokens:?}]")?;
    }
    writer.flush()?;

    let elapsed = started_at.elapsed();
    writeln!(writer)?;
    writeln!(
        writer,
        "{}",
        format_generation_stats(generated_tokens.len(), elapsed)
    )?;

    Ok(response)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn generate_with_dflash_draft<W: Write, M: Model + ?Sized>(
    prompt: &str,
    target_model: &mut M,
    draft_model: &mut oxidize_core::dflash::DFlashDraftModel,
    tokenizer: &LoadedTokenizer,
    max_tokens: usize,
    temperature: f32,
    top_p: Option<f32>,
    top_k: Option<usize>,
    draft_tokens: usize,
    writer: &mut W,
) -> io::Result<String> {
    use futures_core::Stream;
    use std::pin::Pin;
    use std::sync::Arc;
    use std::task::{Context, Poll, Waker};

    let started_at = Instant::now();
    let mut session = Session::new();
    let prompt_tokens = tokenizer.encode_with_special_tokens(
        prompt,
        EncodeOptions {
            add_bos: tokenizer.add_bos_default(),
            add_eos: false,
            pad_to: None,
        },
    );
    let eos_token = tokenizer.special_tokens().eos;
    let suppressed_tokens = suppressed_generation_tokens(tokenizer, target_model.vocab_size());
    let generation = GenerationConfig {
        max_new_tokens: max_tokens,
        stop_token: eos_token,
        suppressed_tokens,
        sampling: SamplingConfig {
            temperature,
            top_p,
            top_k,
            ..SamplingConfig::default()
        },
        ..GenerationConfig::default()
    };
    let config = SpeculativeGenerationConfig {
        generation,
        draft_tokens_per_step: draft_tokens.max(1),
        quantspec_draft_kv: false,
    };

    let mut rng = rand::thread_rng();
    let mut stream = SpeculativeGenerationStream::new(
        target_model,
        draft_model,
        &mut session,
        &prompt_tokens,
        config,
        || rand::Rng::r#gen::<f32>(&mut rng),
    );
    let waker = Waker::from(Arc::new(NoopWaker));
    let mut cx = Context::from_waker(&waker);
    let mut pinned = Pin::new(&mut stream);
    let mut generated_tokens: Vec<u32> = Vec::new();

    loop {
        match Stream::poll_next(pinned.as_mut(), &mut cx) {
            Poll::Ready(Some(Ok(token))) => generated_tokens.push(token),
            Poll::Ready(Some(Err(e))) => {
                return Err(io::Error::other(format!("generation error: {:?}", e)));
            }
            Poll::Ready(None) => break,
            Poll::Pending => break,
        }
    }

    dump_emitted_tokens(&generated_tokens)?;
    let response = tokenizer
        .decode_without_special_tokens(&generated_tokens)
        .unwrap_or_default();
    if !response.is_empty() {
        write!(writer, "{response}")?;
        writer.flush()?;
    }
    let elapsed = started_at.elapsed();
    writeln!(writer)?;
    writeln!(
        writer,
        "{}",
        format_generation_stats(generated_tokens.len(), elapsed)
    )?;
    Ok(response)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn generate_with_mtp_model<W: Write>(
    prompt: &str,
    target_model: &mut InferenceModel,
    tokenizer: &LoadedTokenizer,
    max_tokens: usize,
    temperature: f32,
    top_p: Option<f32>,
    top_k: Option<usize>,
    draft_tokens: usize,
    writer: &mut W,
    quantspec: bool,
) -> io::Result<String> {
    use futures_core::Stream;
    use std::pin::Pin;
    use std::sync::Arc;
    use std::task::{Context, Poll, Waker};

    let started_at = Instant::now();
    let mut session = Session::new();
    let prompt_tokens = tokenizer.encode_with_special_tokens(
        prompt,
        EncodeOptions {
            add_bos: tokenizer.add_bos_default(),
            add_eos: false,
            pad_to: None,
        },
    );
    let eos_token = tokenizer.special_tokens().eos;
    let suppressed_tokens = suppressed_generation_tokens(tokenizer, target_model.vocab_size());
    let generation = GenerationConfig {
        max_new_tokens: max_tokens,
        stop_token: eos_token,
        suppressed_tokens,
        sampling: SamplingConfig {
            temperature,
            top_p,
            top_k,
            ..SamplingConfig::default()
        },
        ..GenerationConfig::default()
    };
    let config = SpeculativeGenerationConfig {
        generation,
        draft_tokens_per_step: draft_tokens.max(1),
        quantspec_draft_kv: quantspec,
    };

    let mut rng = rand::thread_rng();
    let mut stream =
        MtpGenerationStream::new(target_model, &mut session, &prompt_tokens, config, || {
            rand::Rng::r#gen::<f32>(&mut rng)
        });
    let waker = Waker::from(Arc::new(NoopWaker));
    let mut cx = Context::from_waker(&waker);
    let mut pinned = Pin::new(&mut stream);
    let mut generated_tokens: Vec<u32> = Vec::new();

    loop {
        match Stream::poll_next(pinned.as_mut(), &mut cx) {
            Poll::Ready(Some(Ok(token))) => generated_tokens.push(token),
            Poll::Ready(Some(Err(e))) => {
                return Err(io::Error::other(format!("generation error: {:?}", e)));
            }
            Poll::Ready(None) => break,
            Poll::Pending => break,
        }
    }

    dump_emitted_tokens(&generated_tokens)?;
    let response = tokenizer
        .decode_without_special_tokens(&generated_tokens)
        .unwrap_or_default();
    if !response.is_empty() {
        write!(writer, "{response}")?;
    } else if !generated_tokens.is_empty() {
        write!(writer, "[generated token ids: {generated_tokens:?}]")?;
    }
    writer.flush()?;
    let elapsed = started_at.elapsed();
    writeln!(writer)?;
    writeln!(
        writer,
        "{}",
        format_generation_stats(generated_tokens.len(), elapsed)
    )?;
    Ok(response)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn generate_with_eagle3_draft<W: Write>(
    prompt: &str,
    target_model: &mut InferenceModel,
    draft_model: &mut oxidize_core::eagle3::Eagle3DraftModel,
    tokenizer: &LoadedTokenizer,
    max_tokens: usize,
    temperature: f32,
    top_p: Option<f32>,
    top_k: Option<usize>,
    draft_tokens: usize,
    writer: &mut W,
) -> io::Result<String> {
    use futures_core::Stream;
    use std::pin::Pin;
    use std::sync::Arc;
    use std::task::{Context, Poll, Waker};

    let started_at = Instant::now();
    let mut session = Session::new();
    let prompt_tokens = tokenizer.encode_with_special_tokens(
        prompt,
        EncodeOptions {
            add_bos: tokenizer.add_bos_default(),
            add_eos: false,
            pad_to: None,
        },
    );
    let eos_token = tokenizer.special_tokens().eos;
    let suppressed_tokens = suppressed_generation_tokens(tokenizer, target_model.vocab_size());
    let generation = GenerationConfig {
        max_new_tokens: max_tokens,
        stop_token: eos_token,
        suppressed_tokens,
        sampling: SamplingConfig {
            temperature,
            top_p,
            top_k,
            ..SamplingConfig::default()
        },
        ..GenerationConfig::default()
    };
    let config = SpeculativeGenerationConfig {
        generation,
        draft_tokens_per_step: draft_tokens.max(1),
        quantspec_draft_kv: false,
    };

    let mut rng = rand::thread_rng();
    let mut stream = Eagle3GenerationStream::new(
        target_model,
        draft_model,
        &mut session,
        &prompt_tokens,
        config,
        || rand::Rng::r#gen::<f32>(&mut rng),
    );
    let waker = Waker::from(Arc::new(NoopWaker));
    let mut cx = Context::from_waker(&waker);
    let mut pinned = Pin::new(&mut stream);
    let mut generated_tokens: Vec<u32> = Vec::new();

    loop {
        match Stream::poll_next(pinned.as_mut(), &mut cx) {
            Poll::Ready(Some(Ok(token))) => generated_tokens.push(token),
            Poll::Ready(Some(Err(e))) => {
                return Err(io::Error::other(format!("generation error: {:?}", e)));
            }
            Poll::Ready(None) => break,
            Poll::Pending => break,
        }
    }

    dump_emitted_tokens(&generated_tokens)?;
    let response = tokenizer
        .decode_without_special_tokens(&generated_tokens)
        .unwrap_or_default();
    if !response.is_empty() {
        write!(writer, "{response}")?;
        writer.flush()?;
    }
    let elapsed = started_at.elapsed();
    writeln!(writer)?;
    writeln!(
        writer,
        "{}",
        format_generation_stats(generated_tokens.len(), elapsed)
    )?;
    Ok(response)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn generate_with_quantspec<W: Write>(
    prompt: &str,
    target_model: &mut InferenceModel,
    tokenizer: &LoadedTokenizer,
    max_tokens: usize,
    temperature: f32,
    top_p: Option<f32>,
    top_k: Option<usize>,
    draft_tokens: usize,
    writer: &mut W,
) -> io::Result<String> {
    generate_with_mtp_model(
        prompt,
        target_model,
        tokenizer,
        max_tokens,
        temperature,
        top_p,
        top_k,
        draft_tokens,
        writer,
        true,
    )
}

pub(super) struct NoopWaker;

impl Wake for NoopWaker {
    fn wake(self: Arc<Self>) {}
}
