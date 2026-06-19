use super::*;

pub(super) fn resolve_model_spec(spec: &str, hf_file: Option<&str>) -> io::Result<PathBuf> {
    let path = PathBuf::from(spec);
    if path.exists() || !spec.contains('/') {
        return Ok(path);
    }

    let api = HfApi::new()?;
    resolve_hf_model_spec(&api, spec, hf_file)
}

#[derive(Debug, Clone)]
pub(super) struct HfApi {
    cache_dir: PathBuf,
    agent: ureq::Agent,
}

#[derive(Debug, Clone)]
pub(super) struct HfRepo {
    id: String,
    cache_dir: PathBuf,
    agent: ureq::Agent,
}

#[derive(Debug, Deserialize)]
pub(super) struct HfRepoInfo {
    #[serde(default)]
    siblings: Vec<HfSibling>,
}

#[derive(Debug, Deserialize)]
pub(super) struct HfSibling {
    rfilename: String,
}

impl HfApi {
    fn new() -> io::Result<Self> {
        Ok(Self {
            cache_dir: oxidize_cache_dir()?.join("hf"),
            agent: ureq::AgentBuilder::new()
                .timeout_connect(Duration::from_secs(30))
                .timeout_read(Duration::from_secs(600))
                .build(),
        })
    }

    fn model(&self, id: String) -> HfRepo {
        HfRepo {
            id,
            cache_dir: self.cache_dir.clone(),
            agent: self.agent.clone(),
        }
    }
}

impl HfRepo {
    fn info(&self) -> io::Result<HfRepoInfo> {
        let url = format!("https://huggingface.co/api/models/{}", self.id);
        self.agent
            .get(&url)
            .call()
            .map_err(|error| io::Error::other(format!("{error}")))?
            .into_json::<HfRepoInfo>()
            .map_err(|error| io::Error::other(format!("{error}")))
    }

    fn get(&self, filename: &str) -> io::Result<PathBuf> {
        let target = self
            .cache_dir
            .join(cache_safe_name(&self.id))
            .join("main")
            .join(filename);
        if target.exists() {
            return Ok(target);
        }
        if let Some(parent) = target.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let url = format!(
            "https://huggingface.co/{}/resolve/main/{}",
            self.id, filename
        );
        let mut response = self
            .agent
            .get(&url)
            .call()
            .map_err(|error| io::Error::other(format!("{error}")))?
            .into_reader();
        let partial = target.with_extension("partial");
        let mut file = std::fs::File::create(&partial)?;
        io::copy(&mut response, &mut file)?;
        std::fs::rename(&partial, &target)?;
        Ok(target)
    }
}

pub(super) fn model_files_for_repo(repo: &HfRepo, spec: &str) -> io::Result<(Vec<String>, Vec<String>)> {
    let info = repo
        .info()
        .map_err(|error| io::Error::other(format!("failed to inspect HF repo {spec}: {error}")))?;
    let mut ggufs = Vec::new();
    let mut safetensors = Vec::new();
    for sibling in info.siblings {
        let name = sibling.rfilename;
        let lower = name.to_ascii_lowercase();
        if lower.ends_with(".gguf") {
            ggufs.push(name);
        } else if lower.ends_with(".safetensors") {
            // Skip LoRA/PEFT adapter shards — they are low-rank deltas, not full
            // weights. Merging them into the base GGUF as standalone tensors
            // corrupts the model (produces real-token gibberish). Only the base
            // model weights should be converted.
            let base = lower.rsplit('/').next().unwrap_or(&lower);
            if base.starts_with("adapter_model") || base.starts_with("adapter.") {
                continue;
            }
            safetensors.push(name);
        }
    }
    ggufs.sort();
    safetensors.sort();
    Ok((ggufs, safetensors))
}

pub(super) fn select_default_gguf(ggufs: &[String]) -> Option<String> {
    const PREFERRED: &[&str] = &[
        "q4_k_m", "q4_k_s", "q4_0", "q5_k_m", "q5_0", "q3_k_m", "q8_0",
    ];
    for needle in PREFERRED {
        if let Some(name) = ggufs
            .iter()
            .find(|name| name.to_ascii_lowercase().contains(needle))
        {
            return Some(name.clone());
        }
    }
    ggufs.first().cloned()
}

pub(super) fn oxidize_cache_dir() -> io::Result<PathBuf> {
    if let Some(home) = std::env::var_os("HOME") {
        Ok(PathBuf::from(home).join(".cache").join("oxidize"))
    } else {
        Ok(std::env::temp_dir().join("oxidize"))
    }
}

pub(super) fn cache_safe_name(spec: &str) -> String {
    spec.chars()
        .map(|ch| if ch.is_ascii_alphanumeric() { ch } else { '-' })
        .collect()
}

pub(super) fn copy_hf_file_to_dir(repo: &HfRepo, filename: &str, dir: &Path) -> io::Result<PathBuf> {
    let source = repo.get(filename).map_err(|error| {
        io::Error::other(format!("failed to download hf file {filename}: {error}"))
    })?;
    let target = dir.join(filename);
    if let Some(parent) = target.parent() {
        std::fs::create_dir_all(parent)?;
    }
    if !target.exists() {
        std::fs::copy(&source, &target)?;
    }
    Ok(target)
}

pub(super) fn convert_hf_safetensors_repo(
    repo: &HfRepo,
    spec: &str,
    safetensors: &[String],
) -> io::Result<PathBuf> {
    let cache_root = oxidize_cache_dir()?
        .join("hf-converted")
        .join(cache_safe_name(spec));
    let source_dir = cache_root.join("source");
    std::fs::create_dir_all(&source_dir)?;
    // The final model is stored as Q8_0 for fast CPU inference (~3x faster than BF16).
    // A BF16 intermediate is kept only as a conversion scratch file.
    let output = cache_root.join("model-q8.gguf");
    if output.exists() {
        eprintln!("using cached converted GGUF {}", output.display());
        return Ok(output);
    }
    // Fall back to the legacy unquantized name if present (created by older versions).
    let legacy = cache_root.join("model.gguf");
    if legacy.exists() {
        eprintln!(
            "found legacy BF16 GGUF {}; requantizing to Q8_0 for faster inference",
            legacy.display()
        );
        return requantize_gguf_to_q8(&legacy, &output);
    }

    eprintln!(
        "hf://{spec}: no .gguf files, downloading {} SafeTensors file(s) for local GGUF conversion",
        safetensors.len()
    );
    for filename in safetensors {
        copy_hf_file_to_dir(repo, filename, &source_dir)?;
    }
    let config_path = match copy_hf_file_to_dir(repo, "config.json", &source_dir) {
        Ok(path) => Some(path),
        Err(error) => {
            eprintln!("hf://{spec}: config.json unavailable during conversion: {error}");
            None
        }
    };
    // The tokenizer ships separately from the weights on HF. Fetch the standard
    // tokenizer files so the converter can embed tokenizer.ggml.* metadata into
    // the GGUF; without them the model loads but has no usable tokenizer.
    for filename in [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ] {
        if let Err(error) = copy_hf_file_to_dir(repo, filename, &source_dir) {
            eprintln!("hf://{spec}: {filename} unavailable during conversion: {error}");
        }
    }

    // Always convert from the directory so the converter can resolve the
    // architecture and tokenizer from config.json / tokenizer.json sitting
    // alongside the weights. Passing a single .safetensors file would hide
    // those and fall back to deriving the arch from the filename.
    let input = source_dir.clone();
    let config = SafetensorsToGgufConfig {
        config_path,
        ..SafetensorsToGgufConfig::default()
    };
    let intermediate = cache_root.join("model.gguf");
    eprintln!("converting hf://{spec} SafeTensors to BF16 GGUF");
    convert_safetensors_to_gguf(&input, &intermediate, &config).map_err(|error| {
        io::Error::other(format!(
            "failed to convert hf://{spec} SafeTensors to GGUF: {error}"
        ))
    })?;
    requantize_gguf_to_q8(&intermediate, &output)
}

pub(super) fn requantize_gguf_to_q8(input: &std::path::Path, output: &std::path::Path) -> io::Result<PathBuf> {
    use oxidize_core::gguf::GgufQuantizationType;
    use oxidize_core::safetensors_to_gguf::quantize_gguf_to_target;
    eprintln!(
        "requantizing {} → Q8_0 (3x faster CPU inference) → {}",
        input.display(),
        output.display()
    );
    let input_bytes = std::fs::read(input)
        .map_err(|e| io::Error::other(format!("failed to read GGUF for requantization: {e}")))?;
    let quantized = quantize_gguf_to_target(&input_bytes, GgufQuantizationType::Q8_0)
        .map_err(|e| io::Error::other(format!("Q8_0 requantization failed: {e}")))?;
    std::fs::write(output, &quantized)
        .map_err(|e| io::Error::other(format!("failed to write Q8_0 GGUF: {e}")))?;
    eprintln!(
        "Q8_0 GGUF written ({:.1} MB) — model ready",
        quantized.len() as f64 / 1_048_576.0
    );
    Ok(output.to_path_buf())
}

pub(super) fn gguf_repo_candidates(spec: &str) -> Vec<String> {
    let mut candidates = Vec::new();
    if spec.ends_with("-GGUF") || spec.ends_with("-gguf") {
        candidates.push(spec.to_owned());
        return candidates;
    }
    candidates.push(format!("{spec}-GGUF"));
    if let Some((_, model)) = spec.rsplit_once('/') {
        candidates.push(format!("bartowski/{model}-GGUF"));
        candidates.push(format!("unsloth/{model}-GGUF"));
        candidates.push(format!("TheBloke/{model}-GGUF"));
    }
    candidates
}

pub(super) fn resolve_hf_model_spec(api: &HfApi, spec: &str, hf_file: Option<&str>) -> io::Result<PathBuf> {
    let mut attempted = Vec::new();
    for candidate in std::iter::once(spec.to_owned()).chain(gguf_repo_candidates(spec)) {
        if attempted.contains(&candidate) {
            continue;
        }
        attempted.push(candidate.clone());
        let repo = api.model(candidate.clone());
        let (ggufs, safetensors) = match model_files_for_repo(&repo, &candidate) {
            Ok(files) => files,
            Err(error) if candidate != spec => {
                eprintln!("hf://{candidate}: {error}; trying next GGUF mirror");
                continue;
            }
            Err(error) => return Err(error),
        };
        let filename = if let Some(file) = hf_file {
            file.to_owned()
        } else if let Some(filename) = select_default_gguf(&ggufs) {
            if ggufs.len() > 1 {
                eprintln!(
                    "hf://{candidate}: selected {filename} (use --file <name> to choose another quant)"
                );
            }
            filename
        } else {
            if candidate == spec && !safetensors.is_empty() {
                return convert_hf_safetensors_repo(&repo, &candidate, &safetensors);
            }
            eprintln!("hf://{candidate}: no .gguf files; trying known GGUF mirrors");
            continue;
        };

        eprintln!("downloading hf://{candidate}/{filename}");
        return repo.get(&filename).map_err(|error| {
            io::Error::other(format!(
                "failed to download hf://{candidate}/{filename}: {error}"
            ))
        });
    }

    Err(io::Error::other(format!(
        "could not find a downloadable GGUF for {spec}; tried: {}. Pass an exact GGUF repo with --file <name> if needed.",
        attempted.join(", ")
    )))
}
