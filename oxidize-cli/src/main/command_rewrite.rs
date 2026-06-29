use super::*;

pub(super) fn has_flag(args: &[OsString], name: &str) -> bool {
    args.iter()
        .any(|arg| arg == name || arg.to_string_lossy().starts_with(&format!("{name}=")))
}

pub(super) fn rewrite_run_args<I>(input: I) -> io::Result<Vec<OsString>>
where
    I: IntoIterator<Item = OsString>,
{
    let raw = input.into_iter().collect::<Vec<_>>();
    match raw.get(1).and_then(|arg| arg.to_str()) {
        Some("-h" | "--help") => {
            print_ollama_help();
            std::process::exit(0);
        }
        Some("list" | "ls") => {
            print_model_list()?;
            std::process::exit(0);
        }
        Some("serve") => return rewrite_serve_args(raw),
        Some("gpu-cluster") => {
            let rest: Vec<String> = raw
                .iter()
                .skip(2)
                .filter_map(|a| a.to_str().map(str::to_string))
                .collect();
            let code = run_gpu_cluster(&rest);
            std::process::exit(code);
        }
        Some("run") => {}
        _ => {
            return Ok(raw);
        }
    }
    if raw.get(1).and_then(|arg| arg.to_str()) != Some("run") {
        return Ok(raw);
    }
    if raw.len() == 2
        || matches!(
            raw.get(2).and_then(|arg| arg.to_str()),
            Some("-h" | "--help")
        )
    {
        print_run_help();
        std::process::exit(0);
    }

    let program = raw[0].clone();
    let mut model: Option<String> = None;
    let mut hf_file: Option<String> = None;
    let mut prompt: Option<OsString> = None;
    let mut rewritten = vec![program];
    let mut args = raw.into_iter().skip(2).peekable();

    while let Some(arg) = args.next() {
        match arg.to_str() {
            Some("--file") | Some("--hf-file") => {
                let Some(file) = args.next() else {
                    return Err(io::Error::other("--file requires a GGUF filename"));
                };
                hf_file = Some(file.to_string_lossy().into_owned());
            }
            Some(value) if value.starts_with("--file=") => {
                hf_file = Some(value["--file=".len()..].to_owned());
            }
            Some(value) if value.starts_with("--hf-file=") => {
                hf_file = Some(value["--hf-file=".len()..].to_owned());
            }
            Some("--api-host") => {
                rewritten.push("--api-host".into());
                let Some(value) = args.next() else {
                    return Err(io::Error::other("--api-host requires a value"));
                };
                rewritten.push(value);
            }
            Some(value) if value.starts_with("--api-host=") => {
                rewritten.push("--api-host".into());
                rewritten.push(value["--api-host=".len()..].into());
            }
            Some("--api-port") => {
                rewritten.push("--api-port".into());
                let Some(value) = args.next() else {
                    return Err(io::Error::other("--api-port requires a value"));
                };
                rewritten.push(value);
            }
            Some(value) if value.starts_with("--api-port=") => {
                rewritten.push("--api-port".into());
                rewritten.push(value["--api-port=".len()..].into());
            }
            Some(value) if !value.starts_with('-') && model.is_none() => {
                model = Some(value.to_owned());
            }
            Some(value) if !value.starts_with('-') && prompt.is_none() => {
                prompt = Some(arg);
            }
            Some("--no-api") => {
                rewritten.push("--no-api".into());
            }
            Some(
                "--prompt"
                | "--model"
                | "--backend"
                | "--n-gpu-layers"
                | "--gpus"
                | "--parallelism"
                | "--lora"
                | "--profile"
                | "--profile-output"
                | "--max-tokens"
                | "--temperature"
                | "--top-p"
                | "--top-k"
                | "--layer-cache"
                | "--ctx-size"
                | "--threads"
                | "--kv-cache-dtype"
                | "--mesh-port"
                | "--pipe-peer"
                | "--pipe-listen"
                | "--pipe-max-tokens"
                | "--tokenizer-model"
                | "--ram-offload-threads",
            ) => {
                rewritten.push(arg);
                let Some(value) = args.next() else {
                    return Err(io::Error::other("option requires a value"));
                };
                rewritten.push(value);
            }
            _ => rewritten.push(arg),
        }
    }

    let Some(model) = model else {
        return Err(io::Error::other(
            "oxidize run requires a model name or local .gguf path",
        ));
    };
    let model_path = resolve_model_spec(&model, hf_file.as_deref())?;
    rewritten.push("--model".into());
    rewritten.push(model_path.into_os_string());
    let one_shot = prompt.is_some();
    if let Some(prompt) = prompt {
        rewritten.push("--prompt".into());
        rewritten.push(prompt);
    } else if !has_flag(&rewritten, "--chat") {
        rewritten.push("--chat".into());
    }

    for flag in ["--cpu-optimized", "--mmap-prefetch", "--mmap-hugepages"] {
        if !has_flag(&rewritten, flag) {
            rewritten.push(flag.into());
        }
    }
    if !has_flag(&rewritten, "--kv-cache-dtype") {
        // f16/f32 are the KV dtypes decode attention can borrow zero-copy
        // (f16 converts in-kernel via F16C); q8 dequantizes the WHOLE K/V
        // prefix into workspace buffers every layer, every token. f16 also
        // halves attention DRAM reads vs f32 as the context grows. Pass
        // --kv-cache-dtype q8 to trade decode speed for memory.
        rewritten.push("--kv-cache-dtype".into());
        rewritten.push("f16".into());
    }
    // One-shot prompt runs exit right after generation, so a background API
    // server would just load the model a second time (concurrently, stealing
    // memory bandwidth from prefill) and die with the process.
    let skip_api = one_shot
        || has_flag(&rewritten, "--no-api")
        || has_flag(&rewritten, "--mesh")
        || has_flag(&rewritten, "--pipe-head")
        || has_flag(&rewritten, "--pipe-tail");
    if !skip_api && !has_flag(&rewritten, "--serve-api") {
        rewritten.push("--serve-api".into());
    }
    Ok(rewritten)
}

pub(super) fn rewrite_serve_args(raw: Vec<OsString>) -> io::Result<Vec<OsString>> {
    if raw.len() >= 3
        && matches!(
            raw.get(2).and_then(|arg| arg.to_str()),
            Some("-h" | "--help")
        )
    {
        print_serve_help();
        std::process::exit(0);
    }

    let program = raw[0].clone();
    let mut rewritten = vec![program, "--serve-api".into(), "--api-only".into()];
    let mut model: Option<String> = None;
    let mut hf_file: Option<String> = None;
    let mut args = raw.into_iter().skip(2).peekable();

    while let Some(arg) = args.next() {
        match arg.to_str() {
            Some("--host") => {
                rewritten.push("--api-host".into());
                let Some(value) = args.next() else {
                    return Err(io::Error::other("--host requires a value"));
                };
                rewritten.push(value);
            }
            Some(value) if value.starts_with("--host=") => {
                rewritten.push("--api-host".into());
                rewritten.push(value["--host=".len()..].into());
            }
            Some("--port") => {
                rewritten.push("--api-port".into());
                let Some(value) = args.next() else {
                    return Err(io::Error::other("--port requires a value"));
                };
                rewritten.push(value);
            }
            Some(value) if value.starts_with("--port=") => {
                rewritten.push("--api-port".into());
                rewritten.push(value["--port=".len()..].into());
            }
            Some("--file") | Some("--hf-file") => {
                let Some(file) = args.next() else {
                    return Err(io::Error::other("--file requires a GGUF filename"));
                };
                hf_file = Some(file.to_string_lossy().into_owned());
            }
            Some(value) if value.starts_with("--file=") => {
                hf_file = Some(value["--file=".len()..].to_owned());
            }
            Some(value) if value.starts_with("--hf-file=") => {
                hf_file = Some(value["--hf-file=".len()..].to_owned());
            }
            Some(value) if !value.starts_with('-') && model.is_none() => {
                model = Some(value.to_owned());
            }
            Some(
                "--model"
                | "--backend"
                | "--max-tokens"
                | "--temperature"
                | "--top-p"
                | "--top-k"
                | "--ctx-size"
                | "--threads"
                | "--kv-cache-dtype"
                | "--tokenizer-model"
                | "--draft-model"
                | "--draft-tokens"
                | "--layer-cache"
                | "--ram-offload-threads",
            ) => {
                rewritten.push(arg);
                let Some(value) = args.next() else {
                    return Err(io::Error::other("option requires a value"));
                };
                rewritten.push(value);
            }
            _ => rewritten.push(arg),
        }
    }

    if let Some(model) = model {
        let model_path = resolve_model_spec(&model, hf_file.as_deref())?;
        rewritten.push("--model".into());
        rewritten.push(model_path.into_os_string());
    }
    if !has_flag(&rewritten, "--kv-cache-dtype") {
        // Match the `run` rewrite: f16 KV is the zero-copy decode path with
        // half the attention reads of f32 (see the comment there).
        rewritten.push("--kv-cache-dtype".into());
        rewritten.push("f16".into());
    }
    if !has_flag(&rewritten, "--cpu-optimized") {
        rewritten.push("--cpu-optimized".into());
    }
    Ok(rewritten)
}
