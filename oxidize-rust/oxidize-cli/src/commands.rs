use std::io;
use std::path::PathBuf;

use oxidize_core::gguf::GgufMetadataValue;
use oxidize_core::model_loader::{GgufModelLoader, ModelLoader};

use crate::resolve_model_spec;

pub fn run_show(args: &[String]) -> io::Result<()> {
    let (model, hf_file, verbose) = parse_model_flags(args)?;
    let path = resolve_model_for_cli(&model, hf_file.as_deref())?;
    if !path.is_file() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!("model not found: {}", path.display()),
        ));
    }

    let loader = GgufModelLoader;
    let mapped = loader
        .load(&path)
        .map_err(|e| io::Error::other(format!("failed to load {}: {e}", path.display())))?;
    let meta = &mapped.parsed().metadata;
    let name = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or(&model);

    println!("  Model");
    print_row("name", name);
    print_row("file", &path.display().to_string());
    print_row("format", "gguf");
    if let Some(v) = meta_string(meta, "general.architecture") {
        print_row("architecture", &v);
    }
    if let Some(v) = meta_string(meta, "general.file_type") {
        print_row("quantization", &v);
    }
    if let Some(v) = meta_string(meta, "general.size_label") {
        print_row("parameters", &v);
    } else if let Some(v) = meta_string(meta, "general.parameter_count") {
        print_row("parameters", &v);
    }
    for key in [
        "llama.context_length",
        "qwen2.context_length",
        "gemma.context_length",
    ] {
        if let Some(v) = meta_string(meta, key) {
            print_row("context length", &v);
            break;
        }
    }
    if let Some(v) = meta_string(meta, "general.embedding_length") {
        print_row("embedding length", &v);
    }
    if let Some(v) = meta_string(meta, "llama.block_count") {
        print_row("layers", &v);
    }
    if let Ok(st) = std::fs::metadata(&path) {
        print_row("size", &human_bytes(st.len()));
        print_row("modified", &human_time(st.modified().ok()));
    }
    if verbose {
        print_row("tensors", &mapped.mapped_tensor_infos().len().to_string());
        print_row("gguf version", &mapped.parsed().version.to_string());
    }
    Ok(())
}

pub fn run_pull(args: &[String]) -> io::Result<()> {
    let (model, hf_file, _) = parse_model_flags(args)?;
    eprintln!("pulling {model}...");
    let path = resolve_model_spec(&model, hf_file.as_deref())?;
    println!("success {}", path.display());
    Ok(())
}

pub fn run_inspect(args: &[String]) -> io::Result<()> {
    let (model, hf_file, _) = parse_model_flags(args)?;
    let path = resolve_model_for_cli(&model, hf_file.as_deref())?;
    let loader = GgufModelLoader;
    let mapped = loader
        .load(&path)
        .map_err(|e| io::Error::other(format!("failed to load {}: {e}", path.display())))?;
    println!("Metadata in {}:", path.display());
    for (key, value) in mapped.parsed().metadata.iter() {
        println!("  {key} = {value:?}");
    }
    println!("\nTensors in {}:", path.display());
    for tensor in mapped.mapped_tensor_infos() {
        let qtype = oxidize_core::gguf::GgufQuantizationType::from_ggml_type(tensor.ggml_type);
        let count: usize = tensor.dimensions.iter().map(|&d| d as usize).product();
        let size = oxidize_core::quantization::quantized_size(qtype, count).unwrap_or(0);
        println!(
            "  {} dims={:?} type={:?} offset={} qsize={}",
            tensor.name, tensor.dimensions, qtype, tensor.absolute_offset, size
        );
    }
    Ok(())
}

fn parse_model_flags(args: &[String]) -> io::Result<(String, Option<String>, bool)> {
    let mut model = None;
    let mut hf_file = None;
    let mut verbose = false;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--file" | "--hf-file" => {
                i += 1;
                hf_file = Some(
                    args.get(i)
                        .ok_or_else(|| io::Error::other("--file requires a value"))?
                        .clone(),
                );
            }
            v if v.starts_with("--file=") => hf_file = Some(v["--file=".len()..].to_string()),
            v if v.starts_with("--hf-file=") => {
                hf_file = Some(v["--hf-file=".len()..].to_string())
            }
            "--verbose" | "-v" => verbose = true,
            "-h" | "--help" => {
                println!(
                    "Usage: oxidize show MODEL [--file NAME] [--verbose]\n\
                     \n\
                     Show GGUF model metadata in a readable summary."
                );
                std::process::exit(0);
            }
            v if !v.starts_with('-') && model.is_none() => model = Some(v.to_string()),
            other => {
                return Err(io::Error::other(format!("unknown argument: {other}")));
            }
        }
        i += 1;
    }
    let model = model.ok_or_else(|| io::Error::other("requires a model name or .gguf path"))?;
    Ok((model, hf_file, verbose))
}

fn resolve_model_for_cli(spec: &str, hf_file: Option<&str>) -> io::Result<PathBuf> {
    let path = PathBuf::from(spec);
    if path.is_file() {
        return Ok(path);
    }
    if let Ok(cwd) = std::env::current_dir() {
        let local = cwd.join("models").join(format!("{spec}.gguf"));
        if local.is_file() {
            return Ok(local);
        }
    }
    resolve_model_spec(spec, hf_file)
}

fn print_row(label: &str, value: &str) {
    println!("    {label:<18} {value}");
}

fn meta_string(
    meta: &std::collections::BTreeMap<String, GgufMetadataValue>,
    key: &str,
) -> Option<String> {
    meta.get(key).map(format_meta)
}

fn format_meta(value: &GgufMetadataValue) -> String {
    match value {
        GgufMetadataValue::String(s) => s.clone(),
        GgufMetadataValue::Bool(b) => b.to_string(),
        GgufMetadataValue::Uint8(v) => v.to_string(),
        GgufMetadataValue::Uint16(v) => v.to_string(),
        GgufMetadataValue::Uint32(v) => v.to_string(),
        GgufMetadataValue::Uint64(v) => v.to_string(),
        GgufMetadataValue::Int8(v) => v.to_string(),
        GgufMetadataValue::Int16(v) => v.to_string(),
        GgufMetadataValue::Int32(v) => v.to_string(),
        GgufMetadataValue::Int64(v) => v.to_string(),
        GgufMetadataValue::Float32(v) => v.to_string(),
        GgufMetadataValue::Float64(v) => v.to_string(),
        GgufMetadataValue::Array(a) => format!("{:?}", a.values),
    }
}

fn human_bytes(n: u64) -> String {
    const UNIT: u64 = 1024;
    if n < UNIT {
        return format!("{n} B");
    }
    let mut exp = 0usize;
    let mut val = n as f64;
    while val >= UNIT as f64 && exp < 5 {
        val /= UNIT as f64;
        exp += 1;
    }
    let suffix = ["KiB", "MiB", "GiB", "TiB", "PiB"][exp - 1];
    format!("{val:.1} {suffix}")
}

fn human_time(modified: Option<std::time::SystemTime>) -> String {
    let Some(modified) = modified else {
        return "unknown".to_string();
    };
    match std::time::SystemTime::now().duration_since(modified) {
        Ok(d) if d.as_secs() < 60 => "just now".to_string(),
        Ok(d) if d.as_secs() < 3600 => format!("{} minutes ago", d.as_secs() / 60),
        Ok(d) if d.as_secs() < 86_400 => format!("{} hours ago", d.as_secs() / 3600),
        Ok(d) if d.as_secs() < 604_800 => format!("{} days ago", d.as_secs() / 86_400),
        Ok(_) => "over a week ago".to_string(),
        Err(_) => "just now".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn human_bytes_formats_gib() {
        assert_eq!(human_bytes(4 * 1024 * 1024 * 1024), "4.0 GiB");
    }
}
