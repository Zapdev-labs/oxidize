use std::io;

pub fn print_run_help() {
    println!(
        "Usage: oxidize run <model> [prompt] [options]\n\n\
         Models can be local .gguf files or Hugging Face GGUF repos.\n\n\
         Examples:\n\
           oxidize run ./models/model.gguf \"hello\"\n\
           oxidize run Qwen/Qwen2.5-0.5B-Instruct-GGUF --file qwen2.5-0.5b-instruct-q4_k_m.gguf --chat\n\
           oxidize run TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF \"write a haiku\" --max-tokens 128\n\n\
         Common options: --chat, --prompt, --max-tokens, --temperature, --backend, --threads, --no-api"
    );
}

pub fn print_serve_help() {
    println!(
        "Usage: oxidize serve [model] [options]\n\n\
         Starts the OpenAI-compatible API server.\n\n\
         Examples:\n\
           oxidize serve ./models/Qwen3-4B-Q4_K_M.gguf\n\
           oxidize serve --host 0.0.0.0 --port 11434\n\
           oxidize serve ./models/model.gguf --temperature 0 --top-k 1\n\n\
         Common options: --host, --port, --model, --max-tokens, --temperature, --top-p, --top-k, --threads"
    );
}

pub fn print_ollama_help() {
    println!(
        "Usage: oxidize <command> [args]\n\n\
         Commands:\n\
           run <model> [prompt]     Run a model locally\n\
           chat <model>             Interactive chat (alias for run --chat)\n\
           pull <model>             Download a model from Hugging Face\n\
           show <model>             Show model information\n\
           list                     List local GGUF models in ./models\n\
           inspect <model>          Print raw GGUF metadata and tensors\n\
           serve [model]            Start the OpenAI-compatible server\n\n\
         Examples:\n\
           oxidize run ./models/Qwen3-4B-Q4_K_M.gguf \"hello\"\n\
           oxidize pull Qwen/Qwen2.5-0.5B-Instruct-GGUF\n\
           oxidize show qwen2.5-0.5b\n\
           oxidize list"
    );
}

pub fn print_model_list() -> io::Result<()> {
    let models_dir = std::env::current_dir()?.join("models");
    let mut rows = Vec::new();
    if models_dir.is_dir() {
        for entry in std::fs::read_dir(&models_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path
                .extension()
                .and_then(|ext| ext.to_str())
                .is_some_and(|ext| ext.eq_ignore_ascii_case("gguf"))
            {
                let metadata = entry.metadata()?;
                let size_gib = metadata.len() as f64 / 1024.0 / 1024.0 / 1024.0;
                rows.push((path, size_gib));
            }
        }
    }
    rows.sort_by(|a, b| a.0.cmp(&b.0));
    println!("{:<48} {:>9} PATH", "NAME", "SIZE");
    for (path, size_gib) in rows {
        let name = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("<invalid>");
        println!("{name:<48} {size_gib:>8.2}G {}", path.display());
    }
    Ok(())
}
