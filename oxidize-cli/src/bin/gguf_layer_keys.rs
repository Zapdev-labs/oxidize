use oxidize_core::conversion::gguf_layer_tensor_keys;
use oxidize_core::model_loader::ModelLoader;
use std::env;
use std::path::Path;

fn main() {
    let args: Vec<String> = env::args().collect();
    let path = args
        .get(1)
        .expect("Usage: gguf_layer_keys <model.gguf> [layer_idx]");
    let layer_idx: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(0);

    let loader = oxidize_core::model_loader::GgufModelLoader;
    let mapped = loader.load(Path::new(path)).expect("Failed to mmap GGUF");
    let names: Vec<String> = mapped
        .mapped_tensor_infos()
        .iter()
        .map(|t| t.name.clone())
        .collect();
    let keys = gguf_layer_tensor_keys(names, layer_idx);
    println!("Layer {layer_idx} normalized keys ({}):", keys.len());
    for key in keys {
        println!("  {key}");
    }
}
