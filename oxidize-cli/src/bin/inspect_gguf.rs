use std::env;
use std::path::Path;

fn main() {
    let args: Vec<String> = env::args().collect();
    let path = args.get(1).expect("Usage: inspect_gguf <model.gguf>");
    use oxidize_core::model_loader::ModelLoader;
    let loader = oxidize_core::model_loader::GgufModelLoader;
    let mapped = loader.load(Path::new(path)).expect("Failed to load GGUF");
    println!("Metadata in {}:", path);
    for (key, value) in mapped.parsed().metadata.iter() {
        println!("  {} = {:?}", key, value);
    }
    println!("\nTensors in {}:", path);
    for tensor in mapped.mapped_tensor_infos() {
        let qtype = oxidize_core::gguf::GgufQuantizationType::from_ggml_type(tensor.ggml_type);
        let count: usize = tensor.dimensions.iter().map(|&d| d as usize).product();
        let size = oxidize_core::quantization::quantized_size(qtype, count).unwrap_or(0);
        println!(
            "  {} dims={:?} type={:?} offset={} qsize={}",
            tensor.name, tensor.dimensions, qtype, tensor.absolute_offset, size
        );
    }
}
