#![no_main]

use libfuzzer_sys::fuzz_target;
use oxidize_core::gguf::parse_gguf;

fuzz_target!(|data: &[u8]| {
    // Keep parser allocations bounded during fuzzing runs.
    if data.len() > 1 << 20 {
        return;
    }
    let _ = parse_gguf(data);
});
