# oxidize-cli

**Generated:** 2026-06-03
**Domain:** Command-line interface for prompt/chat, benchmarking, profiling

## OVERVIEW
Interactive CLI for LLM inference: prompt mode, chat REPL, benchmarking, profiling, and pipeline execution. Thin frontend over `oxidize-core` APIs. Also contains diagnostic tools in `src/bin/`.

## STRUCTURE
```
oxidize-cli/
├── Cargo.toml
├── src/
│   ├── main.rs          # CLI entry: clap argument parsing, subcommand dispatch (2,457 lines)
│   ├── pipeline.rs      # Pipeline mode: batch processing of prompts from files/stdin
│   └── bin/
│       ├── bench.rs          # Standalone benchmark binary
│       └── inspect_gguf.rs  # GGUF metadata inspection tool
└── tests/
    └── cli_binary.rs    # Integration test for CLI binary
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Add CLI subcommand | `main.rs` | Extend clap `Commands` enum + dispatch match |
| Change prompt mode | `main.rs` | `run_prompt_mode()`, `run_chat_mode()` |
| Add benchmark | `bin/bench.rs` | Or `main.rs` benchmark subcommand |
| Pipeline processing | `pipeline.rs` | Batch prompt processing, file input |
| GGUF inspection | `bin/inspect_gguf.rs` | Standalone diagnostic tool |
| Profiling integration | `main.rs` | `PROFILE_CHILD_ENV` for subprocess profiling |

## CONVENTIONS
- **Single binary, many modes**: `main.rs` dispatches to `run_prompt_mode()`, `run_chat_mode()`, `run_benchmark_mode()`, etc.
- **REPL for chat**: Uses `std::io::BufRead` loop; handles Ctrl-C gracefully.
- **Pipeline for batch**: `pipeline.rs` processes lines from stdin or file, supports JSON output.
- **Aux tools in bin/**: `bench.rs` and `inspect_gguf.rs` are standalone binaries included via `[[bin]]` in `Cargo.toml`.

## RUN
```bash
# Prompt mode
sfw cargo run -p oxidize-cli -- --prompt "Hello, world!" --model path/to/model.gguf

# Chat REPL
sfw cargo run -p oxidize-cli -- --chat --model path/to/model.gguf

# Benchmark
sfw cargo run -p oxidize-cli -- bench --model path/to/model.gguf --iterations 10

# GGUF inspection
sfw cargo run -p oxidize-cli --bin inspect_gguf -- path/to/model.gguf
```

## ANTI-PATTERNS
- `main.rs` at 2,457 lines — mixes argument parsing, mode dispatch, REPL logic, and profiling. Refactor candidate.
- Hardcoded debug log path to `/home/dih/oxidize/.cursor/debug-49b0b9.log` — should use `tracing` or be removed.
- `unwrap()` in CLI argument validation — use `clap`'s built-in validation or `anyhow::bail()`.
- Chat mode does not persist history across sessions — consider readline integration (e.g., `rustyline`).
