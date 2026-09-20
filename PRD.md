# Product Requirements Document

**Project:** oxidize — C11 local LLM inference  
**UI:** TypeScript/Bun TUI (`oxidize-tui`)  
**Runtime:** `oxidize-c` owns load, generate, quantize, prune, convert, and the OpenAI HTTP API.

## Goals

- One C11 binary for CLI and HTTP. No Rust/Go/Python/C++ product dependency.
- TUI talks HTTP/SSE to `oxidize-c serve`; it does not link the engine.
- Default `make build` / `make test` / Docker / GitHub CI are C (+ TUI tests).

## Non-goals

- Shipping other language ports as the engine.
- Requiring cargo, go, or python to build or run inference.
