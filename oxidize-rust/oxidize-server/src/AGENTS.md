# oxidize-server/src/

**Generated:** 2026-05-26
**Domain:** OpenAI-compatible HTTP API (Axum)

## OVERVIEW
Axum-based HTTP server exposing `/v1/*` OpenAI-compatible endpoints with optional mesh clustering, rate limiting, and API-key auth.

## STRUCTURE
```
src/
├── app.rs           # Router assembly (888 lines). AppState holds all middleware state
├── routes/          # 8 endpoint modules: chat, completions, embeddings, models, health, mesh
├── runtime/         # Model runtime: standard + paged attention + generation loop
├── auth.rs          # API-key middleware (x-api-key or Authorization: Bearer)
├── limits.rs        # Rate limiting + continuous batching wait queue
├── mesh_cluster.rs  # Mesh master node state and runner aggregation
├── cli.rs           # Server CLI arguments
├── logging.rs       # Request/response logging middleware
├── metrics.rs       # Prometheus metrics registry + middleware
├── audit.rs         # Audit log middleware
├── schema.rs        # OpenAPI schema types
├── openapi.rs       # OpenAPI JSON document handler
├── shutdown.rs      # Graceful shutdown signal handling
└── lib.rs / main.rs # Crate entry points
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Add endpoint | `routes/` + `app.rs` | Register in `build_app_with_state()` |
| Change auth | `auth.rs` | Constant-time comparison, skips `/healthz` |
| Rate limits | `limits.rs` | `StdMutex` in async context — see anti-patterns |
| Runtime model | `runtime/model.rs` | Bridges core `Model` trait to HTTP |
| Paged attention | `runtime/paged.rs` | vLLM-style scheduler integration |
| Streaming gen | `runtime/generate.rs` | SSE stream construction |
| Mesh routing | `routes/mesh.rs` + `mesh_cluster.rs` | Master-only; workers use core mesh |

## CONVENTIONS
- **Middleware stacking order matters**: auth -> limits -> audit -> metrics -> route
- **Route modules are thin**: business logic delegates to `runtime/`, responses use `routes/responses.rs`
- **AppState is monolithic**: single struct passed to all handlers; no per-route state
- **Test co-location**: `#[cfg(test)]` modules at bottom of files (no `tests/` dir)

## ANTI-PATTERNS
- `StdMutex` in async context (`limits.rs`, `mesh_cluster.rs`) — should be `tokio::sync::Mutex`
- `app.rs` at 888 lines — router construction + state definition mixed; refactor candidate
- `unwrap()`/`expect()` in non-test route handlers — use `?` or proper error mapping
- `Option<Arc<T>>` for optional runtimes (model, paged, mesh) — consider enum dispatch
