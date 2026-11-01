# oxidize — Security Model

Threat model for oxidize, as of `595f5fa` (master). Keep it current as surfaces change; security scans start from this file.

oxidize is local-first. The default posture is a single-user process bound to localhost. Binding beyond localhost, sharing an API key across users, or joining a mesh are operator-opt-in.

## 1. System overview

oxidize is a local-first LLM inference, serving, quantization and training engine. The Rust workspace is the product, with parallel ports in C11, C++20, Go and pure Python. Models are loaded from local GGUF/SafeTensors files or fetched from Hugging Face. The servers expose an OpenAI-compatible HTTP/SSE/WebSocket API. Optionally, nodes form a libp2p mesh to shard inference.

Trust boundaries:
1. **Network → server**: HTTP clients reach the `/v1/*` routes of any of the server implementations.
2. **Model file → parser**: GGUF/SafeTensors/tokenizer/config bytes, possibly from an untrusted Hugging Face repo, are parsed by native (unsafe/C) code.
3. **Mesh peer → node**: LAN peers found over mDNS exchange gossip and tensor data with each other.
4. **Host app → FFI/bindings**: C-ABI (`oxidize-ffi`) and PyO3 (`oxidize-py`) callers pass pointers and buffers in.

Infrastructure: no IaC repository. Deployment artifacts are `Dockerfile.server` (non-root `oxidize` user, `EXPOSE 8080`, entrypoint `oxidize-server` with no `--host` so the process still binds `127.0.0.1` inside the container unless the operator passes `--host 0.0.0.0`) and `Dockerfile.cli`.

## 2. Components

| Component | Code | Notes |
|---|---|---|
| **Rust Server** | `oxidize-server/` | Axum app (`app.rs`): `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `/v1/models`, `/v1/mesh/chat/completions`, `/v1/realtime` (WS), plus `/healthz` `/livez` `/readyz` `/metrics` `/openapi.json` |
| **Core Runtime** | `oxidize-core/src/` except `format/` and `mesh/`; `oxidize-kernels/` | Tensor ops, SIMD/unsafe kernels, KV cache, paged attention, backends (CUDA/Metal/Vulkan/WebGPU) |
| **Model Formats** | `oxidize-core/src/format/` (gguf, safetensors, tokenizer, safetensors_to_gguf), `oxidize-quantize/`, `oxidize-convert/`, `oxidize-prune/`, `oxidize-merge/` | Parsers of untrusted model files |
| **Mesh** | `oxidize-core/src/mesh/`, `oxidize-server/src/mesh_cluster.rs`, `routes/mesh.rs` | libp2p TCP+Noise+Yamux, mDNS discovery, GossipSub, leader election |
| **CLI & Model Download** | `oxidize-cli/` (incl. `src/main/model_resolution.rs`), `oxidize-tui/` | Resolves `org/repo` specs against `huggingface.co`, caches under the oxidize cache dir. `oxidize run` starts the HTTP server by default. TUI drives or attaches to that server. |
| **Bindings** | `oxidize-ffi/`, `oxidize-py/` | C-ABI and PyO3 surfaces over oxidize-core |
| **C/C++ Ports** | `oxidize-c/` (incl. `src/server/` HTTP/WS server, `src/format/` parsers), `oxidize-cpp/` (incl. `src/server/main.cpp`) | Memory-unsafe languages, separate parsers and servers |
| **Go/Python Ports** | `oxidize-golang/` (`internal/server`, `internal/auth`, `hf/`), `oxidize-python/` (`oxidize_python/hf/hub.py`, server) | Parallel implementations with their own servers and HF resolvers |
| **Training & Tooling** | `oxidize-finetuning/`, `oxidize-train/`, `oxidize-training/`, `scripts/`, `notebooks/`, root `modal_*.py`, `bench_oxk.py`, `run_codeneedle.sh` | Offline, operator-run |

## 3. Entry points and who can reach them

- **Rust Server / CLI API**: `oxidize-server` and `oxidize-cli --api-host` default to `127.0.0.1` (`cli.rs`, `--host` / `--api-host`). Operators can bind `0.0.0.0`. API-key auth (`auth.rs`) applies only to `/v1/*` and **only when `OXIDIZE_API_KEY` or `OXIDIZE_API_KEYS` is set**. Otherwise every `/v1/*` route is open to anyone who can reach the port. `/healthz`, `/livez`, `/readyz`, `/metrics` and `/openapi.json` are never gated. `/v1/realtime` also accepts `?api_key=`; other Rust routes do not. The body limit is 10 MiB (`MAX_BODY_SIZE_BYTES`). Rate limits live in `limits.rs` (default 64 req/s, 8 in-flight, queue 128).
- **C Server**: CLI defaults to `127.0.0.1` (`oxidize-c/src/cli/args.c`). `http.c` binds `INADDR_ANY` only when the host is NULL or `0.0.0.0`. Auth is in `middleware.c` (`ct_streq`) and is enabled only when `--api-key` is given.
- **C++ Server**: `oxidize-cpp/src/server/main.cpp` defaults to `127.0.0.1`. There is no API-key auth.
- **Go/Python servers**: `oxidize-golang/internal/server` and `oxidize-python` `serve` default to `127.0.0.1`. Auth is `OXIDIZE_API_KEY` only (not `OXIDIZE_API_KEYS`). Unlike Rust, both accept `?api_key=` on every `/v1/*` route.
- **Mesh**: Rust `run_mesh_node` listens on `0.0.0.0:<mesh_port>`; C `oc_mesh_listen` binds `INADDR_ANY`. Go/Python mesh listeners bind `127.0.0.1`. Rust peers are accepted when their namespace string matches (`accept_peer` → `same_namespace`). Noise encrypts traffic but does not authorize peers: there is no shared secret or peer allow-list.
- **Model files**: CLI/server `--model` paths, and `org/repo` specs downloaded from Hugging Face (`model_resolution.rs`, Go `hf/`, Python `hf/hub.py`). Files are memory-mapped and parsed by `format/*.rs` and the C/C++ parsers.
- **FFI/bindings / TUI**: whatever host process links `oxidize-ffi` or imports `oxidize_py`. `oxidize-tui` is a local bun app that talks to the HTTP API.

## 4. Assets

- Host integrity of the machine running inference (native code; GPU drivers). This is the highest-value asset.
- Prompts and completions of API users (potentially private). Prefix-cache / paged-attention KV snapshots are reused across requests with a matching prefix (`prefix_cache.rs`, paged scheduler).
- API keys (`OXIDIZE_API_KEY(S)`), the HF token (`HF_TOKEN`), and files reachable by the server process (model cache, `~/.cache/oxidize`).
- Availability of an expensive inference host (CPU/GPU/RAM).
- Model weights (usually public; may be private finetunes).

## 5. Attacker profiles

In scope:

1. Remote unauthenticated client on the network, when a server is bound beyond localhost without an API key, or reaching ungated routes (`/metrics`, `/openapi.json`, C++ `/v1/*`).
2. Authenticated API user who tries to read other users' data (shared prefix/KV cache, one shared key — no per-user isolation), exhaust resources, or escape limits.
3. Malicious model publisher: a crafted GGUF/SafeTensors/tokenizer/config from a Hugging Face repo, aiming at memory corruption, path traversal in the cache or downloads, or resource exhaustion.
4. Rogue LAN peer that joins the mesh namespace.
5. Supply chain: dependencies, vendored `vendor/libp2p-*`, CI.

## 6. Existing controls

- Constant-time API-key comparison (Rust `subtle`, C `ct_streq`, Go `crypto/subtle`, Python `hmac.compare_digest`). Rust key rotation via `OXIDIZE_API_KEYS`. The query-string key is accepted only on `/v1/realtime` in Rust; Go/Python accept it on all `/v1/*`.
- Localhost default bind on every HTTP server (Rust, C CLI, C++, Go, Python). The Docker images run as a non-root user; the server image still needs `--host 0.0.0.0` to be reachable from outside the container.
- Rust: 10 MiB request-body limit, rate limits and a batching queue (`limits.rs`). Chat/completions emit audit events; `audit_middleware` exists but is not installed on the router.
- Hardened HF download path and parsers across ports (PR #63).
- `cargo-deny` license/advisory audit (`deny.toml`). A fuzz harness exists at `oxidize-core/fuzz/` (GGUF parser, tokenizer).
- Noise-encrypted Rust mesh transport with namespace-isolated topics and election-clock session invalidation.

## 7. Out of scope / accepted risk

- Physical access to the host, a compromised OS, and bugs in GPU drivers / CUDA / Metal / Vulkan.
- Prompt injection as an application-level LLM concern. The engine generates tokens; it does not enforce app policy on model output.
- Resource exhaustion of a localhost single-user process (the operator is the client). Remote DoS after binding beyond localhost is in scope; rate limits are a best-effort control, not a guarantee.
- Mesh over an untrusted network. Mesh is a trusted-LAN / operator-controlled cluster feature. Namespace match is not authentication.
- `scripts/`, `notebooks/`, Modal/Kaggle helpers, and the training crates as a production serving surface. They are in scope for secrets and supply chain in the repo, not as internet-facing APIs.
- Per-user tenancy. A configured API key (or key list) is a shared secret, not an identity.

## 8. Baseline for scans

Treat the facts above as the committed baseline rather than re-deriving them.

Known gaps to score against, not to rediscover:

- Auth is off unless an API key is configured. C++ has no auth at all.
- Ungated `/metrics` and `/openapi.json` (and C++ `/v1/*` on a non-localhost bind).
- Go/Python `?api_key=` on every `/v1/*` route (keys in URLs, proxies, history).
- Rust/C mesh listen on all interfaces with no peer allow-list.
- Shared prefix/KV cache across requests; no user isolation.
- `audit_middleware` is unused; Docker `EXPOSE 8080` does not change the localhost default bind.

Update this file when adding a server, an auth scheme, a parser, or a mesh transport.
