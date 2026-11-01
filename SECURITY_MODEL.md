# oxidize — Security Model

_Threat model for oxidize, built from the code at `595f5fa` (master). It is still a draft: the deployment, attacker and out-of-scope sections are inferred from the code and have not yet been confirmed by the maintainers (see Open questions). Keep it up to date as surfaces change; security scans start from this file._

## 1. System overview

oxidize is a local-first LLM inference, serving, quantization and training engine. The Rust workspace is the product, with parallel ports in C11, C++20, Go and pure Python. Models are loaded from local GGUF/SafeTensors files or fetched from Hugging Face. The servers expose an OpenAI-compatible HTTP/SSE/WebSocket API. Optionally, nodes form a libp2p mesh to shard inference.

Trust boundaries:
1. **Network → server**: HTTP clients reach the `/v1/*` routes of any of the server implementations.
2. **Model file → parser**: GGUF/SafeTensors/tokenizer/config bytes, possibly from an untrusted Hugging Face repo, are parsed by native (unsafe/C) code.
3. **Mesh peer → node**: LAN peers found over mDNS exchange gossip and tensor data with each other.
4. **Host app → FFI/bindings**: C-ABI (`oxidize-ffi`) and PyO3 (`oxidize-py`) callers pass pointers and buffers in.

Infrastructure: no IaC repository. The only deployment artifacts are `Dockerfile.server` (non-root `oxidize` user, `EXPOSE 8080`) and `Dockerfile.cli`.

## 2. Components

| Component | Code | Notes |
|---|---|---|
| **Rust Server** | `oxidize-server/` | Axum app (`app.rs`): `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `/v1/models`, `/v1/mesh/chat/completions`, `/v1/realtime` (WS), plus `/healthz` `/livez` `/readyz` `/metrics` `/openapi.json` |
| **Core Runtime** | `oxidize-core/src/` except `format/` and `mesh/`; `oxidize-kernels/` | Tensor ops, SIMD/unsafe kernels, KV cache, paged attention, backends (CUDA/Metal/Vulkan/WebGPU) |
| **Model Formats** | `oxidize-core/src/format/` (gguf, safetensors, tokenizer, safetensors_to_gguf), `oxidize-quantize/`, `oxidize-convert/`, `oxidize-prune/`, `oxidize-merge/` | Parsers of untrusted model files |
| **Mesh** | `oxidize-core/src/mesh/`, `oxidize-server/src/mesh_cluster.rs`, `routes/mesh.rs` | libp2p TCP+Noise+Yamux, mDNS discovery, GossipSub, leader election |
| **CLI & Model Download** | `oxidize-cli/` (incl. `src/main/model_resolution.rs`), `oxidize-tui/` | Resolves `org/repo` specs against `huggingface.co`, caches under the oxidize cache dir |
| **Bindings** | `oxidize-ffi/`, `oxidize-py/` | C-ABI and PyO3 surfaces over oxidize-core |
| **C/C++ Ports** | `oxidize-c/` (incl. `src/server/` HTTP/WS server, `src/format/` parsers), `oxidize-cpp/` (incl. `src/server/main.cpp`) | Memory-unsafe languages, separate parsers and servers |
| **Go/Python Ports** | `oxidize-golang/` (`internal/server`, `internal/auth`, `hf/`), `oxidize-python/` (`oxidize_python/hf/hub.py`, server) | Parallel implementations with their own servers and HF resolvers |
| **Training & Tooling** | `oxidize-finetuning/`, `oxidize-train/`, `oxidize-training/`, `scripts/`, `notebooks/`, root `modal_*.py`, `bench_oxk.py`, `run_codeneedle.sh` | Offline, operator-run |

## 3. Entry points and who can reach them

- **Rust Server**: binds `127.0.0.1` by default (`cli.rs`, `--host`). Operators can bind `0.0.0.0`; the Docker image exposes 8080. API-key auth (`auth.rs`) applies only to `/v1/*` and **only when `OXIDIZE_API_KEY(S)` is set**. Otherwise every `/v1/*` route is open to anyone who can reach the port. `/metrics` and `/openapi.json` are never gated. `/v1/realtime` also accepts `?api_key=`. The body limit is 10 MiB (`MAX_BODY_SIZE_BYTES`). Rate limits live in `limits.rs`.
- **C Server**: `oxidize-c/src/server/http.c` binds `INADDR_ANY` when the host is NULL or `0.0.0.0`. Auth is in `middleware.c` (`ct_streq`) and is enabled only when a key is given.
- **C++ Server**: `oxidize-cpp/src/server/main.cpp` defaults to `127.0.0.1`.
- **Go/Python servers**: `oxidize-golang/internal/server`, `oxidize-python` `serve` (default `127.0.0.1`).
- **Mesh**: `run_mesh_node` listens on `0.0.0.0:<mesh_port>`. Peers are accepted when their namespace string matches (`accept_peer` → `same_namespace`). Noise encrypts traffic but does not authorize peers, so there is no shared secret or peer allow-list.
- **Model files**: CLI/server `--model` paths, and `org/repo` specs downloaded from Hugging Face (`model_resolution.rs`, Go `hf/`, Python `hf/hub.py`). Files are memory-mapped and parsed by `format/*.rs` and the C/C++ parsers.
- **FFI/bindings**: whatever host process links `oxidize-ffi` or imports `oxidize_py`.

## 4. Assets

- Host integrity of the machine running inference (native code; GPU drivers). This is the highest-value asset.
- Prompts and completions of API users (potentially private), plus KV/prefix caches shared between requests.
- API keys (`OXIDIZE_API_KEY(S)`), the HF token (`HF_TOKEN`), and files reachable by the server process (model cache, `~/.cache/oxidize`).
- Availability of an expensive inference host (CPU/GPU/RAM).
- Model weights (usually public; may be private finetunes).

## 5. Attacker profiles (assumed; not confirmed)

1. Remote unauthenticated client on the network, when a server is bound beyond localhost without an API key, or reaching ungated routes.
2. Authenticated API user who tries to read other users' data, exhaust resources or escape limits.
3. Malicious model publisher: a crafted GGUF/SafeTensors/tokenizer/config from a Hugging Face repo, aiming at memory corruption, path traversal in the cache or downloads, or resource exhaustion.
4. Rogue LAN peer that joins the mesh namespace.
5. Supply chain: dependencies, vendored `vendor/libp2p-*`, CI.

## 6. Existing controls

- Constant-time API-key comparison (Rust `subtle`, C `ct_streq`). Key rotation via `OXIDIZE_API_KEYS`. The query-string key is accepted only on `/v1/realtime`.
- Localhost default bind (Rust, C++, Python servers). The Docker image runs as a non-root user.
- 10 MiB request-body limit, rate limits and a batching queue (`limits.rs`), audit/metrics middleware.
- Hardened HF download path and parsers across ports (PR #63).
- `cargo-deny` license/advisory audit (`deny.toml`). A fuzz harness exists at `oxidize-core/fuzz/`.
- Noise-encrypted mesh transport with namespace-isolated topics and election-clock session invalidation.

## 7. Out of scope / accepted risk

_None stated by the owner yet._

## 8. Open questions

1. How the server is really deployed: localhost-only, trusted LAN, or internet-facing multi-user?
2. Which attackers matter, and is a malicious Hugging Face model file in scope?
3. Are `scripts/`, `notebooks/`, the Modal/Kaggle helpers, the training crates, or the Go/Python ports out of scope?
4. Is DoS on a local single-user server accepted risk?
5. Is the mesh used only on trusted LANs?
