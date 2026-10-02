# SECURITY-REMEDIATION-SPEC.md

Security remediation specification — deep security review run 2026-10-01 against commit `cbb2b91d`.
Findings source: `~/security-audits/oxidize-20260930/findings.json` (canonical record set); this file converts them into executable work items.

## Conventions

- **Work item:** one `FINDING-ID`, title, target files, expected outcome, test plan, and done-criteria. Each item is scoped to be completed and verified in isolation.
- **Priority order:** P1 first, then P2 grouped by area, then P3 hygiene. Items marked *(deferred)* are recorded here but out of scope for this pass.
- **Verification gate:** every code change must pass `cargo build -p oxidize-cli -p oxidize-server -p oxidize-kernels && cargo test -p oxidize-core -p oxidize-server -p oxidize-kernels` plus the relevant per-crate checks (`make -C oxidize-c test`, `CGO_ENABLED=0 go test ./...`, `uv run pytest`) before merge. Benchmark-gated paths (`oxidize-c` hot path) additionally require a before/after benchmark on `ai@192.168.1.132` — no throughput regression allowed.

---

## P1 — fix immediately

### FINDING-supply-5: Remove committed shared SSH password from bench scripts

**Target:** `scripts/bench-ai-box.sh:7,12`, `scripts/run_kimi_k26_dflash_ai.sh:33`, `scripts/run-minimax-m3-eagle3.sh:5-6,16`

**Fix (code):**
1. Replace the credential fallback with a hard failure and key-based auth:
   ```bash
   HOST="${OXIDIZE_AI_HOST:?OXIDIZE_AI_HOST required}"
   # sshpass fallback kept only for hosts that cannot take keys yet; value never has a default.
   if [[ -z "${OXIDIZE_AI_PASS:-}" ]]; then
     echo "ssh: key-based auth required, or export OXIDIZE_AI_PASS" >&2
     exit 1
   fi
   sshpass -p "$PASS" ssh -o StrictHostKeyChecking=accept-new "$HOST" "$@"
   ```
2. Migrate every affected script to SSH keys; keep `sshpass` behind an explicit env opt-in, never a default.

**Rotations (must run before merge):**
1. Rotate the lab password on every host reachable with it (`ai@192.168.1.132`, `.121`).
2. Re-issue any HF tokens/keys provisioned through these scripts.
3. Purge the literal from git history if the repo was ever pushed with it (`git log -S 'machine' --oneline -- scripts/` first to size it).

**Verification:** `rg 'PASS="-|sshpass|-o StrictHostKeyChecking=no' scripts/` returns nothing after migration; bench scripts still run against the boxes end-to-end.

### FINDING-server-001: Fail-closed server auth when bound to routable interfaces

**Target:** `oxidize-server/src/auth.rs:84-95`, `oxidize-cli/src/main/server.rs:92`, `oxidize-c/src/server/middleware.c` (same posture)

**Fix (code):**
1. At startup in both Rust and C servers, if auth is disabled and the bind address is not loopback, emit `tracing::warn!` / `oc_log(OC_LOG_WARN, ...)` stating exactly what is exposed.
2. Add `--require-auth-on-public-bind` opt-in that refuses to start when auth is disabled on a non-loopback bind (default off; document enabling it for LAN deployments — AGENTS.md says servers run LAN-reachable by design, so the *warning* is default behavior, not the refusal).
3. Document the posture in `AGENTS.md` "Server deployment" note.

**Verification:** integration test: server started with no key + `0.0.0.0` emits the warning line and exits 2 under `--require-auth-on-public-bind`; loopback binds never warn.

---

## P2 — fix this cycle

### FINDING-c-engine-001: Constant-time API-key comparison in C middleware

**Target:** `oxidize-c/src/server/middleware.c:68` (`auth_check`)

**Fix:** replace `strcmp(p, auth->api_key)` with a constant-time byte compare over equal length first:
```c
/* constant_time_eq: lengths checked in constant time via flag fold, bytes XOR-folded */
static bool ct_eq(const char *a, const char *b) {
    volatile unsigned char diff = 0; ...
}
```
Mirror of the Rust side (`auth.rs` already uses `constant_time_eq`); reuse the existing pattern rather than inventing a second one.

**Verification:** unit test asserting equal-length mismatch takes same wall-clock order as full match (loose timing assert acceptable), plus all existing middleware tests green.

### FINDING-rust-core-001: Guard GGUF header allocation sizes against hostile u64 counts

**Target:** `oxidize-core/src/format/gguf.rs:624,628`

**Fix:** before allocating, validate `tensor_count`/`metadata_count` and per-tensor `n_dimensions` against file size, matching the C parser's guard at `oxidize-c/src/format/gguf.c:361`:
```rust
if tensor_count > bytes.len() as u64 || metadata_count > bytes.len() as u64 {
    return Err(GgufParseError::IntegerOverflow /* or InvalidHeader */);
}
```
Also cap `n_dimensions` (C side uses `OC_GGUF_MAX_DIMS`; mirror that constant into Rust so both parsers reject identically).

**Verification:** new fixture `invalid-tensor-count.gguf` (u64 tensor_count = u64::MAX) loads to `Err`, never allocates; add parity test asserting C and Rust parsers reject the same file.

### Remaining P2s — remediation notes

| ID | File | Fix summary | Verify with |
|---|---|---|---|
| server-002 | `auth.rs:96-99` | Prefer header auth; query-string key redacted in access logs (`?api_key=` scrubbed by logging middleware) | unit test on log formatter |
| server-003 | `audit.rs:112-118` | Replace SipHash digest with `sha2::Sha256` truncated-or-full keyed hash (HMAC) of API key | unit test vectors |
| server-004 | `app.rs:60-77` | Layer `audit_middleware` into the router; wire `ConnectInfo` so it functions | existing audit tests must now pass through real routing |
| server-005 | `app.rs:52`, `limits.rs:194-215` | Exclude `/healthz`,`/livez`,`/readyz` from rate limiter; gate `/metrics` behind the auth layer or bind metrics to loopback separately | update rate-limit test expectations |
| server-006 | `realtime/session.rs:67,74` | Cap session item count and total transcript bytes; evict oldest with a counter, matching KV-cache caps elsewhere | property test: N>cap appends bounded memory |
| server-009 | `oxidize-ffi/src/lib.rs:93-107,248` | Null-check all raw pointers before `from_raw_parts`; wrap core calls in `catch_unwind(AssertUnwindSafe)` boundary per existing `oxidize_model_forward` pattern | C-side FFI smoke tests calling with null/zero-len inputs |
| supply-1 | `.github/workflows/*.yml` | Pin every third-party action to a full commit SHA (`actions/checkout@<sha>`); add Dependabot for actions updates | `actionlint` + manual pin diff |
| supply-4 | `scripts/*/setup-node.sh`, `bench_iq_remote.sh` | Replace `curl \| sh` installers with pinned, checksum-pinned downloads (rustup `--profile minimal --toolchain <pin>`, docker via apt repo with keyring) | provisioning dry-run script |
| supply-6 | `scripts/kaggle_tpu_k2/build/bootstrap_cell.py`, `runner.py` | Move `hf_token`/`hmac_key` out of file contents into env/secrets at runtime; move gist payload into `scripts/kaggle_tpu_k2/k2_runner_payload.py` reviewed in-repo | notebook builds without embedded secrets |

---

## P3 — hygiene (schedule normally)

| ID | Area | Fix |
|---|---|---|
| supply-2 | `deny.toml:34-35` | After cleaning existing floating ranges, flip `wildcards = "allow"` → `"deny"`; keep Cargo.lock + checksums as compensating control |
| supply-7 | `[patch.crates-io]` vendor dir | Add a `vendor/UPSTREAM.md` provenance manifest mapping each vendored crate to upstream SHA; CI job diffs vendored trees against upstream tags |
| supply-8 | `Dockerfile.cli:1`, `Dockerfile.server:1` | Pin base images by digest (`rust:1.95-bookworm@sha256:...`); stop publishing mutable `latest`; publish digest-pinned tags |
| server-007 | `realtime/session.rs:99,106-125` | *(deferred)* tool-schema injection is accepted-by-design for local single-tenant inference; document the multi-tenant caveat in the module doc comment only |
| server-008 | `generate.rs:163-168` | *(deferred)* image-content channel documented as LLM01-equivalent of message content; no code change |
| server-010 | `chat.rs`, `generate.rs:206` | Add optional `max_tokens` clamp via runtime config (default unlimited to preserve behavior) |

---

## Verification and reporting

- One branch per finding group; commits reference the finding ID (`fix(auth): constant-time compare (FINDING-c-engine-001)`).
- After each P1/P2 batch: run full CI locally (`make ci`) and record results in this file's Progress table.
- Supply-chain fixes (supply-*) don't touch hot paths — benchmark gate does not apply to them.

## Progress tracker

| Finding | Priority | Owner | Status |
|---|---|---|---|
| supply-5 | P1 | Claude | fixed (code); `.121`/`.132` sold — no rotation needed; history purge declined (literal is on `origin`) |
| server-001 | P1 | Claude | fixed (Rust + C) |
| c-engine-001 | P2 | Claude | fixed |
| rust-core-001 | P2 | Claude | fixed (+ string/array length + nested-array guards, C parity test) |
| server-002..006, 009 | P2 | Claude | fixed |
| supply-1,4,6 | P2 | Claude | fixed; supply-6 needs Kaggle Secrets `K2_HF_TOKEN`/`K2_HMAC_KEY` and deletion of stale `build/` files |
| supply-2,7,8 | P3 | Claude | fixed |
| server-007,008 | P3 deferred | Claude | documented only |
| server-010 | P3 | Claude | fixed (`--max-tokens-cap`, opt-in) |
