# Gemma 4 31B: 1,000 raw TPS for one request

## Metric contract

`raw_tps = target_verified_committed_output_tokens / decode_wall_seconds` for exactly one causal sequence. Timing begins after prompt prefill. A token is counted only after the Gemma 4 31B target verifies and commits it. Draft proposals, rejected tokens, prompt tokens, parallel requests, and loading time are excluded.

## Current evidence

- The existing B=64 CUDA decode probe reaches 1,589.63 slot-tokens/s. It is independent-request batching and is explicitly invalid for this goal.
- The existing scalar target decode is approximately 33 tokens/s on H100. It cannot reach the target alone.
- `oxidize-c` has one CUDA context/device; it has no causal multi-token verifier, NCCL tensor parallelism, learned GPU draft, or CUDA kernels for the project AL low-bit quants.
- Prime currently rejects H100 reservations as `Payment required`; no pod is active.

## Architecture and non-negotiable correctness rules

1. A compact learned Gemma-compatible draft/MTP proposes K=32--64 tokens on GPU.
2. The 31B target verifies the whole causal proposal block using a prefill-style causal attention path. Commit the exact accepted prefix only; on the first mismatch, roll KV state back and generate the target replacement token.
3. Shard target tensors across 2--4 SXM H100s: column-parallel QKV/gate/up; row-parallel output/down/head; NCCL collectives. Require NVLink/NVSwitch topology for the performance claim.
4. Add a calibrated mixed 2--3-bit target representation: higher precision embeddings, norms, output head, attention output, and outlier groups. Use groupwise tensor-core dequant-GEMM, not scalar DP4A. Preserve a target-output parity mode before enabling the compressed path.
5. Use CUDA graphs for fixed K verifier steps once exactness is established.

## Waves

| Wave | Work | Depends on | Exit evidence |
|---|---|---|---|
| 0 | Single-sequence raw benchmark, target-token hashing, corpus manifest and baseline | none | tmux transcript proves only one sequence and separates prefill/decode |
| 1 | Calibrated low-bit format, packer, GEMM kernels, target parity harness | 0 | target logits/token hashes match reference in exact mode; quant tolerance report |
| 2 | Causal K-token target verifier with accepted-prefix commit and KV rollback | 0, 1 | K=1/2/8/32 parity + mismatch rollback transcript |
| 3 | Train/distill MTP/draft, GPU proposal runtime, acceptance instrumentation | 2 | held-out acceptance report and target-only fallback correctness |
| 4 | 2/4 rank TP + NCCL topology gate + CUDA graph capture | 1, 2, 3 | rank-equivalent outputs and topology/collective transcript |
| 5 | Prime 4x H100 proof: two fresh 120s runs and mandatory CLI teardown | 4 | each run >=120,000 committed target tokens in <=120s; all pods absent after teardown |

## Dependency matrix

| Task | Depends on | Blocks |
|---|---|---|
| W0 raw benchmark | none | W1, W2, W5 |
| W1 quant kernels | W0 | W2, W4 |
| W2 verifier/KV | W0, W1 | W3, W4 |
| W3 learned draft | W2 | W4 |
| W4 TP/graphs | W1, W2, W3 | W5 |
| W5 live proof | W4, Prime billing | completion |

## Acceptance tests

- Happy: a tmux-run raw bench reports `sequences=1`, `counted=target_verified_committed`, prefill and decode durations separately, and a deterministic committed token hash.
- Edge: force a draft mismatch at every proposal position and prove rollback, target replacement, and target-only equivalence.
- Regression/adversarial: malformed quant/draft manifests and non-NVLink topology fail closed; repeated cancel/resume cannot count stale/rejected tokens.
- Final: two independent Prime CLI launches on 2--4 H100 SXM/NVLink hardware each produce >=1,000 raw committed TPS for 120 seconds, preserve target verification evidence, and finish with `prime pods list` empty.

## Explicit exclusions

- No aggregate, concurrent-user, slot-token, proposal-token, or prefill throughput may be labelled raw TPS.
- No result from PCIe H100s, CPU fallback, or an unverified draft qualifies.
- No production claim based only on unit tests or simulation.
