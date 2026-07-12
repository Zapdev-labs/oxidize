# scripts

**Domain:** CI benchmark gating, remote NUMA benchmarking, quant/publish/run recipes

## OVERVIEW
Mixed Bash + Python helper scripts. No build target — these support CI regression gating, remote benchmarking on the NUMA boxes, GGUF quant/merge/publish recipes, and one-off model run/setup flows. Python scripts have co-located `test_*.py` unit tests.

## GROUPS
| Group | Files | Purpose |
|-------|-------|---------|
| CI benchmark gate | `ci_benchmark_regression.py`, `ci_benchmark_dashboard.py` (+ `test_*.py`) | Detect perf regressions; render dashboard |
| Remote NUMA bench | `bench-ai-box.sh` (defaults to `ai@192.168.1.132`), `bench_al5_remote.sh`, `bench_iq_remote.sh`, `bench_vs_llamacpp.sh`, `*_al_remote.sh`, `qwen*_remote.sh` | Build + benchmark on remote hosts |
| Quant / merge / publish | `publish_gguf_hf.py`, `publish_gguf_remote_hf.sh`, `glm_merge_exporter.py`, `glm_stream_merge.py`, `run_glm_merge_remote.sh` | Quantize, merge, and publish GGUF to HF |
| DFlash / speculative | `build_kimi26_dflash_from_base.py`, `build_nex_n2_pro_dflash_baseinit.py`, `run_kimi_k26_dflash_ai.sh` | Build/run DFlash draft models |
| Model setup / run | `setup-glm-5.2.sh`, `glm-52-prune.sh`, `glm-vision-download.sh`, `gen-train-sequence.sh`, `run-minimax-m3-eagle3.sh`, `minimax-m3-eagle3-wait-and-run.sh` (+ `minimax-m3-manifest.json`) | Download/prune/run specific models |
| Repo tooling / tests | `check-udeps.sh`, `test_cargo_deny_setup.sh`, `test_makefile_targets.sh` | Dependency + Makefile checks |
| Sub-recipes | `glm-5.2-eagle3/`, `prime-gemma4-31b-int4/` | Model-specific recipe dirs |

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Change CI perf gate | `ci_benchmark_regression.py` | Run its `test_*.py` after edits |
| Remote bench defaults | `bench-ai-box.sh` | Host defaults to `.132` (2× Xeon Gold 5220R) |
| Publish quant to HF | `publish_gguf_hf.py` / `publish_gguf_remote_hf.sh` | Repos private unless user asks otherwise |

## RUN
```bash
python scripts/ci_benchmark_regression.py ...
python -m pytest scripts/test_ci_benchmark_regression.py
bash scripts/bench-ai-box.sh                 # remote NUMA benchmark
```

## NOTES
- Remote hosts: `ai@192.168.1.132` (primary NUMA bench), `ai@192.168.1.121` (~20 TB storage for large-model quant + HF publish), legacy `ai@192.168.1.68`.
- Custom HF repos for quant/model publishing should be private unless the user explicitly requests public.
