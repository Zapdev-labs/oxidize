# oxidize-core/src/video

**Domain:** Video multimodal path — frame sampling, encoding, temporal aggregation, prompt assembly

## OVERVIEW
Video analogue of `vision/`: samples frames from video, preprocesses/encodes each frame, aggregates temporally, and assembles multimodal prompts. Requires `ffmpeg` for decode (shell-out or availability check).

## STRUCTURE
```
video/
├── mod.rs
├── config.rs         # VideoConfig, TemporalConfig, FrameSamplingStrategy, TemporalPool
├── decoder.rs        # video decode (ffmpeg)
├── encoder.rs        # per-frame encoding
├── frame_sampler.rs  # temporal frame sampling
├── preprocess.rs     # per-frame preprocessing
├── prompt.rs         # multimodal prompt assembly (video + text)
├── temporal.rs       # temporal aggregation across frames
└── error.rs
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Frame sampling strategy | `frame_sampler.rs` | How many frames, which timestamps |
| Per-frame encode | `encoder.rs` | Analogous to `vision/encoder.rs` |
| Temporal fusion | `temporal.rs` | Aggregate frame embeddings over time |
| Prompt assembly | `prompt.rs` | Interleave video + text segments |
| ffmpeg integration | `decoder.rs` | Decode video to frames |

## NOTES
- For static-image multimodal, see `oxidize-core/src/vision/AGENTS.md`.
- Training/generation CLI lives in `oxidize-train` (`video` subcommands).
