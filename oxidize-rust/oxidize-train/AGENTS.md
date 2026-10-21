# oxidize-train

**Domain:** CPU CSV classifier + generative video training

## OVERVIEW
Rust CLI + library for CPU training: an MLP classifier over CSV features, a video clip classifier (virality / engagement / creator labels), and an autoregressive next-frame video generator. Video paths require `ffmpeg` on `PATH`.

## STRUCTURE
```
oxidize-train/
├── Cargo.toml
└── src/
    ├── main.rs     # clap CLI: Csv / Video / GenTrain / Generate
    ├── lib.rs      # TrainingConfig, load_csv_dataset, train_classifier
    └── video/      # FrameConfig, VideoTrainingConfig, GenTrainingConfig,
                    # load/train video classifier + generator, generate_video, ffmpeg_available
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| CSV classifier | `src/lib.rs` | MLP; `train_classifier`, `load_csv_dataset` |
| Video classifier | `src/video/` | `train_video_classifier`, `LabelTask` |
| Video generation | `src/video/` | `train_generator`, `generate_video` |
| ffmpeg detection | `src/video/` | `ffmpeg_available()` |

## CLI
```text
oxidize-train csv       --train-csv <f> [--label-column N --epochs 20 --batch-size 32
                         --learning-rate 1e-3 --weight-decay 0.01 --hidden-size 128 --seed 42]
oxidize-train video     ...    # clip classifier (creator|virality|engagement)
oxidize-train gen-train ...    # autoregressive next-frame generator
oxidize-train generate  ...    # generate a video from a trained checkpoint
```

## BUILD / TEST / RUN
```bash
cargo build -p oxidize-train
cargo test  -p oxidize-train
cargo run   -p oxidize-train -- csv --train-csv data.csv --label-column 0
```

## NOTES
- The `video`, `gen-train`, and `generate` subcommands shell out to `ffmpeg`; they fail early via `ffmpeg_available()` if it is missing.
