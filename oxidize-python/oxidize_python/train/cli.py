"""oxidize-train CLI mirroring oxidize-train."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from oxidize_python.train.lib import MlpClassifier, TrainingConfig, load_csv_dataset, save_report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="oxidize-train")
    p.add_argument("--csv", required=True, help="training CSV path")
    p.add_argument("--output", default="training-report.json")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    ns = p.parse_args(argv)

    dataset = load_csv_dataset(Path(ns.csv))
    clf = MlpClassifier.new(dataset.inputs.cols, ns.hidden, dataset.classes, ns.seed)
    report = clf.train(
        dataset,
        TrainingConfig(
            epochs=ns.epochs,
            batch_size=ns.batch_size,
            learning_rate=ns.lr,
            hidden_size=ns.hidden,
            seed=ns.seed,
        ),
    )
    save_report(Path(ns.output), report)
    print(
        f"accuracy={report.accuracy:.4f} loss={report.final_loss:.4f} samples={report.samples}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
