#!/usr/bin/env python3
"""
Export quantile-map and hybrid calibration artifacts from evaluation data.

Expected columns in both train and eval inputs:
- predicted_level
- level_mean

Optional in eval input:
- forecast_step
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from scripts.realtime_eval_metrics import export_calibration_bundle


def load_frame(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported input format: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export calibration artifacts.")
    parser.add_argument("--train-input", type=Path, required=True)
    parser.add_argument("--eval-input", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/artifacts/calibration"),
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="*",
        default=(0.6, 0.8),
        help="Hybrid thresholds to evaluate.",
    )
    parser.add_argument(
        "--selection-metric",
        default="accuracy_exact",
        choices=["accuracy_exact", "accuracy_pm1", "mae"],
        help="Metric used to choose the best exported variant.",
    )
    args = parser.parse_args()

    train_df = load_frame(args.train_input)
    eval_df = load_frame(args.eval_input)

    summary = export_calibration_bundle(
        train_df=train_df,
        eval_df=eval_df,
        output_dir=args.output_dir,
        thresholds=tuple(args.thresholds),
        selection_metric=args.selection_metric,
    )
    print(json.dumps(summary, indent=2))
    print(f"Calibration artifacts written to {args.output_dir}")


if __name__ == "__main__":
    main()
