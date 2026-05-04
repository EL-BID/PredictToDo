#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def metric_bundle(actual: pd.Series, pred: pd.Series) -> dict[str, float]:
    if len(actual) == 0:
        return {
            "count": 0,
            "mae": float("nan"),
            "rmse": float("nan"),
            "bias": float("nan"),
            "acc_exact": float("nan"),
            "acc_pm1": float("nan"),
        }
    err = pred - actual
    rounded_pred = pred.round().clip(0, 5)
    rounded_actual = actual.round().clip(0, 5)
    class_diff = (rounded_pred - rounded_actual).abs()
    return {
        "count": int(len(actual)),
        "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt(np.mean(np.square(err)))),
        "bias": float(err.mean()),
        "acc_exact": float((rounded_pred == rounded_actual).mean()),
        "acc_pm1": float((class_diff <= 1).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("data/artifacts/realtime_last7d_eval"),
    )
    parser.add_argument(
        "--limit-pred-files",
        type=int,
        default=None,
        help="Optional cap to evaluate only the most recent N prediction files.",
    )
    args = parser.parse_args()

    pred_dir = args.base_dir / "downloads_pred"
    jam_dir = args.base_dir / "downloads_jam"
    out_dir = args.base_dir / "dense_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_paths = sorted(pred_dir.glob("predictions_*.parquet"))
    jam_paths = sorted(jam_dir.glob("JamsService_*.parquet"))
    if args.limit_pred_files is not None:
        pred_paths = pred_paths[-args.limit_pred_files :]

    if not pred_paths:
        raise SystemExit("No prediction files available.")
    if not jam_paths:
        raise SystemExit("No jams files available.")

    jams_frames = []
    for path in jam_paths:
        df = pd.read_parquet(path)
        if df.empty:
            continue
        df = df.rename(columns={"id_mvi": "road_id", "timeframe": "timestamp"})
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_localize(None)
        jams_frames.append(df[["road_id", "timestamp", "level_mean", "vel_mean", "count_jams"]])
    if not jams_frames:
        raise SystemExit("All jams files were empty.")
    jams_df = pd.concat(jams_frames, ignore_index=True).drop_duplicates(["road_id", "timestamp"])

    overall_rows = []
    by_level_rows = []
    by_step_rows = []

    for path in pred_paths:
        pred_df = pd.read_parquet(
            path,
            columns=["road_id", "timestamp", "predicted_level", "predicted_velocity"],
        )
        if pred_df.empty:
            continue
        pred_df["timestamp"] = pd.to_datetime(pred_df["timestamp"], utc=True).dt.tz_localize(None)
        ts_order = sorted(pred_df["timestamp"].drop_duplicates().tolist())
        step_map = {ts: idx + 1 for idx, ts in enumerate(ts_order)}
        pred_df["forecast_step"] = pred_df["timestamp"].map(step_map)

        merged = pred_df.merge(
            jams_df,
            on=["road_id", "timestamp"],
            how="left",
        )
        merged["level_mean"] = merged["level_mean"].fillna(0.0)
        merged["vel_mean"] = merged["vel_mean"].fillna(0.0)
        merged["count_jams"] = merged["count_jams"].fillna(0.0)
        merged["actual_level_bucket"] = merged["level_mean"].round().clip(0, 5).astype(int)
        merged["source_file"] = path.name

        overall = metric_bundle(merged["level_mean"], merged["predicted_level"])
        overall["source_file"] = path.name
        overall["num_rows"] = int(len(merged))
        overall_rows.append(overall)

        for level, group in merged.groupby("actual_level_bucket", sort=True):
            metrics = metric_bundle(group["level_mean"], group["predicted_level"])
            metrics["actual_level_bucket"] = int(level)
            metrics["source_file"] = path.name
            by_level_rows.append(metrics)

        for step, group in merged.groupby("forecast_step", sort=True):
            metrics = metric_bundle(group["level_mean"], group["predicted_level"])
            metrics["forecast_step"] = int(step)
            metrics["source_file"] = path.name
            by_step_rows.append(metrics)

    overall_df = pd.DataFrame(overall_rows)
    by_level_df = pd.DataFrame(by_level_rows)
    by_step_df = pd.DataFrame(by_step_rows)

    overall_summary = metric_bundle(
        pd.concat(
            [
                pd.read_parquet(path, columns=["road_id", "timestamp", "predicted_level"])
                .assign(
                    timestamp=lambda df: pd.to_datetime(df["timestamp"], utc=True).dt.tz_localize(None)
                )
                .merge(jams_df[["road_id", "timestamp", "level_mean"]], on=["road_id", "timestamp"], how="left")
                .assign(level_mean=lambda df: df["level_mean"].fillna(0.0))["level_mean"]
                for path in pred_paths
            ],
            ignore_index=True,
        ),
        pd.concat(
            [
                pd.read_parquet(path, columns=["predicted_level"])["predicted_level"]
                for path in pred_paths
            ],
            ignore_index=True,
        ),
    )

    summary = {
        "prediction_files_evaluated": len(pred_paths),
        "jam_files_loaded": len(jam_paths),
        "overall": overall_summary,
        "latest_prediction_file": pred_paths[-1].name,
        "latest_jam_file": jam_paths[-1].name,
    }

    overall_df.to_csv(out_dir / "metrics_by_prediction_file.csv", index=False)
    by_level_df.to_csv(out_dir / "metrics_by_level_per_file.csv", index=False)
    by_step_df.to_csv(out_dir / "metrics_by_step_per_file.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))
    print(f"Wrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
