#!/usr/bin/env python3
"""
Backfill real-time evaluation from S3.

This script identifies prediction files that already have ground truth available,
downloads a selected sample of them, aligns them with jams observations, and
computes metrics by forecast step and by observed jam level bucket.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


S3_BUCKET = "ine-projects-incoming-data"
S3_PREFIX = "congestion-prediction/co/movilidad-bogota/real-time"
INFERENCE_PREFIX = f"s3://{S3_BUCKET}/{S3_PREFIX}/inference/"
JAMS_PREFIX = f"s3://{S3_BUCKET}/{S3_PREFIX}/jams/"
PRED_RE = re.compile(r"predictions_(\d{8})_(\d{4})\.parquet$")
JAMS_RE = re.compile(r"JamsService_(\d{8})_(\d{4})\.parquet$")


@dataclass
class S3Object:
    listed_at: datetime
    size_bytes: int
    key: str

    @property
    def name(self) -> str:
        return Path(self.key).name


def run_aws(args: list[str]) -> str:
    result = subprocess.run(
        ["aws", *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def parse_s3_ls(output: str) -> list[S3Object]:
    objects: list[S3Object] = []
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("PRE "):
            continue
        parts = raw_line.split(maxsplit=3)
        if len(parts) != 4:
            continue
        date_str, time_str, size_str, key = parts
        objects.append(
            S3Object(
                listed_at=datetime.strptime(
                    f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S"
                ),
                size_bytes=int(size_str),
                key=key,
            )
        )
    return objects


def list_objects(prefix: str, profile: str) -> list[S3Object]:
    output = run_aws(["s3", "ls", prefix, "--recursive", "--profile", profile])
    return parse_s3_ls(output)


def parse_prediction_file_dt(name: str) -> datetime:
    match = PRED_RE.match(name)
    if not match:
        raise ValueError(f"Unexpected prediction file name: {name}")
    date_part, time_part = match.groups()
    return datetime.strptime(f"{date_part}_{time_part}", "%d%m%Y_%H%M")


def parse_jams_file_dt(name: str) -> datetime:
    match = JAMS_RE.match(name)
    if not match:
        raise ValueError(f"Unexpected jams file name: {name}")
    date_part, time_part = match.groups()
    return datetime.strptime(f"{date_part}_{time_part}", "%d%m%Y_%H%M")


def prediction_start_utc(name: str) -> datetime:
    # Observed from the files inspected: predictions begin 5 minutes after the file timestamp.
    return parse_prediction_file_dt(name) + timedelta(minutes=5)


def latest_observed_utc_from_jams_name(name: str) -> datetime:
    # Observed from the files inspected: latest observed timeframe is file timestamp minus 15 min.
    return parse_jams_file_dt(name) - timedelta(minutes=15)


def overlap_steps(pred_name: str, latest_observed_utc: datetime) -> int:
    start_ts = prediction_start_utc(pred_name)
    diff_minutes = (latest_observed_utc - start_ts).total_seconds() / 60.0
    if diff_minutes < 0:
        return 0
    return max(0, min(50, int(math.floor(diff_minutes / 15.0)) + 1))


def select_prediction_files(
    inference_objects: list[S3Object],
    latest_observed_utc: datetime,
    min_overlap_steps: int,
    stride: int,
    max_files: int | None,
) -> list[S3Object]:
    eligible = [
        obj
        for obj in inference_objects
        if overlap_steps(obj.name, latest_observed_utc) >= min_overlap_steps
    ]
    if stride > 1:
        eligible = eligible[::stride]
    if max_files is not None:
        eligible = eligible[-max_files:]
    return eligible


def select_jams_files(
    jams_objects: list[S3Object],
    earliest_needed_utc: datetime,
    latest_needed_utc: datetime,
) -> list[S3Object]:
    selected: list[S3Object] = []
    for obj in jams_objects:
        file_dt = parse_jams_file_dt(obj.name)
        # File contains a rolling window that reaches file_dt - 15min.
        if file_dt < earliest_needed_utc:
            continue
        if file_dt > latest_needed_utc + timedelta(minutes=15):
            continue
        selected.append(obj)
    return selected


def download(profile: str, key: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / Path(key).name
    if not path.exists():
        run_aws(["s3", "cp", f"s3://{S3_BUCKET}/{key}", str(path), "--profile", profile])
    return path


def load_jams(jams_paths: list[Path]) -> pd.DataFrame:
    frames = [pd.read_parquet(path) for path in jams_paths]
    jams = pd.concat(frames, ignore_index=True).drop_duplicates()
    jams = jams.rename(columns={"id_mvi": "road_id", "timeframe": "observed_time"})
    jams["timestamp"] = (
        jams["observed_time"].dt.tz_convert("UTC").dt.tz_localize(None)
    )
    jams = jams.sort_values(["timestamp", "road_id"]).reset_index(drop=True)
    return jams


def metric_bundle(actual: pd.Series, pred: pd.Series) -> dict[str, float]:
    if len(actual) == 0:
        return {
            "count": 0,
            "mae": float("nan"),
            "rmse": float("nan"),
            "bias": float("nan"),
            "corr": float("nan"),
            "rounded_exact_acc": float("nan"),
            "rounded_within_1_acc": float("nan"),
        }
    err = pred - actual
    rounded_pred = pred.round().clip(0, 5)
    rounded_actual = actual.round().clip(0, 5)
    corr = float(np.corrcoef(actual, pred)[0, 1]) if len(actual) > 1 else float("nan")
    return {
        "count": int(len(actual)),
        "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt(np.mean(np.square(err)))),
        "bias": float(err.mean()),
        "corr": corr,
        "rounded_exact_acc": float((rounded_pred == rounded_actual).mean()),
        "rounded_within_1_acc": float((np.abs(rounded_pred - rounded_actual) <= 1).mean()),
    }


def fit_linear_calibration(df: pd.DataFrame) -> tuple[float, float]:
    x = df["predicted_level"].to_numpy(dtype=float)
    y = df["level_mean"].to_numpy(dtype=float)
    a, b = np.polyfit(x, y, 1)
    return float(a), float(b)


def apply_calibration(series: pd.Series, a: float, b: float) -> pd.Series:
    return (series * a + b).clip(0, 5)


def build_report(summary: dict, horizon_df: pd.DataFrame, level_df: pd.DataFrame) -> str:
    top_horizon = horizon_df.sort_values("forecast_step").head(10)
    best_steps = horizon_df.sort_values(["mae", "forecast_step"]).head(5)
    worst_steps = horizon_df.sort_values(["mae", "forecast_step"], ascending=[False, True]).head(5)

    lines = [
        "# Real-Time Evaluation Snapshot",
        "",
        f"- Prediction files analyzed: {summary['prediction_file_count']}",
        f"- Jams files analyzed: {summary['jams_file_count']}",
        f"- Rows matched: {summary['matched_rows']}",
        f"- Useful UTC window: {summary['matched_timestamp_min_utc']} to {summary['matched_timestamp_max_utc']}",
        f"- Latest observed UTC available: {summary['latest_observed_utc']}",
        f"- Level MAE raw: {summary['overall_raw']['mae']:.4f}",
        f"- Level MAE calibrated: {summary['overall_calibrated']['mae']:.4f}",
        f"- Calibration: actual ~= {summary['calibration']['slope']:.4f} * pred + {summary['calibration']['intercept']:.4f}",
        "",
        "## First Horizon Rows",
        "",
        top_horizon.to_markdown(index=False),
        "",
        "## Best Steps By MAE",
        "",
        best_steps[["forecast_step", "count", "mae", "corr", "rounded_within_1_acc"]].to_markdown(index=False),
        "",
        "## Worst Steps By MAE",
        "",
        worst_steps[["forecast_step", "count", "mae", "corr", "rounded_within_1_acc"]].to_markdown(index=False),
        "",
        "## By Observed Level",
        "",
        level_df.to_markdown(index=False),
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill real-time evaluation.")
    parser.add_argument("--profile", default="default")
    parser.add_argument("--min-overlap-steps", type=int, default=50)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--max-pred-files", type=int, default=16)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/realtime_eval_backfill"),
    )
    args = parser.parse_args()

    inference_objects = sorted(
        list_objects(INFERENCE_PREFIX, args.profile), key=lambda obj: obj.listed_at
    )
    jams_objects = sorted(
        list_objects(JAMS_PREFIX, args.profile), key=lambda obj: obj.listed_at
    )

    latest_observed_utc = latest_observed_utc_from_jams_name(jams_objects[-1].name)
    prediction_selection = select_prediction_files(
        inference_objects,
        latest_observed_utc=latest_observed_utc,
        min_overlap_steps=args.min_overlap_steps,
        stride=args.stride,
        max_files=args.max_pred_files,
    )
    if not prediction_selection:
        raise SystemExit("No prediction files matched the overlap criteria.")

    earliest_pred_start = min(prediction_start_utc(obj.name) for obj in prediction_selection)
    jams_selection = select_jams_files(
        jams_objects,
        earliest_needed_utc=earliest_pred_start,
        latest_needed_utc=latest_observed_utc,
    )

    download_dir = args.output_dir / "downloads"
    pred_paths = [download(args.profile, obj.key, download_dir) for obj in prediction_selection]
    jams_paths = [download(args.profile, obj.key, download_dir) for obj in jams_selection]

    jams_df = load_jams(jams_paths)
    road_filter = set(jams_df["road_id"].unique())
    latest_obs = jams_df["timestamp"].max()

    merged_parts: list[pd.DataFrame] = []
    for path in pred_paths:
        pred_df = pd.read_parquet(
            path,
            columns=["road_id", "timestamp", "predicted_level", "predicted_velocity"],
        )
        pred_df = pred_df[pred_df["road_id"].isin(road_filter)]
        pred_df = pred_df[pred_df["timestamp"] <= latest_obs]
        if pred_df.empty:
            continue

        ts_order = sorted(pred_df["timestamp"].drop_duplicates().tolist())
        step_map = {ts: idx + 1 for idx, ts in enumerate(ts_order)}
        pred_df["forecast_step"] = pred_df["timestamp"].map(step_map)
        pred_df["source_file"] = path.name

        merged = pred_df.merge(
            jams_df[
                [
                    "road_id",
                    "timestamp",
                    "observed_time",
                    "level_mean",
                    "vel_mean",
                    "count_jams",
                ]
            ],
            on=["road_id", "timestamp"],
            how="inner",
        )
        if not merged.empty:
            merged_parts.append(merged)

    if not merged_parts:
        raise SystemExit("No matched rows after joining predictions and jams.")

    merged_df = pd.concat(merged_parts, ignore_index=True)
    merged_df["actual_level_bucket"] = merged_df["level_mean"].round().clip(0, 5).astype(int)

    slope, intercept = fit_linear_calibration(merged_df)
    merged_df["predicted_level_calibrated"] = apply_calibration(
        merged_df["predicted_level"], slope, intercept
    )

    overall_raw = metric_bundle(merged_df["level_mean"], merged_df["predicted_level"])
    overall_cal = metric_bundle(
        merged_df["level_mean"], merged_df["predicted_level_calibrated"]
    )

    horizon_rows = []
    for step, group in merged_df.groupby("forecast_step", sort=True):
        raw_metrics = metric_bundle(group["level_mean"], group["predicted_level"])
        cal_metrics = metric_bundle(
            group["level_mean"], group["predicted_level_calibrated"]
        )
        horizon_rows.append(
            {
                "forecast_step": int(step),
                "count": raw_metrics["count"],
                "mae": raw_metrics["mae"],
                "rmse": raw_metrics["rmse"],
                "corr": raw_metrics["corr"],
                "rounded_within_1_acc": raw_metrics["rounded_within_1_acc"],
                "mae_calibrated": cal_metrics["mae"],
                "rmse_calibrated": cal_metrics["rmse"],
            }
        )
    horizon_df = pd.DataFrame(horizon_rows).sort_values("forecast_step")

    level_rows = []
    for level, group in merged_df.groupby("actual_level_bucket", sort=True):
        raw_metrics = metric_bundle(group["level_mean"], group["predicted_level"])
        cal_metrics = metric_bundle(
            group["level_mean"], group["predicted_level_calibrated"]
        )
        level_rows.append(
            {
                "actual_level_bucket": int(level),
                "count": raw_metrics["count"],
                "mae": raw_metrics["mae"],
                "rmse": raw_metrics["rmse"],
                "bias": raw_metrics["bias"],
                "rounded_exact_acc": raw_metrics["rounded_exact_acc"],
                "rounded_within_1_acc": raw_metrics["rounded_within_1_acc"],
                "mae_calibrated": cal_metrics["mae"],
            }
        )
    level_df = pd.DataFrame(level_rows).sort_values("actual_level_bucket")

    summary = {
        "latest_observed_utc": latest_observed_utc.isoformat(),
        "prediction_file_count": len(pred_paths),
        "jams_file_count": len(jams_paths),
        "prediction_files": [path.name for path in pred_paths],
        "jams_files": [path.name for path in jams_paths],
        "matched_rows": int(len(merged_df)),
        "matched_timestamp_min_utc": merged_df["timestamp"].min().isoformat(),
        "matched_timestamp_max_utc": merged_df["timestamp"].max().isoformat(),
        "matched_unique_roads": int(merged_df["road_id"].nunique()),
        "matched_unique_steps": int(merged_df["forecast_step"].nunique()),
        "selection": {
            "min_overlap_steps": args.min_overlap_steps,
            "stride": args.stride,
            "max_pred_files": args.max_pred_files,
            "earliest_prediction_start_utc": earliest_pred_start.isoformat(),
        },
        "overall_raw": overall_raw,
        "overall_calibrated": overall_cal,
        "calibration": {"slope": slope, "intercept": intercept},
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    merged_df.to_parquet(args.output_dir / "matched_eval_rows.parquet", index=False)
    horizon_df.to_csv(args.output_dir / "metrics_by_forecast_step.csv", index=False)
    level_df.to_csv(args.output_dir / "metrics_by_observed_level.csv", index=False)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (args.output_dir / "report.md").write_text(build_report(summary, horizon_df, level_df))

    print(json.dumps(summary, indent=2))
    print(f"Wrote outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
