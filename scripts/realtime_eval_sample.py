#!/usr/bin/env python3
"""
Build a recent real-time evaluation sample from S3.

The script:
1. Lists recent inference and jams parquet files from S3
2. Downloads a configurable window of recent files
3. Aligns predictions with observations on (road_id, timestamp)
4. Writes an evaluation sample plus a compact JSON summary
"""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd



@dataclass
class S3Object:
    timestamp: datetime
    size_bytes: int
    key: str


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
                timestamp=datetime.strptime(
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


def download_object(profile: str, key: str, output_dir: Path, bucket: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    local_path = output_dir / Path(key).name
    run_aws(["s3", "cp", f"s3://{bucket}/{key}", str(local_path), "--profile", profile])
    return local_path


def load_jams(paths: list[Path]) -> pd.DataFrame:
    frames = [pd.read_parquet(path) for path in paths]
    jams = pd.concat(frames, ignore_index=True)
    jams = jams.rename(columns={"id_mvi": "road_id", "timeframe": "observed_time"})
    jams["timestamp"] = (
        jams["observed_time"].dt.tz_convert("UTC").dt.tz_localize(None)
    )
    return jams


def load_predictions(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        pred = pd.read_parquet(path)
        pred["source_file"] = path.name
        frames.append(pred)
    return pd.concat(frames, ignore_index=True)


def build_eval_dataset(predictions: pd.DataFrame, jams: pd.DataFrame) -> pd.DataFrame:
    keep_obs = [
        "road_id",
        "timestamp",
        "observed_time",
        "level_mean",
        "vel_mean",
        "count_jams",
    ]
    keep_pred = [
        "road_id",
        "timestamp",
        "predicted_level",
        "predicted_velocity",
        "source_file",
    ]
    merged = jams[keep_obs].merge(
        predictions[keep_pred],
        on=["road_id", "timestamp"],
        how="inner",
    )
    merged["level_abs_error"] = (
        merged["predicted_level"] - merged["level_mean"]
    ).abs()
    merged["velocity_abs_error"] = (
        merged["predicted_velocity"] - merged["vel_mean"]
    ).abs()
    return merged.sort_values(["timestamp", "road_id"]).reset_index(drop=True)


def summarize_dataset(
    merged: pd.DataFrame,
    prediction_files: list[Path],
    jams_files: list[Path],
) -> dict:
    if merged.empty:
        return {
            "prediction_files": [path.name for path in prediction_files],
            "jams_files": [path.name for path in jams_files],
            "rows": 0,
        }

    return {
        "prediction_files": [path.name for path in prediction_files],
        "jams_files": [path.name for path in jams_files],
        "rows": int(len(merged)),
        "unique_roads": int(merged["road_id"].nunique()),
        "unique_timestamps": int(merged["timestamp"].nunique()),
        "timestamp_min_utc": merged["timestamp"].min().isoformat(),
        "timestamp_max_utc": merged["timestamp"].max().isoformat(),
        "mae_level_raw": float(merged["level_abs_error"].mean()),
        "mae_velocity_raw": float(merged["velocity_abs_error"].mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a recent real-time eval sample.")
    parser.add_argument("--profile", default="default", help="AWS profile name")
    parser.add_argument(
        "--inference-count",
        type=int,
        default=2,
        help="How many recent inference files to download",
    )
    parser.add_argument(
        "--jams-count",
        type=int,
        default=4,
        help="How many recent jams files to download",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/realtime_eval_sample"),
        help="Directory for downloaded files and outputs",
    )
    parser.add_argument("--bucket", required=True, help="S3 bucket containing predictions and observations")
    parser.add_argument("--s3-prefix", required=True, help="Key prefix containing inference/ and jams/ subdirectories")
    args = parser.parse_args()
    s3_base = f"s3://{args.bucket}/{args.s3_prefix.strip('/')}"

    inference_prefix = f"{s3_base}/inference/"
    jams_prefix = f"{s3_base}/jams/"

    inference_objects = sorted(
        list_objects(inference_prefix, args.profile), key=lambda obj: obj.timestamp
    )
    jams_objects = sorted(
        list_objects(jams_prefix, args.profile), key=lambda obj: obj.timestamp
    )

    selected_inference = inference_objects[-args.inference_count :]
    selected_jams = jams_objects[-args.jams_count :]

    download_dir = args.output_dir / "downloads"
    inference_paths = [
        download_object(args.profile, obj.key, download_dir, args.bucket) for obj in selected_inference
    ]
    jams_paths = [download_object(args.profile, obj.key, download_dir, args.bucket) for obj in selected_jams]

    predictions = load_predictions(inference_paths)
    jams = load_jams(jams_paths)
    merged = build_eval_dataset(predictions, jams)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    merged_path = args.output_dir / "eval_sample.parquet"
    summary_path = args.output_dir / "summary.json"

    merged.to_parquet(merged_path, index=False)
    summary = summarize_dataset(merged, inference_paths, jams_paths)
    summary_path.write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))
    print(f"Wrote eval sample to {merged_path}")


if __name__ == "__main__":
    main()
