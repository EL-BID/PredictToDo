from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge shard parquet outputs and assign global buckets.")
    parser.add_argument("--input-dir", required=True, help="Directory containing shard parquet outputs")
    parser.add_argument("--output-parquet", required=True)
    parser.add_argument("--output-metadata", required=True)
    parser.add_argument("--bucket-quantiles", type=str, default="0.5,0.75")
    return parser.parse_args()


def _parse_bucket_quantiles(raw: str) -> tuple[float, float]:
    values = [float(v.strip()) for v in raw.split(",") if v.strip()]
    if len(values) != 2:
        raise ValueError("bucket_quantiles must contain exactly two comma-separated values")
    low_q, high_q = values
    if not (0.0 < low_q < high_q < 1.0):
        raise ValueError("bucket_quantiles must satisfy 0 < low_q < high_q < 1")
    return low_q, high_q


def main() -> None:
    args = parse_args()
    low_q, high_q = _parse_bucket_quantiles(args.bucket_quantiles)
    input_dir = Path(args.input_dir)
    parquet_files = sorted(input_dir.glob("*.parquet"))
    if not parquet_files:
        raise ValueError(f"No parquet shard files found in {input_dir}")

    parts = [pd.read_parquet(p) for p in parquet_files]
    df = pd.concat(parts, ignore_index=True)
    low_thr = float(df["score"].quantile(low_q))
    high_thr = float(df["score"].quantile(high_q))
    bucket = pd.Series("medium", index=df.index, dtype=object)
    bucket[df["score"] <= low_thr] = "low"
    bucket[df["score"] >= high_thr] = "high"
    df["bucket"] = bucket

    output_parquet = Path(args.output_parquet)
    output_metadata = Path(args.output_metadata)
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    output_metadata.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_parquet, index=False)

    metadata = {
        "num_rows": int(len(df)),
        "num_seeds": int(df["seed_id"].nunique()),
        "num_time_indices": int(df["time_idx"].nunique()),
        "score_min": float(df["score"].min()),
        "score_mean": float(df["score"].mean()),
        "score_max": float(df["score"].max()),
        "bucket_quantiles": [low_q, high_q],
        "bucket_score_thresholds": [low_thr, high_thr],
        "bucket_counts": {str(k): int(v) for k, v in df["bucket"].value_counts().sort_index().items()},
        "num_input_parts": int(len(parquet_files)),
    }
    output_metadata.write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
