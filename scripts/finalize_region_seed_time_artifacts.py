#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import boto3
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download new regional seed-time shards, merge with reused rows, and rebucket per region."
    )
    parser.add_argument("--bucket", required=True)
    parser.add_argument(
        "--s3-prefix",
        default="congestion-bogota/region-seed-time-artifacts",
    )
    parser.add_argument(
        "--reuse-dir",
        type=Path,
        default=Path("data/artifacts/region_seed_time_reuse"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/artifacts/region_seed_time_final"),
    )
    parser.add_argument("--bucket-quantiles", default="0.5,0.75")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--upload-final", action="store_true")
    return parser.parse_args()


def parse_quantiles(raw: str) -> tuple[float, float]:
    values = [float(v.strip()) for v in raw.split(",") if v.strip()]
    if len(values) != 2:
        raise ValueError("bucket quantiles must contain exactly two comma-separated values")
    low_q, high_q = values
    if not (0 < low_q < high_q < 1):
        raise ValueError("bucket quantiles must satisfy 0 < low < high < 1")
    return low_q, high_q


def download_region_outputs(s3, bucket: str, prefix: str, region: str, output_dir: Path) -> list[Path]:
    region_prefix = f"{prefix.rstrip('/')}/outputs/{region}/"
    paginator = s3.get_paginator("list_objects_v2")
    downloaded: list[Path] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=region_prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith(".parquet"):
                continue
            local_path = output_dir / Path(key).name
            if not local_path.exists():
                s3.download_file(bucket, key, str(local_path))
            downloaded.append(local_path)
    return downloaded


def select_region_input_parts(merge_dir: Path, region: str) -> tuple[list[Path], dict]:
    reusable = merge_dir / f"seed_time_reusable_existing_{region}.parquet"
    selected: list[Path] = []
    diagnostics = {
        "reusable_existing": str(reusable) if reusable.exists() else None,
        "generated_shards": {},
        "ignored_parquets": [],
    }
    if reusable.exists():
        selected.append(reusable)

    final_pattern = re.compile(rf"seed_time_bucket_rows_{region}_shard_(\d+)\.parquet$")
    checkpoint_pattern = re.compile(
        rf"seed_time_bucket_rows_{region}_shard_(\d+)\.checkpoint_(\d+)\.parquet$"
    )
    partial_pattern = re.compile(rf"seed_time_bucket_rows_{region}_shard_(\d+)\.partial\.parquet$")

    candidates: dict[int, dict] = {}
    for path in sorted(merge_dir.glob("*.parquet")):
        if path == reusable:
            continue
        final_match = final_pattern.match(path.name)
        if final_match:
            shard_id = int(final_match.group(1))
            candidates.setdefault(shard_id, {})["final"] = path
            continue
        checkpoint_match = checkpoint_pattern.match(path.name)
        if checkpoint_match:
            shard_id = int(checkpoint_match.group(1))
            processed = int(checkpoint_match.group(2))
            checkpoints = candidates.setdefault(shard_id, {}).setdefault("checkpoints", [])
            checkpoints.append((processed, path))
            continue
        partial_match = partial_pattern.match(path.name)
        if partial_match:
            shard_id = int(partial_match.group(1))
            candidates.setdefault(shard_id, {})["partial"] = path
            continue
        diagnostics["ignored_parquets"].append(str(path))

    for shard_id, shard_candidates in sorted(candidates.items()):
        source = ""
        if "final" in shard_candidates:
            chosen = shard_candidates["final"]
            source = "final"
        elif shard_candidates.get("checkpoints"):
            processed, chosen = max(shard_candidates["checkpoints"], key=lambda item: item[0])
            source = f"checkpoint_{processed}"
        elif "partial" in shard_candidates:
            chosen = shard_candidates["partial"]
            source = "partial"
        else:
            continue
        selected.append(chosen)
        diagnostics["generated_shards"][str(shard_id)] = {
            "selected": str(chosen),
            "source": source,
        }

    if not selected:
        raise ValueError(f"No usable parquet inputs found for {region} in {merge_dir}")
    return selected, diagnostics


def rebucket(df: pd.DataFrame, low_q: float, high_q: float) -> tuple[pd.DataFrame, dict]:
    low_thr = float(df["score"].quantile(low_q))
    high_thr = float(df["score"].quantile(high_q))
    bucket = pd.Series("medium", index=df.index, dtype=object)
    bucket[df["score"] <= low_thr] = "low"
    bucket[df["score"] >= high_thr] = "high"
    df = df.copy()
    df["bucket"] = bucket
    metadata = {
        "num_rows": int(len(df)),
        "num_seeds": int(df["seed_id"].nunique()),
        "num_time_indices": int(df["time_idx"].nunique()),
        "score_min": float(df["score"].min()),
        "score_mean": float(df["score"].mean()),
        "score_max": float(df["score"].max()),
        "bucket_quantiles": [low_q, high_q],
        "bucket_score_thresholds": [low_thr, high_thr],
        "bucket_counts": {
            str(k): int(v) for k, v in df["bucket"].value_counts().sort_index().items()
        },
    }
    return df, metadata


def upload_file(s3, local_path: Path, bucket: str, key: str) -> str:
    s3.upload_file(str(local_path), bucket, key)
    return f"s3://{bucket}/{key}"


def main() -> None:
    args = parse_args()
    low_q, high_q = parse_quantiles(args.bucket_quantiles)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    s3 = boto3.client("s3") if (args.download or args.upload_final) else None

    manifest = {"regions": {}, "download": bool(args.download), "upload_final": bool(args.upload_final)}
    for region in ["central", "noncentral"]:
        merge_dir = args.reuse_dir / "merge_inputs" / region
        merge_dir.mkdir(parents=True, exist_ok=True)
        if args.download:
            download_region_outputs(s3, args.bucket, args.s3_prefix, region, merge_dir)

        parquet_files, selection_diagnostics = select_region_input_parts(merge_dir, region)
        df = pd.concat([pd.read_parquet(path) for path in parquet_files], ignore_index=True)
        df["seed_id"] = df["seed_id"].astype(str)
        df["time_idx"] = df["time_idx"].astype(int)
        df["score"] = pd.to_numeric(df["score"], errors="raise")
        df, metadata = rebucket(df, low_q, high_q)

        output_parquet = args.output_dir / f"seed_time_bucket_rows_{region}.parquet"
        output_metadata = args.output_dir / f"seed_time_bucket_metadata_{region}.json"
        df.to_parquet(output_parquet, index=False)
        metadata["region"] = region
        metadata["input_parts"] = [str(path) for path in parquet_files]
        metadata["input_selection"] = selection_diagnostics
        output_metadata.write_text(json.dumps(metadata, indent=2))

        uploads = {}
        if args.upload_final:
            base_key = f"{args.s3_prefix.rstrip('/')}/final/{region}"
            uploads["parquet"] = upload_file(
                s3,
                output_parquet,
                args.bucket,
                f"{base_key}/{output_parquet.name}",
            )
            uploads["metadata"] = upload_file(
                s3,
                output_metadata,
                args.bucket,
                f"{base_key}/{output_metadata.name}",
            )

        manifest["regions"][region] = {
            "output_parquet": str(output_parquet),
            "output_metadata": str(output_metadata),
            "metadata": metadata,
            "uploads": uploads,
        }

    manifest_path = args.output_dir / "final_region_seed_time_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
