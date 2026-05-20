#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reuse existing seed-time rows for regional seed strategies and emit "
            "missing seed strategies for only the seeds that still need scoring."
        )
    )
    parser.add_argument(
        "--existing-seed-time",
        type=Path,
        default=Path("data/artifacts/seed_time_merged_all/seed_time_bucket_rows_merged_all.parquet"),
    )
    parser.add_argument(
        "--seed-strategy-dir",
        type=Path,
        default=Path("data/artifacts/region_seed_strategy"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/artifacts/region_seed_time_reuse"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    missing_strategy_dir = output_dir / "missing_seed_strategy"
    reusable_dir = output_dir / "reusable_existing"
    merge_input_dir = output_dir / "merge_inputs"
    missing_strategy_dir.mkdir(parents=True, exist_ok=True)
    reusable_dir.mkdir(parents=True, exist_ok=True)
    merge_input_dir.mkdir(parents=True, exist_ok=True)

    existing = pd.read_parquet(args.existing_seed_time)
    required = {"seed_id", "time_idx", "score", "bucket"}
    missing_cols = required - set(existing.columns)
    if missing_cols:
        raise ValueError(f"Existing seed-time artifact missing columns: {sorted(missing_cols)}")
    existing["seed_id"] = existing["seed_id"].astype(str)
    existing_seed_ids = set(existing["seed_id"].unique())

    manifest = {
        "existing_seed_time": str(args.existing_seed_time),
        "seed_strategy_dir": str(args.seed_strategy_dir),
        "regions": {},
    }
    for region in ["central", "noncentral"]:
        strategy_path = args.seed_strategy_dir / f"seed_strategy_{region}.csv"
        strategy = pd.read_csv(strategy_path)
        if "seed_id" not in strategy.columns:
            raise ValueError(f"{strategy_path} must contain seed_id")
        strategy["seed_id"] = strategy["seed_id"].astype(str)
        desired_seed_ids = set(strategy["seed_id"])
        reusable_seed_ids = desired_seed_ids & existing_seed_ids
        missing_seed_ids = desired_seed_ids - existing_seed_ids

        reusable = existing[existing["seed_id"].isin(reusable_seed_ids)].copy()
        # The final regional merge will recompute buckets, so keep scores and
        # preserve the old bucket only as a diagnostic input column.
        reusable_path = reusable_dir / f"seed_time_reusable_existing_{region}.parquet"
        reusable.to_parquet(reusable_path, index=False)

        region_merge_dir = merge_input_dir / region
        region_merge_dir.mkdir(parents=True, exist_ok=True)
        reusable_merge_path = region_merge_dir / f"seed_time_reusable_existing_{region}.parquet"
        reusable.to_parquet(reusable_merge_path, index=False)

        missing_strategy = strategy[strategy["seed_id"].isin(missing_seed_ids)].copy()
        missing_strategy_path = missing_strategy_dir / f"seed_strategy_{region}.csv"
        missing_strategy.to_csv(missing_strategy_path, index=False)
        missing_metadata = {
            "region": region,
            "num_desired_seeds": int(len(desired_seed_ids)),
            "num_reused_seeds": int(len(reusable_seed_ids)),
            "num_missing_seeds": int(len(missing_seed_ids)),
            "num_reused_rows": int(len(reusable)),
            "expected_missing_rows_at_2000_time_indices": int(len(missing_seed_ids) * 2000),
            "reusable_existing_path": str(reusable_path),
            "merge_input_dir": str(region_merge_dir),
            "missing_strategy_path": str(missing_strategy_path),
        }
        (missing_strategy_dir / f"seed_strategy_{region}_metadata.json").write_text(
            json.dumps(missing_metadata, indent=2)
        )
        manifest["regions"][region] = missing_metadata

    manifest_path = output_dir / "reuse_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
