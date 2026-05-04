from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split a seed strategy CSV into shard CSV files.")
    parser.add_argument("--seed-strategy-csv", required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.seed_strategy_csv)
    if "seed_id" not in df.columns:
        raise ValueError("seed strategy CSV must contain `seed_id`")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "num_shards": int(args.num_shards),
        "total_seeds": int(len(df)),
        "shards": [],
    }

    for shard_id in range(int(args.num_shards)):
        shard_df = df.iloc[shard_id:: int(args.num_shards)].copy()
        shard_path = output_dir / f"seed_strategy_shard_{shard_id:02d}.csv"
        shard_df.to_csv(shard_path, index=False)
        manifest["shards"].append(
            {
                "shard_id": shard_id,
                "path": str(shard_path),
                "num_seeds": int(len(shard_df)),
            }
        )

    (output_dir / "seed_shard_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
