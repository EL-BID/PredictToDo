from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def build_time_index_strategy(
    train_time_count: int,
    total_selected: int,
    seed: int = 42,
) -> pd.DataFrame:
    if total_selected <= 0:
        raise ValueError("total_selected must be > 0")
    if train_time_count <= 0:
        raise ValueError("train_time_count must be > 0")

    rng = np.random.default_rng(seed)
    total_selected = min(int(total_selected), int(train_time_count))
    chosen = np.sort(rng.choice(train_time_count, size=total_selected, replace=False))
    return pd.DataFrame({"time_idx": chosen})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a simple time_idx strategy artifact.")
    parser.add_argument("--train-time-count", type=int, required=True)
    parser.add_argument("--total-selected", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-metadata", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = build_time_index_strategy(
        train_time_count=args.train_time_count,
        total_selected=args.total_selected,
        seed=args.seed,
    )
    output_csv = Path(args.output_csv)
    output_meta = Path(args.output_metadata)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_meta.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    metadata = {
        "train_time_count": int(args.train_time_count),
        "total_selected": int(len(df)),
        "seed": int(args.seed),
    }
    output_meta.write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
