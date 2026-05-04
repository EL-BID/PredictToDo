from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class SeedStrategyArtifact:
    seeds: pd.DataFrame
    metadata: dict

    def to_csv(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.seeds.to_csv(path, index=False)

    def write_metadata(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.metadata, indent=2))


def load_optimal_seeds(path: str | Path) -> list[str]:
    with Path(path).open("rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, list):
        raise ValueError(f"Expected list in {path}, got {type(payload)}")
    return [str(x) for x in payload]


def load_top_congestion_nodes(
    path: str | Path,
    top_k: int | None = None,
    top_fraction: float | None = None,
) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "road_id" not in df.columns or "level_mean" not in df.columns:
        raise ValueError("nodes_traffic_info.csv must contain columns: road_id, level_mean")
    df = df[["road_id", "level_mean"]].copy()
    df["road_id"] = df["road_id"].astype(str)
    df["level_mean"] = pd.to_numeric(df["level_mean"], errors="coerce").fillna(0.0)
    df = df.sort_values("level_mean", ascending=False, ignore_index=True)

    if top_k is not None:
        return df.head(int(top_k)).copy()
    if top_fraction is not None:
        n = max(1, int(len(df) * float(top_fraction)))
        return df.head(n).copy()
    return df.copy()


def build_seed_strategy_artifact(
    coverage_seeds: list[str],
    top_congestion_df: pd.DataFrame,
    extra_seed_limit: int = 0,
) -> SeedStrategyArtifact:
    coverage_set = set(str(x) for x in coverage_seeds)

    coverage_rows = pd.DataFrame(
        {
            "seed_id": [str(x) for x in coverage_seeds],
            "source": "coverage",
            "priority": 0,
            "level_mean": pd.NA,
        }
    )

    top_df = top_congestion_df.copy()
    top_df = top_df.rename(columns={"road_id": "seed_id"})
    top_df["seed_id"] = top_df["seed_id"].astype(str)
    top_df = top_df[~top_df["seed_id"].isin(coverage_set)].copy()
    if extra_seed_limit > 0:
        top_df = top_df.head(int(extra_seed_limit)).copy()
    top_df["source"] = "top_congestion"
    top_df["priority"] = range(1, len(top_df) + 1)

    combined = pd.concat(
        [
            coverage_rows,
            top_df[["seed_id", "source", "priority", "level_mean"]],
        ],
        ignore_index=True,
    )

    metadata = {
        "num_coverage_seeds": int(len(coverage_rows)),
        "num_top_congestion_extra": int(len(top_df)),
        "num_total_seeds": int(len(combined)),
        "coverage_fraction": float(len(coverage_rows) / len(combined)) if len(combined) else 0.0,
    }
    return SeedStrategyArtifact(seeds=combined, metadata=metadata)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an operational seed strategy artifact.")
    parser.add_argument(
        "--optimal-seeds-path",
        default="data/datos_procesados/optimal_seeds.pkl",
    )
    parser.add_argument(
        "--nodes-traffic-info-path",
        default="data/datos_procesados/nodes_traffic_info.csv",
    )
    parser.add_argument(
        "--extra-seed-limit",
        type=int,
        default=0,
        help="How many top congestion seeds to add on top of coverage seeds.",
    )
    parser.add_argument(
        "--output-csv",
        default="data/artifacts/seed_strategy/seed_strategy.csv",
    )
    parser.add_argument(
        "--output-metadata",
        default="data/artifacts/seed_strategy/seed_strategy_metadata.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    coverage_seeds = load_optimal_seeds(args.optimal_seeds_path)
    top_df = load_top_congestion_nodes(args.nodes_traffic_info_path)
    artifact = build_seed_strategy_artifact(
        coverage_seeds=coverage_seeds,
        top_congestion_df=top_df,
        extra_seed_limit=args.extra_seed_limit,
    )
    artifact.to_csv(args.output_csv)
    artifact.write_metadata(args.output_metadata)
    print(json.dumps(artifact.metadata, indent=2))


if __name__ == "__main__":
    main()
