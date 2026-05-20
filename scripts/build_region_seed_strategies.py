#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build separate seed strategy CSVs for central and non-central corridor regions."
    )
    parser.add_argument(
        "--region-map",
        type=Path,
        default=Path("data/artifacts/corridor_regions/node_region_map.csv"),
    )
    parser.add_argument(
        "--coverage-seeds-path",
        type=Path,
        default=Path("data/datos_procesados/optimal_seeds.pkl"),
    )
    parser.add_argument(
        "--nodes-traffic-info-path",
        type=Path,
        default=Path("data/datos_procesados/nodes_traffic_info.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/artifacts/region_seed_strategy"),
    )
    parser.add_argument("--central-extra-limit", type=int, default=1200)
    parser.add_argument("--noncentral-extra-limit", type=int, default=2800)
    return parser.parse_args()


def load_coverage_seeds(path: Path) -> list[str]:
    if path.suffix == ".json":
        payload = json.loads(path.read_text())
        if isinstance(payload, dict):
            values = payload.get("seeds", [])
        else:
            values = payload
        return [str(value) for value in values]
    with path.open("rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, list):
        raise ValueError(f"Expected a list of coverage seeds in {path}")
    return [str(value) for value in payload]


def build_region_strategy(
    *,
    region_name: str,
    region_nodes: set[str],
    coverage_seeds: list[str],
    top_congestion: pd.DataFrame,
    extra_limit: int,
) -> tuple[pd.DataFrame, dict]:
    coverage_rows = pd.DataFrame(
        {
            "seed_id": [seed for seed in coverage_seeds if seed in region_nodes],
            "source": f"coverage_{region_name}",
            "priority": 0,
            "level_mean": pd.NA,
            "region": region_name,
        }
    )
    coverage_set = set(coverage_rows["seed_id"].astype(str))

    top_df = top_congestion[top_congestion["road_id"].isin(region_nodes)].copy()
    top_df = top_df[~top_df["road_id"].isin(coverage_set)].copy()
    if extra_limit >= 0:
        top_df = top_df.head(extra_limit).copy()
    top_df = top_df.rename(columns={"road_id": "seed_id"})
    top_df["source"] = f"top_congestion_{region_name}"
    top_df["priority"] = range(1, len(top_df) + 1)
    top_df["region"] = region_name

    strategy = pd.concat(
        [
            coverage_rows,
            top_df[["seed_id", "source", "priority", "level_mean", "region"]],
        ],
        ignore_index=True,
    )
    strategy = strategy.drop_duplicates(subset=["seed_id"], keep="first")

    metadata = {
        "region": region_name,
        "num_region_nodes": int(len(region_nodes)),
        "num_coverage_seeds": int(len(coverage_rows)),
        "num_top_congestion_extra": int(len(top_df)),
        "num_total_seeds": int(len(strategy)),
        "extra_limit": int(extra_limit),
        "coverage_fraction": (
            float(len(coverage_rows) / len(strategy)) if len(strategy) else 0.0
        ),
    }
    return strategy, metadata


def main() -> None:
    args = parse_args()
    region_map = pd.read_csv(args.region_map)
    if "road_id" not in region_map.columns or "is_central" not in region_map.columns:
        raise ValueError("region map must contain road_id and is_central columns")
    region_map["road_id"] = region_map["road_id"].astype(str)
    region_map["is_central"] = region_map["is_central"].astype(bool)

    coverage_seeds = load_coverage_seeds(args.coverage_seeds_path)
    top_congestion = pd.read_csv(args.nodes_traffic_info_path)
    if "road_id" not in top_congestion.columns or "level_mean" not in top_congestion.columns:
        raise ValueError("nodes traffic info must contain road_id and level_mean columns")
    top_congestion = top_congestion[["road_id", "level_mean"]].copy()
    top_congestion["road_id"] = top_congestion["road_id"].astype(str)
    top_congestion["level_mean"] = pd.to_numeric(
        top_congestion["level_mean"],
        errors="coerce",
    ).fillna(0.0)
    top_congestion = top_congestion.sort_values(
        "level_mean",
        ascending=False,
        ignore_index=True,
    )

    central_nodes = set(region_map.loc[region_map["is_central"], "road_id"])
    noncentral_nodes = set(region_map.loc[~region_map["is_central"], "road_id"])
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs = {}
    for region_name, nodes, extra_limit in [
        ("central", central_nodes, args.central_extra_limit),
        ("noncentral", noncentral_nodes, args.noncentral_extra_limit),
    ]:
        strategy, metadata = build_region_strategy(
            region_name=region_name,
            region_nodes=nodes,
            coverage_seeds=coverage_seeds,
            top_congestion=top_congestion,
            extra_limit=extra_limit,
        )
        csv_path = output_dir / f"seed_strategy_{region_name}.csv"
        metadata_path = output_dir / f"seed_strategy_{region_name}_metadata.json"
        strategy.to_csv(csv_path, index=False)
        metadata_path.write_text(json.dumps(metadata, indent=2))
        outputs[region_name] = {
            "csv": str(csv_path),
            "metadata": metadata,
        }

    manifest = {
        "region_map": str(args.region_map),
        "coverage_seeds_path": str(args.coverage_seeds_path),
        "nodes_traffic_info_path": str(args.nodes_traffic_info_path),
        "central_extra_limit": int(args.central_extra_limit),
        "noncentral_extra_limit": int(args.noncentral_extra_limit),
        "outputs": outputs,
    }
    (output_dir / "region_seed_strategy_manifest.json").write_text(
        json.dumps(manifest, indent=2)
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
