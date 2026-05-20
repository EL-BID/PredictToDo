#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
import shutil
import subprocess
from pathlib import Path

import geopandas as gpd
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build central/non-central node map from prioritized corridor geometries."
    )
    parser.add_argument(
        "--lpkx-path",
        type=Path,
        default=Path("artifacts/21 CORREDORES PRIORIZADOS.lpkx"),
        help="ArcGIS layer package containing the prioritized corridor shapefile.",
    )
    parser.add_argument(
        "--corridors-shp",
        type=Path,
        default=None,
        help="Optional already-extracted corridor shapefile path.",
    )
    parser.add_argument(
        "--extract-dir",
        type=Path,
        default=Path("data/artifacts/corridors_prioritized_lpkx"),
        help="Directory where the LPKX package is extracted when needed.",
    )
    parser.add_argument(
        "--mvi-geojson",
        type=Path,
        default=Path("data/2_Estructura_via/estructura_via_completa/MVI.geojson"),
    )
    parser.add_argument(
        "--graph-path",
        type=Path,
        default=Path("data/datos_procesados/conectivity_graph/road_graph.pickle"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/artifacts/corridor_regions"),
    )
    parser.add_argument("--buffer-meters", type=float, default=20.0)
    parser.add_argument("--target-crs", default="EPSG:3116")
    return parser.parse_args()


def extract_lpkx(lpkx_path: Path, extract_dir: Path) -> Path:
    shp_path = extract_dir / "commondata" / "corredores_principales" / "PLMB_V2.shp"
    if shp_path.exists():
        return shp_path
    if not lpkx_path.exists():
        raise FileNotFoundError(f"LPKX not found: {lpkx_path}")
    bsdtar = shutil.which("bsdtar")
    if bsdtar is None:
        raise RuntimeError("bsdtar is required to extract .lpkx packages in this environment")
    extract_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [bsdtar, "-xf", str(lpkx_path), "-C", str(extract_dir)],
        check=True,
    )
    if not shp_path.exists():
        raise FileNotFoundError(f"Expected corridor shapefile after extraction: {shp_path}")
    return shp_path


def load_graph_nodes(graph_path: Path) -> set[str]:
    with graph_path.open("rb") as f:
        graph = pickle.load(f)
    return {str(node) for node in graph.nodes()}


def build_mvi_road_ids(mvi: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    mvi = mvi.copy()
    mvi["id_base"] = (
        mvi["MVICIV"].astype(str)
        + "_"
        + mvi["MVISVIA"].astype(str).str.replace(" ", "", regex=False)
    )
    mvi["contador"] = mvi.groupby("id_base").cumcount() + 1
    mvi["road_id"] = mvi["id_base"].astype(str) + "." + mvi["contador"].astype(str)
    return mvi


def main() -> None:
    args = parse_args()
    corridor_shp = args.corridors_shp or extract_lpkx(args.lpkx_path, args.extract_dir)

    graph_nodes = load_graph_nodes(args.graph_path)
    corridors = gpd.read_file(corridor_shp).to_crs(args.target_crs)
    if "CORREDOR" not in corridors.columns:
        raise ValueError("Corridor shapefile must contain a CORREDOR column")

    corridor_buffers = corridors[["CORREDOR", "geometry"]].copy()
    corridor_buffers["geometry"] = corridor_buffers.geometry.buffer(args.buffer_meters)

    mvi = gpd.read_file(args.mvi_geojson)
    mvi = build_mvi_road_ids(mvi)
    mvi = mvi[mvi["road_id"].astype(str).isin(graph_nodes)].copy()
    mvi = mvi[["road_id", "geometry"]].to_crs(args.target_crs)

    joined = gpd.sjoin(mvi, corridor_buffers, how="inner", predicate="intersects")
    corridor_names = (
        joined.groupby("road_id")["CORREDOR"]
        .agg(lambda values: "|".join(sorted(set(str(v) for v in values))))
        .rename("corridor_names")
    )

    region_map = pd.DataFrame({"road_id": sorted(graph_nodes)})
    region_map = region_map.merge(corridor_names, on="road_id", how="left")
    region_map["is_central"] = region_map["corridor_names"].notna()
    region_map["corridor_names"] = region_map["corridor_names"].fillna("")
    region_map["corridor_count"] = region_map["corridor_names"].map(
        lambda raw: 0 if raw == "" else len(raw.split("|"))
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    region_map_path = output_dir / "node_region_map.csv"
    central_nodes_path = output_dir / "central_nodes.csv"
    noncentral_nodes_path = output_dir / "noncentral_nodes.csv"
    metadata_path = output_dir / "node_region_map_metadata.json"

    region_map.to_csv(region_map_path, index=False)
    region_map.loc[region_map["is_central"], ["road_id"]].to_csv(
        central_nodes_path,
        index=False,
    )
    region_map.loc[~region_map["is_central"], ["road_id"]].to_csv(
        noncentral_nodes_path,
        index=False,
    )

    corridor_counts = (
        joined.groupby("CORREDOR")["road_id"].nunique().sort_values(ascending=False)
    )
    metadata = {
        "lpkx_path": str(args.lpkx_path),
        "corridors_shp": str(corridor_shp),
        "mvi_geojson": str(args.mvi_geojson),
        "graph_path": str(args.graph_path),
        "target_crs": args.target_crs,
        "buffer_meters": float(args.buffer_meters),
        "num_corridors": int(len(corridors)),
        "num_graph_nodes": int(len(graph_nodes)),
        "num_mvi_nodes_in_graph": int(len(mvi)),
        "num_central_nodes": int(region_map["is_central"].sum()),
        "num_noncentral_nodes": int((~region_map["is_central"]).sum()),
        "corridor_node_counts": {str(k): int(v) for k, v in corridor_counts.items()},
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
