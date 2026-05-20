#!/usr/bin/env python3
"""
Compute minimal regional seed lists for inference coverage.

This is a greedy set-cover approximation. For each region, candidate seeds are
scored by how many still-uncovered nodes from that same region they can reach
within K graph hops. The selected lists are meant for inference, so no
top-congestion extra seeds are added.
"""

from __future__ import annotations

import argparse
import heapq
import json
import pickle
import statistics
import time
from collections import deque
from pathlib import Path
from typing import Iterable

import pandas as pd
from tqdm import tqdm


def load_graph(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def bfs_k_hop_targets(graph, seed: str, k: int, target_nodes: set[str]) -> set[str]:
    visited = {seed}
    covered = {seed} if seed in target_nodes else set()
    q: deque[tuple[str, int]] = deque([(seed, 0)])
    while q:
        node, depth = q.popleft()
        if depth >= k:
            continue
        for nb in graph.neighbors(node):
            if nb in visited:
                continue
            visited.add(nb)
            if nb in target_nodes:
                covered.add(nb)
            q.append((nb, depth + 1))
    return covered


def greedy_max_cover(
    *,
    graph,
    target_nodes: set[str],
    candidate_nodes: Iterable[str],
    k: int,
    region_name: str,
) -> tuple[list[str], dict]:
    t0 = time.time()
    candidate_nodes = [str(node) for node in candidate_nodes if str(node) in graph]
    target_nodes = {str(node) for node in target_nodes if str(node) in graph}

    coverage_by_seed: dict[str, set[str]] = {}
    heap: list[tuple[int, str]] = []
    for seed in tqdm(candidate_nodes, desc=f"{region_name}: precompute coverage"):
        covered = bfs_k_hop_targets(graph, seed, k, target_nodes)
        if covered:
            coverage_by_seed[seed] = covered
            heapq.heappush(heap, (-len(covered), seed))

    uncovered = set(target_nodes)
    selected: list[str] = []
    selected_cover_sizes: list[int] = []
    stale_pops = 0
    while uncovered and heap:
        neg_score, seed = heapq.heappop(heap)
        old_score = -neg_score
        current_cover = coverage_by_seed[seed] & uncovered
        current_score = len(current_cover)
        if current_score == 0:
            continue
        if current_score < old_score:
            stale_pops += 1
            heapq.heappush(heap, (-current_score, seed))
            continue
        selected.append(seed)
        selected_cover_sizes.append(current_score)
        uncovered.difference_update(current_cover)
        if len(selected) % 25 == 0 or not uncovered:
            pct = 100.0 * (len(target_nodes) - len(uncovered)) / max(len(target_nodes), 1)
            print(
                f"{region_name}: seeds={len(selected)} "
                f"covered={len(target_nodes) - len(uncovered)}/{len(target_nodes)} "
                f"({pct:.2f}%) last_gain={current_score}"
            )

    gain_stats = {}
    if selected_cover_sizes:
        gain_stats = {
            "min": int(min(selected_cover_sizes)),
            "max": int(max(selected_cover_sizes)),
            "mean": float(round(statistics.fmean(selected_cover_sizes), 4)),
            "median": float(statistics.median(selected_cover_sizes)),
            "num_single_node_gains": int(sum(1 for value in selected_cover_sizes if value == 1)),
            "first_10": [int(value) for value in selected_cover_sizes[:10]],
            "last_10": [int(value) for value in selected_cover_sizes[-10:]],
        }

    metadata = {
        "region": region_name,
        "k": int(k),
        "candidate_scope": "region",
        "num_region_nodes": int(len(target_nodes)),
        "num_candidate_nodes": int(len(candidate_nodes)),
        "num_candidate_nodes_with_coverage": int(len(coverage_by_seed)),
        "num_seeds": int(len(selected)),
        "num_covered_nodes": int(len(target_nodes) - len(uncovered)),
        "num_uncovered_nodes": int(len(uncovered)),
        "coverage_pct": float(
            100.0 * (len(target_nodes) - len(uncovered)) / max(len(target_nodes), 1)
        ),
        "selected_gain_stats": gain_stats,
        "stale_heap_pops": int(stale_pops),
        "elapsed_seconds": round(time.time() - t0, 3),
    }
    return selected, metadata


def write_seed_json(path: Path, *, graph_path: Path, region_map_path: Path, metadata: dict, seeds: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        **metadata,
        "graph_path": str(graph_path),
        "region_map_path": str(region_map_path),
        "seeds": seeds,
    }
    path.write_text(json.dumps(payload, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute central/noncentral minimal seed cover lists.")
    parser.add_argument(
        "--graph-path",
        type=Path,
        default=Path("data/datos_procesados/conectivity_graph/road_graph.pickle"),
    )
    parser.add_argument(
        "--region-map-path",
        type=Path,
        default=Path("data/artifacts/corridor_regions/node_region_map.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/artifacts/regional_seed_lists"),
    )
    parser.add_argument("--k", type=int, default=6)
    args = parser.parse_args()

    graph = load_graph(args.graph_path)
    region_map = pd.read_csv(args.region_map_path)
    required = {"road_id", "is_central"}
    missing = required - set(region_map.columns)
    if missing:
        raise ValueError(f"Region map missing required columns: {sorted(missing)}")
    region_map["road_id"] = region_map["road_id"].astype(str)
    region_map["is_central"] = region_map["is_central"].astype(bool)

    graph_nodes = {str(node) for node in graph.nodes()}
    central_nodes = set(region_map.loc[region_map["is_central"], "road_id"]) & graph_nodes
    noncentral_nodes = set(region_map.loc[~region_map["is_central"], "road_id"]) & graph_nodes

    outputs = {}
    for region_name, nodes in [("central", central_nodes), ("noncentral", noncentral_nodes)]:
        seeds, metadata = greedy_max_cover(
            graph=graph,
            target_nodes=nodes,
            candidate_nodes=nodes,
            k=args.k,
            region_name=region_name,
        )
        output_path = args.output_dir / f"seeds_{region_name}_k{args.k}_minimal.json"
        write_seed_json(
            output_path,
            graph_path=args.graph_path,
            region_map_path=args.region_map_path,
            metadata=metadata,
            seeds=seeds,
        )
        outputs[region_name] = {
            "path": str(output_path),
            "metadata": metadata,
        }

    combined = {
        "k": int(args.k),
        "graph_path": str(args.graph_path),
        "region_map_path": str(args.region_map_path),
        "outputs": outputs,
        "total_seeds": int(sum(item["metadata"]["num_seeds"] for item in outputs.values())),
    }
    manifest_path = args.output_dir / f"regional_seed_cover_manifest_k{args.k}.json"
    manifest_path.write_text(json.dumps(combined, indent=2))
    print(json.dumps(combined, indent=2))


if __name__ == "__main__":
    main()
