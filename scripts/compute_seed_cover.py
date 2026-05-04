#!/usr/bin/env python3
"""
Compute a seed list that covers all nodes within K hops.
Greedy sweep: pick the next uncovered node as a seed, cover its K-hop neighborhood.
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from collections import deque
from pathlib import Path

import networkx as nx
from tqdm import tqdm


def bfs_k_hop_nodes(graph: nx.Graph, seed: str, k: int) -> set:
    visited = {seed}
    q: deque[tuple[str, int]] = deque([(seed, 0)])
    while q:
        node, depth = q.popleft()
        if depth >= k:
            continue
        for nb in graph.neighbors(node):
            if nb in visited:
                continue
            visited.add(nb)
            q.append((nb, depth + 1))
    return visited


def compute_seed_cover(graph: nx.Graph, k: int) -> list:
    nodes = list(graph.nodes())
    covered: set = set()
    seeds: list = []
    total = len(nodes)
    t0 = time.time()
    for idx, node in enumerate(tqdm(nodes, desc="Seed sweep"), start=1):
        if node in covered:
            continue
        seeds.append(node)
        covered.update(bfs_k_hop_nodes(graph, node, k))
        if len(seeds) % 50 == 0:
            pct = 100.0 * len(covered) / max(total, 1)
            print(
                f"seeds={len(seeds)} covered={len(covered)}/{total} ({pct:.2f}%) "
                f"elapsed={time.time() - t0:.1f}s"
            )
    return seeds


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute K-hop seed cover list.")
    parser.add_argument(
        "--graph-path",
        type=Path,
        required=True,
        help="Path to graph pickle/graphml/json",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        required=True,
        help="Path to write seeds json",
    )
    parser.add_argument("--k", type=int, default=6, help="Hop distance")
    parser.add_argument(
        "--undirected",
        action="store_true",
        help="Use undirected view of the graph",
    )
    args = parser.parse_args()

    graph_path = args.graph_path
    if graph_path.suffix == ".pickle":
        with graph_path.open("rb") as f:
            g = pickle.load(f)
    elif graph_path.suffix == ".graphml":
        g = nx.read_graphml(graph_path)
    elif graph_path.suffix == ".json":
        g = nx.readwrite.json_graph.node_link_graph(
            json.load(graph_path.open("r"))
        )
    else:
        raise ValueError(f"Unsupported graph format: {graph_path}")

    if args.undirected and g.is_directed():
        g = g.to_undirected(as_view=True)

    seeds = compute_seed_cover(g, args.k)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "k": args.k,
        "num_nodes": g.number_of_nodes(),
        "num_edges": g.number_of_edges(),
        "num_seeds": len(seeds),
        "graph_path": str(graph_path),
        "seeds": seeds,
    }
    args.output_path.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {len(seeds)} seeds -> {args.output_path}")


if __name__ == "__main__":
    main()
