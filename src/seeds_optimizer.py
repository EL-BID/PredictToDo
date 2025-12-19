import math
import pickle
from pathlib import Path

import networkx as nx
from tqdm import tqdm


class SeedsOptimizer:
    """
    Optimizer to find the minimum set of seed nodes for traffic dataset batching.

    Given a graph and a node batch size, this class finds the smallest set of
    seed nodes such that their BFS neighborhoods cover all nodes in the graph.

    Attributes:
        graph: The NetworkX graph.
        node_batch_size: Number of nodes per BFS neighborhood.
        output_path: Path to save the optimal seeds pickle file.
        verbose: Whether to print progress information.
        neighborhoods: Precomputed BFS neighborhoods for all nodes.
        optimal_seeds: The computed list of optimal seed nodes.
    """

    def __init__(
        self,
        graph_path: str,
        node_batch_size: int = 500,
        output_path: str = "data/datos_procesados/optimal_seeds.pkl",
        verbose: bool = True,
    ):
        """
        Initialize the SeedsOptimizer.

        Args:
            graph_path: Path to the NetworkX graph pickle file.
            node_batch_size: Number of nodes per BFS neighborhood (default: 500).
            output_path: Path to save the output pickle file with seed node IDs.
            verbose: Whether to print progress information (default: True).
        """
        self.graph_path = graph_path
        self.node_batch_size = node_batch_size
        self.output_path = output_path
        self.verbose = verbose

        # Load graph
        if self.verbose:
            print(f"Loading graph from {self.graph_path}...")
        self.graph = self._load_graph(graph_path)

        # Initialize computed attributes
        self.neighborhoods: dict[str, set[str]] | None = None
        self.optimal_seeds: list[str] | None = None

    def _load_graph(self, graph_path: str) -> nx.Graph:
        """Load a NetworkX graph from a pickle file."""
        with open(graph_path, "rb") as f:
            graph = pickle.load(f)
        return graph

    def _compute_bfs_neighborhood(self, seed_node: str) -> set[str]:
        """
        Compute the BFS neighborhood of a seed node up to `node_batch_size` nodes.

        This mirrors the logic in TrafficDataset.get_first_n_reachable() but for a
        single node.

        Args:
            seed_node: The node to start BFS from.

        Returns:
            A set of node IDs reachable from the seed node (including the seed itself).
        """
        bfs_edges_generator = nx.bfs_edges(self.graph, source=seed_node)

        reachable = {seed_node}
        for _, v in bfs_edges_generator:
            reachable.add(v)
            if len(reachable) >= self.node_batch_size:
                break

        return reachable

    def _compute_all_neighborhoods(self) -> dict[str, set[str]]:
        """
        Precompute BFS neighborhoods for all nodes in the graph.

        Returns:
            Dictionary mapping each node to its BFS neighborhood set.
        """
        neighborhoods = {}
        for node in tqdm(self.graph.nodes()):
            neighborhoods[node] = self._compute_bfs_neighborhood(node)
        return neighborhoods

    def find_optimal_seeds(self) -> list[str]:
        """
        Find the minimum set of seed nodes whose BFS neighborhoods cover all graph nodes.

        Uses a greedy set cover algorithm that repeatedly selects the seed covering
        the most uncovered nodes. This provides a ln(n)-approximation to the optimal
        solution.

        Returns:
            List of seed node IDs that together cover all nodes in the graph.
        """
        all_nodes = set(self.graph.nodes())
        num_nodes = len(all_nodes)

        if self.verbose:
            print(f"Graph has {num_nodes} nodes")
            print(f"Node batch size: {self.node_batch_size}")
            print(
                f"Theoretical minimum seeds: {math.ceil(num_nodes / self.node_batch_size)}"
            )
            print("\nPrecomputing BFS neighborhoods for all nodes...")

        # Precompute all neighborhoods
        self.neighborhoods = self._compute_all_neighborhoods()

        if self.verbose:
            print("Done precomputing neighborhoods.")
            print("\nRunning greedy set cover algorithm...")

        # Greedy set cover
        uncovered = all_nodes.copy()
        selected_seeds: list[str] = []

        iteration = 0
        while uncovered:
            iteration += 1

            # Find the seed that covers the most uncovered nodes
            best_seed = None
            best_coverage = 0
            best_newly_covered = set()

            for node, neighborhood in self.neighborhoods.items():
                newly_covered = neighborhood & uncovered
                coverage = len(newly_covered)
                if coverage > best_coverage:
                    best_coverage = coverage
                    best_seed = node
                    best_newly_covered = newly_covered

            if best_seed is None or best_coverage == 0:
                # This should not happen if the graph is connected
                raise RuntimeError(
                    "Could not find a seed to cover remaining nodes. "
                    "The graph may be disconnected."
                )

            # Select this seed
            selected_seeds.append(best_seed)
            uncovered -= best_newly_covered

            if self.verbose:
                print(
                    f"  Iteration {iteration}: Selected seed '{best_seed}' "
                    f"covering {best_coverage} new nodes. "
                    f"Remaining uncovered: {len(uncovered)}"
                )

        if self.verbose:
            print(f"\nOptimization complete!")
            print(
                f"Selected {len(selected_seeds)} seed nodes to cover all {num_nodes} nodes"
            )
            print(
                f"Efficiency: {len(selected_seeds)} seeds vs "
                f"{math.ceil(num_nodes / self.node_batch_size)} theoretical minimum"
            )

        self.optimal_seeds = selected_seeds
        return selected_seeds

    def verify_coverage(self) -> tuple[bool, set[str]]:
        """
        Verify that the selected seeds cover all nodes in the graph.

        Returns:
            Tuple of (is_complete_coverage, uncovered_nodes).

        Raises:
            RuntimeError: If find_optimal_seeds() has not been called yet.
        """
        if self.optimal_seeds is None:
            raise RuntimeError(
                "No seeds computed yet. Call find_optimal_seeds() first."
            )

        all_nodes = set(self.graph.nodes())
        covered = set()

        for seed in self.optimal_seeds:
            neighborhood = self._compute_bfs_neighborhood(seed)
            covered |= neighborhood

        uncovered = all_nodes - covered
        is_complete = len(uncovered) == 0

        if self.verbose:
            if is_complete:
                print("✓ All nodes are covered by the selected seeds.")
            else:
                print(f"✗ {len(uncovered)} nodes are NOT covered: {uncovered}")

        return is_complete, uncovered

    def save_seeds(self, output_path: str | None = None) -> None:
        """
        Save the list of seed nodes to a pickle file.

        Args:
            output_path: Path to save the file. If None, uses self.output_path.

        Raises:
            RuntimeError: If find_optimal_seeds() has not been called yet.
        """
        if self.optimal_seeds is None:
            raise RuntimeError(
                "No seeds computed yet. Call find_optimal_seeds() first."
            )

        path = Path(output_path or self.output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump(self.optimal_seeds, f)

        if self.verbose:
            print(f"Seeds saved to {path}")

    def print_summary(self) -> None:
        """Print a summary of the optimization results."""
        if self.optimal_seeds is None:
            print("No optimization performed yet. Call find_optimal_seeds() first.")
            return

        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Graph nodes: {len(self.graph.nodes())}")
        print(f"Node batch size: {self.node_batch_size}")
        print(f"Optimal seeds found: {len(self.optimal_seeds)}")
        print(f"Output path: {self.output_path}")
        print("=" * 60)


if __name__ == "__main__":
    optimizer = SeedsOptimizer(
        graph_path="data/datos_procesados/conectivity_graph/road_graph.pickle",
        node_batch_size=500,
        output_path="data/datos_procesados/optimal_seeds.pkl",
        verbose=True,
    )

    seeds = optimizer.find_optimal_seeds()
    optimizer.verify_coverage()
    optimizer.save_seeds()
    optimizer.print_summary()
