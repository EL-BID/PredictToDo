import os
import pickle
from typing import Dict, Set

import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm

# ==============================================================================
# 1. DATA LOADING FUNCTIONS
# ==============================================================================


def load_graph(graph_path: str) -> nx.Graph:
    """Loads a graph from a pickle file."""
    print(f"Loading graph from {graph_path}...")
    with open(graph_path, "rb") as f:
        graph = pickle.load(f)
    return graph


def load_and_process_irregularities(irregularities_dir: str) -> pd.DataFrame:
    """
    Loads all parquet files from a directory, concatenates them,
    and computes the mean trend and regularSpeed for each road segment.
    """
    print(f"Loading and processing irregularities from {irregularities_dir}...")
    use_cols = ["id_mvi", "timeframe", "trend", "speed", "regularSpeed"]

    parquet_files = [
        os.path.join(irregularities_dir, f)
        for f in os.listdir(irregularities_dir)
        if f.endswith(".parquet")
    ]

    if not parquet_files:
        raise FileNotFoundError(
            f"No .parquet files found in directory: {irregularities_dir}"
        )

    df = pd.concat(
        [pd.read_parquet(f, columns=use_cols) for f in parquet_files], ignore_index=True
    )

    # Compute the mean features for each road segment
    mean_features = df.groupby("id_mvi")[["trend", "regularSpeed"]].mean()
    return mean_features


# ==============================================================================
# 2. GRAPH ANALYSIS FUNCTIONS
# ==============================================================================


def get_disconnected_nodes(graph: nx.Graph) -> Set[str]:
    """Identifies all nodes not in the largest connected component."""
    print("Identifying disconnected nodes...")
    # Find all connected components in the undirected version of the graph
    components = list(nx.connected_components(graph.to_undirected()))

    if not components:
        return set()

    # The largest component is the one with the most nodes
    largest_component = max(components, key=len)

    # Disconnected nodes are all nodes not in the largest component
    all_nodes = set(graph.nodes())
    disconnected_nodes = all_nodes - largest_component

    return disconnected_nodes


def get_street_types(graph: nx.Graph) -> pd.DataFrame:
    """Extracts the 'MVITIPO' attribute for each node into a DataFrame."""
    print("Extracting street types from graph nodes...")
    street_types_dict = nx.get_node_attributes(graph, "MVITIPO")

    street_types_df = (
        pd.DataFrame(street_types_dict.items(), columns=["node_id", "street_type"])
        .replace({" ": "Null"})
        .set_index("node_id")
    )
    return street_types_df


# ==============================================================================
# 3. FEATURE COMPUTATION FUNCTIONS
# ==============================================================================


def calculate_mean_features_by_street_type(
    mean_features: pd.DataFrame, street_types: pd.DataFrame
) -> pd.DataFrame:
    """
    Computes the mean 'trend' and 'regularSpeed' for each street type.
    This is used for later imputation of missing values.
    """
    print("Calculating mean features per street type for imputation...")
    mean_features_per_street_type = (
        mean_features.merge(street_types, how="left", left_index=True, right_index=True)
        .groupby("street_type")[["trend", "regularSpeed"]]
        .mean()
    )
    return mean_features_per_street_type


def compute_neighbor_features(
    graph: nx.Graph,
    mean_features: pd.DataFrame,
    street_types: pd.DataFrame,
    disconnected_nodes: Set[str],
    k_hops: int = 10,
) -> pd.DataFrame:
    """
    For each node, calculates the mean features of its neighbors within k_hops
    that share the same street type.
    """
    print(f"Computing neighbor features within k={k_hops} hops...")
    neighbor_features: Dict[str, Dict[str, float]] = {}

    for node in tqdm(graph.nodes(), desc="Processing nodes"):
        if node in disconnected_nodes:
            neighbor_features[node] = {
                "mean_trend": np.nan,
                "mean_regularSpeed": np.nan,
            }
            continue

        # Find all neighbors within the k-hop radius
        k_hop_subgraph = nx.ego_graph(graph, node, radius=k_hops)

        # Get street type for the current node
        node_street_type = street_types.loc[node, "street_type"]

        # Filter neighbors to those with data and the same street type
        valid_neighbors = [
            n
            for n in k_hop_subgraph.nodes()
            if n in mean_features.index
            and street_types.loc[n, "street_type"] == node_street_type
        ]

        if valid_neighbors:
            neighbor_df = mean_features.loc[valid_neighbors]
            neighbor_features[node] = {
                "mean_trend": neighbor_df["trend"].mean(),
                "mean_regularSpeed": neighbor_df["regularSpeed"].mean(),
            }
        else:
            # Assign NaN if no valid neighbors are found
            neighbor_features[node] = {
                "mean_trend": np.nan,
                "mean_regularSpeed": np.nan,
            }

    return pd.DataFrame.from_dict(neighbor_features, orient="index")


def impute_missing_features(
    features_df: pd.DataFrame,
    street_types: pd.DataFrame,
    mean_features_by_type: pd.DataFrame,
) -> pd.DataFrame:
    """
    Fills NaN values in the feature DataFrame using the mean values
    calculated for each street type.
    """
    print("Imputing missing values...")
    # Add street type information to the features DataFrame
    df = features_df.merge(street_types, how="left", left_index=True, right_index=True)

    # Add the mean values for imputation
    df = df.merge(
        mean_features_by_type,
        how="left",
        left_on="street_type",
        right_index=True,
        suffixes=("", "_impute"),
    )

    # Fill NaN values using the imputation columns
    df["mean_trend"] = df["mean_trend"].fillna(df["trend"])
    df["mean_regularSpeed"] = df["mean_regularSpeed"].fillna(df["regularSpeed"])

    # Clean up and return the final DataFrame
    final_df = df.drop(columns=["street_type", "trend", "regularSpeed"])
    return final_df


# ==============================================================================
# 4. MAIN ORCHESTRATOR FUNCTION
# ==============================================================================


def generate_neighbor_features_from_files(
    graph_path: str,
    irregularities_dir: str,
    k_hops: int = 10,
) -> pd.DataFrame:
    """
    Full pipeline to load data and generate the neighbor features DataFrame.

    Args:
        graph_path (str): Path to the graph pickle file.
        irregularities_dir (str): Directory containing irregularities parquet files.
        k_hops (int): The radius for neighbor search.

    Returns:
        pd.DataFrame: A DataFrame with computed and imputed neighbor features
                      for each node in the graph.
    """
    # Load original data
    base_graph = load_graph(graph_path)
    mean_features = load_and_process_irregularities(irregularities_dir)

    # Analyze graph structure
    disconnected_nodes = get_disconnected_nodes(base_graph)
    street_types = get_street_types(base_graph)

    # Compute raw neighbor features
    neighbor_features_df = compute_neighbor_features(
        graph=base_graph,
        mean_features=mean_features,
        street_types=street_types,
        disconnected_nodes=disconnected_nodes,
        k_hops=k_hops,
    )

    prop_missing = neighbor_features_df.isna().iloc[:, 0].sum() / len(
        neighbor_features_df
    )
    print(f"Proportion of nodes without direct neighbor data: {prop_missing:.2%}")

    # Impute missing values
    mean_features_by_type = calculate_mean_features_by_street_type(
        mean_features, street_types
    )
    final_features_df = impute_missing_features(
        features_df=neighbor_features_df,
        street_types=street_types,
        mean_features_by_type=mean_features_by_type,
    )

    print("Feature generation complete.")
    return final_features_df


# ==============================================================================
# 5. SCRIPT EXECUTION
# ==============================================================================

if __name__ == "__main__":
    # Paths and parameters
    PROCESSED_DATA_DIR = "data/datos_procesados"
    GRAPH_FILENAME = os.path.join(
        PROCESSED_DATA_DIR, "conectivity_graph/road_graph.pickle"
    )
    IRREGULARITIES_DIR = os.path.join(PROCESSED_DATA_DIR, "irregularidades_procesadas/")
    OUTPUT_FILENAME = os.path.join(PROCESSED_DATA_DIR, "neighbor_features.pkl")
    K_HOPS = 10

    # Generate the features
    final_df = generate_neighbor_features_from_files(
        graph_path=GRAPH_FILENAME, irregularities_dir=IRREGULARITIES_DIR, k_hops=K_HOPS
    )

    # Save the final DataFrame to a pickle file
    print(f"Saving final DataFrame to {OUTPUT_FILENAME}...")
    with open(OUTPUT_FILENAME, "wb") as f:
        pickle.dump(final_df, f)

    print("Process finished successfully.")
