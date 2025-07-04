import pickle

import networkx as nx
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

import src.preprocessing.imputation as imp


@pytest.fixture
def sample_graph():
    """
    Creates a sample graph with distinct components and node attributes.
    - Component 1: A-B-C (AVENUE)
    - Component 2: D-E (STREET)
    - Isolated Node: F (Null type)
    """
    G = nx.Graph()
    G.add_node("A", MVITIPO="AVENUE")
    G.add_node("B", MVITIPO="AVENUE")
    G.add_node("C", MVITIPO="AVENUE")
    G.add_node("D", MVITIPO="STREET")
    G.add_node("E", MVITIPO="STREET")
    G.add_node("F", MVITIPO=" ")  # Test space replacement
    G.add_edges_from([("A", "B"), ("B", "C"), ("D", "E")])
    return G


@pytest.fixture
def sample_mean_features():
    """Creates a sample DataFrame of mean features for specific nodes."""
    data = {"trend": [0.5, 0.8, 1.2, 1.5], "regularSpeed": [50, 55, 30, 32]}
    index = ["A", "C", "D", "E"]  # Node B has no features
    return pd.DataFrame(data, index=index)


@pytest.fixture
def temp_dir(tmp_path):
    """Creates a temporary directory for file-based tests."""
    return tmp_path


def test_load_graph(sample_graph, temp_dir):
    """Tests loading a graph from a pickle file."""
    graph_path = temp_dir / "test_graph.pkl"
    with open(graph_path, "wb") as f:
        pickle.dump(sample_graph, f)

    loaded_graph = imp.load_graph(str(graph_path))

    assert isinstance(loaded_graph, nx.Graph)
    assert list(loaded_graph.nodes()) == list(sample_graph.nodes())
    assert list(loaded_graph.edges()) == list(sample_graph.edges())


def test_load_and_process_irregularities(temp_dir):
    """Tests loading and aggregating data from multiple parquet files."""
    # Create mock data that includes all necessary columns
    df1 = pd.DataFrame(
        {
            "id_mvi": ["A", "B", "A"],
            "timeframe": ["day", "night", "day"],
            "speed": [110, 60, 100],
            "trend": [10, 5, 2],
            "regularSpeed": [100, 50, 98],
        }
    )
    df2 = pd.DataFrame(
        {
            "id_mvi": ["B", "C"],
            "timeframe": ["day", "night"],
            "speed": [65, 40],
            "trend": [7, 20],
            "regularSpeed": [54, 30],
        }
    )

    # Save to parquet files in the temp directory
    df1.to_parquet(temp_dir / "data1.parquet")
    df2.to_parquet(temp_dir / "data2.parquet")

    # The expected result only depends on trend and regularSpeed
    expected_data = {
        "trend": {"A": 6.0, "B": 6.0, "C": 20.0},
        "regularSpeed": {"A": 99.0, "B": 52.0, "C": 30.0},
    }
    expected_df = pd.DataFrame(expected_data)
    expected_df.index.name = "id_mvi"

    result_df = imp.load_and_process_irregularities(str(temp_dir))

    assert_frame_equal(result_df.sort_index(), expected_df.sort_index())


def test_load_irregularities_no_files(temp_dir):
    """Tests that a FileNotFoundError is raised if no parquet files exist."""
    with pytest.raises(FileNotFoundError):
        imp.load_and_process_irregularities(str(temp_dir))


def test_get_disconnected_nodes(sample_graph):
    """Tests the identification of nodes outside the largest component."""
    # Largest component is {A, B, C}. Disconnected should be {D, E, F}
    expected_disconnected = {"D", "E", "F"}
    disconnected_nodes = imp.get_disconnected_nodes(sample_graph)
    assert disconnected_nodes == expected_disconnected


def test_get_street_types(sample_graph):
    """Tests the extraction of street type attributes from the graph."""
    expected_data = {
        "node_id": ["A", "B", "C", "D", "E", "F"],
        "street_type": ["AVENUE", "AVENUE", "AVENUE", "STREET", "STREET", "Null"],
    }
    expected_df = pd.DataFrame(expected_data).set_index("node_id")

    street_types_df = imp.get_street_types(sample_graph)
    assert_frame_equal(street_types_df.sort_index(), expected_df.sort_index())


def test_calculate_mean_features_by_street_type(sample_mean_features):
    """Tests the aggregation of features by street type for imputation."""
    street_types_data = {
        "node_id": ["A", "C", "D", "E"],
        "street_type": ["AVENUE", "AVENUE", "STREET", "STREET"],
    }
    street_types_df = pd.DataFrame(street_types_data).set_index("node_id")

    # Expected averages
    # AVENUE: trend=(0.5+0.8)/2=0.65, speed=(50+55)/2=52.5
    # STREET: trend=(1.2+1.5)/2=1.35, speed=(30+32)/2=31.0
    expected_data = {
        "trend": {"AVENUE": 0.65, "STREET": 1.35},
        "regularSpeed": {"AVENUE": 52.5, "STREET": 31.0},
    }
    expected_df = pd.DataFrame(expected_data)
    expected_df.index.name = "street_type"

    result_df = imp.calculate_mean_features_by_street_type(
        sample_mean_features, street_types_df
    )
    assert_frame_equal(
        result_df.sort_index(), expected_df.sort_index(), check_dtype=False
    )


def test_compute_neighbor_features(sample_graph, sample_mean_features):
    """Tests the core logic of computing features from neighbors."""
    street_types = imp.get_street_types(sample_graph)
    disconnected_nodes = imp.get_disconnected_nodes(sample_graph)

    # Calculate features for all nodes
    result_df = imp.compute_neighbor_features(
        graph=sample_graph,
        mean_features=sample_mean_features,
        street_types=street_types,
        disconnected_nodes=disconnected_nodes,
        k_hops=1,
    )

    # Node 'B': Neighbors are 'A', 'C'. Both have data and same street type
    # Expected trend: (0.5 + 0.8) / 2 = 0.65
    # Expected speed: (50 + 55) / 2 = 52.5
    assert result_df.loc["B", "mean_trend"] == pytest.approx(0.65)
    assert result_df.loc["B", "mean_regularSpeed"] == pytest.approx(52.5)

    # Node 'A': Neighbors 'B'. 'B' has no data in sample_mean_features
    # Ego graph includes 'A' itself, which has data
    # So the result should be the features of 'A' itself
    assert result_df.loc["A", "mean_trend"] == 0.5
    assert result_df.loc["A", "mean_regularSpeed"] == 50

    # Node 'D': Disconnected, so should be NaN
    assert pd.isna(result_df.loc["D", "mean_trend"])
    assert pd.isna(result_df.loc["D", "mean_regularSpeed"])


def test_impute_missing_features():
    """Tests the filling of NaN values using mean values by street type."""
    features_df = pd.DataFrame(
        {
            "mean_trend": [1.0, np.nan, 3.0, np.nan],
            "mean_regularSpeed": [50, np.nan, 30, np.nan],
        },
        index=["A", "B", "C", "D"],
    )

    street_types = pd.DataFrame(
        {"street_type": ["TYPE1", "TYPE2", "TYPE1", "TYPE2"]},
        index=["A", "B", "C", "D"],
    )

    mean_features_by_type = pd.DataFrame(
        {
            "trend": {"TYPE1": 2.0, "TYPE2": 5.0},
            "regularSpeed": {"TYPE1": 40, "TYPE2": 20},
        }
    )
    mean_features_by_type.index.name = "street_type"

    # Expected result: B and D should be filled with TYPE2 values
    expected_data = {
        "mean_trend": [1.0, 5.0, 3.0, 5.0],
        "mean_regularSpeed": [50.0, 20.0, 30.0, 20.0],
    }
    expected_df = pd.DataFrame(expected_data, index=["A", "B", "C", "D"])

    result_df = imp.impute_missing_features(
        features_df, street_types, mean_features_by_type
    )
    assert_frame_equal(result_df.sort_index(), expected_df.sort_index())


def test_generate_neighbor_features_integration(
    mocker, sample_graph, sample_mean_features
):
    """
    Tests the main orchestrator function by mocking the helper functions.
    This ensures the pipeline flows correctly without actual file I/O.
    """
    # Mock all helper functions
    mocker.patch("src.preprocessing.imputation.load_graph", return_value=sample_graph)
    mocker.patch(
        "src.preprocessing.imputation.load_and_process_irregularities",
        return_value=sample_mean_features,
    )

    # The other functions are pure and fast, so we can call them directly

    # Call the main function with dummy paths
    final_df = imp.generate_neighbor_features_from_files(
        graph_path="dummy_path.pkl", irregularities_dir="dummy_dir/", k_hops=1
    )

    # Assert the final output is as expected
    # Node B has its features calculated from A and C
    # Expected trend: (0.5 + 0.8) / 2 = 0.65
    # Expected speed: (50 + 55) / 2 = 52.5
    assert final_df.loc["B", "mean_trend"] == pytest.approx(0.65)
    assert final_df.loc["B", "mean_regularSpeed"] == pytest.approx(52.5)

    # Nodes D, E, F are disconnected/have no neighbors with data
    # They should be imputed with the mean of their street type
    # STREET mean trend: (1.2+1.5)/2=1.35
    # Null (from F) has no features, so it will remain NaN
    assert final_df.loc["D", "mean_trend"] == pytest.approx(1.35)
    assert final_df.loc["E", "mean_trend"] == pytest.approx(1.35)
    assert pd.isna(final_df.loc["F", "mean_trend"])
