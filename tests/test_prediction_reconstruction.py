import pickle
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))
from prediction_reconstruction import reconstruct_predictions


@pytest.fixture
def mock_dataset():
    """Create a mock dataset with minimal required attributes."""

    class MockDataset:
        def __init__(self):
            self.in_timesteps = 6
            self.out_timesteps = 3
            self.num_nodes = 5
            self.node_seed = "node_0"

            # Create simple node mappings
            self.node_ids = [f"node_{i}" for i in range(self.num_nodes)]
            self.node_to_id = {node: i for i, node in enumerate(self.node_ids)}
            self.id_to_node = {i: node for i, node in enumerate(self.node_ids)}

            # Simple batch mapping - each node maps to itself and neighbors
            self.batch_mapping = {
                "node_0": ["node_0", "node_1"],
                "node_1": ["node_1", "node_2"],
                "node_2": ["node_2", "node_3"],
                "node_3": ["node_3", "node_4"],
                "node_4": ["node_4"],
            }

        def _get_time_and_node_indices(self, idx):
            """Simple implementation that returns predictable values."""
            # For simplicity, time_idx = idx, and use the current seed's neighbors
            time_idx = idx
            neighbors = self.batch_mapping.get(self.node_seed, [])
            node_indices = [self.node_to_id[n] for n in neighbors]
            return time_idx, node_indices

    return MockDataset()


@pytest.fixture
def mock_metadata(tmp_path):
    """Create mock metadata file with time mappings."""
    # Create timestamp mappings
    timeframes = pd.date_range(
        start="2024-01-01 00:00:00", periods=20, freq="15min"
    ).to_list()
    time_idx_map = {ts: i for i, ts in enumerate(timeframes)}

    metadata = {
        "time_idx_map": time_idx_map,
        "node_idx_map": {f"node_{i}": i for i in range(5)},
        "feature_idx_map": {"level": 0},
    }

    metadata_path = tmp_path / "test_metadata.pkl"
    with open(metadata_path, "wb") as f:
        pickle.dump(metadata, f)

    return str(metadata_path)


def test_reconstruct_single_batch(mock_dataset, mock_metadata):
    """Test reconstruction of a single batch of predictions."""
    # Create mock predictions: batch_size=2, num_nodes=2, out_timesteps=3
    predictions = torch.tensor(
        [
            [[1.2, 2.3, 3.4], [0.5, 1.5, 2.5]],  # Sample 0, 2 nodes, 3 timesteps
            [[2.1, 3.2, 4.3], [1.4, 2.4, 3.4]],  # Sample 1, 2 nodes, 3 timesteps
        ]
    )

    batch_indices = [0, 1]

    df = reconstruct_predictions(
        predictions=predictions,
        dataset=mock_dataset,
        batch_indices=batch_indices,
        metadata_path=mock_metadata,
        value_column_name="predicted_level",
    )

    # Check structure
    assert isinstance(df, pd.DataFrame)
    assert isinstance(df.index, pd.MultiIndex)
    assert df.index.names == ["NodeID", "Timestamp"]
    assert "predicted_level" in df.columns

    # Check number of rows
    # 2 samples × 2 nodes × 3 timesteps = 12 rows
    assert len(df) == 12

    # Check that NodeIDs are present
    node_ids = df.index.get_level_values("NodeID").unique()
    assert "node_0" in node_ids
    assert "node_1" in node_ids

    # Check that timestamps are datetime
    timestamps = df.index.get_level_values("Timestamp")
    assert pd.api.types.is_datetime64_any_dtype(timestamps)


def test_reconstruct_single_sample(mock_dataset, mock_metadata):
    """Test reconstruction of a single sample (no batch dimension)."""
    # Single sample: num_nodes=2, out_timesteps=3
    predictions = torch.tensor([[1.2, 2.3, 3.4], [0.5, 1.5, 2.5]])

    batch_indices = [0]

    df = reconstruct_predictions(
        predictions=predictions,
        dataset=mock_dataset,
        batch_indices=batch_indices,
        metadata_path=mock_metadata,
    )

    # Check structure
    assert len(df) == 2 * 3  # 2 nodes × 3 timesteps


def test_reconstruction_temporal_alignment(mock_dataset, mock_metadata):
    """Test that predictions align correctly with future timestamps."""
    predictions = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])
    batch_indices = [0]

    df = reconstruct_predictions(
        predictions=predictions,
        dataset=mock_dataset,
        batch_indices=batch_indices,
        metadata_path=mock_metadata,
    )

    # Load metadata to verify timestamps
    with open(mock_metadata, "rb") as f:
        metadata = pickle.load(f)
    idx_to_timestamp = {idx: ts for ts, idx in metadata["time_idx_map"].items()}

    # Sample 0: time_idx=0, so predictions start at time_idx=6 (after in_timesteps=6)
    expected_start_time = 0 + mock_dataset.in_timesteps
    expected_timestamps = [
        idx_to_timestamp[expected_start_time + i]
        for i in range(mock_dataset.out_timesteps)
    ]

    # Get actual timestamps for first node
    node_0_data = df.xs("node_0", level="NodeID")
    actual_timestamps = node_0_data.index.tolist()

    assert actual_timestamps == expected_timestamps


def test_reconstruction_preserves_prediction_values(mock_dataset, mock_metadata):
    """Test that prediction values are correctly preserved."""
    predictions = torch.tensor([[[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]]])
    batch_indices = [0]

    df = reconstruct_predictions(
        predictions=predictions,
        dataset=mock_dataset,
        batch_indices=batch_indices,
        metadata_path=mock_metadata,
    )

    # Check values for node_0
    node_0_values = df.xs("node_0", level="NodeID")["predicted_level"].values.tolist()
    assert node_0_values == pytest.approx([1.1, 2.2, 3.3])

    # Check values for node_1
    node_1_values = df.xs("node_1", level="NodeID")["predicted_level"].values.tolist()
    assert node_1_values == pytest.approx([4.4, 5.5, 6.6])


def test_reconstruction_with_custom_column_name(mock_dataset, mock_metadata):
    """Test that custom column names work correctly."""
    # Mock dataset returns 2 nodes (node_0 and node_1), so predictions need 2 nodes
    predictions = torch.tensor([[[1.0, 2.0, 3.0], [1.5, 2.5, 3.5]]])
    batch_indices = [0]

    df = reconstruct_predictions(
        predictions=predictions,
        dataset=mock_dataset,
        batch_indices=batch_indices,
        metadata_path=mock_metadata,
        value_column_name="traffic_level",
    )

    assert "traffic_level" in df.columns
    assert "predicted_level" not in df.columns


def test_reconstruction_multiple_samples(mock_dataset, mock_metadata):
    """Test reconstruction with multiple samples in a batch."""
    # Batch of 4 samples, 2 nodes each, 3 timesteps
    batch_size = 4
    predictions = torch.randn(batch_size, 2, 3)
    batch_indices = [0, 1, 2, 3]

    df = reconstruct_predictions(
        predictions=predictions,
        dataset=mock_dataset,
        batch_indices=batch_indices,
        metadata_path=mock_metadata,
    )

    # Should have 4 samples × 2 nodes × 3 timesteps = 24 rows
    assert len(df) == 24

    # Check that we have data for different timestamps
    timestamps = df.index.get_level_values("Timestamp").unique()
    # Each sample advances time by 1, predictions span 3 timesteps
    # Sample 0: times 6, 7, 8
    # Sample 1: times 7, 8, 9
    # Sample 2: times 8, 9, 10
    # Sample 3: times 9, 10, 11
    # Unique timestamps: 6, 7, 8, 9, 10, 11 = 6 timestamps
    assert len(timestamps) == 6


def test_reconstruction_node_ordering(mock_dataset, mock_metadata):
    """Test that node ordering is preserved correctly."""
    predictions = torch.tensor(
        [[[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]]  # node_0  # node_1
    )
    batch_indices = [0]

    df = reconstruct_predictions(
        predictions=predictions,
        dataset=mock_dataset,
        batch_indices=batch_indices,
        metadata_path=mock_metadata,
    )

    # All values for node_0 should be 1.0
    node_0_values = df.xs("node_0", level="NodeID")["predicted_level"].unique()
    assert len(node_0_values) == 1
    assert node_0_values[0] == pytest.approx(1.0)

    # All values for node_1 should be 2.0
    node_1_values = df.xs("node_1", level="NodeID")["predicted_level"].unique()
    assert len(node_1_values) == 1
    assert node_1_values[0] == pytest.approx(2.0)
