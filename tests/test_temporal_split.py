import sys
from pathlib import Path

import pytest
from torch import Tensor

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

# Import test fixtures from the existing test file
from temporal_split import (
    TemporalSubset,
    temporal_train_test_split,
    temporal_train_val_test_split,
)
from tests.test_sparse_dataloader import TestTrafficDataset


class TestTemporalSplit(TestTrafficDataset):
    """Test suite for temporal splitting functionality."""

    def test_temporal_train_test_split_basic(self, test_data_setup):
        """Test basic train-test split functionality."""
        setup = test_data_setup

        # Create dataset
        from sparse_dataloader import TrafficDataset

        dataset = TrafficDataset(
            graph_path=setup["graph_path"],
            static_data_path=setup["static_path"],
            dynamic_data_paths=setup["dynamic_paths"],
            metadata_paths=setup["metadata_paths"],
            target_data_path=setup["target_path"],
            in_timesteps=10,
            out_timesteps=5,
            future_features=["coi_puntos"],
            node_batch_size=5,
            node_seed="node_0",
        )

        # Split 80-20
        train_ds, test_ds = temporal_train_test_split(dataset, train_ratio=0.8)

        # Check lengths
        total_len = len(dataset)
        expected_train_len = int(total_len * 0.8)
        expected_test_len = total_len - expected_train_len

        assert len(train_ds) == expected_train_len
        assert len(test_ds) == expected_test_len
        assert len(train_ds) + len(test_ds) == total_len

    def test_temporal_subset_indices(self, test_data_setup):
        """Test that temporal subsets access correct time indices."""
        setup = test_data_setup

        from sparse_dataloader import TrafficDataset

        dataset = TrafficDataset(
            graph_path=setup["graph_path"],
            static_data_path=setup["static_path"],
            dynamic_data_paths=setup["dynamic_paths"],
            metadata_paths=setup["metadata_paths"],
            target_data_path=setup["target_path"],
            in_timesteps=10,
            out_timesteps=5,
            future_features=["coi_puntos"],
            node_batch_size=5,
            node_seed="node_0",
        )

        # Create subset for time windows [10, 20)
        subset = TemporalSubset(dataset, start_time_idx=10, end_time_idx=20)

        assert len(subset) == 10

        # When we access subset[0], it should fetch the same data as dataset[10]
        subset_sample_0 = subset[0]
        dataset_sample_10 = dataset[10]

        # Compare key tensor shapes and values
        assert subset_sample_0.x_static.shape == dataset_sample_10.x_static.shape
        assert isinstance(subset_sample_0.y, Tensor) and isinstance(
            dataset_sample_10.y, Tensor
        )
        assert subset_sample_0.y.shape == dataset_sample_10.y.shape
        assert (subset_sample_0.x_static == dataset_sample_10.x_static).all()

    def test_temporal_split_no_overlap_in_indices(self, test_data_setup):
        """Test that train and test sets have non-overlapping time indices."""
        setup = test_data_setup

        from sparse_dataloader import TrafficDataset

        dataset = TrafficDataset(
            graph_path=setup["graph_path"],
            static_data_path=setup["static_path"],
            dynamic_data_paths=setup["dynamic_paths"],
            metadata_paths=setup["metadata_paths"],
            target_data_path=setup["target_path"],
            in_timesteps=10,
            out_timesteps=5,
            future_features=["coi_puntos"],
            node_batch_size=5,
            node_seed="node_0",
        )

        train_ds, test_ds = temporal_train_test_split(dataset, train_ratio=0.8)

        # The time indices should not overlap
        assert train_ds.end_time_idx == test_ds.start_time_idx
        assert train_ds.start_time_idx == 0
        assert test_ds.end_time_idx == len(dataset)

    def test_seed_reassignment_with_subset(self, test_data_setup):
        """Test that seed reassignment works through temporal subsets."""
        setup = test_data_setup

        from sparse_dataloader import TrafficDataset

        dataset = TrafficDataset(
            graph_path=setup["graph_path"],
            static_data_path=setup["static_path"],
            dynamic_data_paths=setup["dynamic_paths"],
            metadata_paths=setup["metadata_paths"],
            target_data_path=setup["target_path"],
            in_timesteps=10,
            out_timesteps=5,
            future_features=["coi_puntos"],
            node_batch_size=5,
            node_seed="node_0",
        )

        train_ds, test_ds = temporal_train_test_split(dataset, train_ratio=0.8)

        # Initial seed
        assert train_ds.node_seed == "node_0"
        assert test_ds.node_seed == "node_0"

        # Reassign through train subset
        train_ds.reasign_seed("node_3")

        # Should affect both subsets and underlying dataset
        assert train_ds.node_seed == "node_3"
        assert test_ds.node_seed == "node_3"
        assert dataset.node_seed == "node_3"

        # Reassign through test subset
        test_ds.reasign_seed("node_5")

        # Should affect everything
        assert train_ds.node_seed == "node_5"
        assert test_ds.node_seed == "node_5"
        assert dataset.node_seed == "node_5"

    def test_train_val_test_split(self, test_data_setup):
        """Test three-way split into train, val, and test."""
        setup = test_data_setup

        from sparse_dataloader import TrafficDataset

        dataset = TrafficDataset(
            graph_path=setup["graph_path"],
            static_data_path=setup["static_path"],
            dynamic_data_paths=setup["dynamic_paths"],
            metadata_paths=setup["metadata_paths"],
            target_data_path=setup["target_path"],
            in_timesteps=10,
            out_timesteps=5,
            future_features=["coi_puntos"],
            node_batch_size=5,
            node_seed="node_0",
        )

        # Split 70-15-15
        train_ds, val_ds, test_ds = temporal_train_val_test_split(
            dataset, train_ratio=0.7, val_ratio=0.15
        )

        # Check lengths
        total_len = len(dataset)
        expected_train_len = int(total_len * 0.7)
        expected_val_len = int(total_len * 0.15)
        expected_test_len = total_len - expected_train_len - expected_val_len

        assert len(train_ds) == expected_train_len
        assert len(val_ds) == expected_val_len
        assert len(test_ds) == expected_test_len
        assert len(train_ds) + len(val_ds) + len(test_ds) == total_len

        # Check that splits are contiguous and non-overlapping
        assert train_ds.start_time_idx == 0
        assert train_ds.end_time_idx == val_ds.start_time_idx
        assert val_ds.end_time_idx == test_ds.start_time_idx
        assert test_ds.end_time_idx == len(dataset)

    def test_temporal_subset_properties(self, test_data_setup):
        """Test that temporal subset exposes necessary properties."""
        setup = test_data_setup

        from sparse_dataloader import TrafficDataset

        dataset = TrafficDataset(
            graph_path=setup["graph_path"],
            static_data_path=setup["static_path"],
            dynamic_data_paths=setup["dynamic_paths"],
            metadata_paths=setup["metadata_paths"],
            target_data_path=setup["target_path"],
            in_timesteps=10,
            out_timesteps=5,
            future_features=["coi_puntos"],
            node_batch_size=5,
            node_seed="node_0",
        )

        train_ds, test_ds = temporal_train_test_split(dataset, train_ratio=0.8)

        # Check that subsets expose important properties
        assert train_ds.num_nodes == dataset.num_nodes
        assert train_ds.node_batch_size == dataset.node_batch_size
        assert (
            train_ds.total_possible_node_batches == dataset.total_possible_node_batches
        )

        # Check total_training_samples
        expected_train_samples = len(train_ds) * dataset.total_possible_node_batches
        assert train_ds.total_training_samples == expected_train_samples

        expected_test_samples = len(test_ds) * dataset.total_possible_node_batches
        assert test_ds.total_training_samples == expected_test_samples

    def test_temporal_subset_index_bounds(self, test_data_setup):
        """Test that temporal subset raises errors for out-of-bounds indices."""
        setup = test_data_setup

        from sparse_dataloader import TrafficDataset

        dataset = TrafficDataset(
            graph_path=setup["graph_path"],
            static_data_path=setup["static_path"],
            dynamic_data_paths=setup["dynamic_paths"],
            metadata_paths=setup["metadata_paths"],
            target_data_path=setup["target_path"],
            in_timesteps=10,
            out_timesteps=5,
            future_features=["coi_puntos"],
            node_batch_size=5,
            node_seed="node_0",
        )

        subset = TemporalSubset(dataset, start_time_idx=0, end_time_idx=10)

        # Valid indices
        _ = subset[0]
        _ = subset[9]

        # Out of bounds indices
        with pytest.raises(IndexError):
            _ = subset[10]

        with pytest.raises(IndexError):
            _ = subset[-1]

    def test_invalid_split_ratios(self, test_data_setup):
        """Test that invalid split ratios raise errors."""
        setup = test_data_setup

        from sparse_dataloader import TrafficDataset

        dataset = TrafficDataset(
            graph_path=setup["graph_path"],
            static_data_path=setup["static_path"],
            dynamic_data_paths=setup["dynamic_paths"],
            metadata_paths=setup["metadata_paths"],
            target_data_path=setup["target_path"],
            in_timesteps=10,
            out_timesteps=5,
            future_features=["coi_puntos"],
            node_batch_size=5,
            node_seed="node_0",
        )

        # Invalid train_ratio values
        with pytest.raises(ValueError):
            temporal_train_test_split(dataset, train_ratio=0.0)

        with pytest.raises(ValueError):
            temporal_train_test_split(dataset, train_ratio=1.0)

        with pytest.raises(ValueError):
            temporal_train_test_split(dataset, train_ratio=1.5)

        # Invalid train+val ratio for three-way split
        with pytest.raises(ValueError):
            temporal_train_val_test_split(dataset, train_ratio=0.9, val_ratio=0.2)

    def test_data_consistency_across_split(self, test_data_setup):
        """Test that the same absolute time index produces the same data."""
        setup = test_data_setup

        from sparse_dataloader import TrafficDataset

        dataset = TrafficDataset(
            graph_path=setup["graph_path"],
            static_data_path=setup["static_path"],
            dynamic_data_paths=setup["dynamic_paths"],
            metadata_paths=setup["metadata_paths"],
            target_data_path=setup["target_path"],
            in_timesteps=10,
            out_timesteps=5,
            future_features=["coi_puntos"],
            node_batch_size=5,
            node_seed="node_0",
        )

        train_ds, test_ds = temporal_train_test_split(dataset, train_ratio=0.8)

        # Access the same absolute time index through different paths
        # test_ds[0] should correspond to dataset[train_ds.end_time_idx]
        test_sample = test_ds[0]
        dataset_sample = dataset[train_ds.end_time_idx]

        # They should produce identical data
        assert (test_sample.x_static == dataset_sample.x_static).all()
        assert Tensor(test_sample.y == dataset_sample.y).all()
        assert (test_sample.x_dynamic_past == dataset_sample.x_dynamic_past).all()
        assert (test_sample.x_dynamic_future == dataset_sample.x_dynamic_future).all()
