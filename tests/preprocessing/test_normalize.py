import pickle as pkl

import h5py
import numpy as np

from src.preprocessing.normalize import (
    compute_all_normalization_constants,
    compute_normalization_constants,
    create_flat_normalization_dict,
)


class TestComputeNormalizationConstants:
    """Tests for compute_normalization_constants function."""

    def test_temporal_dataset_basic(self, tmp_path):
        """Test normalization constants for a temporal dataset."""
        # Create a temporal dataset with known values
        # Format: [node_idx, time_idx, feature_idx, value]
        data = np.array(
            [
                [0, 0, 0, 10.0],
                [0, 1, 0, 20.0],
                [0, 2, 0, 30.0],
                [1, 0, 0, 40.0],
                [1, 1, 0, 50.0],
                [0, 0, 1, 100.0],
                [0, 1, 1, 200.0],
            ]
        )

        # Save to HDF5
        h5_path = tmp_path / "temporal_dataset.h5"
        with h5py.File(h5_path, "w") as f:
            f.create_dataset("data", data=data)

        # Save metadata
        metadata = {"feature_idx_map": {"feature_a": 0, "feature_b": 1}}
        metadata_path = tmp_path / "temporal_dataset_metadata.pkl"
        with open(metadata_path, "wb") as f:
            pkl.dump(metadata, f)

        # Compute normalization constants
        norm_constants = compute_normalization_constants(
            str(h5_path), str(metadata_path), percentile=100.0
        )

        # With 100th percentile, should get max absolute values
        assert "feature_a" in norm_constants
        assert "feature_b" in norm_constants
        assert norm_constants["feature_a"] == 50.0
        assert norm_constants["feature_b"] == 200.0

    def test_static_dataset_basic(self, tmp_path):
        """Test normalization constants for a static dataset."""
        # Create a static dataset with known values
        # Format: [node_idx, feature_idx, value]
        data = np.array(
            [
                [0, 0, 5.0],
                [1, 0, 10.0],
                [2, 0, 15.0],
                [0, 1, 100.0],
                [1, 1, 50.0],
            ]
        )

        # Save to HDF5
        h5_path = tmp_path / "static_dataset.h5"
        with h5py.File(h5_path, "w") as f:
            f.create_dataset("data", data=data)

        # Save metadata
        metadata = {"feature_idx_map": {"static_a": 0, "static_b": 1}}
        metadata_path = tmp_path / "static_dataset_metadata.pkl"
        with open(metadata_path, "wb") as f:
            pkl.dump(metadata, f)

        # Compute normalization constants
        norm_constants = compute_normalization_constants(
            str(h5_path), str(metadata_path), percentile=100.0
        )

        assert norm_constants["static_a"] == 15.0
        assert norm_constants["static_b"] == 100.0

    def test_percentile_handling(self, tmp_path):
        """Test that percentile-based normalization handles outliers."""
        # Create data with an outlier
        # 100 values: 99 values of 10.0 and 1 outlier of 1000.0
        values = [10.0] * 99 + [1000.0]
        data = np.array([[0, i, 0, v] for i, v in enumerate(values)])

        h5_path = tmp_path / "outlier_dataset.h5"
        with h5py.File(h5_path, "w") as f:
            f.create_dataset("data", data=data)

        metadata = {"feature_idx_map": {"feature": 0}}
        metadata_path = tmp_path / "outlier_dataset_metadata.pkl"
        with open(metadata_path, "wb") as f:
            pkl.dump(metadata, f)

        # With 99th percentile, the outlier should have minimal impact
        norm_constants = compute_normalization_constants(
            str(h5_path), str(metadata_path), percentile=99.0
        )

        # The 99th percentile should be much closer to 10 than to 1000
        # (due to interpolation, it won't be exactly 10, but should be << 1000)
        assert norm_constants["feature"] < 100.0  # Much less than outlier
        assert norm_constants["feature"] >= 10.0  # At least the normal value

    def test_negative_values(self, tmp_path):
        """Test that negative values are handled correctly via absolute value."""
        data = np.array(
            [
                [0, 0, 0, -50.0],
                [0, 1, 0, 30.0],
                [0, 2, 0, -20.0],
            ]
        )

        h5_path = tmp_path / "negative_dataset.h5"
        with h5py.File(h5_path, "w") as f:
            f.create_dataset("data", data=data)

        metadata = {"feature_idx_map": {"feature": 0}}
        metadata_path = tmp_path / "negative_dataset_metadata.pkl"
        with open(metadata_path, "wb") as f:
            pkl.dump(metadata, f)

        norm_constants = compute_normalization_constants(
            str(h5_path), str(metadata_path), percentile=100.0
        )

        # Should use absolute values, so max is 50.0
        assert norm_constants["feature"] == 50.0

    def test_min_scale_enforcement(self, tmp_path):
        """Test that min_scale prevents division by very small numbers."""
        # All zeros or very small values
        data = np.array(
            [
                [0, 0, 0, 1e-10],
                [0, 1, 0, 1e-11],
            ]
        )

        h5_path = tmp_path / "small_dataset.h5"
        with h5py.File(h5_path, "w") as f:
            f.create_dataset("data", data=data)

        metadata = {"feature_idx_map": {"feature": 0}}
        metadata_path = tmp_path / "small_dataset_metadata.pkl"
        with open(metadata_path, "wb") as f:
            pkl.dump(metadata, f)

        min_scale = 1e-6
        norm_constants = compute_normalization_constants(
            str(h5_path), str(metadata_path), percentile=100.0, min_scale=min_scale
        )

        # Should use min_scale since actual values are smaller
        assert norm_constants["feature"] == min_scale

    def test_empty_feature(self, tmp_path):
        """Test handling of features with no data."""
        # Only feature 0 has data, feature 1 has none
        data = np.array(
            [
                [0, 0, 0, 10.0],
                [0, 1, 0, 20.0],
            ]
        )

        h5_path = tmp_path / "partial_dataset.h5"
        with h5py.File(h5_path, "w") as f:
            f.create_dataset("data", data=data)

        # Metadata includes a feature that has no data
        metadata = {"feature_idx_map": {"feature_a": 0, "feature_b": 1}}
        metadata_path = tmp_path / "partial_dataset_metadata.pkl"
        with open(metadata_path, "wb") as f:
            pkl.dump(metadata, f)

        norm_constants = compute_normalization_constants(
            str(h5_path), str(metadata_path), percentile=100.0
        )

        # feature_a should be computed, feature_b should not exist (no data)
        assert "feature_a" in norm_constants
        assert "feature_b" not in norm_constants


class TestComputeAllNormalizationConstants:
    """Tests for compute_all_normalization_constants function."""

    def test_processes_multiple_datasets(self, tmp_path):
        """Test that function processes static and dynamic datasets."""
        # Create static dataset
        static_data = np.array([[0, 0, 100.0], [1, 0, 200.0]])
        static_path = tmp_path / "static_dataset.h5"
        with h5py.File(static_path, "w") as f:
            f.create_dataset("data", data=static_data)

        static_metadata = {"feature_idx_map": {"static_feat": 0}}
        with open(tmp_path / "static_dataset_metadata.pkl", "wb") as f:
            pkl.dump(static_metadata, f)

        # Create a dynamic dataset
        dynamic_data = np.array([[0, 0, 0, 50.0], [0, 1, 0, 75.0]])
        dynamic_path = tmp_path / "events_dataset.h5"
        with h5py.File(dynamic_path, "w") as f:
            f.create_dataset("data", data=dynamic_data)

        dynamic_metadata = {"feature_idx_map": {"event_feat": 0}}
        with open(tmp_path / "events_dataset_metadata.pkl", "wb") as f:
            pkl.dump(dynamic_metadata, f)

        # Create target dataset
        target_data = np.array([[0, 0, 0, 5.0], [0, 1, 0, 4.0]])
        target_path = tmp_path / "jams_dataset.h5"
        with h5py.File(target_path, "w") as f:
            f.create_dataset("data", data=target_data)

        target_metadata = {"feature_idx_map": {"mean_jam_level": 0}}
        with open(tmp_path / "jams_dataset_metadata.pkl", "wb") as f:
            pkl.dump(target_metadata, f)

        # Run the function
        output_path = tmp_path / "norm_constants.pkl"
        result = compute_all_normalization_constants(
            sparse_data_dir=str(tmp_path),
            output_path=str(output_path),
            dynamic_filenames=["events_dataset.h5"],
            weather_filenames=[],
            percentile=100.0,
        )

        # Verify results
        assert "static" in result
        assert "events_dataset" in result
        assert "target" in result

        assert result["static"]["static_feat"] == 200.0
        assert result["events_dataset"]["event_feat"] == 75.0
        assert result["target"]["mean_jam_level"] == 5.0

    def test_saves_output_files(self, tmp_path):
        """Test that output pickle and summary files are created."""
        # Create minimal static dataset
        static_data = np.array([[0, 0, 10.0]])
        with h5py.File(tmp_path / "static_dataset.h5", "w") as f:
            f.create_dataset("data", data=static_data)

        with open(tmp_path / "static_dataset_metadata.pkl", "wb") as f:
            pkl.dump({"feature_idx_map": {"feat": 0}}, f)

        output_path = tmp_path / "norm_constants.pkl"
        compute_all_normalization_constants(
            sparse_data_dir=str(tmp_path),
            output_path=str(output_path),
            dynamic_filenames=[],
            weather_filenames=[],
        )

        # Check files exist
        assert output_path.exists()
        summary_path = tmp_path / "norm_constants_summary.txt"
        assert summary_path.exists()

        # Verify pickle can be loaded
        with open(output_path, "rb") as f:
            loaded = pkl.load(f)
        assert "static" in loaded

    def test_skips_missing_datasets(self, tmp_path):
        """Test that missing datasets are gracefully skipped."""
        # Create only static dataset, skip dynamic
        static_data = np.array([[0, 0, 10.0]])
        with h5py.File(tmp_path / "static_dataset.h5", "w") as f:
            f.create_dataset("data", data=static_data)

        with open(tmp_path / "static_dataset_metadata.pkl", "wb") as f:
            pkl.dump({"feature_idx_map": {"feat": 0}}, f)

        output_path = tmp_path / "norm_constants.pkl"
        result = compute_all_normalization_constants(
            sparse_data_dir=str(tmp_path),
            output_path=str(output_path),
            dynamic_filenames=["nonexistent.h5"],
            weather_filenames=[],
        )

        # Should have static but not the missing dynamic
        assert "static" in result
        assert "nonexistent" not in result


class TestCreateFlatNormalizationDict:
    """Tests for create_flat_normalization_dict function."""

    def test_flattens_nested_dict(self):
        """Test that nested dictionary is properly flattened."""
        nested = {
            "static": {"feat_a": 10.0, "feat_b": 20.0},
            "dynamic": {"feat_c": 30.0},
            "target": {"mean_jam_level": 5.0},
        }

        flat = create_flat_normalization_dict(nested)

        assert flat["feat_a"] == 10.0
        assert flat["feat_b"] == 20.0
        assert flat["feat_c"] == 30.0
        assert flat["mean_jam_level"] == 5.0

    def test_handles_duplicate_feature_names(self):
        """Test that later values override earlier ones for duplicates."""
        nested = {
            "dataset_1": {"common_feat": 10.0},
            "dataset_2": {"common_feat": 20.0},
        }

        flat = create_flat_normalization_dict(nested)

        # Later value should win (dataset_2 processed after dataset_1)
        assert flat["common_feat"] == 20.0

    def test_empty_input(self):
        """Test handling of empty input."""
        flat = create_flat_normalization_dict({})
        assert flat == {}


class TestNormalizationIntegration:
    """Integration tests for the normalization workflow."""

    def test_normalized_values_in_range(self, tmp_path):
        """Test that applying normalization constants keeps values in expected range."""
        # Create dataset with values that span a wide range
        np.random.seed(42)
        values = np.concatenate(
            [
                np.random.uniform(0, 100, 95),  # Normal values
                np.array([500.0, 600.0, 700.0, 800.0, 1000.0]),  # Outliers
            ]
        )
        data = np.array([[0, i, 0, v] for i, v in enumerate(values)])

        h5_path = tmp_path / "test_dataset.h5"
        with h5py.File(h5_path, "w") as f:
            f.create_dataset("data", data=data)

        metadata = {"feature_idx_map": {"feature": 0}}
        metadata_path = tmp_path / "test_dataset_metadata.pkl"
        with open(metadata_path, "wb") as f:
            pkl.dump(metadata, f)

        # Compute normalization constant at 99th percentile
        norm_constants = compute_normalization_constants(
            str(h5_path), str(metadata_path), percentile=99.0
        )

        norm_constant = norm_constants["feature"]

        # Normalize the values
        normalized = values / norm_constant

        # 99% of values should be <= 1.0 (before clipping)
        within_range = np.sum(np.abs(normalized) <= 1.0)
        assert within_range >= 95  # At least 95 out of 100 should be in range

        # After clipping, all should be in [-1, 1]
        clipped = np.clip(normalized, -1.0, 1.0)
        assert np.all(np.abs(clipped) <= 1.0)
