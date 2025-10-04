import os
import pickle
import shutil
import sys
import tempfile
from pathlib import Path

import h5py
import networkx as nx
import numpy as np
import pandas as pd
import pytest
import torch
from torch_geometric.data import Data

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))
from sparse_dataloader import TrafficDataset


class TestDataGenerator:
    """Helper class to generate dummy test data in the correct format."""

    def __init__(self, num_nodes=10, num_timesteps=50, num_static_features=5):
        self.num_nodes = num_nodes
        self.num_timesteps = num_timesteps
        self.num_static_features = num_static_features

        # Create node IDs as strings (similar to real data)
        self.node_ids = [f"node_{i}" for i in range(num_nodes)]

        # Create time mappings
        self.timeframes = pd.date_range(
            start="2024-01-01 00:00:00", periods=num_timesteps, freq="15min"
        ).to_list()

        self.node_idx_map = {node: i for i, node in enumerate(self.node_ids)}
        self.time_idx_map = {ts: i for i, ts in enumerate(self.timeframes)}

    def create_graph(self, temp_dir: str) -> str:
        """Create a simple test graph."""
        # Create a connected graph (path graph for simplicity)
        G = nx.path_graph(self.num_nodes)
        # Convert to string node IDs to match real data
        mapping = {i: self.node_ids[i] for i in range(self.num_nodes)}
        G = nx.relabel_nodes(G, mapping)

        # Add some basic node attributes
        for node in G.nodes():
            G.nodes[node]["test_attr"] = np.random.rand()

        graph_path = os.path.join(temp_dir, "test_graph.pkl")
        with open(graph_path, "wb") as f:
            pickle.dump(G, f)

        return graph_path

    def create_static_features(self, temp_dir: str) -> str:
        """Create static features HDF5 file."""
        # Create sparse static data
        data_list = []
        for node_idx in range(self.num_nodes):
            for feature_idx in range(self.num_static_features):
                if np.random.rand() > 0.3:  # 70% sparsity
                    value = np.random.rand() * 10
                    data_list.append([node_idx, feature_idx, value])

        data_array = np.array(data_list)

        static_path = os.path.join(temp_dir, "static_features.h5")
        with h5py.File(static_path, "w") as f:
            f.create_dataset("data", data=data_array)

        return static_path

    def create_dynamic_features(
        self, temp_dir: str, filename: str, feature_names: list
    ) -> tuple[str, str]:
        """Create dynamic features HDF5 file and corresponding metadata."""
        data_list = []

        for time_idx in range(self.num_timesteps):
            for node_idx in range(self.num_nodes):
                for feature_idx, feature_name in enumerate(feature_names):
                    if np.random.rand() > 0.8:  # 80% sparsity (more sparse than static)
                        value = np.random.rand() * 5
                        data_list.append([node_idx, time_idx, feature_idx, value])

        # Sort by time_idx as expected by the dataloader
        data_array = np.array(data_list)
        if len(data_array) > 0:
            data_array = data_array[data_array[:, 1].argsort()]  # Sort by time_idx

        # Create HDF5 file
        h5_path = os.path.join(temp_dir, f"{filename}.h5")
        with h5py.File(h5_path, "w") as f:
            f.create_dataset("data", data=data_array)

        # Create metadata file
        feature_idx_map = {name: i for i, name in enumerate(feature_names)}
        metadata = {
            "node_idx_map": self.node_idx_map,
            "time_idx_map": self.time_idx_map,
            "feature_idx_map": feature_idx_map,
        }

        metadata_path = os.path.join(temp_dir, f"{filename}_metadata.pkl")
        with open(metadata_path, "wb") as f:
            pickle.dump(metadata, f)

        return h5_path, metadata_path

    def create_target_data(self, temp_dir: str) -> str:
        """Create target data (jams) HDF5 file with single feature."""
        data_list = []

        for time_idx in range(self.num_timesteps):
            for node_idx in range(self.num_nodes):
                if np.random.rand() > 0.5:  # 50% sparsity for target
                    value = np.random.rand() * 100  # Jam level 0-100
                    data_list.append(
                        [node_idx, time_idx, 0, value]
                    )  # feature_idx always 0 for target

        # Sort by time_idx
        data_array = np.array(data_list)
        if len(data_array) > 0:
            data_array = data_array[data_array[:, 1].argsort()]

        target_path = os.path.join(temp_dir, "target_data.h5")
        with h5py.File(target_path, "w") as f:
            f.create_dataset("data", data=data_array)

        return target_path

    def create_weather_stations(self, temp_dir: str) -> str:
        """Create weather stations mapping for testing."""
        # Create a simple mapping where each node uses a predictable weather station
        weather_stations = {}

        for i, node_id in enumerate(self.node_ids):
            # Assign stations in a round-robin fashion
            wind_station = f"WindStation_{i % 3}"  # 3 wind stations
            humidity_station = f"HumidityStation_{i % 2}"  # 2 humidity stations
            speed_station = f"SpeedStation_{i % 3}"
            precip_station = f"PrecipStation_{i % 2}"

            weather_stations[node_id] = {
                "estacion_direccion_viento": wind_station,
                "estacion_humedad": humidity_station,
                "estacion_velocidad_viento": speed_station,
                "estacion_precipitacion": precip_station,
            }

        stations_path = os.path.join(temp_dir, "weather_stations.pkl")
        with open(stations_path, "wb") as f:
            pickle.dump(weather_stations, f)

        return stations_path

    def create_weather_data(self, temp_dir: str, feature_name: str) -> str:
        """Create deterministic weather data for testing."""
        # Map feature names to column names
        station_col_map = {
            "wind_direction": "estacion_direccion_viento",
            "humidity": "estacion_humedad",
            "wind_speed": "estacion_velocidad_viento",
            "precipitation": "estacion_precipitacion",
        }
        value_col_map = {
            "wind_direction": "direccion_viento",
            "humidity": "humedad",
            "wind_speed": "velocidad_viento",
            "precipitation": "precipitacion",
        }

        station_col = station_col_map[feature_name]
        value_col = value_col_map[feature_name]

        # Create deterministic weather data
        data_rows = []

        # For each timestep, create weather data for all relevant stations
        for time_idx in range(self.num_timesteps):
            # Get unique stations for this feature
            if feature_name == "wind_direction":
                stations = [f"WindStation_{i}" for i in range(3)]
                base_value = 100
            elif feature_name == "humidity":
                stations = [f"HumidityStation_{i}" for i in range(2)]
                base_value = 50
            elif feature_name == "wind_speed":
                stations = [f"SpeedStation_{i}" for i in range(3)]
                base_value = 10
            else:  # precipitation
                stations = [f"PrecipStation_{i}" for i in range(2)]
                base_value = 0

            for station_idx, station_name in enumerate(stations):
                # Create predictable values: base_value + time_idx + station_idx
                value = base_value + time_idx + station_idx

                row = {
                    station_col: station_name,
                    value_col: float(value),
                    "time_idx": time_idx,
                    "timestamp": self.timeframes[time_idx],
                }
                data_rows.append(row)

        # Create DataFrame
        df = pd.DataFrame(data_rows)

        # Save as parquet
        weather_path = os.path.join(temp_dir, f"weather_{feature_name}.parquet")
        df.to_parquet(weather_path, index=False)

        return weather_path


class TestTrafficDataset:
    """Test suite for TrafficDataset class."""

    @pytest.fixture
    def test_data_setup(self):
        """Create test data files in a temporary directory."""
        temp_dir = tempfile.mkdtemp()
        generator = TestDataGenerator(num_nodes=10, num_timesteps=50)

        # Create all necessary files
        graph_path = generator.create_graph(temp_dir)
        static_path = generator.create_static_features(temp_dir)

        # Create dynamic feature files
        dynamic_paths = []
        metadata_paths = []

        # File 1: coi_points with future features
        h5_path, meta_path = generator.create_dynamic_features(
            temp_dir, "coi_points", ["coi_puntos", "other_feature"]
        )
        dynamic_paths.append(h5_path)
        metadata_paths.append(meta_path)

        # File 2: events with future features
        h5_path, meta_path = generator.create_dynamic_features(
            temp_dir, "events", ["EMERGENCIA", "MANIFESTACION"]
        )
        dynamic_paths.append(h5_path)
        metadata_paths.append(meta_path)

        # File 3: incidents (no future features)
        h5_path, meta_path = generator.create_dynamic_features(
            temp_dir, "incidents", ["incident_type1", "incident_type2"]
        )
        dynamic_paths.append(h5_path)
        metadata_paths.append(meta_path)

        target_path = generator.create_target_data(temp_dir)

        # Create weather data files
        weather_stations_path = generator.create_weather_stations(temp_dir)
        weather_data_paths = {}
        for feature in ["wind_direction", "humidity", "wind_speed", "precipitation"]:
            weather_data_paths[feature] = generator.create_weather_data(
                temp_dir, feature
            )

        setup_data = {
            "temp_dir": temp_dir,
            "graph_path": graph_path,
            "static_path": static_path,
            "dynamic_paths": dynamic_paths,
            "metadata_paths": metadata_paths,
            "target_path": target_path,
            "generator": generator,
            "weather_stations_path": weather_stations_path,
            "weather_data_paths": weather_data_paths,
        }

        yield setup_data

        # Cleanup
        shutil.rmtree(temp_dir)

    def test_initialization_basic(self, test_data_setup):
        """Test basic dataset initialization."""
        setup = test_data_setup

        dataset = TrafficDataset(
            graph_path=setup["graph_path"],
            static_data_path=setup["static_path"],
            dynamic_data_paths=setup["dynamic_paths"],
            metadata_paths=setup["metadata_paths"],
            target_data_path=setup["target_path"],
            in_timesteps=10,
            out_timesteps=5,
            future_features=["coi_puntos", "EMERGENCIA"],
            node_batch_size=5,
            node_seed="node_0",
        )

        # Test basic properties
        assert dataset.in_timesteps == 10
        assert dataset.out_timesteps == 5
        assert dataset.total_timesteps_window == 15
        assert dataset.node_batch_size == 5
        assert dataset.num_nodes == 10
        assert len(dataset.future_features) == 2

        # Test that data structures are properly initialized
        assert dataset.graph is not None
        assert dataset.edge_index is not None
        assert dataset.static_features is not None
        assert len(dataset.node_to_id) == 10
        assert len(dataset.id_to_node) == 10
        assert len(dataset.batch_mapping) == 10

    def test_initialization_invalid_future_features(self, test_data_setup):
        """Test that an exception is raised with invalid future features."""
        setup = test_data_setup

        # Should raise a ValueError due to non-existent future feature
        with pytest.raises(
            ValueError,
            match="not found in any metadata file",
        ):
            TrafficDataset(
                graph_path=setup["graph_path"],
                static_data_path=setup["static_path"],
                dynamic_data_paths=setup["dynamic_paths"],
                metadata_paths=setup["metadata_paths"],
                target_data_path=setup["target_path"],
                in_timesteps=5,
                out_timesteps=3,
                future_features=["non_existent_feature"],  # Invalid feature
                node_batch_size=5,
                node_seed="node_0",
            )

    def test_invalid_node_seed(self, test_data_setup):
        """Test that initialization raises error with invalid default node_seed."""
        setup = test_data_setup

        # Should raise ValueError due to invalid node_seed
        with pytest.raises(
            ValueError, match="Node seed InvalidNodeID not found in graph nodes"
        ):
            TrafficDataset(
                graph_path=setup["graph_path"],
                static_data_path=setup["static_path"],
                dynamic_data_paths=setup["dynamic_paths"],
                metadata_paths=setup["metadata_paths"],
                target_data_path=setup["target_path"],
                in_timesteps=5,
                out_timesteps=3,
                future_features=["coi_puntos"],
                node_batch_size=5,
                node_seed="InvalidNodeID",  # Invalid seed
            )

    def test_dataset_length(self, test_data_setup):
        """Test dataset length calculation."""
        setup = test_data_setup

        dataset = TrafficDataset(
            graph_path=setup["graph_path"],
            static_data_path=setup["static_path"],
            dynamic_data_paths=setup["dynamic_paths"],
            metadata_paths=setup["metadata_paths"],
            target_data_path=setup["target_path"],
            in_timesteps=4,
            out_timesteps=1,
            future_features=["coi_puntos"],
            node_batch_size=5,
            node_seed="node_0",
        )

        # With 50 timesteps, window of 5, we should have 46 possible windows
        # For a fixed seed, the dataset length is just the number of time windows
        expected_windows = 50 - 5 + 1  # 46 windows
        assert len(dataset) == expected_windows

    def test_dataset_length_insufficient_data(self, test_data_setup):
        """Test dataset length when there's insufficient data."""
        setup = test_data_setup

        dataset = TrafficDataset(
            graph_path=setup["graph_path"],
            static_data_path=setup["static_path"],
            dynamic_data_paths=setup["dynamic_paths"],
            metadata_paths=setup["metadata_paths"],
            target_data_path=setup["target_path"],
            in_timesteps=60,  # More than available timesteps (50)
            out_timesteps=5,
            future_features=["coi_puntos"],
            node_batch_size=5,
            node_seed="node_0",
        )

        assert len(dataset) == 0

    def test_get_item_structure(self, test_data_setup):
        """Test that __getitem__ returns correctly structured Data objects."""
        setup = test_data_setup

        in_timesteps = 6
        out_timesteps = 3
        node_batch_size = 5
        future_features = ["coi_puntos", "EMERGENCIA"]
        dataset = TrafficDataset(
            graph_path=setup["graph_path"],
            static_data_path=setup["static_path"],
            dynamic_data_paths=setup["dynamic_paths"],
            metadata_paths=setup["metadata_paths"],
            target_data_path=setup["target_path"],
            in_timesteps=in_timesteps,
            out_timesteps=out_timesteps,
            future_features=future_features,
            node_batch_size=node_batch_size,
            node_seed="node_0",
        )

        assert len(dataset) > 0
        sample = dataset[0]
        assert isinstance(sample, Data)

        # Check tensor shapes and types
        assert isinstance(sample.x_static, torch.Tensor)
        assert sample.x_static.dtype == torch.float32
        assert isinstance(sample.x_dynamic_past, torch.Tensor)
        assert sample.x_dynamic_past.dtype == torch.float32
        assert isinstance(sample.x_dynamic_future, torch.Tensor)
        assert sample.x_dynamic_future.dtype == torch.float32
        assert isinstance(sample.y, torch.Tensor)
        assert sample.y.dtype == torch.float32

        # Check dimensions
        batch_size = sample.x_static.shape[0]
        assert (
            batch_size == node_batch_size
        )  # Should match node_batch_size since ceil(10/5)=2 batches

        # Could be less than batch size if last batch and num_nodes not divisible
        # assert batch_size <= 5  # Should not exceed node_batch_size

        assert sample.x_static.shape[1] == 5  # num_static_features
        assert sample.x_dynamic_past.shape == (
            batch_size,
            in_timesteps,
            dataset.num_dynamic_features,
        )
        assert sample.x_dynamic_future.shape == (
            batch_size,
            out_timesteps,
            len(future_features),
        )
        assert sample.y.shape == (batch_size, out_timesteps)

        # Check edge_index
        assert isinstance(sample.edge_index, torch.Tensor)
        assert sample.edge_index.shape[0] == 2  # Source and target nodes
        assert (
            sample.edge_index.shape[1] == (node_batch_size - 1) * 2
        )  # Number of bi-directional edges in a path subgraph with 5 nodes
        assert sample.edge_index.dtype == torch.long

    def test_node_batch_mapping(self, test_data_setup):
        """Test that node batch mapping works correctly."""
        setup = test_data_setup

        node_batch_size = 2
        dataset = TrafficDataset(
            graph_path=setup["graph_path"],
            static_data_path=setup["static_path"],
            dynamic_data_paths=setup["dynamic_paths"],
            metadata_paths=setup["metadata_paths"],
            target_data_path=setup["target_path"],
            in_timesteps=5,
            out_timesteps=3,
            future_features=["coi_puntos"],
            node_batch_size=node_batch_size,
            node_seed="node_0",
        )

        # Test that batch_mapping contains all nodes
        assert len(dataset.batch_mapping) == 10

        # Test that each node maps to a list of neighbors
        for node_id, neighbors in dataset.batch_mapping.items():
            assert isinstance(neighbors, list)
            assert (
                len(neighbors) <= node_batch_size
            )  # Should not exceed node_batch_size
            # Neighbors should be valid node IDs
            for neighbor in neighbors:
                assert neighbor in dataset.node_to_id

    def test_seed_node_neighborhood(self, test_data_setup):
        """Tests that the neighborhood of the seed node is computed correctly."""
        setup = test_data_setup

        dataset = TrafficDataset(
            graph_path=setup["graph_path"],
            static_data_path=setup["static_path"],
            dynamic_data_paths=setup["dynamic_paths"],
            metadata_paths=setup["metadata_paths"],
            target_data_path=setup["target_path"],
            in_timesteps=5,
            out_timesteps=3,
            future_features=["coi_puntos"],
            node_batch_size=3,
            node_seed="node_2",  # Neighbors are node_1 and node_3
        )

        _, node_indices = dataset._get_time_and_node_indices(0)
        seed_node_id = dataset.node_to_id["node_2"]

        # Check if seed node is included in its own batch
        assert seed_node_id in node_indices
        # Check if all neighbors are included (node batch size is 3 so it's only 1-jump neighbors)
        assert set(node_indices) == {seed_node_id - 1, seed_node_id, seed_node_id + 1}

    def test_time_and_node_indices(self, test_data_setup):
        """Test the _get_time_and_node_indices method."""
        setup = test_data_setup

        dataset = TrafficDataset(
            graph_path=setup["graph_path"],
            static_data_path=setup["static_path"],
            dynamic_data_paths=setup["dynamic_paths"],
            metadata_paths=setup["metadata_paths"],
            target_data_path=setup["target_path"],
            in_timesteps=5,
            out_timesteps=3,
            future_features=["coi_puntos"],
            node_batch_size=3,
            node_seed="node_0",
        )

        assert len(dataset) > 0

        # Test with different indices
        for idx in [0, 1, len(dataset) // 2]:
            if idx < len(dataset):
                time_idx, node_indices = dataset._get_time_and_node_indices(idx)

                assert isinstance(time_idx, int)
                assert time_idx >= 0
                assert isinstance(node_indices, list)
                assert all(isinstance(node_idx, int) for node_idx in node_indices)
                assert all(
                    0 <= node_idx < dataset.num_nodes for node_idx in node_indices
                )

    def test_static_features_loading(self, test_data_setup):
        """Test static features are loaded correctly with exact numerical validation."""
        setup = test_data_setup

        # Create controlled test with known static feature values
        temp_dir = setup["temp_dir"]
        static_path = self._create_deterministic_static_features(temp_dir)

        dataset = TrafficDataset(
            graph_path=setup["graph_path"],
            static_data_path=static_path,
            dynamic_data_paths=setup["dynamic_paths"][:1],  # Use minimal dynamic data
            metadata_paths=setup["metadata_paths"][:1],
            target_data_path=setup["target_path"],
            in_timesteps=2,
            out_timesteps=1,
            future_features=[],
            node_batch_size=3,
            node_seed="node_0",
        )

        # Check static features tensor dimensions and type
        assert dataset.static_features.shape[0] == 10  # num_nodes (original graph size)
        assert (
            dataset.static_features.shape[1] == 5
        )  # num_static_features (original size)
        assert dataset.static_features.dtype == torch.float32

        # Verify exact values based on our deterministic pattern
        # Pattern: value = node_idx * 100 + feature_idx * 10
        # Sparse filtering: include only values where (value + 5) % 7 != 0

        expected_static_values = {}

        # Calculate expected values considering sparse filtering
        for node_idx in range(10):  # 10 nodes
            for feature_idx in range(5):  # 5 features
                value = node_idx * 100 + feature_idx * 10
                if (value + 5) % 7 != 0:  # This value should be present
                    expected_static_values[(node_idx, feature_idx)] = float(value)
                # else: value should be 0.0 (filtered out)

        # Verify exact non-zero values
        for (node_idx, feat_idx), expected_val in expected_static_values.items():
            actual_val = dataset.static_features[node_idx, feat_idx].item()
            assert abs(actual_val - expected_val) < 1e-5, (
                f"Expected static {expected_val} at node[{node_idx}, {feat_idx}], got {actual_val}"
            )

        # Verify zero values where we expect them (filtered out by sparsity)
        zero_positions = []
        for node_idx in range(10):  # 10 nodes
            for feature_idx in range(5):  # 5 features
                value = node_idx * 100 + feature_idx * 10
                if (value + 5) % 7 == 0:  # This value should be filtered out
                    zero_positions.append((node_idx, feature_idx))

        for node_idx, feat_idx in zero_positions:
            actual_val = dataset.static_features[node_idx, feat_idx].item()
            assert abs(actual_val) < 1e-5, (
                f"Expected zero at node[{node_idx}, {feat_idx}], got {actual_val}"
            )

        # Test that static features are consistently loaded in dataset samples
        sample = dataset[0]
        batch_static = sample.x_static

        # Sample should contain static features for the nodes in the batch
        time_idx, node_indices = dataset._get_time_and_node_indices(0)
        assert batch_static.shape[0] == len(node_indices), (
            "Batch static features should match batch size"
        )
        assert batch_static.shape[1] == 5, (
            "Batch static features should have 5 features"
        )

        # Verify that batch static features match the corresponding rows from the full tensor
        for i, global_node_idx in enumerate(node_indices):
            for feat_idx in range(5):  # 5 features
                expected_val = dataset.static_features[global_node_idx, feat_idx].item()
                actual_val = batch_static[i, feat_idx].item()
                assert abs(actual_val - expected_val) < 1e-5, (
                    f"Batch static feature mismatch at batch[{i}, {feat_idx}]: "
                    f"expected {expected_val}, got {actual_val}"
                )

    def test_node_batching_logic(self, test_data_setup):
        """Tests the node batching logic to ensure different batches have different nodes."""
        setup = test_data_setup

        dataset = TrafficDataset(
            graph_path=setup["graph_path"],
            static_data_path=setup["static_path"],
            dynamic_data_paths=setup["dynamic_paths"],
            metadata_paths=setup["metadata_paths"],
            target_data_path=setup["target_path"],
            in_timesteps=5,
            out_timesteps=3,
            future_features=["coi_puntos"],
            node_batch_size=4,
            node_seed="node_0",
        )

        # With 10 nodes and batch size 4, we could theoretically have 3 node batches (of sizes 4, 4, 2)
        assert dataset.total_possible_node_batches == 3
        # However, for a fixed seed, the dataset length is just the number of time windows
        expected_windows = 50 - (5 + 3) + 1  # 43 windows
        assert len(dataset) == expected_windows

        # The batches will have the same set of nodes if the seed node doesn't change
        t_0_0, nodes_0_0 = dataset._get_time_and_node_indices(0)  # seed 0, time 0
        t_0_42, nodes_0_42 = dataset._get_time_and_node_indices(42)  # seed 0, time 42
        expected_nodes_seed_0 = {0, 1, 2, 3}  # For seed node_0 and batch size 4
        assert set(nodes_0_0) == expected_nodes_seed_0
        assert set(nodes_0_42) == expected_nodes_seed_0
        assert t_0_0 == 0
        assert t_0_42 == 42
        # Time indices wrap around after reaching the end of available windows (43 total)
        # This is normal behavior - wrapping back to time 0 with the same node set
        t_0_wrap, nodes_0_wrap = dataset._get_time_and_node_indices(
            43
        )  # time wraps to 0
        assert set(nodes_0_wrap) == expected_nodes_seed_0
        assert t_0_wrap == 0

        # Setting another seed node should change the node batch but maintain time behavior
        dataset.reasign_seed("node_5")
        expected_nodes_seed_5 = {
            4,
            5,
            6,
            3,
        }  # Could also be {4, 5, 6, 7} if it picks the higher neighbor
        t_5_0, nodes_5_0 = dataset._get_time_and_node_indices(0)  # seed 5, time 0
        t_5_42, nodes_5_42 = dataset._get_time_and_node_indices(42)  # seed 5, time 42
        assert set(nodes_5_42) == expected_nodes_seed_5
        assert t_5_42 == 42
        assert set(nodes_5_0) == expected_nodes_seed_5
        assert t_5_0 == 0
        # Time wrapping works the same regardless of seed
        t_5_wrap, nodes_5_wrap = dataset._get_time_and_node_indices(
            43
        )  # time wraps to 0
        assert set(nodes_5_wrap) == expected_nodes_seed_5
        assert t_5_wrap == 0

    def test_dynamic_past_features_loading(self, test_data_setup):
        """Test that dynamic past features are loaded with correct values."""
        setup = test_data_setup

        # Create a controlled test: manually create known HDF5 data and verify it's loaded correctly
        temp_dir = setup["temp_dir"]

        # Create a new test with deterministic values
        test_h5_path, test_meta_path = self._create_deterministic_dynamic_features(
            temp_dir, "test_deterministic", ["feature_1", "feature_2"]
        )

        dataset = TrafficDataset(
            graph_path=setup["graph_path"],
            static_data_path=setup["static_path"],
            dynamic_data_paths=[test_h5_path],
            metadata_paths=[test_meta_path],
            target_data_path=setup["target_path"],
            in_timesteps=2,
            out_timesteps=1,
            future_features=[],  # No future features for this test
            node_batch_size=3,
            node_seed="node_1",  # Start with node_1 (neighbors: node_0, node_2)
        )

        sample = dataset[0]  # time=0

        # Based on our deterministic data:
        # Pattern: value = (node_idx * 10 + time_idx) * 10 + feature_idx
        # Node batch for node_1 with batch_size=3 includes [node_1, node_0, node_2] (BFS from node_1)
        # Time window [0, 1] for in_timesteps=2

        # Verify specific values are loaded correctly
        past_features = sample.x_dynamic_past

        # Check dimensions
        # (batch_size=3, timesteps=2, features=2)
        assert past_features.shape == (3, 2, 2)

        # Get actual node mapping to understand batch order
        time_idx, node_indices = dataset._get_time_and_node_indices(0)
        node_mapping = [dataset.id_to_node[idx] for idx in node_indices]

        # Based on the actual batch order [node_1, node_0, node_2], verify values:
        # Batch index 0 = node_1 (global index 1)
        # Batch index 1 = node_0 (global index 0)
        # Batch index 2 = node_2 (global index 2)

        # Expected values based on our pattern and sparsity (value % 3 != 0):
        expected_values = {
            # Batch index 0 (node_1):
            (0, 0, 0): 100.0,  # node_1, time_0, feature_0: (1*10 + 0)*10 + 0 = 100
            (0, 0, 1): 101.0,  # node_1, time_0, feature_1: (1*10 + 0)*10 + 1 = 101
            (0, 1, 0): 110.0,  # node_1, time_1, feature_0: (1*10 + 1)*10 + 0 = 110
            # node_1, time_1, feature_1: (1*10 + 1)*10 + 1 = 111 -> filtered out (111 % 3 == 0)
            # Batch index 1 (node_0):
            # node_0, time_0, feature_0: (0*10 + 0)*10 + 0 = 0 -> filtered out (0 % 3 == 0)
            (1, 0, 1): 1.0,  # node_0, time_0, feature_1: (0*10 + 0)*10 + 1 = 1
            (1, 1, 0): 10.0,  # node_0, time_1, feature_0: (0*10 + 1)*10 + 0 = 10
            (1, 1, 1): 11.0,  # node_0, time_1, feature_1: (0*10 + 1)*10 + 1 = 11
            # Batch index 2 (node_2):
            (2, 0, 0): 200.0,  # node_2, time_0, feature_0: (2*10 + 0)*10 + 0 = 200
            # node_2, time_0, feature_1: (2*10 + 0)*10 + 1 = 201 -> filtered out (201 % 3 == 0)
            # node_2, time_1, feature_0: (2*10 + 1)*10 + 0 = 210 -> filtered out (210 % 3 == 0)
            (2, 1, 1): 211.0,  # node_2, time_1, feature_1: (2*10 + 1)*10 + 1 = 211
        }

        # Verify expected non-zero values
        for (
            batch_idx,
            time_idx,
            feat_idx,
        ), expected_val in expected_values.items():
            actual_val = past_features[batch_idx, time_idx, feat_idx].item()
            assert abs(actual_val - expected_val) < 1e-5, (
                f"Expected {expected_val} at batch[{batch_idx}, {time_idx}, {feat_idx}], got {actual_val}"
            )

        # Verify zero values where we expect them (filtered out by sparsity)
        assert (
            past_features[0, 1, 1].item() == 0.0
        )  # node_1, time_1, feature_1 -> 111 % 3 == 0
        assert (
            past_features[1, 0, 0].item() == 0.0
        )  # node_0, time_0, feature_0 -> 0 % 3 == 0
        assert (
            past_features[2, 0, 1].item() == 0.0
        )  # node_2, time_0, feature_1 -> 201 % 3 == 0
        assert (
            past_features[2, 1, 0].item() == 0.0
        )  # node_2, time_1, feature_0 -> 210 % 3 == 0

    def test_dynamic_future_features_loading(self, test_data_setup):
        """Test that dynamic future features are loaded with correct values."""
        setup = test_data_setup

        # Create controlled test with known future feature values
        temp_dir = setup["temp_dir"]

        future_h5_path, future_meta_path = self._create_deterministic_dynamic_features(
            temp_dir,
            "test_future",
            ["future_feat_1", "future_feat_2", "other_feat"],
        )

        dataset = TrafficDataset(
            graph_path=setup["graph_path"],
            static_data_path=setup["static_path"],
            dynamic_data_paths=[future_h5_path],
            metadata_paths=[future_meta_path],
            target_data_path=setup["target_path"],
            in_timesteps=1,
            out_timesteps=2,
            future_features=[
                "future_feat_1",
                "future_feat_2",
            ],  # Only these should be loaded as future
            node_batch_size=2,
            node_seed="node_0",
        )

        sample = dataset[0]
        future_features = sample.x_dynamic_future

        # Shape should be (batch_size, out_timesteps, num_future_features)
        assert future_features.shape == (2, 2, 2)

        # Verify that only future features are loaded and others are filtered out
        # The third feature "other_feat" should not appear in the future tensor

        # Verify exact values based on our deterministic pattern
        # Expected values for future features (filtered by sparsity and future feature selection)
        # We're looking at time window [1, 2] for the future (after in_timesteps=1)
        # Need to understand the actual batch composition first

        time_idx, node_indices = dataset._get_time_and_node_indices(0)

        # Based on actual batch order (node_0 seed with batch_size=2 gives [node_0, node_1])
        # IMPORTANT: The current implementation has target_start_time = input_end_time
        # For sample 0: time_idx=0, input=[0], target=[0,1] (overlapping!)
        # This means future features come from time window [0, 1], not [1, 2]
        # Pattern: value = (node_idx * 10 + time_idx) * 10 + feature_idx
        # Only future_feat_1 (feature_idx=0) and future_feat_2 (feature_idx=1) should be loaded
        # Only values where value % 3 != 0 are included

        expected_future_values = {
            # Batch index 0 (node_0):
            # node_0, time_0, future_feat_1: (0*10 + 0)*10 + 0 = 0 -> filtered out (0 % 3 = 0)
            # node_0, time_0, future_feat_2: (0*10 + 0)*10 + 1 = 1 -> passes filter (1 % 3 = 1)
            (0, 0, 1): 1.0,
            # node_0, time_1, future_feat_1: (0*10 + 1)*10 + 0 = 10 -> passes filter (10 % 3 = 1)
            (0, 1, 0): 10.0,
            # node_0, time_1, future_feat_2: (0*10 + 1)*10 + 1 = 11 -> passes filter (11 % 3 = 2)
            (0, 1, 1): 11.0,
            # Batch index 1 (node_1):
            # node_1, time_0, future_feat_1: (1*10 + 0)*10 + 0 = 100 -> filtered out (100 % 3 = 1)
            (1, 0, 0): 100.0,
            # node_1, time_0, future_feat_2: (1*10 + 0)*10 + 1 = 101 -> passes filter (101 % 3 = 2)
            (1, 0, 1): 101.0,
            # node_1, time_1, future_feat_1: (1*10 + 1)*10 + 0 = 110 -> passes filter (110 % 3 = 2)
            (1, 1, 0): 110.0,
            # node_1, time_1, future_feat_2: (1*10 + 1)*10 + 1 = 111 -> filtered out (111 % 3 = 0)
        }

        # Verify expected non-zero values
        for (
            batch_idx,
            time_idx,
            feat_idx,
        ), expected_val in expected_future_values.items():
            actual_val = future_features[batch_idx, time_idx, feat_idx].item()
            assert abs(actual_val - expected_val) < 1e-5, (
                f"Expected future {expected_val} at batch[{batch_idx}, {time_idx}, {feat_idx}], got {actual_val}"
            )

        # Verify zero values where we expect them (filtered out by sparsity)
        zero_positions = [
            (0, 0, 0),  # node_0, time_0, future_feat_1: 0 % 3 = 0
            (1, 1, 1),  # node_1, time_1, future_feat_2: 111 % 3 = 0
        ]

        for batch_idx, time_idx, feat_idx in zero_positions:
            actual_val = future_features[batch_idx, time_idx, feat_idx].item()
            assert abs(actual_val) < 1e-5, (
                f"Expected zero at batch[{batch_idx}, {time_idx}, {feat_idx}], got {actual_val}"
            )

        assert (
            future_features[0, 0, 0].item() == 0.0
        )  # node_0, time_0, future_feat_1 -> 0 % 3 = 0
        assert (
            future_features[1, 1, 1].item() == 0.0
        )  # node_1, time_1, future_feat_2 -> 111 % 3 = 0

        # Check that future_features_idxs mapping is working correctly
        assert (
            len(dataset.future_features_idxs) == 1
        )  # Only one file has future features
        for file_path, indices in dataset.future_features_idxs.items():
            assert indices == [0, 1]  # future_feat_1 and future_feat_2 indices

    def test_target_data_loading(self, test_data_setup):
        """Test that target data (y) is loaded with correct values."""
        setup = test_data_setup

        # Create controlled test with known target values first
        temp_dir = setup["temp_dir"]
        target_path = self._create_deterministic_target_data(temp_dir)

        dataset_controlled = TrafficDataset(
            graph_path=setup["graph_path"],
            static_data_path=setup["static_path"],
            dynamic_data_paths=setup["dynamic_paths"][:1],  # Use minimal dynamic data
            metadata_paths=setup["metadata_paths"][:1],
            target_data_path=target_path,
            in_timesteps=1,
            out_timesteps=2,
            future_features=[],
            node_batch_size=3,
            node_seed="node_0",
        )

        controlled_sample = dataset_controlled[
            0
        ]  # time_idx=0, so target window is [0, 1]
        targets = controlled_sample.y

        # Shape should be (batch_size, out_timesteps)
        assert isinstance(targets, torch.Tensor)
        assert targets.shape == (3, 2)

        # Based on our deterministic target data:
        # Pattern: value = node_idx * 10 + time_idx + 100
        # Sample 0: time window [0, 1] (target comes after in_timesteps=1)

        time_idx, node_indices = dataset_controlled._get_time_and_node_indices(0)

        # Verify that targets have the expected shape and contain reasonable values
        assert targets.shape == (3, 2), f"Expected shape (3, 2), got {targets.shape}"

        # Based on actual batch order (node_0 seed with batch_size=3 gives [node_0, node_1, node_2])
        # Time window [0, 1] for targets (after in_timesteps=1)
        # Pattern: value = node_idx * 10 + time_idx + 100

        expected_target_values = {
            # Batch index 0 (node_0):
            (0, 0): 100.0,  # node_0, time_0: 0*10 + 0 + 100 = 100
            (0, 1): 101.0,  # node_0, time_1: 0*10 + 1 + 100 = 101
            # Batch index 1 (node_1):
            (1, 0): 110.0,  # node_1, time_0: 1*10 + 0 + 100 = 110
            (1, 1): 111.0,  # node_1, time_1: 1*10 + 1 + 100 = 111
            # Batch index 2 (node_2):
            (2, 0): 120.0,  # node_2, time_0: 2*10 + 0 + 100 = 120
            (2, 1): 121.0,  # node_2, time_1: 2*10 + 1 + 100 = 121
        }

        # Verify exact target values
        for (batch_idx, time_idx), expected_val in expected_target_values.items():
            actual_val = targets[batch_idx, time_idx].item()
            assert abs(actual_val - expected_val) < 1e-5, (
                f"Expected target {expected_val} at batch[{batch_idx}, {time_idx}], got {actual_val}"
            )

        # Test time window shifting - different samples should have different target windows
        if len(dataset_controlled) > 1:
            sample_1 = dataset_controlled[1]  # time_idx=1, so target window is [1, 2]
            targets_1 = sample_1.y
            assert isinstance(targets_1, torch.Tensor)

            # Verify the time shift worked correctly for sample 1
            # Time window [1, 2] for targets (time_idx=1)
            expected_shifted_values = {
                # Batch index 0 (node_0):
                (0, 0): 101.0,  # node_0, time_1: 0*10 + 1 + 100 = 101
                (0, 1): 102.0,  # node_0, time_2: 0*10 + 2 + 100 = 102
                # Batch index 1 (node_1):
                (1, 0): 111.0,  # node_1, time_1: 1*10 + 1 + 100 = 111
                (1, 1): 112.0,  # node_1, time_2: 1*10 + 2 + 100 = 112
                # Batch index 2 (node_2):
                (2, 0): 121.0,  # node_2, time_1: 2*10 + 1 + 100 = 121
                (2, 1): 122.0,  # node_2, time_2: 2*10 + 2 + 100 = 122
            }

            # Verify exact shifted target values
            for (batch_idx, time_idx), expected_val in expected_shifted_values.items():
                actual_val = targets_1[batch_idx, time_idx].item()
                assert abs(actual_val - expected_val) < 1e-5, (
                    f"Expected shifted target {expected_val} at batch[{batch_idx}, {time_idx}], got {actual_val}"
                )

            # The values should be different due to different time windows
            assert not torch.equal(targets, targets_1), (
                "Different time windows should produce different targets"
            )

    def test_future_feature_indices(self, test_data_setup):
        """Test future feature index mapping."""
        setup = test_data_setup

        dataset = TrafficDataset(
            graph_path=setup["graph_path"],
            static_data_path=setup["static_path"],
            dynamic_data_paths=setup["dynamic_paths"],
            metadata_paths=setup["metadata_paths"],
            target_data_path=setup["target_path"],
            in_timesteps=5,
            out_timesteps=3,
            future_features=["coi_puntos", "EMERGENCIA"],
            node_batch_size=5,
            node_seed="node_0",
        )

        # Should have mappings for files that contain future features
        assert isinstance(dataset.future_features_idxs, dict)

        # Check that files with future features are in the mapping
        future_feature_files = 0
        for file_path in dataset.dynamic_data_paths:
            if file_path in dataset.future_features_idxs:
                future_feature_files += 1
                indices = dataset.future_features_idxs[file_path]
                assert isinstance(indices, list)
                assert all(isinstance(idx, int) for idx in indices)

        # Should have at least one file with future features
        assert future_feature_files > 0

        # Check the exact indices for our test features
        for file_path, indices in dataset.future_features_idxs.items():
            if "coi_points" in file_path:
                assert indices == [0]  # coi_puntos is index 0 in that file
            if "events" in file_path:
                assert indices == [0]  # EMERGENCIA is index 0 in that file

    def _create_deterministic_dynamic_features(
        self, temp_dir: str, filename: str, feature_names: list
    ) -> tuple[str, str]:
        """Create dynamic features with deterministic, predictable values for testing."""
        data_list = []

        # Create predictable pattern: value = (node_idx * 10 + time_idx) * 10 + feature_idx
        # Use same dimensions as the test graph (10 nodes, 50 timesteps) to avoid index mismatches
        num_nodes = 10  # Match the test graph
        num_timesteps = 50  # Match the test data generator

        for time_idx in range(num_timesteps):
            for node_idx in range(num_nodes):
                for feature_idx, _ in enumerate(feature_names):
                    # Create predictable values that we can verify in tests
                    value = (node_idx * 10 + time_idx) * 10 + feature_idx
                    if value % 3 != 0:  # Add some sparsity but keep it predictable
                        data_list.append(
                            [node_idx, time_idx, feature_idx, float(value)]
                        )

        # Sort by time_idx
        data_array = np.array(data_list)
        if len(data_array) > 0:
            data_array = data_array[data_array[:, 1].argsort()]

        # Create HDF5 file
        h5_path = os.path.join(temp_dir, f"{filename}.h5")
        with h5py.File(h5_path, "w") as f:
            f.create_dataset("data", data=data_array)

        # Create metadata - use same dimensions as the test graph
        node_ids = [f"node_{i}" for i in range(num_nodes)]  # Match test graph
        node_idx_map = {node: i for i, node in enumerate(node_ids)}
        timeframes = pd.date_range(
            start="2024-01-01", periods=num_timesteps, freq="15min"
        ).to_list()
        time_idx_map = {ts: i for i, ts in enumerate(timeframes)}
        feature_idx_map = {name: i for i, name in enumerate(feature_names)}

        metadata = {
            "node_idx_map": node_idx_map,
            "time_idx_map": time_idx_map,
            "feature_idx_map": feature_idx_map,
        }

        metadata_path = os.path.join(temp_dir, f"{filename}_metadata.pkl")
        with open(metadata_path, "wb") as f:
            pickle.dump(metadata, f)

        return h5_path, metadata_path

    def _create_deterministic_target_data(self, temp_dir: str) -> str:
        """Create target data with deterministic values for testing."""
        data_list = []

        # Create predictable target pattern: value = node_idx * 10 + time_idx + 100
        # Use same dimensions as the test graph (10 nodes, 50 timesteps) to avoid index mismatches
        num_nodes = 10  # Match the test graph
        num_timesteps = 50  # Match the test data generator

        for time_idx in range(num_timesteps):
            for node_idx in range(num_nodes):
                value = node_idx * 10 + time_idx + 100
                data_list.append(
                    [node_idx, time_idx, 0, float(value)]
                )  # feature_idx always 0 for targets

        # Sort by time_idx
        data_array = np.array(data_list)
        data_array = data_array[data_array[:, 1].argsort()]

        target_path = os.path.join(temp_dir, "deterministic_target.h5")
        with h5py.File(target_path, "w") as f:
            f.create_dataset("data", data=data_array)

        return target_path

    def _create_deterministic_static_features(self, temp_dir: str) -> str:
        """Create static features with deterministic, predictable values for testing."""
        data_list = []

        # Create predictable static pattern: value = node_idx * 100 + feature_idx * 10
        # This ensures each node-feature combination has a unique, predictable value
        # Use 10 nodes to match the original test graph and 5 features to match original static data
        for node_idx in range(10):  # 10 nodes to match graph
            for feature_idx in range(5):  # 5 static features to match original
                value = node_idx * 100 + feature_idx * 10
                # Add sparse filtering: include only values where (value + 5) % 7 != 0
                # This gives us a predictable but sparse pattern
                if (value + 5) % 7 != 0:
                    data_list.append([node_idx, feature_idx, float(value)])

        data_array = np.array(data_list)

        static_path = os.path.join(temp_dir, "deterministic_static.h5")
        with h5py.File(static_path, "w") as f:
            f.create_dataset("data", data=data_array)

        return static_path

    def test_weather_features_initialization(self, test_data_setup):
        """Test that weather features are properly initialized when include_weather=True."""
        setup = test_data_setup

        dataset = TrafficDataset(
            graph_path=setup["graph_path"],
            static_data_path=setup["static_path"],
            dynamic_data_paths=setup["dynamic_paths"],
            metadata_paths=setup["metadata_paths"],
            target_data_path=setup["target_path"],
            in_timesteps=5,
            out_timesteps=3,
            future_features=["coi_puntos"],
            node_batch_size=3,
            node_seed="node_0",
            weather_data_dir=setup["temp_dir"],
            include_weather=True,
        )

        # Check that weather components are properly initialized
        assert dataset.include_weather is True
        assert dataset.weather_data_dir == setup["temp_dir"]
        assert dataset.weather_stations is not None
        assert dataset.weather_data is not None
        assert (
            len(dataset.weather_features) == 4
        )  # wind_direction, humidity, wind_speed, precipitation
        assert dataset.num_weather_features == 4

        # Check that weather data was loaded
        for feature in dataset.weather_features:
            assert feature in dataset.weather_data
            assert isinstance(dataset.weather_data[feature], pd.DataFrame)
            assert not dataset.weather_data[feature].empty

        # Check weather stations mapping
        for node_id in setup["generator"].node_ids:
            assert node_id in dataset.weather_stations
            station_mapping = dataset.weather_stations[node_id]
            assert "estacion_direccion_viento" in station_mapping
            assert "estacion_humedad" in station_mapping
            assert "estacion_velocidad_viento" in station_mapping
            assert "estacion_precipitacion" in station_mapping

    def test_get_weather_features_method(self, test_data_setup):
        """Test the _get_weather_features method in isolation."""
        setup = test_data_setup

        dataset = TrafficDataset(
            graph_path=setup["graph_path"],
            static_data_path=setup["static_path"],
            dynamic_data_paths=setup["dynamic_paths"],
            metadata_paths=setup["metadata_paths"],
            target_data_path=setup["target_path"],
            in_timesteps=5,
            out_timesteps=3,
            future_features=["coi_puntos"],
            node_batch_size=3,
            node_seed="node_0",
            weather_data_dir=setup["temp_dir"],
            include_weather=True,
        )

        # Test parameters
        batch_size = 3
        total_timesteps = 8  # in_timesteps + out_timesteps
        time_start = 0
        time_end = total_timesteps - 1
        node_indices = [0, 1, 2]  # First 3 nodes
        node_idx_to_batch_idx = {0: 0, 1: 1, 2: 2}

        # Create empty weather tensor
        x_weather = torch.zeros(
            (batch_size, total_timesteps, dataset.num_weather_features)
        )

        # Call the method
        result_tensor = dataset._get_weather_features(
            x_weather=x_weather,
            time_start=time_start,
            time_end=time_end,
            node_indices=node_indices,
            node_idx_to_batch_idx=node_idx_to_batch_idx,
        )

        # Check that result is the same tensor object
        assert result_tensor is x_weather
        assert result_tensor.shape == (
            batch_size,
            total_timesteps,
            dataset.num_weather_features,
        )

        # Check that values were populated (at least some non-zero values)
        assert torch.any(result_tensor != 0), (
            "Weather tensor should have non-zero values"
        )

        # Test specific values for deterministic data
        # Node 0 uses:
        # - WindStation_0 for wind_direction
        # - HumidityStation_0 for humidity
        # - SpeedStation_0 for wind_speed
        # - PrecipStation_0 for precipitation

        # Check first timestep (time_idx=0) for first node (batch_idx=0)
        # Expected values based on our deterministic formula: base_value + time_idx + station_idx
        expected_wind = 100 + 0 + 0  # base=100, time=0, station=0
        expected_humidity = 50 + 0 + 0  # base=50, time=0, station=0
        expected_speed = 10 + 0 + 0  # base=10, time=0, station=0
        expected_precip = 0 + 0 + 0  # base=0, time=0, station=0

        assert result_tensor[0, 0, 0].item() == expected_wind
        assert result_tensor[0, 0, 1].item() == expected_humidity
        assert result_tensor[0, 0, 2].item() == expected_speed
        assert result_tensor[0, 0, 3].item() == expected_precip

        # Check second timestep (time_idx=1) for first node
        expected_wind_t1 = 100 + 1 + 0  # base=100, time=1, station=0
        expected_humidity_t1 = 50 + 1 + 0  # base=50, time=1, station=0
        assert result_tensor[0, 1, 0].item() == expected_wind_t1
        assert result_tensor[0, 1, 1].item() == expected_humidity_t1

        # Check different node (node_1 uses different stations)
        # Node 1 uses:
        # - WindStation_1 for wind_direction
        # - HumidityStation_1 for humidity
        expected_wind_n1 = 100 + 0 + 1  # base=100, time=0, station=1
        expected_humidity_n1 = 50 + 0 + 1  # base=50, time=0, station=1
        assert result_tensor[1, 0, 0].item() == expected_wind_n1
        assert result_tensor[1, 0, 1].item() == expected_humidity_n1

    def test_getitem_with_weather_enabled(self, test_data_setup):
        """Test __getitem__ method when weather is enabled."""
        setup = test_data_setup

        dataset = TrafficDataset(
            graph_path=setup["graph_path"],
            static_data_path=setup["static_path"],
            dynamic_data_paths=setup["dynamic_paths"],
            metadata_paths=setup["metadata_paths"],
            target_data_path=setup["target_path"],
            in_timesteps=5,
            out_timesteps=3,
            future_features=["coi_puntos"],
            node_batch_size=4,
            node_seed="node_0",
            weather_data_dir=setup["temp_dir"],
            include_weather=True,
        )

        # Get a sample
        sample = dataset[0]

        # Check that sample contains weather data
        assert hasattr(sample, "x_weather")
        assert isinstance(sample.x_weather, torch.Tensor)

        # Check weather tensor dimensions
        batch_size = sample.x_static.shape[0]
        total_timesteps = dataset.total_timesteps_window  # in_timesteps + out_timesteps
        assert sample.x_weather.shape == (
            batch_size,
            total_timesteps,
            dataset.num_weather_features,
        )
        assert sample.x_weather.dtype == torch.float32

        # Check that weather data is populated (not all zeros)
        assert torch.any(sample.x_weather != 0), (
            "Weather tensor should contain non-zero values"
        )

        # Verify that all other data components are still present
        assert hasattr(sample, "x_static")
        assert hasattr(sample, "x_dynamic_past")
        assert hasattr(sample, "x_dynamic_future")
        assert hasattr(sample, "y")
        assert hasattr(sample, "edge_index")

    def test_getitem_without_weather(self, test_data_setup):
        """Test __getitem__ method when weather is disabled."""
        setup = test_data_setup

        dataset = TrafficDataset(
            graph_path=setup["graph_path"],
            static_data_path=setup["static_path"],
            dynamic_data_paths=setup["dynamic_paths"],
            metadata_paths=setup["metadata_paths"],
            target_data_path=setup["target_path"],
            in_timesteps=5,
            out_timesteps=3,
            future_features=["coi_puntos"],
            node_batch_size=4,
            node_seed="node_0",
            include_weather=False,  # Weather disabled
        )

        # Get a sample
        sample = dataset[0]

        # Check that sample does NOT contain weather data
        assert not hasattr(sample, "x_weather")

        # Verify that all other data components are still present
        assert hasattr(sample, "x_static")
        assert hasattr(sample, "x_dynamic_past")
        assert hasattr(sample, "x_dynamic_future")
        assert hasattr(sample, "y")
        assert hasattr(sample, "edge_index")

    def test_weather_data_consistency(self, test_data_setup):
        """Test that weather data loaded in dataset matches the generated test data."""
        setup = test_data_setup

        dataset = TrafficDataset(
            graph_path=setup["graph_path"],
            static_data_path=setup["static_path"],
            dynamic_data_paths=setup["dynamic_paths"],
            metadata_paths=setup["metadata_paths"],
            target_data_path=setup["target_path"],
            in_timesteps=5,
            out_timesteps=3,
            future_features=["coi_puntos"],
            node_batch_size=4,
            node_seed="node_0",
            weather_data_dir=setup["temp_dir"],
            include_weather=True,
        )

        # Check that loaded weather data matches our generated data
        for feature_name in [
            "wind_direction",
            "humidity",
            "wind_speed",
            "precipitation",
        ]:
            # Load the original data file
            original_df = pd.read_parquet(setup["weather_data_paths"][feature_name])
            dataset_df = dataset.weather_data[feature_name]

            # Check that DataFrames are equivalent
            pd.testing.assert_frame_equal(original_df, dataset_df)

        # Test specific node-station mappings
        generator = setup["generator"]
        for i, node_id in enumerate(generator.node_ids[:5]):  # Test first 5 nodes
            assert dataset.weather_stations is not None
            station_mapping = dataset.weather_stations[node_id]

            # Verify the round-robin assignment matches our test data generation
            expected_wind_station = f"WindStation_{i % 3}"
            expected_humidity_station = f"HumidityStation_{i % 2}"
            expected_speed_station = f"SpeedStation_{i % 3}"
            expected_precip_station = f"PrecipStation_{i % 2}"

            assert station_mapping["estacion_direccion_viento"] == expected_wind_station
            assert station_mapping["estacion_humedad"] == expected_humidity_station
            assert (
                station_mapping["estacion_velocidad_viento"] == expected_speed_station
            )
            assert station_mapping["estacion_precipitacion"] == expected_precip_station

    def test_weather_features_with_different_time_ranges(self, test_data_setup):
        """Test weather features with different time windows."""
        setup = test_data_setup

        # Test with larger time window
        dataset = TrafficDataset(
            graph_path=setup["graph_path"],
            static_data_path=setup["static_path"],
            dynamic_data_paths=setup["dynamic_paths"],
            metadata_paths=setup["metadata_paths"],
            target_data_path=setup["target_path"],
            in_timesteps=10,
            out_timesteps=5,
            future_features=["coi_puntos"],
            node_batch_size=3,
            node_seed="node_0",
            weather_data_dir=setup["temp_dir"],
            include_weather=True,
        )

        sample = dataset[0]

        # Check weather tensor dimensions match the new time window
        batch_size = sample.x_static.shape[0]
        total_timesteps = 15  # in_timesteps + out_timesteps
        assert sample.x_weather.shape == (
            batch_size,
            total_timesteps,
            dataset.num_weather_features,
        )

        # Check that data is populated across the full time range
        assert torch.any(sample.x_weather[:, :10, :] != 0), (
            "Past weather data should be non-zero"
        )
        assert torch.any(sample.x_weather[:, 10:, :] != 0), (
            "Future weather data should be non-zero"
        )

    def test_weather_features_edge_cases(self, test_data_setup):
        """Test weather features with edge cases."""
        setup = test_data_setup

        dataset = TrafficDataset(
            graph_path=setup["graph_path"],
            static_data_path=setup["static_path"],
            dynamic_data_paths=setup["dynamic_paths"],
            metadata_paths=setup["metadata_paths"],
            target_data_path=setup["target_path"],
            in_timesteps=2,
            out_timesteps=1,
            future_features=["coi_puntos"],
            node_batch_size=2,
            node_seed="node_0",
            weather_data_dir=setup["temp_dir"],
            include_weather=True,
        )

        # Test near the end of the time series
        last_valid_idx = len(dataset) - 1
        if last_valid_idx > 0:
            sample = dataset[last_valid_idx]

            # Should still have proper weather tensor shape
            batch_size = sample.x_static.shape[0]
            total_timesteps = 3  # in_timesteps + out_timesteps
            assert sample.x_weather.shape == (
                batch_size,
                total_timesteps,
                dataset.num_weather_features,
            )

    def test_weather_loading_without_weather_dir(self, test_data_setup):
        """Test that dataset works correctly when weather is disabled."""
        setup = test_data_setup

        dataset = TrafficDataset(
            graph_path=setup["graph_path"],
            static_data_path=setup["static_path"],
            dynamic_data_paths=setup["dynamic_paths"],
            metadata_paths=setup["metadata_paths"],
            target_data_path=setup["target_path"],
            in_timesteps=5,
            out_timesteps=3,
            future_features=["coi_puntos"],
            node_batch_size=4,
            node_seed="node_0",
            include_weather=False,  # Weather disabled
        )

        # Should not have weather components
        assert dataset.include_weather is False
        assert dataset.weather_stations is None
        assert not hasattr(dataset, "weather_data") or dataset.weather_data is None
        assert dataset.num_weather_features == 0

        # Should work normally without weather
        sample = dataset[0]
        assert not hasattr(sample, "x_weather")

        # All other functionality should work
        assert hasattr(sample, "x_static")
        assert hasattr(sample, "x_dynamic_past")
        assert hasattr(sample, "x_dynamic_future")
        assert hasattr(sample, "y")
        assert hasattr(sample, "edge_index")

    def test_node_index_safeguard(self, test_data_setup):
        """Test that the node index safeguard properly raises an exception when node indices are out of bounds."""
        setup = test_data_setup

        # Create a dataset first to get access to its internals
        dataset = TrafficDataset(
            graph_path=setup["graph_path"],
            static_data_path=setup["static_path"],
            dynamic_data_paths=setup["dynamic_paths"],
            metadata_paths=setup["metadata_paths"],
            target_data_path=setup["target_path"],
            in_timesteps=5,
            out_timesteps=3,
            future_features=["coi_puntos"],
            node_batch_size=4,
            node_seed="node_0",
        )

        # Artificially create an out-of-bounds scenario by manipulating the batch_mapping
        # to include a node index that exceeds the number of nodes in the graph
        original_mapping = dataset.batch_mapping[dataset.node_seed].copy()

        # Add an invalid node that maps to an out-of-bounds index
        invalid_node_id = "invalid_node"
        dataset.node_to_id[invalid_node_id] = (
            dataset.num_nodes + 5
        )  # Out of bounds index
        dataset.batch_mapping[dataset.node_seed] = original_mapping + [invalid_node_id]

        # Now try to access the dataset, which should trigger the safeguard
        with pytest.raises(
            ValueError,
            match=r"Node index \d+ is out of bounds for graph with \d+ nodes",
        ):
            # This should trigger _get_time_and_node_indices and hit the safeguard
            dataset._get_time_and_node_indices(0)

        # Restore original state for cleanup
        del dataset.node_to_id[invalid_node_id]
        dataset.batch_mapping[dataset.node_seed] = original_mapping
