import math
import pickle

import h5py
import networkx as nx
import numpy as np
import pandas as pd
import torch
from scipy.sparse import coo_matrix
from torch_geometric.data import Data, Dataset
from torch_geometric.utils import subgraph
from torch_geometric.utils.convert import from_networkx


class TrafficDataset(Dataset):
    """
    Custom Dataset class for the traffic forecasting problem with memory-efficient
    lazy loading. It loads a graph and static node features into memory, but
    fetches dynamic features from HDF5 files on-the-fly for each sample.
    """

    def __init__(
        self,
        graph_path: str,
        static_data_path: str,
        dynamic_data_paths: list[str],
        metadata_paths: list[str],
        target_data_path: str,
        in_timesteps: int,
        out_timesteps: int,
        future_features: list[str],
        node_batch_size: int = 500,
        node_seed: str = "0_FT.5",
        weather_data_dir: str | None = None,
        include_weather: bool = False,
        neighbor_features_path: str | None = None,
        free_flow_estimates_path: str | None = None,
    ):
        """
        Args:
            graph_path (str): Path to the NetworkX graph file (.pkl).
            static_data_path (str): Path to the HDF5 file with static node features.
            dynamic_data_paths (list of str): List of paths to HDF5 files with dynamic features.
            target_data_path (str): Path to the HDF5 file with target variable.
            in_timesteps (int): Number of time steps in the input sequence.
            out_timesteps (int): Number of time steps in the output sequence.
            future_features (list of str): List of dynamic feature names available in the future.
            node_batch_size (int): Number of reachable neighboring nodes to include in each batch.
            node_seed (str): The node ID to use as the seed for the first batch.
            weather_data_dir (str, optional): Directory containing processed weather data files.
            include_weather (bool): Whether to include weather features in the dataset.
            neighbor_features_path (str, optional): Path to parquet file with neighbor-aggregated features for imputation.
            free_flow_estimates_path (str, optional): Path to parquet file with free-flow speed estimates.
        """
        super().__init__()
        self.node_seed = node_seed
        self.in_timesteps = in_timesteps
        self.out_timesteps = out_timesteps
        self.total_timesteps_window = in_timesteps + out_timesteps
        self.dynamic_data_paths = dynamic_data_paths
        self.target_data_path = target_data_path
        self.node_batch_size = node_batch_size
        self.future_features = future_features
        self.include_weather = include_weather
        self.weather_data_dir = weather_data_dir

        # Load static data and graph
        # These are assumed to be small enough to fit in memory
        self.graph, self.edge_index, self.node_to_id, self.id_to_node = (
            self._load_graph(graph_path)
        )
        if node_seed not in self.graph.nodes():
            raise ValueError(
                f"Node seed {node_seed} not found in graph nodes. "
                "Please choose a valid node ID."
            )
        self.batch_mapping = self.get_first_n_reachable()
        self.static_features = self._load_static_features(static_data_path)
        self.num_nodes = self.static_features.shape[0]
        # Number of different seed nodes needed to cover all nodes (for external training logic)
        self.total_possible_node_batches = math.ceil(
            self.num_nodes / self.node_batch_size
        )
        self.num_static_features = self.static_features.shape[1]

        # Load weather data if requested
        self.weather_stations = None
        self.weather_features = [
            "wind_direction",
            "humidity",
            "wind_speed",
            "precipitation",
        ]
        self.station_col_map = {
            "wind_direction": "estacion_direccion_viento",
            "humidity": "estacion_humedad",
            "wind_speed": "estacion_velocidad_viento",
            "precipitation": "estacion_precipitacion",
        }
        self.value_col_map = {
            "wind_direction": "direccion_viento",
            "humidity": "humedad",
            "wind_speed": "velocidad_viento",
            "precipitation": "precipitacion",
        }
        self.num_weather_features = 0
        if self.include_weather and self.weather_data_dir:
            self.weather_stations, self.weather_data = self._load_weather_data()
            self.num_weather_features = len(self.weather_features)

        # Load neighbor features for imputation of special features
        self.neighbor_features = None
        self.imputation_mapping = {
            "trend": "mean_trend",
            "speed": "mean_regularSpeed",
            "regularSpeed": "mean_regularSpeed",
        }
        self.imputation_features = list(self.imputation_mapping.keys())
        self.imputation_feature_indices: dict[str, dict[str, int]] = {}
        if neighbor_features_path:
            self.neighbor_features = pd.read_parquet(neighbor_features_path)
            # Identify which HDF5 files contain the imputation features
            self._identify_imputation_features(metadata_paths)

        # Open dynamic data files and get metadata
        self.total_timesteps_in_data = 0
        self.dynamic_feature_dims = []
        for file_path in self.dynamic_data_paths + [self.target_data_path]:
            with h5py.File(file_path, "r") as hf:
                hf_data = hf["data"]

                # Assuming all files have the same number of nodes and time steps
                if hf_data.shape[0] > 0:  # type: ignore
                    self.total_timesteps_in_data = max(
                        self.total_timesteps_in_data,
                        int(hf_data[:, 1].max() + 1),  # type: ignore
                    )
                    if file_path != self.target_data_path:
                        # Only count dynamic feature files
                        self.dynamic_feature_dims.append(
                            int(hf_data[:, 2].max() + 1)  # type: ignore
                        )
                    else:
                        # Check that the target data has just one feature_idx
                        if hf_data[:, 2].max() != 0:  # type: ignore
                            raise ValueError(
                                f"Target data file {file_path} should have only one feature index (0)."
                            )
                else:
                    if file_path == self.target_data_path:
                        raise ValueError(
                            f"Target data file {file_path} is empty. "
                            "Ensure it contains valid data."
                        )
                    else:
                        self.dynamic_feature_dims.append(0)

        self.num_dynamic_features = sum(self.dynamic_feature_dims)
        self.num_dynamic_future_features = len(self.future_features)
        self.dynamic_data_indices, self.target_data_indices = (
            self._precompute_time_indices()
        )
        self.future_features_idxs = self._get_future_dynamic_feature_idxs(
            metadata_paths
        )

    def get_first_n_reachable(self) -> dict[str, list[str]]:
        """
        For every node in a graph, finds the first N reachable neighboring nodes
        ordered by the fewest number of jumps (BFS order). This function is
        efficient for large graphs.
        Args:
            graph (nx.Graph): The input NetworkX graph.
            n_neighbors (int): The number of reachable nodes to find for each node.
        Returns:
            dict: A dictionary where each key is a node ID and the value is a
                list of the first n_neighbors reachable nodes.
        """
        # Initialize the dictionary to store the results
        all_reachable_nodes = {}

        # Iterate over every node in the graph to treat it as a starting point
        for start_node in self.graph.nodes():
            # nx.bfs_edges provides an efficient generator that yields edges
            # in the order of a BFS
            bfs_edges_generator = nx.bfs_edges(self.graph, source=start_node)

            reachable = [start_node]  # Include the start node itself
            # Since bfs_edges_generator is a generator, it's much more efficient
            # to iterate over it directly and break when we have enough neighbors
            for _, v in bfs_edges_generator:
                reachable.append(v)
                if len(reachable) >= self.node_batch_size:
                    break
            all_reachable_nodes[start_node] = reachable

        return all_reachable_nodes

    def _get_future_dynamic_feature_idxs(
        self, metadata_paths: list[str]
    ) -> dict[str, list[int]]:
        """
        Maps future feature names to their corresponding indices in each dynamic
        feature file.

        Args:
            metadata_paths (list[str]): List of paths to the metadata files.

        Returns:
            dict[str, list[int]]: A mapping from dataset file paths to lists of feature indices.
        """
        future_features_idxs: dict[str, list[int]] = {}
        found_features = set()

        for dataset_file, metadata_file in zip(self.dynamic_data_paths, metadata_paths):
            with open(metadata_file, "rb") as f:
                metadata = pickle.load(f)
            features_in_this_file = [
                feature
                for feature in self.future_features
                if feature in metadata["feature_idx_map"]
            ]
            if features_in_this_file:
                feature_idxs = [
                    metadata["feature_idx_map"][feature]
                    for feature in features_in_this_file
                ]
                future_features_idxs.update({dataset_file: feature_idxs})
                found_features.update(features_in_this_file)

        # Check if any future features were not found
        missing_features = set(self.future_features) - found_features
        if missing_features:
            raise ValueError(
                f"The following future features were not found in any metadata file: {missing_features}"
            )

        return future_features_idxs

    def reasign_seed(self, node_seed: str) -> None:
        """Reassigns the node seed for batch selection."""
        if node_seed not in self.graph.nodes():
            raise ValueError(
                f"Node seed {node_seed} not found in graph nodes. "
                "Please choose a valid node ID."
            )
        if node_seed != self.node_seed:
            self.node_seed = node_seed

    def _load_graph(
        self, graph_path
    ) -> tuple[nx.Graph, torch.Tensor, dict[str, int], dict[int, str]]:
        """Loads the graph from a pickle file.
        Args:
            graph_path (str): The path to the graph pickle file.
        Returns:
            tuple[nx.Graph, torch.Tensor, dict[str, int], dict[int, str]]: The loaded graph,
                edge index tensor, node to ID mapping, and ID to node mapping.
        """
        with open(graph_path, "rb") as f:
            nx_graph = pickle.load(f)

        node_to_id = {node: i for i, node in enumerate(nx_graph.nodes())}
        id_to_node = {i: node for i, node in enumerate(nx_graph.nodes())}
        return nx_graph, from_networkx(nx_graph).edge_index, node_to_id, id_to_node

    def _load_static_features(self, file_path):
        """Loads static features from HDF5 and returns a dense tensor."""
        with h5py.File(file_path, "r") as f:
            data = f["data"][:]  # type: ignore
            assert isinstance(data, np.ndarray)

        node_idx, feature_idx, values = data[:, 0], data[:, 1], data[:, 2]
        num_nodes = int(node_idx.max() + 1)
        num_features = int(feature_idx.max() + 1)

        sparse_matrix = coo_matrix(
            (values, (node_idx, feature_idx)), shape=(num_nodes, num_features)
        )
        del data  # Free memory

        return torch.from_numpy(sparse_matrix.toarray()).float()

    def _load_weather_data(self) -> tuple[dict, dict[str, pd.DataFrame]]:
        """
        Load weather data for efficient access during training.

        Returns:
            tuple: (weather_stations mapping, weather_data dict)
        """
        # Load stations mapping
        stations_path = f"{self.weather_data_dir}/weather_stations.pkl"
        with open(stations_path, "rb") as f:
            weather_stations = pickle.load(f)

        # Load weather dataframes
        weather_data = {}
        for feature in self.weather_features:
            weather_path = f"{self.weather_data_dir}/weather_{feature}.parquet"
            weather_data[feature] = pd.read_parquet(weather_path)

        return weather_stations, weather_data

    def _identify_imputation_features(self, metadata_paths: list[str]) -> None:
        """
        Identify which HDF5 files and feature indices correspond to the imputation features.

        Args:
            metadata_paths (list[str]): List of paths to metadata files.
        """
        for dataset_file, metadata_file in zip(self.dynamic_data_paths, metadata_paths):
            with open(metadata_file, "rb") as f:
                metadata = pickle.load(f)

            # Check which imputation features are in this file
            features_in_file = {}
            for feature_name in self.imputation_features:
                if feature_name in metadata["feature_idx_map"]:
                    feature_idx = metadata["feature_idx_map"][feature_name]
                    features_in_file[feature_name] = feature_idx

            if features_in_file:
                self.imputation_feature_indices[dataset_file] = features_in_file

    def _precompute_time_indices(
        self,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """
        Pre-computes the start and end row indices for every possible sample
        in each HDF5 file. This is done once at initialization to speed up
        __getitem__.

        Returns:
            tuple[dict[str, np.ndarray], dict[str, np.ndarray]]: The precomputed
                start and end indices for dynamic and target data.
        """
        dynamic_data_indices = {}
        target_data_indices = {}

        for file_path in self.dynamic_data_paths + [self.target_data_path]:
            # print(f"Precomputing time indices for {file_path}...")
            file_indices = np.zeros((self.total_timesteps_in_data - 1, 2), dtype=int)
            # This assumes that the time column is the second column in the data
            # and that it is sorted. If not, we would need to sort it first.
            with h5py.File(file_path, "r") as f:
                time_column = f["data"][:, 1]  # type: ignore
                assert isinstance(time_column, np.ndarray)
                file_indices[:, 0] = np.searchsorted(
                    time_column,
                    np.arange(self.total_timesteps_in_data - 1),
                    side="left",
                )
                file_indices[:, 1] = np.searchsorted(
                    time_column,
                    np.arange(self.total_timesteps_in_data - 1),
                    side="right",
                )
            if file_path in self.dynamic_data_paths:
                dynamic_data_indices[file_path] = file_indices
            else:
                target_data_indices[file_path] = file_indices
        # print("Time indices precomputation completed.")
        return dynamic_data_indices, target_data_indices

    def __len__(self):
        """
        Returns the total number of samples for the current node seed.
        Since the dataset only provides samples for the current node_seed's
        neighborhood, the length is just the number of time windows.
        Node batching is handled externally by reassigning the seed.
        """
        if self.total_timesteps_in_data < self.total_timesteps_window:
            return 0
        num_time_windows = (
            self.total_timesteps_in_data - self.total_timesteps_window + 1
        )
        return num_time_windows

    def _get_time_and_node_indices(self, idx) -> tuple[int, list[int]]:
        """
        Given a global index, computes the corresponding time index and
        node indices.
        """
        num_time_windows = (
            self.total_timesteps_in_data - self.total_timesteps_window + 1
        )
        time_idx = idx % num_time_windows
        neighbors = self.batch_mapping.get(self.node_seed, [])
        node_indices = [self.node_to_id[n] for n in neighbors]

        # Validate that all node indices are within bounds
        for node_idx in node_indices:
            if node_idx >= self.num_nodes:
                raise ValueError(
                    f"Node index {node_idx} is out of bounds for graph with {self.num_nodes} nodes. "
                    f"This indicates an inconsistency between the graph and the data mappings."
                )

        return time_idx, node_indices

    def _get_dynamic_past_features(
        self,
        x_dynamic_past: torch.Tensor,
        file_path: str,
        input_start_time: int,
        input_end_time: int,
        feature_offset_past: int,
        node_idx_to_batch_idx: dict[int, int],
    ) -> torch.Tensor:
        """
        Fetches and populates the past dynamic features for the given time window
        and node batch into the provided dense tensor.
        The first step is to read the relevant rows from the HDF5 file using
        the precomputed indices, in order to avoid loading the entire file into memory.
        Then, we filter and populate the dense tensor only for the nodes in the current batch.
        Args:
            x_dynamic_past (torch.Tensor): The dense tensor to populate.
            file_path (str): Path to the HDF5 file.
            input_start_time (int): Start time index of the input window.
            input_end_time (int): End time index of the input window.
            feature_offset_past (int): Offset for feature indices in the dense tensor.
            node_idx_to_batch_idx (dict[int, int]): Mapping from global node indices to batch indices.
        Returns:
            torch.Tensor: The populated dense tensor with past dynamic features.
        """
        with h5py.File(file_path, "r") as f:
            # We first fetch the relevant data for the past
            start_index = self.dynamic_data_indices[file_path][input_start_time, 0]
            end_index = self.dynamic_data_indices[file_path][input_end_time, 1]
            relevant_data_past = f["data"][start_index:end_index]  # type: ignore
            assert isinstance(relevant_data_past, np.ndarray)

        if relevant_data_past.shape[0] > 0:
            node_idx, time_idx, feature_idx, values = (
                relevant_data_past[:, 0].astype(int),
                relevant_data_past[:, 1].astype(int) - input_start_time,
                # Add offset to feature index (each file has its own feature set
                # so we need to offset them to avoid collisions)
                relevant_data_past[:, 2].astype(int) + feature_offset_past,
                relevant_data_past[:, 3],
            )
            del relevant_data_past  # Free memory

            # Only populate features for nodes in the current batch
            batch_mask = np.isin(node_idx, list(node_idx_to_batch_idx.keys()))
            if batch_mask.sum() > 0:
                batch_node_idx = node_idx[batch_mask]
                batch_time_idx = time_idx[batch_mask]
                batch_feature_idx = feature_idx[batch_mask]
                batch_values = values[batch_mask]

                # Convert global node indices to batch indices
                batch_node_indices = [
                    node_idx_to_batch_idx[idx] for idx in batch_node_idx
                ]

                # Populate the dense tensor for the window
                x_dynamic_past[
                    batch_node_indices, batch_time_idx, batch_feature_idx
                ] = torch.from_numpy(batch_values).float()

        # Impute missing values for special features using neighbor aggregates
        if (
            self.neighbor_features is not None
            and file_path in self.imputation_feature_indices
        ):
            features_to_impute = self.imputation_feature_indices[file_path]

            for feature_name, feature_idx_in_file in features_to_impute.items():
                # Get the imputation column name from the mapping
                imputation_col = self.imputation_mapping[feature_name]

                # Calculate the global feature index (with offset)
                global_feature_idx = feature_idx_in_file + feature_offset_past

                # For each node in the batch, impute missing values
                for global_node_idx, batch_idx in node_idx_to_batch_idx.items():
                    # Get the node ID from the global node index
                    node_id = self.id_to_node[global_node_idx]

                    # Check if this node has neighbor features
                    if node_id not in self.neighbor_features.index:
                        continue

                    # Get the imputation value for this node
                    imputation_value = float(
                        self.neighbor_features.loc[node_id, imputation_col]  # type: ignore
                    )

                    # Find timesteps where this feature is missing (zero) for this node
                    # We check the entire time window for this node and feature
                    missing_mask = x_dynamic_past[batch_idx, :, global_feature_idx] == 0

                    # Impute the missing values
                    x_dynamic_past[batch_idx, missing_mask, global_feature_idx] = (
                        imputation_value
                    )

        return x_dynamic_past

    def _get_dynamic_future_features(
        self,
        x_dynamic_future,
        file_path: str,
        target_start_time: int,
        target_end_time: int,
        feature_offset_future: int,
        node_idx_to_batch_idx: dict[int, int],
    ) -> torch.Tensor:
        """
        Fetches and populates the future dynamic features for the given time window
        and node batch into the provided dense tensor.
        The first step is to read the relevant rows from the HDF5 file using
        the precomputed indices, in order to avoid loading the entire file into memory.
        Then, we filter and populate the dense tensor only for the nodes in the current batch.
        Args:
            x_dynamic_future (torch.Tensor): The dense tensor to populate.
            file_path (str): Path to the HDF5 file.
            target_start_time (int): Start time index of the target window.
            target_end_time (int): End time index of the target window.
            feature_offset_future (int): Offset for feature indices in the dense tensor.
            node_idx_to_batch_idx (dict[int, int]): Mapping from global node indices to batch indices.
        Returns:
            torch.Tensor: The populated dense tensor with future dynamic features.
        """
        with h5py.File(file_path, "r") as f:
            # Now the same but for the future
            future_feature_idxs = self.future_features_idxs[file_path]

            start_index = self.dynamic_data_indices[file_path][target_start_time, 0]
            end_index = self.dynamic_data_indices[file_path][target_end_time, 1]
            relevant_data_future = f["data"][start_index:end_index]  # type: ignore
            assert isinstance(relevant_data_future, np.ndarray)
            # Get only the relevant features
            relevant_data_future = relevant_data_future[
                np.isin(relevant_data_future[:, 2], future_feature_idxs)
            ]
            # Overwrite the feature_idx column to be 0, 1, ..., len(future_feature_idxs) - 1
            relevant_data_future[:, 2] = np.searchsorted(
                future_feature_idxs, relevant_data_future[:, 2]
            )

        if relevant_data_future.shape[0] > 0:
            node_idx, time_idx, feature_idx, values = (
                relevant_data_future[:, 0].astype(int),
                relevant_data_future[:, 1].astype(int) - target_start_time,
                # Add offset to feature index (each file has its own feature set
                # so we need to offset them to avoid collisions)
                relevant_data_future[:, 2].astype(int) + feature_offset_future,
                relevant_data_future[:, 3],
            )
            del relevant_data_future

            # Only populate features for nodes in the current batch
            batch_mask = np.isin(node_idx, list(node_idx_to_batch_idx.keys()))
            if batch_mask.sum() > 0:
                batch_node_idx = node_idx[batch_mask]
                batch_time_idx = time_idx[batch_mask]
                batch_feature_idx = feature_idx[batch_mask]
                batch_values = values[batch_mask]

                # Convert global node indices to batch indices
                batch_node_indices = [
                    node_idx_to_batch_idx[idx] for idx in batch_node_idx
                ]

                # Populate the dense tensor for the window
                x_dynamic_future[
                    batch_node_indices, batch_time_idx, batch_feature_idx
                ] = torch.from_numpy(batch_values).float()

        return x_dynamic_future

    def _get_weather_features(
        self,
        x_weather: torch.Tensor,
        time_start: int | np.ndarray,
        time_end: int | np.ndarray,
        node_indices: list[int],
    ) -> torch.Tensor:
        """
        Fetches and populates weather features for the given time window and node batch.

        Args:
            x_weather (torch.Tensor): The dense tensor to populate with weather features.
            time_start (int | np.ndarray): Start time index.
            time_end (int | np.ndarray): End time index.
            node_indices (list[int]): Global node indices for the current batch.

        Returns:
            torch.Tensor: The populated tensor with weather features.
        """
        # Ensure time indices are integers
        time_start_int = int(np.asarray(time_start).item())
        time_end_int = int(np.asarray(time_end).item())

        # Get the relevant node IDs (convert from indices back to node IDs)
        node_ids = [self.id_to_node[idx] for idx in node_indices]

        for feature_idx, feature_name in enumerate(self.weather_features):
            weather_df = self.weather_data[feature_name]

            # Filter by time range once
            time_filtered = weather_df[
                (weather_df["time_idx"] >= time_start_int)
                & (weather_df["time_idx"] <= time_end_int)
            ]

            if time_filtered.empty:
                continue

            station_col = self.station_col_map[feature_name]
            value_col = self.value_col_map[feature_name]

            # Collect all relevant station names for this batch
            relevant_stations = []
            node_to_batch_idx = {}

            for batch_idx, node_id in enumerate(node_ids):
                if self.weather_stations and node_id in self.weather_stations:
                    station_name = self.weather_stations[node_id][station_col]
                    relevant_stations.append(station_name)
                    node_to_batch_idx[node_id] = batch_idx

            if not relevant_stations:
                continue

            # Filter for all relevant stations at once using isin()
            relevant_data = time_filtered[
                time_filtered[station_col].isin(relevant_stations)
            ]

            if relevant_data.empty:
                continue

            # Extract arrays for faster operations
            station_names = relevant_data[station_col].values
            time_indices = relevant_data["time_idx"].values
            values = relevant_data[value_col].values

            # Create a reverse mapping from station to nodes for efficient lookup
            station_to_nodes = {}
            for node_id in node_ids:
                if self.weather_stations and node_id in self.weather_stations:
                    station_name = self.weather_stations[node_id][station_col]
                    if station_name not in station_to_nodes:
                        station_to_nodes[station_name] = []
                    station_to_nodes[station_name].append(node_id)

            # Collect all indices and values for vectorized assignment
            batch_indices = []
            time_offsets = []
            tensor_values = []

            for i in range(len(station_names)):
                station_name = station_names[i]
                time_idx = time_indices[i]
                value = values[i]

                time_offset = int(time_idx) - time_start_int
                if 0 <= time_offset < self.total_timesteps_window:
                    # Collect indices for all nodes that use this station
                    if station_name in station_to_nodes:
                        for node_id in station_to_nodes[station_name]:
                            batch_idx = node_to_batch_idx[node_id]
                            batch_indices.append(batch_idx)
                            time_offsets.append(time_offset)
                            tensor_values.append(float(value))

            # Vectorized tensor assignment (much faster than individual assignments)
            if batch_indices:
                x_weather[batch_indices, time_offsets, feature_idx] = torch.tensor(
                    tensor_values, dtype=x_weather.dtype, device=x_weather.device
                )

        return x_weather

    def __getitem__(self, idx) -> Data:
        """
        Returns a single sample, which may be a subset of nodes for a specific
        time window if node_batch_size is set.
        """
        time_idx, node_indices = self._get_time_and_node_indices(idx)
        # Past relevant time indices
        input_start_time = time_idx
        input_end_time = input_start_time + self.in_timesteps - 1
        # Future relevant time indices
        target_start_time = input_end_time
        target_end_time = target_start_time + self.out_timesteps - 1

        # Initialize a dense tensor for the dynamic features of this window
        # Only allocate for the nodes we actually need (batch size, not all nodes)
        batch_size = len(node_indices)
        # Create a mapping from global node index to batch index
        node_idx_to_batch_idx = {node_idx: i for i, node_idx in enumerate(node_indices)}

        x_dynamic_past = torch.zeros(
            (batch_size, self.in_timesteps, self.num_dynamic_features)
        )
        # The same for the dynamic future features
        x_dynamic_future = torch.zeros(
            (batch_size, self.out_timesteps, self.num_dynamic_future_features)
        )

        # Weather features tensor (for both past and future since weather is always available)
        x_weather = (
            torch.zeros(
                (batch_size, self.total_timesteps_window, self.num_weather_features)
            )
            if self.include_weather
            else None
        )

        feature_offset_past, feature_offset_future = 0, 0
        for i, file_path in enumerate(self.dynamic_data_paths):
            # print(f"Loading dynamic features from {file_path}...")
            # Get the number of features for the current file
            num_features_in_file = self.dynamic_feature_dims[i]
            if num_features_in_file == 0:
                continue

            x_dynamic_past = self._get_dynamic_past_features(
                x_dynamic_past,
                file_path,
                input_start_time,
                input_end_time,
                feature_offset_past,
                node_idx_to_batch_idx,
            )
            # Update the feature offset for the next file
            feature_offset_past += num_features_in_file

            num_future_features_in_file = len(
                self.future_features_idxs.get(file_path, [])
            )
            if num_future_features_in_file == 0:
                continue

            x_dynamic_future = self._get_dynamic_future_features(
                x_dynamic_future,
                file_path,
                target_start_time,
                target_end_time,
                feature_offset_future,
                node_idx_to_batch_idx,
            )
            # Update the feature offset for the next file
            feature_offset_future += num_future_features_in_file

        y = torch.zeros((batch_size, self.out_timesteps))

        with h5py.File(self.target_data_path, "r") as f:
            start_index = self.target_data_indices[self.target_data_path][
                target_start_time, 0
            ]
            end_index = self.target_data_indices[self.target_data_path][
                target_end_time, 1
            ]
            target_relevant_data = f["data"][start_index:end_index]  # type: ignore
            assert isinstance(target_relevant_data, np.ndarray)

        # print("Found", target_relevant_data.shape[0], "relevant target entries.\n")

        if target_relevant_data.shape[0] > 0:
            target_node_idx, target_time_idx, target_values = (
                target_relevant_data[:, 0].astype(int),
                target_relevant_data[:, 1].astype(int) - target_start_time,
                # target_relevant_data[:, 2].astype(int), # feature_idx is always 0
                target_relevant_data[:, 3],
            )

            # Only populate targets for nodes in the current batch
            batch_mask = np.isin(target_node_idx, list(node_idx_to_batch_idx.keys()))
            if batch_mask.sum() > 0:
                batch_node_idx = target_node_idx[batch_mask]
                batch_time_idx = target_time_idx[batch_mask]
                batch_values = target_values[batch_mask]

                # Convert global node indices to batch indices
                batch_node_indices = [
                    node_idx_to_batch_idx[idx] for idx in batch_node_idx
                ]

                y[batch_node_indices, batch_time_idx] = torch.from_numpy(
                    batch_values
                ).float()

        # Process weather features if enabled
        if (
            self.include_weather
            and x_weather is not None
            and self.weather_stations is not None
        ):
            # time_idx should already be an int from _get_time_and_node_indices
            x_weather = self._get_weather_features(
                x_weather=x_weather,
                time_start=time_idx,
                time_end=time_idx + self.total_timesteps_window - 1,
                node_indices=node_indices,
            )

        # Use torch_geometric's subgraph utility to get the edges for the
        # node subset and automatically relabel the node indices from 0
        edge_index_sub, _ = subgraph(
            subset=node_indices,
            edge_index=self.edge_index,
            relabel_nodes=True,
            num_nodes=self.num_nodes,
        )

        # Create the data object with weather features if available
        data_dict = {
            "x_static": self.static_features[node_indices],
            "x_dynamic_past": x_dynamic_past,
            "x_dynamic_future": x_dynamic_future,
            "y": y,
            "edge_index": edge_index_sub,
            "num_nodes": self.num_nodes,
        }

        if self.include_weather and x_weather is not None:
            data_dict["x_weather"] = x_weather

        return Data(**data_dict)


if __name__ == "__main__":
    import time

    from torch_geometric.loader import DataLoader

    from training_utils import pad_collate

    # Example usage
    processed_data_dir = "data/datos_procesados"
    sparse_data_dir = f"{processed_data_dir}/sparse_datasets"
    graph_filename = f"{processed_data_dir}/conectivity_graph/road_graph.pickle"
    static_data_filename = f"{sparse_data_dir}/static_dataset.h5"
    dynamic_data_filenames = [
        f"{sparse_data_dir}/coi_points_dataset.h5",
        f"{sparse_data_dir}/coi_tracts_dataset.h5",
        f"{sparse_data_dir}/irregularities_dataset.h5",
        f"{sparse_data_dir}/events_dataset.h5",
        f"{sparse_data_dir}/incidents_dataset.h5",
    ]
    metadata_filenames = [
        f"{sparse_data_dir}/coi_points_dataset_metadata.pkl",
        f"{sparse_data_dir}/coi_tracts_dataset_metadata.pkl",
        f"{sparse_data_dir}/irregularities_dataset_metadata.pkl",
        f"{sparse_data_dir}/events_dataset_metadata.pkl",
        f"{sparse_data_dir}/incidents_dataset_metadata.pkl",
    ]
    target_data_filename = f"{sparse_data_dir}/jams_dataset.h5"
    in_timesteps = 150
    out_timesteps = 50

    print("Initializing dataset and dataloader...")
    traffic_dataset = TrafficDataset(
        graph_path=graph_filename,
        static_data_path=static_data_filename,
        dynamic_data_paths=dynamic_data_filenames,
        metadata_paths=metadata_filenames,
        target_data_path=target_data_filename,
        in_timesteps=in_timesteps,
        out_timesteps=out_timesteps,
        future_features=[
            "coi_puntos",
            "coi_tramo",
            "AGLOMERACION",
            "EMERGENCIA",
            "MANIFESTACION",
        ],
        node_batch_size=500,
        weather_data_dir=f"{processed_data_dir}/sparse_datasets",
        include_weather=True,
        neighbor_features_path=f"{processed_data_dir}/neighbor_features.parquet",
    )
    print("Dataset initialized.")
    print(f"Total samples in dataset: {len(traffic_dataset)}")

    batch_size = 16
    traffic_dataloader = DataLoader(
        traffic_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=pad_collate,
        # persistent_workers=True,  # Keep workers alive between epochs, needs num_workers > 0
    )

    print(f"\nDataset size: {len(traffic_dataset)} samples")
    print(f"Number of batches: {len(traffic_dataloader)}")
    print("-" * 40)
    print("-" * 40)

    # Initialize prev_end before the loop
    prev_end = time.perf_counter()

    # Time the dataloader over multiple batches
    for step, batch in enumerate(traffic_dataloader):
        iter_start = time.perf_counter()
        data_wait_s = iter_start - prev_end
        print(f"Batch {step}: dataloader wait = {data_wait_s * 1000:.1f} ms")
        prev_end = time.perf_counter()
        if step >= 10:
            break
