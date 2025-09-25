import math
import pickle

import h5py
import networkx as nx
import numpy as np
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
        node_batch_size: int | None = None,
        node_seed: str = "0_FT.5"
    ):
        """
        Args:
            graph_path (str): Path to the NetworkX graph file (.pkl).
            static_data_path (str): Path to the HDF5 file with static node features.
            dynamic_data_paths (list of str): List of paths to HDF5 files with dynamic features.
            target_data_path (str): Path to the HDF5 file with target variable.
            in_timesteps (int): Number of time steps in the input sequence.
            out_timesteps (int): Number of time steps in the output sequence.
            node_batch_size (int | None, optional): The number of nodes per batch. If None, each sample will contain all nodes.
        """
        super().__init__()
        self.node_seed = node_seed
        self._cached_seed = None
        self._cached_nodes = None
        self._cached_edge_index_sub = None
        self.in_timesteps = in_timesteps
        self.out_timesteps = out_timesteps
        self.total_timesteps_window = in_timesteps + out_timesteps
        self.dynamic_data_paths = dynamic_data_paths
        self.target_data_path = target_data_path
        self.node_batch_size = node_batch_size
        self.future_features = future_features

        # Load static data and graph
        # These are assumed to be small enough to fit in memory
        self.graph, self.edge_index, self.node_to_id, self.id_to_node = (
            self._load_graph(graph_path)
        )
        self.static_features = self._load_static_features(static_data_path)
        self.num_nodes = self.static_features.shape[0]
        self.num_static_features = self.static_features.shape[1]

        self.num_node_batches = 1
        if self.node_batch_size is not None:
            if self.node_batch_size <= 0:
                raise ValueError("node_batch_size must be a positive integer.")
            self.num_node_batches = math.ceil(self.num_nodes / self.node_batch_size)

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

    def _get_future_dynamic_feature_idxs(
        self, metadata_paths: list[str]
    ) -> dict[str, list[int]]:
        future_features_idxs: dict[str, list[int]] = {}
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
        return future_features_idxs

    def reasign_seed(self, node_seed):
        if node_seed != self.node_seed:
            self.node_seed = node_seed

    def _load_graph(
        self, graph_path
    ) -> tuple[nx.Graph, torch.Tensor, dict[str, int], dict[int, str]]:
        """Loads the graph from a pickle file."""
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

    def _precompute_time_indices(
        self,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """
        Pre-computes the start and end row indices for every possible sample
        in each HDF5 file. This is done once at initialization to speed up
        __getitem__.
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
        Returns the total number of samples, which is the product of
        the number of possible time windows and the number of node batches.
        """
        if self.total_timesteps_in_data < self.total_timesteps_window:
            return 0
        num_time_windows = (
            self.total_timesteps_in_data - self.total_timesteps_window + 1
        )
        return num_time_windows * self.num_node_batches

    def get_roads_within_hops(self, k=15):
        # returns dict {road: distance_in_hops}
        dists = nx.single_source_shortest_path_length(self.R, self.node_seed, cutoff=k)
        return dists

    def _compute_subset_for_seed(self):
        roads = list(self.get_roads_within_hops().keys())
        node_indices = torch.tensor(sorted(self.node_to_id[r] for r in roads), dtype=torch.long)
        edge_index_sub, _ = subgraph(
            subset=node_indices, edge_index=self.edge_index,
            relabel_nodes=True, num_nodes=self.num_nodes
        )
        self._cached_seed = self.node_seed
        self._cached_nodes = node_indices
        self._cached_edge_index_sub = edge_index_sub

    def _get_time_and_node_indices(self, idx):
        if self.node_batch_size is None:
            return idx, torch.arange(self.num_nodes)
        time_idx = idx // self.num_node_batches
        if self._cached_seed != self.node_seed or self._cached_nodes is None:
            self._compute_subset_for_seed()
        return time_idx, self._cached_nodes

    def _get_dynamic_past_features(
        self,
        x_dynamic_past: torch.Tensor,
        file_path: str,
        input_start_time: int,
        input_end_time: int,
        feature_offset_past: int,
    ) -> torch.Tensor:
        with h5py.File(file_path, "r") as f:
            # We first fetch the relevant data for the past
            start_index = self.dynamic_data_indices[file_path][input_start_time, 0]
            end_index = self.dynamic_data_indices[file_path][input_end_time, 1]
            relevant_data_past = f["data"][start_index:end_index]  # type: ignore
            assert isinstance(relevant_data_past, np.ndarray)

        # print("Found", relevant_data.shape[0], "relevant entries.\n")

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
            # Populate the dense tensor for the window
            x_dynamic_past[node_idx, time_idx, feature_idx] = torch.from_numpy(
                values
            ).float()

        return x_dynamic_past

    def _get_dynamic_future_features(
        self,
        x_dynamic_future,
        file_path: str,
        target_start_time: int,
        target_end_time: int,
        feature_offset_future: int,
    ) -> torch.Tensor:
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
            # Populate the dense tensor for the window
            x_dynamic_future[node_idx, time_idx, feature_idx] = torch.from_numpy(
                values
            ).float()

        return x_dynamic_future

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
        x_dynamic_past = torch.zeros(
            (self.num_nodes, self.in_timesteps, self.num_dynamic_features)
        )
        # The same for the dynamic future features
        x_dynamic_future = torch.zeros(
            (self.num_nodes, self.out_timesteps, self.num_dynamic_future_features)
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
            )
            # Update the feature offset for the next file
            feature_offset_future += num_future_features_in_file

        y = torch.zeros((self.num_nodes, self.out_timesteps))

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
            node_idx, time_idx, values = (
                target_relevant_data[:, 0].astype(int),
                target_relevant_data[:, 1].astype(int) - target_start_time,
                # target_relevant_data[:, 2].astype(int), # feature_idx is always 0
                target_relevant_data[:, 3],
            )
            y[node_idx, time_idx] = torch.from_numpy(values).float()

        # Use torch_geometric's subgraph utility to get the edges for the
        # node subset and automatically relabel the node indices from 0
        edge_index_sub, _ = subgraph(
            subset=node_indices,
            edge_index=self.edge_index,
            relabel_nodes=True,
            num_nodes=self.num_nodes,
        )
        return Data(
            x_static=self.static_features[node_indices],
            x_dynamic_past=x_dynamic_past[node_indices],
            x_dynamic_future=x_dynamic_future[node_indices],
            y=y[node_indices],
            edge_index=edge_index_sub,
            num_nodes=self.num_nodes,
        )
