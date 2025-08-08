import math
import pickle

import h5py
import numpy as np
import torch
from scipy.sparse import coo_matrix
from torch.utils.data import Dataset
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
        target_data_path: str,
        in_timesteps: int,
        out_timesteps: int,
        node_batch_size: int | None = None,
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
        self.in_timesteps = in_timesteps
        self.out_timesteps = out_timesteps
        self.total_timesteps_window = in_timesteps + out_timesteps
        self.dynamic_data_paths = dynamic_data_paths
        self.target_data_path = target_data_path
        self.node_batch_size = node_batch_size

        # Load static data and graph
        # These are assumed to be small enough to fit in memory
        self.edge_index = self._load_graph(graph_path)
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
        self.dynamic_data_indices, self.target_data_indices = (
            self._precompute_time_indices()
        )

        print(f"Dataset initialized.")
        print(f"Number of nodes: {self.num_nodes}")
        print(f"Node batch size: {self.node_batch_size or 'All nodes'}")
        print(f"Number of node batches: {self.num_node_batches}")
        print(f"Static features per node: {self.num_static_features}")
        print(f"Dynamic features per node: {self.num_dynamic_features}")
        print(f"Total time steps in data: {self.total_timesteps_in_data}")

    def _load_graph(self, graph_path):
        """Loads the graph from a pickle file."""
        with open(graph_path, "rb") as f:
            nx_graph = pickle.load(f)
        return from_networkx(nx_graph).edge_index

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
            print(f"Precomputing time indices for {file_path}...")
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
        print("Time indices precomputation completed.")
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

    def _get_time_and_node_indices(self, idx: int) -> tuple[int, torch.Tensor]:
        """
        Calculates the time window index and the tensor of node indices
        for a given flat dataset index.
        """
        # If not batching by nodes, idx is just the time index and we use all nodes
        if self.node_batch_size is None:
            return idx, torch.arange(self.num_nodes)

        # If batching by nodes, decode the flat index `idx`
        time_idx = idx // self.num_node_batches
        node_batch_idx = idx % self.num_node_batches

        start_node = node_batch_idx * self.node_batch_size
        end_node = min((node_batch_idx + 1) * self.node_batch_size, self.num_nodes)
        node_indices = torch.arange(start_node, end_node)

        return time_idx, node_indices

    def __getitem__(self, idx):
        """
        Returns a single sample, which may be a subset of nodes for a specific
        time window if node_batch_size is set.
        """
        time_idx, node_indices = self._get_time_and_node_indices(idx)
        input_start_time = time_idx
        input_end_time = input_start_time + self.in_timesteps - 1

        # Initialize a dense tensor for the dynamic features of this window
        x_dynamic = torch.zeros(
            (self.num_nodes, self.in_timesteps, self.num_dynamic_features)
        )

        feature_offset = 0
        for i, file_path in enumerate(self.dynamic_data_paths):
            print(f"Loading dynamic features from {file_path}...")
            # Get the number of features for the current file
            num_features_in_file = self.dynamic_feature_dims[i]
            if num_features_in_file == 0:
                continue

            with h5py.File(file_path, "r") as f:
                start_index = self.dynamic_data_indices[file_path][input_start_time, 0]
                end_index = self.dynamic_data_indices[file_path][input_end_time, 1]
                relevant_data = f["data"][start_index:end_index]  # type: ignore
                assert isinstance(relevant_data, np.ndarray)

            print("Found", relevant_data.shape[0], "relevant entries.\n")

            if relevant_data.shape[0] > 0:
                node_idx, time_idx, feature_idx, values = (
                    relevant_data[:, 0].astype(int),
                    relevant_data[:, 1].astype(int) - input_start_time,
                    # Add offset to feature index (each file has its own feature set
                    # so we need to offset them to avoid collisions)
                    relevant_data[:, 2].astype(int) + feature_offset,
                    relevant_data[:, 3],
                )
                del relevant_data  # Free memory
                # Populate the dense tensor for the window
                x_dynamic[node_idx, time_idx, feature_idx] = torch.from_numpy(
                    values
                ).float()
            # Update the feature offset for the next file
            feature_offset += num_features_in_file

        # Now we load the target variable
        target_start_time = input_end_time
        target_end_time = target_start_time + self.out_timesteps - 1

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

        print("Found", target_relevant_data.shape[0], "relevant target entries.\n")

        if target_relevant_data.shape[0] > 0:
            node_idx, time_idx, values = (
                target_relevant_data[:, 0].astype(int),
                target_relevant_data[:, 1].astype(int) - target_start_time,
                # target_relevant_data[:, 2].astype(int), # feature_idx is always 0
                target_relevant_data[:, 3],
            )
            y[node_idx, time_idx] = torch.from_numpy(values).float()

        print(
            "Dynamic features loaded for time window:",
            input_start_time,
            "to",
            input_end_time,
        )
        print(
            "Target variable loaded for time window:",
            target_start_time,
            "to",
            target_end_time,
        )
        print("-" * 40)
        if self.node_batch_size is not None:
            # Use torch_geometric's subgraph utility to get the edges for the
            # node subset and automatically relabel the node indices from 0
            edge_index_sub, _ = subgraph(
                subset=node_indices,
                edge_index=self.edge_index,
                relabel_nodes=True,
                num_nodes=self.num_nodes,
            )
            return {
                "x_static": self.static_features[node_indices],
                "x_dynamic": x_dynamic[node_indices],
                "y": y[node_indices],
                "edge_index": edge_index_sub,
            }
        else:
            return {
                "x_static": self.static_features,
                "x_dynamic": x_dynamic,
                "y": y,
                "edge_index": self.edge_index,
            }
