import pickle

import h5py
import networkx as nx
import numpy as np
import torch
from scipy.sparse import coo_matrix
from torch.utils.data import DataLoader, Dataset
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
    ):
        """
        Args:
            graph_path (str): Path to the NetworkX graph file (.pkl).
            static_data_path (str): Path to the HDF5 file with static node features.
            dynamic_data_paths (list of str): List of paths to HDF5 files with dynamic features.
            in_timesteps (int): Number of time steps in the input sequence.
            out_timesteps (int): Number of time steps in the output sequence.
        """
        super().__init__()
        self.in_timesteps = in_timesteps
        self.out_timesteps = out_timesteps
        self.total_timesteps_window = in_timesteps + out_timesteps
        self.dynamic_data_paths = dynamic_data_paths
        self.target_data_path = target_data_path

        # Load static data and graph
        # These are assumed to be small enough to fit in memory
        self.edge_index = self._load_graph(graph_path)
        self.static_features = self._load_static_features(static_data_path)
        self.num_nodes = self.static_features.shape[0]
        self.num_static_features = self.static_features.shape[1]

        # Open dynamic data files and get metadata
        self.total_timesteps_in_data = 0
        self.dynamic_feature_dims = []
        for file_path in self.dynamic_data_paths + [self.target_data_path]:
            with h5py.File(file_path, "r") as hf:
                # Assuming all files have the same number of nodes and time steps
                if hf["data"].shape[0] > 0:
                    self.total_timesteps_in_data = max(
                        self.total_timesteps_in_data, int(hf["data"][:, 1].max() + 1)
                    )
                    if file_path != self.target_data_path:
                        # Only count dynamic feature files
                        self.dynamic_feature_dims.append(
                            int(hf["data"][:, 2].max() + 1)
                        )
                    else:
                        # Check that the target data has just one feature_idx
                        if hf["data"][:, 2].max() != 0:
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

        print(f"Dataset initialized.")
        print(f"Number of nodes: {self.num_nodes}")
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
            data = f["data"][:]

        node_idx, feature_idx, values = data[:, 0], data[:, 1], data[:, 2]
        num_nodes = int(node_idx.max() + 1)
        num_features = int(feature_idx.max() + 1)

        sparse_matrix = coo_matrix(
            (values, (node_idx, feature_idx)), shape=(num_nodes, num_features)
        )
        del data  # Free memory

        return torch.from_numpy(sparse_matrix.toarray()).float()

    def __len__(self):
        """Returns the total number of samples (sliding windows) in the dataset."""
        if self.total_timesteps_in_data < self.total_timesteps_window:
            return 0
        return self.total_timesteps_in_data - self.total_timesteps_window + 1

    def __getitem__(self, idx):
        """
        Returns a single sample from the dataset by fetching the required
        time window from the HDF5 files on-the-fly.
        """
        input_start_time = idx
        input_end_time = input_start_time + self.in_timesteps

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
                # Reading the entire time_idx column can be slow
                all_time_idx = f["data"][:, 1]
                # Find the rows that fall within our time window
                # TODO: If the data is sorted by time, np.searchsorted can make this much faster
                rows_in_window_mask = (all_time_idx >= input_start_time) & (
                    all_time_idx < input_end_time
                )
                relevant_data = f["data"][rows_in_window_mask]
                del all_time_idx, rows_in_window_mask  # Free memory

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
        target_end_time = target_start_time + self.out_timesteps

        y = torch.zeros((self.num_nodes, self.out_timesteps))

        with h5py.File(file_path, "r") as f:
            target_all_time_idx = f["data"][:, 1]
            target_rows_mask = (target_all_time_idx >= target_start_time) & (
                target_all_time_idx < target_end_time
            )
            target_relevant_data = f["data"][target_rows_mask]
            del target_all_time_idx, target_rows_mask  # Free memory

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

        return {
            "x_static": self.static_features,
            "x_dynamic": x_dynamic,
            "y": y,
            "edge_index": self.edge_index,
        }

    # TODO: This would be a safe way to ensure files are closed, but it returns
    # an error as stated in this GitHub issue: https://github.com/h5py/h5py/issues/2237
    # def __del__(self):
    #     """Ensures HDF5 files are closed when the dataset object is deleted."""
    #     for f in self.dynamic_files:
    #         f.close()
