import os
import pickle as pkl
import re
from datetime import timedelta

import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class GNNTimeSeriesDataset(Dataset):
    def __init__(self, loader, base_graph, samples, num_neighbors=10, history_length=8):
        """
        A PyTorch Dataset that wraps the GraphTimeSeriesLoader.

        Args:
            loader (GraphTimeSeriesLoader): The initialized data loader.
            base_graph (nx.DiGraph): The base graph, used for neighborhood sampling.
            samples (list): A list of training samples, e.g., tuples of (node_id, start_time_str).
            num_neighbors (int): The number of neighbors to sample for each root node.
            history_length (int): The number of 15-minute intervals to fetch for the time window.
        """
        self.loader = loader
        self.base_graph = base_graph
        self.samples = samples
        self.num_neighbors = num_neighbors
        self.history_length = history_length

    def __len__(self):
        """Returns the total number of samples."""
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Fetches one training sample, processes it, and returns it as tensors.
        """
        # Get the root node and start time for this sample
        root_node, start_time = self.samples[idx]

        # Perform neighborhood sampling
        neighbors = list(
            self.base_graph.predecessors(root_node)
        )  # Or successors, or both
        if len(neighbors) > self.num_neighbors:
            # If too many neighbors, sample a fixed number
            neighbors = np.random.choice(
                neighbors, self.num_neighbors, replace=False
            ).tolist()

        subgraph_nodes = [root_node] + neighbors

        # Use the loader to get data for the subgraph
        # The time window is defined by the start_time and the desired history length
        time_window = [(start_time, self.history_length)]

        static_df, temporal_df = self.loader.load_data(
            node_ids=subgraph_nodes, time_windows=time_window
        )

        # TODO: The output must be converted into the tensor format that the GNN needs
        # This part is highly specific to the GNN library (PyG, DGL, etc.)
        return static_df, temporal_df


class GraphTimeSeriesLoader:
    """
    A DataLoader for graph time series data from multiple sources.

    This class is designed to efficiently load node features for a Graph Neural Network (GNN)
    on-demand. It handles static node attributes from a NetworkX graph and temporal
    features from both in-memory pandas DataFrames and large, time-partitioned
    Parquet files.

    Attributes:
        base_graph (nx.DiGraph): A NetworkX graph containing static node attributes.
        events_df (pd.DataFrame): DataFrame with event data.
        incidents_df (pd.DataFrame): DataFrame with incident data.
        road_state_df (pd.DataFrame): DataFrame with road state information over time windows.
        irregularities_files (list): A list of tuples (start_date, end_date, filepath) for irregularity data.
        coi_points_files (list): A list of tuples for COI points data.
        coi_tracts_files (list): A list of tuples for COI tracts data.
    """

    def __init__(
        self,
        graph_pkl_filename: str,
        buildings_csv_filename: str,
        bus_stops_parquet_filename: str,
        events_csv_filename: str,
        incidents_csv_filename: str,
        road_state_parquet_filename: str,
        irregularities_filenames: list[str],
        coi_points_filenames: list[str],
        coi_tracts_filenames: list[str],
    ):
        """
        Initializes the GraphTimeSeriesLoader with all data sources.

        Args:
            base_graph (nx.DiGraph): The base graph with node attributes.
            events_df (pd.DataFrame): DataFrame containing event data. Must have 'road_id' and 'timeframe' columns.
            incidents_df (pd.DataFrame): DataFrame with incident data. Must have 'road_id' and 'timeframe' columns.
            road_state_df (pd.DataFrame): DataFrame with road state data. Must have 'road_id',
                                          'timeframe_start', and 'timeframe_end' columns.
            irregularities_filenames (list): List of file paths for irregularity parquet files.
            coi_points_filenames (list): List of file paths for COI points parquet files.
            coi_tracts_filenames (list): List of file paths for COI tracts parquet files.
        """
        with open(graph_pkl_filename, "rb") as f:
            base_graph = pkl.load(f)
        all_nodes = list(base_graph.nodes(data=False))
        all_nodes_df = pd.DataFrame(all_nodes, columns=["road_id"])

        # Many nodes are missing in the buildings dataframe, so we join on road_id
        # and fill missing values with 0
        buildings_df = pd.read_csv(buildings_csv_filename)
        buildings_df = buildings_df.merge(
            all_nodes_df, on="road_id", how="right"
        ).fillna(0)
        # We do the same with the bus stops dataframe
        bus_stops_df = pd.read_parquet(bus_stops_parquet_filename)
        bus_stops_df = bus_stops_df.merge(
            all_nodes_df, on="road_id", how="right"
        ).fillna(0)

        self.base_graph = base_graph
        # Update the base graph nodes with the buildings data
        self._add_node_attributes(buildings_df, node_id_col="road_id")
        # Update the base graph nodes with the bus stops data
        self._add_node_attributes(bus_stops_df, node_id_col="road_id")

        # Load temporal data from CSV and Parquet files
        self.events_df = pd.read_csv(events_csv_filename)
        self.incidents_df = pd.read_csv(incidents_csv_filename)
        self.road_state_df = pd.read_parquet(road_state_parquet_filename)

        # Ensure time columns are in datetime format for efficient filtering
        self.events_df["timeframe"] = pd.to_datetime(self.events_df["timeframe"])
        self.incidents_df["timeframe"] = pd.to_datetime(self.incidents_df["timeframe"])
        self.road_state_df["timeframe_start"] = pd.to_datetime(
            self.road_state_df["timeframe_start"]
        )
        self.road_state_df["timeframe_end"] = pd.to_datetime(
            self.road_state_df["timeframe_end"]
        )

        # Pre-process file lists to extract date ranges for quick lookups
        self.irregularities_files = self._preprocess_filenames(
            irregularities_filenames,
            r"irregs_proc_(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})\.parquet",
        )
        self.coi_points_files = self._preprocess_filenames(
            coi_points_filenames,
            r"coi_puntos_dataset_(\d{8})_(\d{8})\.parquet",
            date_format="%Y%m%d",
        )
        self.coi_tracts_files = self._preprocess_filenames(
            coi_tracts_filenames,
            r"coi_tramo_dataset_(\d{8})_(\d{8})\.parquet",
            date_format="%Y%m%d",
        )

    def _add_node_attributes(self, df: pd.DataFrame, node_id_col: str):
        """
        Adds static node attributes to a DataFrame based on the base graph.

        Args:
            df (pd.DataFrame): The DataFrame to which attributes will be added.
            node_id_col (str): The name of the column containing node IDs in the DataFrame.

        Returns:
            pd.DataFrame: The DataFrame with static attributes added.
        """
        if node_id_col not in df.columns:
            raise ValueError(f"Column '{node_id_col}' not found in DataFrame.")

        # Add static attributes to the base graph
        for _, row in df.iterrows():
            node_id = row[node_id_col]
            if node_id in self.base_graph.nodes:
                self.base_graph.nodes[node_id].update(row.to_dict())

    def _preprocess_filenames(self, filenames, pattern, date_format="%Y-%m-%d"):
        """
        Parses filenames to extract start and end dates.

        Args:
            filenames (list): A list of file paths.
            pattern (str): A regex pattern to extract date strings from filenames.
            date_format (str): The format of the date strings in the filenames.

        Returns:
            list: A list of tuples, where each tuple contains (start_date, end_date, filepath).
                  Returns an empty list if a file's dates cannot be parsed.
        """
        processed_files = []
        for f in filenames:
            match = re.search(pattern, os.path.basename(f))
            if match:
                start_date_str, end_date_str = match.groups()
                start_date = pd.to_datetime(start_date_str, format=date_format)
                # Set end_date to the end of the day for correct range comparisons
                end_date = pd.to_datetime(end_date_str, format=date_format) + timedelta(
                    days=1, microseconds=-1
                )
                processed_files.append((start_date, end_date, f))
            else:
                print(f"Warning: Could not parse dates from filename: {f}")
        return processed_files

    def _find_relevant_files(self, file_list, query_start_date, query_end_date):
        """
        Finds files whose date ranges overlap with the query time window.

        Args:
            file_list (list): A preprocessed list of (start_date, end_date, filepath) tuples.
            query_start_date (pd.Timestamp): The start of the query time window.
            query_end_date (pd.Timestamp): The end of the query time window.

        Returns:
            list: A list of filepaths relevant to the query time window.
        """
        relevant_files = []
        for start_date, end_date, filepath in file_list:
            # An overlap occurs if (FileStart <= QueryEnd) and (FileEnd >= QueryStart)
            if start_date <= query_end_date and end_date >= query_start_date:
                relevant_files.append(filepath)
        return relevant_files

    def _load_data_from_parquets(
        self, filepaths, node_ids, start_date, end_date, node_id_col, time_col
    ):
        """
        Loads and filters data from a list of parquet files.

        Args:
            filepaths (list): List of parquet file paths to load.
            node_ids (list): List of node IDs to filter for.
            start_date (pd.Timestamp): The start of the time window.
            end_date (pd.Timestamp): The end of the time window.
            node_id_col (str): The name of the node ID column in the parquet files.
            time_col (str): The name of the timeframe column in the parquet files.

        Returns:
            pd.DataFrame: A DataFrame containing the filtered data from all files.
        """
        if not filepaths:
            return pd.DataFrame()

        df_list = []
        for f in filepaths:
            try:
                df = pd.read_parquet(f)
                df[time_col] = pd.to_datetime(df[time_col])

                # Filter by node IDs and then by the time window
                filtered_df = df[
                    df[node_id_col].isin(node_ids)
                    & (df[time_col] >= start_date)
                    & (df[time_col] <= end_date)
                ]
                if not filtered_df.empty:
                    df_list.append(filtered_df)
            except Exception as e:
                print(f"Warning: Could not read or process file {f}. Error: {e}")

        if not df_list:
            return pd.DataFrame()

        return pd.concat(df_list, ignore_index=True)

    def load_data(self, node_ids, time_windows):
        """
        Loads static and temporal data for specified nodes and time windows.

        Args:
            node_ids (list): A list of node IDs to fetch data for.
            time_windows (list): A list of tuples, where each tuple is
                                 (timeframe_start, num_15_min_periods).
                                 'timeframe_start' is a string like 'YYYY-MM-DD HH:MM:SS'.
                                 'num_15_min_periods' is the number of 15-minute intervals.

        Returns:
            tuple: A tuple containing two DataFrames:
                   - static_df (pd.DataFrame): Static attributes for the requested nodes.
                   - temporal_df (pd.DataFrame): All temporal features for the nodes
                                                 within the specified time windows.
        """
        print(f"\n--- Starting data load for {len(node_ids)} nodes ---")
        # Get static data from the base graph
        static_data = []
        for node_id in node_ids:
            if self.base_graph.has_node(node_id):
                # Add node_id to the dictionary of attributes
                attrs = self.base_graph.nodes[node_id].copy()
                attrs["node_id"] = node_id
                static_data.append(attrs)

        static_df = pd.DataFrame(static_data)
        if "node_id" in static_df.columns:
            static_df = static_df.set_index("node_id")

        # Get temporal data from all sources
        all_temporal_dfs = []

        for start_str, num_15_min_periods in time_windows:
            start_date = pd.to_datetime(start_str)
            # The last timestamp in the window is (n-1)*15 minutes after the start
            # This makes the window inclusive of `num_15_min_periods` points
            duration_minutes = (num_15_min_periods - 1) * 15
            end_date = start_date + timedelta(minutes=duration_minutes)

            # Filter in-memory DataFrames
            # Events
            events_temporal = self.events_df[
                self.events_df["road_id"].isin(node_ids)
                & (self.events_df["timeframe"] >= start_date)
                & (self.events_df["timeframe"] <= end_date)
            ].copy()
            if not events_temporal.empty:
                events_temporal["source"] = "events"
                all_temporal_dfs.append(events_temporal)

            # Incidents
            incidents_temporal = self.incidents_df[
                self.incidents_df["road_id"].isin(node_ids)
                & (self.incidents_df["timeframe"] >= start_date)
                & (self.incidents_df["timeframe"] <= end_date)
            ].copy()
            if not incidents_temporal.empty:
                incidents_temporal["source"] = "incidents"
                all_temporal_dfs.append(incidents_temporal)

            # Road State (overlap condition)
            road_state_temporal = self.road_state_df[
                self.road_state_df["road_id"].isin(node_ids)
                & (self.road_state_df["timeframe_start"] <= end_date)
                & (self.road_state_df["timeframe_end"] >= start_date)
            ].copy()
            if not road_state_temporal.empty:
                road_state_temporal["source"] = "road_state"
                all_temporal_dfs.append(road_state_temporal)

            # Load data from Parquet files
            # Irregularities
            irreg_files = self._find_relevant_files(
                self.irregularities_files, start_date, end_date
            )
            irreg_df = self._load_data_from_parquets(
                irreg_files,
                node_ids,
                start_date,
                end_date,
                node_id_col="id_mvi",
                time_col="timeframe",
            )
            if not irreg_df.empty:
                irreg_df["source"] = "irregularities"
                # Standardize column names for concatenation
                irreg_df = irreg_df.rename(columns={"id_mvi": "road_id"})
                all_temporal_dfs.append(irreg_df)

            # COI Points
            coi_points_files = self._find_relevant_files(
                self.coi_points_files, start_date, end_date
            )
            coi_points_df = self._load_data_from_parquets(
                coi_points_files,
                node_ids,
                start_date,
                end_date,
                node_id_col="road_id",
                time_col="timeframe",
            )
            if not coi_points_df.empty:
                coi_points_df["source"] = "coi_points"
                all_temporal_dfs.append(coi_points_df)

            # COI Tracts
            coi_tracts_files = self._find_relevant_files(
                self.coi_tracts_files, start_date, end_date
            )
            coi_tracts_df = self._load_data_from_parquets(
                coi_tracts_files,
                node_ids,
                start_date,
                end_date,
                node_id_col="road_id",
                time_col="timeframe",
            )
            if not coi_tracts_df.empty:
                coi_tracts_df["source"] = "coi_tracts"
                all_temporal_dfs.append(coi_tracts_df)

        if not all_temporal_dfs:
            return static_df, pd.DataFrame()

        temporal_df = pd.concat(all_temporal_dfs, ignore_index=True)

        if "timeframe" in temporal_df.columns:
            temporal_df = temporal_df.sort_values(by="timeframe").reset_index(drop=True)

        print(
            f"--- Load complete. Found {len(static_df)} static records and {len(temporal_df)} temporal records. ---"
        )
        return static_df, temporal_df
