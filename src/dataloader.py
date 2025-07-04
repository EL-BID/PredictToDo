import os
import pickle as pkl
import re
import warnings
from datetime import timedelta

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class GraphTimeSeriesDataset(Dataset):
    """
    A custom PyTorch Dataset that wraps the GraphTimeSeriesLoader.

    This dataset holds the node IDs and time windows, and for each sample,
    it retrieves the corresponding graph data using the provided loader.
    """

    def __init__(
        self,
        loader: "GraphTimeSeriesLoader",
        node_ids: list[str],
        time_windows: list[tuple[str, int, int]],
    ):
        """
        Initializes the GraphDataset.

        Args:
            loader (GraphTimeSeriesLoader): An instance of the custom data loader.
            node_ids (list[str]): The list of all node IDs in the dataset.
            time_windows (list[tuple[str, int, int]]): The list of all time windows.
        """
        self.loader = loader
        self.node_ids = node_ids
        self.time_windows = time_windows

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return len(self.time_windows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Retrieves a single sample from the dataset at the given index.

        A sample consists of the data for all nodes at a specific time window.
        """
        # Get the specific time window for this sample
        time_window = [self.time_windows[idx]]

        # Use the loader to fetch the data for all nodes at this time window
        (static_df, temporal_df), jam_target = self.loader.load_data(
            self.node_ids, time_window
        )

        # Convert the pandas DataFrames to PyTorch tensors
        # TODO: The exact conversion logic will depend on the GNN model's input requirements
        static_tensor = torch.tensor(static_df.values, dtype=torch.float32)
        temporal_tensor = torch.tensor(temporal_df.values, dtype=torch.float32)
        target_tensor = torch.tensor(jam_target.values, dtype=torch.float32)

        return static_tensor, temporal_tensor, target_tensor


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
        default_speeds_df (pd.DataFrame): DataFrame with default speed and trend values for nodes.
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
        default_speeds_parquet_filename: str,
    ):
        """
        Initializes the GraphTimeSeriesLoader with all data sources.

        Args:
            base_graph (nx.DiGraph): The base graph with node attributes.

            events_df (pd.DataFrame): DataFrame containing event data. Must have 'road_id'
            and 'timeframe' columns.

            incidents_df (pd.DataFrame): DataFrame with incident data. Must have 'road_id'
            and 'timeframe' columns.

            road_state_df (pd.DataFrame): DataFrame with road state data. Must have 'road_id',
            'timeframe_start', and 'timeframe_end' columns.

            irregularities_filenames (list): List of file paths for irregularity parquet files.

            coi_points_filenames (list): List of file paths for COI points parquet files.

            coi_tracts_filenames (list): List of file paths for COI tracts parquet files.

            default_speeds_parquet_filename (str): Path to the Parquet file with default speeds
            and trends. Must have columns 'mean_trend' and 'mean_regularSpeed'. The index must
            correspond to the node IDs in the base graph.
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

        # Load the default speeds and trends, assuming index is the node ID
        self.default_speeds_df = pd.read_parquet(default_speeds_parquet_filename)

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

        self.column_names = {
            "events": self.events_df.columns.tolist(),
            "incidents": self.incidents_df.columns.tolist(),
            # For road_state we generat the list that could be returned by `_expand_road_state_data`
            "road_state": ["road_id", "timeframe"]
            + [
                col
                for col in self.road_state_df.columns
                if "timeframe" not in col and col != "road_id"
            ],
            # For these three we can load one of the files to get the columns
            "irregularities": [],
            "coi_points": [],
            "coi_tracts": [],
        }
        # Load one file from each of the last three to get their columns
        if self.irregularities_files:
            sample_irreg_df = pd.read_parquet(self.irregularities_files[0][2])
            self.column_names["irregularities"] = sample_irreg_df.columns.tolist()
        if self.coi_points_files:
            sample_coi_df = pd.read_parquet(self.coi_points_files[0][2])
            self.column_names["coi_points"] = sample_coi_df.columns.tolist()
        if self.coi_tracts_files:
            sample_coi_tracts_df = pd.read_parquet(self.coi_tracts_files[0][2])
            self.column_names["coi_tracts"] = sample_coi_tracts_df.columns.tolist()

    def _add_node_attributes(self, df: pd.DataFrame, node_id_col: str) -> None:
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

    def _preprocess_filenames(
        self, filenames: list[str], pattern: str, date_format: str = "%Y-%m-%d"
    ) -> list[tuple[pd.Timestamp, pd.Timestamp, str]]:
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

    def _find_relevant_files(
        self,
        file_list: list[tuple[pd.Timestamp, pd.Timestamp, str]],
        query_start_date: pd.Timestamp,
        query_end_date: pd.Timestamp,
    ) -> list[str]:
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
        self,
        filepaths: list[str],
        node_ids: list[str],
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        node_id_col: str,
        time_col: str,
    ) -> pd.DataFrame:
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

    def _expand_road_state_data(
        self, state_df: pd.DataFrame, all_timeframes: list[str], node_ids: list[str]
    ) -> pd.DataFrame:
        """
        Expands state data (defined by start/end) to discrete 15-min timeframes.
        """
        if state_df.empty:
            return pd.DataFrame(columns=self.column_names["road_state"])

        # Create a DataFrame of all combinations of relevant nodes and timestamps
        target_index_df = pd.MultiIndex.from_product(
            [node_ids, all_timeframes], names=["road_id", "timeframe"]
        ).to_frame(index=False)

        # Prepare for merge_asof by sorting by time
        target_index_df = target_index_df.sort_values("timeframe")
        state_df = state_df.sort_values("timeframe_start")

        # For each 15-min point, find the last state that had started
        expanded_df = pd.merge_asof(
            target_index_df,
            state_df.drop(columns="source", errors="ignore"),
            left_on="timeframe",
            right_on="timeframe_start",
            by="road_id",
            direction="backward",
        )

        # A state is valid if the timeframe is also before the state's end time
        valid_state_mask = expanded_df["timeframe"] <= expanded_df["timeframe_end"]

        # Keep only the valid rows and the columns we need
        result_df = expanded_df[valid_state_mask][
            self.column_names["road_state"]
        ].copy()

        return result_df

    def _to_snake_case(self, string: str) -> str:
        """
        Converts a string to snake_case by:
        - Always adding underscore after "mvi" or "MVI"
        - Replacing non-alphanumeric with underscore
        - Splitting camelCase
        - Converting to lowercase
        - Removing leading/trailing underscores
        - Collapsing multiple underscores

        Args:
            colnames (list): A list of column names in any case format.

        Returns:
            list: A list of column names converted to snake_case.
        """
        string = string.replace("mvi", "mvi_").replace(
            "MVI", "MVI_"
        )  # Add underscore after "mvi" or "MVI"
        string = re.sub(
            r"[^a-zA-Z0-9]+", "_", string
        )  # Replace non-alphanumeric with underscore
        string = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", string)  # Split camelCase
        string = string.lower()  # Convert to lowercase
        string = re.sub(r"_+", "_", string)  # Collapse multiple underscores
        return string.strip("_")

    def load_data(
        self, node_ids: list[str], time_windows: list[tuple[str, int, int]]
    ) -> tuple[tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        Loads static and temporal data, creating a dense DataFrame for temporal features.

        Args:
            node_ids (list): A list of node IDs to fetch data for.
            time_windows (list): A list of tuples, where each tuple is
                                 (timeframe, num_15_min_periods_forwards, num_15_min_periods_backwards).

        Returns:
            tuple: A tuple containing:
                - A tuple of DataFrames: (static_df, temporal_df)
                    - static_df (pd.DataFrame): Static attributes for the requested nodes.
                    - temporal_df (pd.DataFrame): A dense DataFrame with all temporal features,
                                                  indexed by (road_id, timeframe), with missing
                                                  values filled with 0. (except for trend, speed,
                                                  and regularSpeed which are filled with mean values
                                                  of neighboring nodes)
                - A target DataFrame: jam_target
                    - jam_target (pd.DataFrame): The target variable 'jam_level',
                                                 indexed by (road_id, timeframe).
        """
        if not node_ids:
            print("No node IDs provided. Returning empty DataFrames.")
            return (pd.DataFrame(), pd.DataFrame()), pd.DataFrame()

        print(f"\n--- Starting data load for {len(node_ids)} nodes ---")
        # Get static data from the base graph
        static_data = []
        for node_id in node_ids:
            if self.base_graph.has_node(node_id):
                attrs = self.base_graph.nodes[node_id].copy()
                attrs["node_id"] = node_id
                static_data.append(attrs)
        static_df = pd.DataFrame(static_data).set_index("node_id")

        if not time_windows:
            print(
                f"--- Load complete. Found {len(static_df)} static records. No temporal data requested. ---"
            )
            return (static_df, pd.DataFrame()), pd.DataFrame()

        # Generate all expected timeframes and the master index
        all_expected_timeframes = []
        for timeframe_str, num_forwards, num_backwards in time_windows:
            center_time = pd.to_datetime(timeframe_str)
            # Round center_time to the nearest 15 minutes
            center_time = center_time.round("15min")
            start_date = center_time - timedelta(minutes=num_backwards * 15)
            total_periods = num_backwards + num_forwards + 1
            all_expected_timeframes.extend(
                [start_date + timedelta(minutes=i * 15) for i in range(total_periods)]
            )
        all_expected_timeframes = sorted(list(set(all_expected_timeframes)))

        master_index = pd.MultiIndex.from_product(
            [node_ids, all_expected_timeframes], names=["road_id", "timeframe"]
        )
        master_temporal_df = pd.DataFrame(index=master_index)

        # Load data from all sources
        source_dfs = {}
        raw_data_fragments = {
            "events": [],
            "incidents": [],
            "road_state": [],
            "irregularities": [],
            "coi_points": [],
            "coi_tracts": [],
        }

        for timeframe_str, num_forwards, num_backwards in time_windows:
            center_time = pd.to_datetime(timeframe_str)
            start_date = center_time - timedelta(minutes=num_backwards * 15)
            end_date = center_time + timedelta(minutes=num_forwards * 15)

            # Filter in-memory DataFrames
            raw_data_fragments["events"].append(
                self.events_df[
                    self.events_df["road_id"].isin(node_ids)
                    & self.events_df["timeframe"].between(start_date, end_date)
                ]
            )
            raw_data_fragments["incidents"].append(
                self.incidents_df[
                    self.incidents_df["road_id"].isin(node_ids)
                    & self.incidents_df["timeframe"].between(start_date, end_date)
                ]
            )
            raw_data_fragments["road_state"].append(
                self.road_state_df[
                    self.road_state_df["road_id"].isin(node_ids)
                    & (self.road_state_df["timeframe_start"] <= end_date)
                    & (self.road_state_df["timeframe_end"] >= start_date)
                ]
            )

            irreg_files = self._find_relevant_files(
                self.irregularities_files, start_date, end_date
            )
            raw_data_fragments["irregularities"].append(
                self._load_data_from_parquets(
                    irreg_files, node_ids, start_date, end_date, "id_mvi", "timeframe"
                )
            )

            coi_p_files = self._find_relevant_files(
                self.coi_points_files, start_date, end_date
            )
            raw_data_fragments["coi_points"].append(
                self._load_data_from_parquets(
                    coi_p_files, node_ids, start_date, end_date, "road_id", "timeframe"
                )
            )

            coi_t_files = self._find_relevant_files(
                self.coi_tracts_files, start_date, end_date
            )
            raw_data_fragments["coi_tracts"].append(
                self._load_data_from_parquets(
                    coi_t_files, node_ids, start_date, end_date, "road_id", "timeframe"
                )
            )

        # Consolidate and process each source
        for name, fragments in raw_data_fragments.items():
            if fragments:
                source_dfs[name] = pd.concat(
                    fragments, ignore_index=True
                ).drop_duplicates()
            else:
                source_dfs[name] = pd.DataFrame()

        # Join all sources into the master DataFrame
        # Process and join point-in-time sources
        for name, df in source_dfs.items():
            if df.empty or name == "road_state":
                continue

            # Standardize ID column and set index
            if "id_mvi" in df.columns:
                df = df.rename(columns={"id_mvi": "road_id"})

            indexed_df = df.set_index(["road_id", "timeframe"])
            master_temporal_df = master_temporal_df.join(indexed_df, how="left")

        # Special handling for road_state data
        expanded_state_df = self._expand_road_state_data(
            source_dfs["road_state"], all_expected_timeframes, node_ids
        )
        if not expanded_state_df.empty:
            master_temporal_df = master_temporal_df.join(
                expanded_state_df.set_index(["road_id", "timeframe"]), how="left"
            )

        # Now we must fill missing values and complete with all the expected columns
        # First we deal with the non-zeroable columns (trend, speed and regularSpeed)
        master_temporal_df = master_temporal_df.merge(
            self.default_speeds_df, left_on="road_id", right_index=True, how="left"
        )
        fill_map = {
            "trend": "mean_trend",
            "speed": "mean_regularSpeed",
            "regularSpeed": "mean_regularSpeed",
        }
        for col, mean_col in fill_map.items():
            # If the column exists but it has missing values, we fill them with the mean column
            if col in master_temporal_df.columns:
                master_temporal_df[col] = master_temporal_df[col].fillna(
                    master_temporal_df[mean_col]
                )
            # If the column does not exist, we create it with the mean column values
            else:
                master_temporal_df[col] = master_temporal_df[mean_col].copy()
        # Drop the helper mean columns
        master_temporal_df.drop(
            columns=["mean_trend", "mean_regularSpeed"], inplace=True, errors="ignore"
        )

        # Fill the rest of the missing values with 0
        master_temporal_df = master_temporal_df.fillna(0)

        # Ensure all expected columns are present
        # We fill all columns that are expected but not present in the final DataFrame
        # with 0, so that the final DataFrame has all the expected columns
        for column_names in self.column_names.values():
            for col in column_names:
                if col not in master_temporal_df.columns:
                    master_temporal_df[col] = 0

        print(
            f"--- Load complete. Found {len(static_df)} static records. Temporal DF shape: {master_temporal_df.shape} ---"
        )

        # Separate temporal features from target variable (jamLevel)
        if "jamLevel" in master_temporal_df.columns:
            jam_target = master_temporal_df["jamLevel"]
            master_temporal_df = master_temporal_df.drop(columns=["jamLevel"])
        else:
            jam_target = pd.Series(
                dtype=np.float64, name="jamLevel", index=master_temporal_df.index
            )

        # Convert column names to snake_case
        master_temporal_df.columns = [
            self._to_snake_case(col) for col in master_temporal_df.columns
        ]
        static_df.columns = [self._to_snake_case(col) for col in static_df.columns]
        jam_target.name = self._to_snake_case("jamLevel")

        # static_df already has a 'id_mvi' column and a 'road_id' column,
        # so we can drop the index and drop the 'id_mvi' column
        static_df = static_df.reset_index(drop=True).drop(
            columns=["id_mvi"], errors="ignore"
        )
        # Put it first for consistency
        static_df = static_df[
            ["road_id"] + [col for col in static_df.columns if col != "road_id"]
        ]
        # We drop the 'road_id' and 'timeframe' columns from the temporal DataFrame
        # since they are already in the index
        master_temporal_df = master_temporal_df.drop(
            columns=["road_id", "timeframe"], errors="ignore"
        )
        # We reset the indices to match the static DataFrame
        master_temporal_df = master_temporal_df.reset_index()
        jam_target = jam_target.reset_index()

        if len(jam_target) != len(master_temporal_df):
            warnings.warn("jam_target length does not match master_temporal_df length.")
        if set(static_df["road_id"]) != set(master_temporal_df["road_id"]):
            warnings.warn(
                "Static DataFrame road_id set does not match temporal DataFrame road_id set."
            )
        if not jam_target.index.equals(master_temporal_df.index):
            warnings.warn("jam_target index does not match master_temporal_df index.")

        return (static_df, master_temporal_df), jam_target
