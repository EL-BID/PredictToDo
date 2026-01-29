"""
Inference DataLoader for real-time traffic forecasting.

This module provides a DataLoader that processes real-time data from various sources
(jams, irregularities, weather, PMTs, incidents) and generates batches compatible with the
trained Spatio-Temporal GNN model.

Key differences from training DataLoader:
1. No target data (y) - used for prediction only
2. Loads data from real-time parquet files instead of HDF5
3. Uses Ray for parallel processing of data sources
4. Focuses on latest time window for inference
"""

import json
import logging
import pickle
import time
from pathlib import Path
from typing import Optional

import networkx as nx
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.utils.convert import from_networkx

try:
    import ray

    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

logger = logging.getLogger(__name__)


class InferenceDataLoader:
    """
    DataLoader for inference that loads real-time data and generates batches
    for prediction.

    This class:
    - Loads real-time data from parquet files (jams, irregularities, weather, PMTs, incidents)
    - Uses the same graph structure and static features as training
    - Supports Ray parallelization for faster data processing
    - Generates batches compatible with the trained model
    """

    def __init__(
        self,
        graph_path: str,
        static_data_path: str,
        in_timesteps: int = 150,
        out_timesteps: int = 50,
        node_batch_size: int = 500,
        normalization_constants_path: Optional[str] = None,
        neighbor_features_path: Optional[str] = None,
        weather_stations_path: Optional[str] = None,
        expected_feature_map_path: Optional[str] = None,
        use_ray: bool = False,
    ):
        """
        Initialize the inference DataLoader.

        Args:
            graph_path: Path to the road network graph (.pkl)
            static_data_path: Path to static features HDF5 file
            in_timesteps: Number of input timesteps (default: 150 = 37.5 hours)
            out_timesteps: Number of output timesteps for prediction (default: 50 = 12.5 hours)
            node_batch_size: Number of nodes per batch
            normalization_constants_path: Path to normalization constants pickle
            neighbor_features_path: Path to neighbor features for imputation
            weather_stations_path: Path to weather stations mapping
            expected_feature_map_path: Path to JSON with expected feature columns
            use_ray: Whether to use Ray for parallel processing
        """
        self.in_timesteps = in_timesteps
        self.out_timesteps = out_timesteps
        self.node_batch_size = node_batch_size
        self.use_ray = use_ray and RAY_AVAILABLE

        if self.use_ray and not RAY_AVAILABLE:
            print("Warning: Ray not available. Falling back to sequential processing.")
            print("Install Ray with: uv add ray")

        # Load graph and static features
        self.graph, self.edge_index, self.node_to_id, self.id_to_node = (
            self._load_graph(graph_path)
        )
        self.static_features = self._load_static_features(static_data_path)
        self.num_nodes = self.static_features.shape[0]
        self.num_static_features = self.static_features.shape[1]

        # Load normalization constants
        self.norm_constants = None
        self.static_norm_tensor = None
        if normalization_constants_path:
            self._load_normalization_constants(
                normalization_constants_path,
                static_data_path,
            )

        # Load neighbor features for imputation
        self.neighbor_features = None
        if neighbor_features_path and Path(neighbor_features_path).exists():
            self.neighbor_features = pd.read_parquet(neighbor_features_path)

        # Load weather stations mapping
        self.weather_stations = None
        if weather_stations_path and Path(weather_stations_path).exists():
            with open(weather_stations_path, "rb") as f:
                self.weather_stations = pickle.load(f)

        # Load expected feature map (optional)
        self.expected_feature_map = None
        if expected_feature_map_path and Path(expected_feature_map_path).exists():
            with open(expected_feature_map_path, "r") as f:
                self.expected_feature_map = json.load(f)

        # Initialize Ray if requested
        if self.use_ray:
            if not ray.is_initialized():
                ray.init(ignore_reinit_error=True)

        # Feature dimensions (will be updated when loading data)
        self.num_dynamic_features = 0
        self.num_weather_features = (
            4  # wind_direction, humidity, wind_speed, precipitation
        )

        # Canonical feature sets for minimal loading
        self._jams_feature_cols = [
            "vel_min",
            "vel_mean",
            "vel_max",
            "level_min",
            "level_mean",
            "level_max",
            "delay_min",
            "delay_sum",
            "delay_max",
            "count_jams",
        ]

    def _columns_for_dataset(self, key: str) -> Optional[list[str]]:
        """Return minimal columns required for each dataset."""
        if key == "jams":
            return ["id_mvi", "timeframe"] + self._jams_feature_cols
        if key == "irregularities":
            expected = self._get_expected_cols("irregularities")
            if expected is None:
                return None
            return ["id_mvi", "timeframe"] + expected
        if key == "events":
            expected = self._get_expected_cols("events")
            if expected is None:
                return None
            # events may use road_id or id_vial
            return ["timeframe", "road_id", "id_vial"] + expected
        if key == "incidents":
            expected = self._get_expected_cols("incidents")
            if expected is None:
                return None
            return ["timeframe", "road_id", "id_mvi"] + expected
        if key == "weather":
            return ["timestamp_local", "tipo_sensor", "VALORLECTURA_mean"]
        if key == "pmt_puntos":
            return ["road_id", "timeframe", "coi_puntos"]
        if key == "pmt_tramos":
            return ["road_id", "timeframe", "coi_tramo"]
        return None

    def _load_graph(
        self, graph_path: str
    ) -> tuple[nx.Graph, torch.Tensor, dict[str, int], dict[int, str]]:
        """Load the road network graph."""
        with open(graph_path, "rb") as f:
            nx_graph = pickle.load(f)

        node_to_id = {node: i for i, node in enumerate(nx_graph.nodes())}
        id_to_node = {i: node for i, node in enumerate(nx_graph.nodes())}

        return nx_graph, from_networkx(nx_graph).edge_index, node_to_id, id_to_node

    def _load_static_features(self, file_path: str) -> torch.Tensor:
        """Load static features from HDF5."""
        import h5py
        from scipy.sparse import coo_matrix

        with h5py.File(file_path, "r") as f:
            data = f["data"][:]

        node_idx, feature_idx, values = data[:, 0], data[:, 1], data[:, 2]
        num_nodes = int(node_idx.max() + 1)
        num_features = int(feature_idx.max() + 1)

        sparse_matrix = coo_matrix(
            (values, (node_idx, feature_idx)),
            shape=(num_nodes, num_features),
        )

        return torch.from_numpy(sparse_matrix.toarray()).float()

    def _load_normalization_constants(
        self,
        norm_path: str,
        static_data_path: str,
    ) -> None:
        """Load normalization constants for feature scaling."""
        with open(norm_path, "rb") as f:
            self.norm_constants = pickle.load(f)

        # Build static normalization tensor
        static_metadata_path = static_data_path.replace(".h5", "_metadata.pkl")
        with open(static_metadata_path, "rb") as f:
            static_metadata = pickle.load(f)

        static_feature_idx_map = static_metadata.get("feature_idx_map", {})
        static_norm = self.norm_constants.get("static", {})
        num_static_features = len(static_feature_idx_map)
        static_norm_values = torch.ones(num_static_features)

        for feature_name, feature_idx in static_feature_idx_map.items():
            if feature_name in static_norm:
                static_norm_values[feature_idx] = static_norm[feature_name]

        self.static_norm_tensor = static_norm_values

    def _get_expected_cols(self, key: str) -> Optional[list[str]]:
        """Return expected feature columns for a dataset key, if provided."""
        if not self.expected_feature_map:
            return None
        cols = self.expected_feature_map.get(key)
        return cols if isinstance(cols, list) else None

    def load_and_prepare_batch(
        self,
        jams_path: str,
        irregularities_path: str,
        weather_path: str,
        pmt_puntos_path: str,
        pmt_tramos_path: str,
        events_path: str,
        incidents_path: str,
        node_seeds: Optional[list[str]] = None,
        current_timestamp: Optional[pd.Timestamp] = None,
    ) -> list[Data]:
        """
        Load real-time data and prepare batches for inference.

        Args:
            jams_path: Path to jams parquet file
            irregularities_path: Path to irregularities parquet file
            weather_path: Path to weather parquet file
            pmt_puntos_path: Path to PMT points parquet file
            pmt_tramos_path: Path to PMT tracts parquet file
            events_path: Path to events parquet file
            incidents_path: Path to incidents parquet file
            node_seeds: List of seed nodes for batching (if None, use all nodes)
            current_timestamp: Current timestamp for inference (if None, use latest)

        Returns:
            List of PyTorch Geometric Data objects ready for model inference
        """
        # Load all data sources
        if self.use_ray:
            data_dict = self._load_data_parallel(
                jams_path,
                irregularities_path,
                weather_path,
                pmt_puntos_path,
                pmt_tramos_path,
                events_path,
                incidents_path,
            )
        else:
            data_dict = self._load_data_sequential(
                jams_path,
                irregularities_path,
                weather_path,
                pmt_puntos_path,
                pmt_tramos_path,
                events_path,
                incidents_path,
            )

        # Determine time window
        if current_timestamp is None:
            # Use the latest timestamp from the data
            current_timestamp = self._get_latest_timestamp(data_dict)

        # Process and align data to graph nodes
        features_dict = self._process_features(data_dict, current_timestamp)

        # Create batches
        batches = self._create_batches(features_dict, node_seeds)

        return batches

    def _load_data_parallel(
        self,
        jams_path: str,
        irregularities_path: str,
        weather_path: str,
        pmt_puntos_path: str,
        pmt_tramos_path: str,
        events_path: str,
        incidents_path: str,
    ) -> dict[str, pd.DataFrame]:
        """Load data in parallel using Ray."""

        # Convert to absolute paths to ensure Ray workers can access them
        from pathlib import Path

        jams_path = str(Path(jams_path).resolve())
        irregularities_path = str(Path(irregularities_path).resolve())
        weather_path = str(Path(weather_path).resolve())
        pmt_puntos_path = str(Path(pmt_puntos_path).resolve())
        pmt_tramos_path = str(Path(pmt_tramos_path).resolve())
        events_path = str(Path(events_path).resolve())
        incidents_path = str(Path(incidents_path).resolve())

        @ray.remote
        def load_parquet(path: str, columns: Optional[list[str]]) -> pd.DataFrame:
            """Ray remote function to load parquet file."""
            import pandas as pd
            import pyarrow.parquet as pq

            if columns:
                schema_cols = set(pq.ParquetFile(path).schema.names)
                filtered = [c for c in columns if c in schema_cols]
                return pd.read_parquet(path, columns=filtered)
            return pd.read_parquet(path)

        # Launch parallel loading
        expected_cols = {
            "jams": self._columns_for_dataset("jams"),
            "irregularities": self._columns_for_dataset("irregularities"),
            "weather": self._columns_for_dataset("weather"),
            "pmt_puntos": self._columns_for_dataset("pmt_puntos"),
            "pmt_tramos": self._columns_for_dataset("pmt_tramos"),
            "events": self._columns_for_dataset("events"),
            "incidents": self._columns_for_dataset("incidents"),
        }

        futures = {
            "jams": load_parquet.remote(jams_path, expected_cols["jams"]),
            "irregularities": load_parquet.remote(
                irregularities_path, expected_cols["irregularities"]
            ),
            "weather": load_parquet.remote(weather_path, expected_cols["weather"]),
            "pmt_puntos": load_parquet.remote(
                pmt_puntos_path, expected_cols["pmt_puntos"]
            ),
            "pmt_tramos": load_parquet.remote(
                pmt_tramos_path, expected_cols["pmt_tramos"]
            ),
            "events": load_parquet.remote(events_path, expected_cols["events"]),
            "incidents": load_parquet.remote(
                incidents_path, expected_cols["incidents"]
            ),
        }

        # Collect results
        data_dict = {}
        for key, future in futures.items():
            t0 = time.time()
            df = ray.get(future)
            df = self._ensure_expected_columns(df, expected_cols.get(key))
            data_dict[key] = df
            logger.info(
                "Loaded %s: rows=%s cols=%s in %.2fs",
                key,
                len(df),
                len(df.columns),
                time.time() - t0,
            )

        return data_dict

    def _load_data_sequential(
        self,
        jams_path: str,
        irregularities_path: str,
        weather_path: str,
        pmt_puntos_path: str,
        pmt_tramos_path: str,
        events_path: str,
        incidents_path: str,
    ) -> dict[str, pd.DataFrame]:
        """Load data sequentially (fallback when Ray is not available)."""
        data_dict = {}
        for key, path in [
            ("jams", jams_path),
            ("irregularities", irregularities_path),
            ("weather", weather_path),
            ("pmt_puntos", pmt_puntos_path),
            ("pmt_tramos", pmt_tramos_path),
            ("events", events_path),
            ("incidents", incidents_path),
        ]:
            cols = self._columns_for_dataset(key)
            t0 = time.time()
            if cols:
                cols = self._filter_existing_columns(path, cols)
                df = pd.read_parquet(path, columns=cols)
            else:
                df = pd.read_parquet(path)
            df = self._ensure_expected_columns(df, self._columns_for_dataset(key))
            data_dict[key] = df
            logger.info(
                "Loaded %s: rows=%s cols=%s in %.2fs",
                key,
                len(df),
                len(df.columns),
                time.time() - t0,
            )

        return data_dict

    @staticmethod
    def _filter_existing_columns(path: str, cols: list[str]) -> list[str]:
        import pyarrow.parquet as pq

        if not cols:
            return cols
        schema_cols = set(pq.ParquetFile(path).schema.names)
        return [c for c in cols if c in schema_cols]

    def _ensure_expected_columns(
        self, df: pd.DataFrame, expected_cols: Optional[list[str]]
    ) -> pd.DataFrame:
        if not expected_cols:
            return df

        # Never auto-fill ID/index columns; missing IDs should not be synthesized.
        id_cols = {"road_id", "id_vial", "id_mvi"}
        cols_to_check = [c for c in expected_cols if c not in id_cols]
        missing = [c for c in cols_to_check if c not in df.columns]
        if missing:
            logger.warning("Missing columns filled with 0: %s", missing[:10])
            for col in missing:
                df[col] = 0.0
        return df

    def _get_latest_timestamp(self, data_dict: dict[str, pd.DataFrame]) -> pd.Timestamp:
        """Get the latest timestamp across all data sources."""
        timestamps = []

        # Jams
        if "timeframe" in data_dict["jams"].columns:
            ts = pd.to_datetime(data_dict["jams"]["timeframe"]).max()
            # Remove timezone if present
            if ts.tz is not None:
                ts = ts.tz_localize(None)
            timestamps.append(ts)

        # Irregularities
        if "timeframe" in data_dict["irregularities"].columns:
            ts = pd.to_datetime(data_dict["irregularities"]["timeframe"]).max()
            if ts.tz is not None:
                ts = ts.tz_localize(None)
            timestamps.append(ts)

        # Weather
        if "timestamp_local" in data_dict["weather"].columns:
            # Convert from format like "20251227_0945" to timestamp
            weather_ts = data_dict["weather"]["timestamp_local"].max()
            ts = pd.to_datetime(weather_ts, format="%Y%m%d_%H%M")
            timestamps.append(ts)

        # PMTs
        for key in ["pmt_puntos", "pmt_tramos"]:
            if "timeframe" in data_dict[key].columns:
                ts = pd.to_datetime(data_dict[key]["timeframe"]).max()
                if ts.tz is not None:
                    ts = ts.tz_localize(None)
                timestamps.append(ts)

        # Events
        if "timeframe" in data_dict["events"].columns:
            ts = pd.to_datetime(data_dict["events"]["timeframe"]).max()
            if ts.tz is not None:
                ts = ts.tz_localize(None)
            timestamps.append(ts)

        # Incidents
        if "timeframe" in data_dict["incidents"].columns:
            ts = pd.to_datetime(data_dict["incidents"]["timeframe"]).max()
            if ts.tz is not None:
                ts = ts.tz_localize(None)
            timestamps.append(ts)

        return max(timestamps)

    def _process_features(
        self,
        data_dict: dict[str, pd.DataFrame],
        current_timestamp: pd.Timestamp,
    ) -> dict[str, torch.Tensor]:
        """
        Process raw data into feature tensors aligned with graph nodes and time.

        Returns a dictionary with:
        - x_dynamic_past: (num_nodes, in_timesteps, num_features)
        - x_dynamic_future: (num_nodes, out_timesteps, num_future_features)
        - x_weather: (num_nodes, in_timesteps, num_weather_features)
        """
        # Create time range for input window
        freq = pd.Timedelta(minutes=15)
        input_end = current_timestamp
        input_start = input_end - freq * (self.in_timesteps - 1)

        # Create time index
        time_index = pd.date_range(
            start=input_start,
            end=input_end,
            freq=freq,
        )

        # Future time index (for known features like PMTs)
        future_start = input_end + freq
        future_end = future_start + freq * (self.out_timesteps - 1)
        future_time_index = pd.date_range(
            start=future_start,
            end=future_end,
            freq=freq,
        )

        # Process each data source
        if self.use_ray:
            features = self._process_features_parallel(
                data_dict,
                time_index,
                future_time_index,
            )
        else:
            features = self._process_features_sequential(
                data_dict,
                time_index,
                future_time_index,
            )

        return features

    def _process_features_sequential(
        self,
        data_dict: dict[str, pd.DataFrame],
        time_index: pd.DatetimeIndex,
        future_time_index: pd.DatetimeIndex,
    ) -> dict[str, torch.Tensor]:
        """Process features sequentially."""
        # Initialize tensors
        # We'll dynamically determine feature counts as we process

        # Process jams (past only)
        jams_features = self._process_jams(
            data_dict["jams"],
            time_index,
            expected_cols=self._get_expected_cols("jams"),
        )

        # Process irregularities (past only)
        irreg_features = self._process_irregularities(
            data_dict["irregularities"],
            time_index,
            expected_cols=self._get_expected_cols("irregularities"),
        )

        # Process weather (past only)
        weather_features = self._process_weather(data_dict["weather"], time_index)

        # Process PMTs (both past and future)
        pmt_features_past, pmt_features_future = self._process_pmts(
            data_dict["pmt_puntos"],
            data_dict["pmt_tramos"],
            time_index,
            future_time_index,
        )

        # Process events (past + future)
        events_features = self._process_events(
            data_dict["events"],
            time_index,
            expected_cols=self._get_expected_cols("events"),
        )
        events_features_future = self._process_events(
            data_dict["events"],
            future_time_index,
            expected_cols=self._get_expected_cols("events"),
        )

        # Process incidents
        incidents_features = self._process_incidents(
            data_dict["incidents"],
            time_index,
            expected_cols=self._get_expected_cols("incidents"),
        )

        # Concatenate all past features
        x_dynamic_past = torch.cat(
            [
                jams_features,
                irreg_features,
                pmt_features_past,
                events_features,
                incidents_features,
            ],
            dim=2,  # Feature dimension
        )

        # Future features: PMTs + events
        x_dynamic_future = torch.cat(
            [pmt_features_future, events_features_future],
            dim=2,
        )

        return {
            "x_dynamic_past": x_dynamic_past,
            "x_dynamic_future": x_dynamic_future,
            "x_weather": weather_features,
        }

    def _process_features_parallel(
        self,
        data_dict: dict[str, pd.DataFrame],
        time_index: pd.DatetimeIndex,
        future_time_index: pd.DatetimeIndex,
    ) -> dict[str, torch.Tensor]:
        """Process features in parallel using Ray."""

        @ray.remote
        def process_jams_remote(df, time_idx, node_to_id, num_nodes, expected_cols):
            return InferenceDataLoader._process_jams_static(
                df, time_idx, node_to_id, num_nodes, expected_cols
            )

        @ray.remote
        def process_irreg_remote(df, time_idx, node_to_id, num_nodes, expected_cols):
            return InferenceDataLoader._process_irregularities_static(
                df, time_idx, node_to_id, num_nodes, expected_cols
            )

        @ray.remote
        def process_weather_remote(df, time_idx, weather_stations, num_nodes):
            return InferenceDataLoader._process_weather_static(
                df, time_idx, weather_stations, num_nodes
            )

        # Launch parallel processing
        jams_future = process_jams_remote.remote(
            data_dict["jams"],
            time_index,
            self.node_to_id,
            self.num_nodes,
            self._get_expected_cols("jams"),
        )

        irreg_future = process_irreg_remote.remote(
            data_dict["irregularities"],
            time_index,
            self.node_to_id,
            self.num_nodes,
            self._get_expected_cols("irregularities"),
        )

        weather_future = process_weather_remote.remote(
            data_dict["weather"],
            time_index,
            self.weather_stations,
            self.num_nodes,
        )

        # Process PMTs sequentially (requires both datasets)
        pmt_features_past, pmt_features_future = self._process_pmts(
            data_dict["pmt_puntos"],
            data_dict["pmt_tramos"],
            time_index,
            future_time_index,
        )

        # Collect parallel results
        jams_features = ray.get(jams_future)
        irreg_features = ray.get(irreg_future)
        weather_features = ray.get(weather_future)

        # Process events (past + future)
        events_features = self._process_events(
            data_dict["events"],
            time_index,
            expected_cols=self._get_expected_cols("events"),
        )
        events_features_future = self._process_events(
            data_dict["events"],
            future_time_index,
            expected_cols=self._get_expected_cols("events"),
        )

        # Process incidents
        incidents_features = self._process_incidents(
            data_dict["incidents"],
            time_index,
            expected_cols=self._get_expected_cols("incidents"),
        )

        # Concatenate all past features
        x_dynamic_past = torch.cat(
            [
                jams_features,
                irreg_features,
                pmt_features_past,
                events_features,
                incidents_features,
            ],
            dim=2,
        )

        return {
            "x_dynamic_past": x_dynamic_past,
            "x_dynamic_future": torch.cat(
                [pmt_features_future, events_features_future],
                dim=2,
            ),
            "x_weather": weather_features,
        }

    def _process_jams(
        self,
        df: pd.DataFrame,
        time_index: pd.DatetimeIndex,
        expected_cols: Optional[list[str]] = None,
    ) -> torch.Tensor:
        """Process jams data into tensor format."""
        return self._process_jams_static(
            df, time_index, self.node_to_id, self.num_nodes, expected_cols
        )

    @staticmethod
    def _process_jams_static(
        df: pd.DataFrame,
        time_index: pd.DatetimeIndex,
        node_to_id: dict[str, int],
        num_nodes: int,
        expected_cols: Optional[list[str]] = None,
    ) -> torch.Tensor:
        """Static method for processing jams (can be called by Ray)."""
        # Jams features: vel_min, vel_mean, vel_max, level_min, level_mean,
        # level_max, delay_min, delay_sum, delay_max, count_jams
        jam_feature_cols = expected_cols or [
            "vel_min",
            "vel_mean",
            "vel_max",
            "level_min",
            "level_mean",
            "level_max",
            "delay_min",
            "delay_sum",
            "delay_max",
            "count_jams",
        ]

        num_features = len(jam_feature_cols)
        num_timesteps = len(time_index)

        # Initialize tensor
        tensor = torch.zeros(num_nodes, num_timesteps, num_features)

        # Convert timeframe to datetime
        df = df.copy()
        df["timeframe"] = pd.to_datetime(df["timeframe"])

        # Filter to time window
        df = df[df["timeframe"].isin(time_index)]

        if len(df) == 0:
            return tensor

        # Create time index mapping
        time_to_idx = {t: i for i, t in enumerate(time_index)}

        # Populate tensor
        for _, row in df.iterrows():
            road_id = row["id_mvi"]
            if road_id not in node_to_id:
                continue

            node_idx = node_to_id[road_id]
            time_idx = time_to_idx.get(row["timeframe"])

            if time_idx is None:
                continue

            for feat_idx, col in enumerate(jam_feature_cols):
                value = row.get(col, 0)
                if pd.notna(value):
                    tensor[node_idx, time_idx, feat_idx] = float(value)

        return tensor

    def _process_irregularities(
        self,
        df: pd.DataFrame,
        time_index: pd.DatetimeIndex,
        expected_cols: Optional[list[str]] = None,
    ) -> torch.Tensor:
        """Process irregularities data into tensor format."""
        return self._process_irregularities_static(
            df, time_index, self.node_to_id, self.num_nodes, expected_cols
        )

    @staticmethod
    def _process_irregularities_static(
        df: pd.DataFrame,
        time_index: pd.DatetimeIndex,
        node_to_id: dict[str, int],
        num_nodes: int,
        expected_cols: Optional[list[str]] = None,
    ) -> torch.Tensor:
        """Static method for processing irregularities."""
        # Use all columns except id_mvi and timeframe
        exclude_cols = {"id_mvi", "timeframe"}
        feature_cols = (
            expected_cols
            if expected_cols is not None
            else [col for col in df.columns if col not in exclude_cols]
        )

        num_features = len(feature_cols)
        num_timesteps = len(time_index)

        # Initialize tensor
        tensor = torch.zeros(num_nodes, num_timesteps, num_features)

        # Convert timeframe to datetime
        df = df.copy()
        df["timeframe"] = pd.to_datetime(df["timeframe"])

        # Filter to time window
        df = df[df["timeframe"].isin(time_index)]

        if len(df) == 0:
            return tensor

        # Create time index mapping
        time_to_idx = {t: i for i, t in enumerate(time_index)}

        # Populate tensor
        for _, row in df.iterrows():
            road_id = row["id_mvi"]
            if road_id not in node_to_id:
                continue

            node_idx = node_to_id[road_id]
            time_idx = time_to_idx.get(row["timeframe"])

            if time_idx is None:
                continue

            for feat_idx, col in enumerate(feature_cols):
                value = row.get(col, 0)
                if pd.notna(value):
                    tensor[node_idx, time_idx, feat_idx] = float(value)

        return tensor

    def _process_events(
        self,
        df: pd.DataFrame,
        time_index: pd.DatetimeIndex,
        expected_cols: Optional[list[str]] = None,
    ) -> torch.Tensor:
        """Process events data into tensor format."""
        return self._process_events_static(
            df, time_index, self.node_to_id, self.num_nodes, expected_cols
        )

    @staticmethod
    def _process_events_static(
        df: pd.DataFrame,
        time_index: pd.DatetimeIndex,
        node_to_id: dict[str, int],
        num_nodes: int,
        expected_cols: Optional[list[str]] = None,
    ) -> torch.Tensor:
        """Static method for processing events data."""
        # Determine the road id column name (can be 'road_id' or 'id_vial')
        road_id_col = None
        for col in ["road_id", "id_vial"]:
            if col in df.columns:
                road_id_col = col
                break

        if road_id_col is None:
            # No road id column found, return zeros
            return torch.zeros(num_nodes, len(time_index), 3)

        # Use all columns except road_id and timeframe
        exclude_cols = {road_id_col, "timeframe"}
        feature_cols = (
            expected_cols
            if expected_cols is not None
            else [col for col in df.columns if col not in exclude_cols]
        )

        # If no feature columns, return zeros
        if not feature_cols:
            num_features = 3  # Default to 3 features
        else:
            num_features = min(
                len(feature_cols), 3
            )  # Limit to 3 features to match architecture

        num_timesteps = len(time_index)

        # Initialize tensor
        tensor = torch.zeros(num_nodes, num_timesteps, num_features)

        # Convert timeframe to datetime
        df = df.copy()
        df["timeframe"] = pd.to_datetime(df["timeframe"])

        # Filter to time window
        df = df[df["timeframe"].isin(time_index)]

        if len(df) == 0:
            return tensor

        # Create time index mapping
        time_to_idx = {t: i for i, t in enumerate(time_index)}

        # Populate tensor
        for _, row in df.iterrows():
            road_id = row.get(road_id_col)
            if road_id is None or road_id not in node_to_id:
                continue

            node_idx = node_to_id[road_id]
            time_idx = time_to_idx.get(row["timeframe"])

            if time_idx is None:
                continue

            for feat_idx, col in enumerate(feature_cols[:num_features]):
                value = row.get(col, 0)
                if pd.notna(value):
                    tensor[node_idx, time_idx, feat_idx] = float(value)

        return tensor

    def _process_incidents(
        self,
        df: pd.DataFrame,
        time_index: pd.DatetimeIndex,
        expected_cols: Optional[list[str]] = None,
    ) -> torch.Tensor:
        """Process incidents data into tensor format."""
        return self._process_incidents_static(
            df, time_index, self.node_to_id, self.num_nodes, expected_cols
        )

    @staticmethod
    def _process_incidents_static(
        df: pd.DataFrame,
        time_index: pd.DatetimeIndex,
        node_to_id: dict[str, int],
        num_nodes: int,
        expected_cols: Optional[list[str]] = None,
    ) -> torch.Tensor:
        """Static method for processing incidents data.

        Incidents features include:
        - incident_total: Total number of incidents
        - incident_type_*: Various incident type indicators
        - incident_case_specifics_*: Specific case indicators
        """
        # Determine the road id column name
        road_id_col = None
        for col in ["road_id", "id_mvi"]:
            if col in df.columns:
                road_id_col = col
                break

        if road_id_col is None:
            # No road id column found, return zeros
            # Using 11 features to match the test parquet structure
            return torch.zeros(num_nodes, len(time_index), 11)

        # Use all columns except road_id and timeframe
        exclude_cols = {road_id_col, "timeframe"}
        feature_cols = (
            expected_cols
            if expected_cols is not None
            else [col for col in df.columns if col not in exclude_cols]
        )

        # Sort feature columns for consistent ordering (only when inferred)
        if expected_cols is None:
            feature_cols = sorted(feature_cols)

        # If no feature columns, return zeros
        if not feature_cols:
            num_features = 11  # Default to match test data structure
        else:
            num_features = len(feature_cols)

        num_timesteps = len(time_index)

        # Initialize tensor
        tensor = torch.zeros(num_nodes, num_timesteps, num_features)

        # Convert timeframe to datetime
        df = df.copy()
        df["timeframe"] = pd.to_datetime(df["timeframe"])

        # Remove timezone if present
        if df["timeframe"].dt.tz is not None:
            df["timeframe"] = df["timeframe"].dt.tz_localize(None)

        # Filter to time window
        df = df[df["timeframe"].isin(time_index)]

        if len(df) == 0:
            return tensor

        # Create time index mapping
        time_to_idx = {t: i for i, t in enumerate(time_index)}

        # Populate tensor
        for _, row in df.iterrows():
            road_id = row.get(road_id_col)
            if road_id is None or road_id not in node_to_id:
                continue

            node_idx = node_to_id[road_id]
            time_idx = time_to_idx.get(row["timeframe"])

            if time_idx is None:
                continue

            for feat_idx, col in enumerate(feature_cols):
                value = row.get(col, 0)
                if pd.notna(value):
                    tensor[node_idx, time_idx, feat_idx] = float(value)

        return tensor

    def _process_weather(
        self, df: pd.DataFrame, time_index: pd.DatetimeIndex
    ) -> torch.Tensor:
        """Process weather data into tensor format."""
        return self._process_weather_static(
            df, time_index, self.weather_stations, self.num_nodes
        )

    @staticmethod
    def _process_weather_static(
        df: pd.DataFrame,
        time_index: pd.DatetimeIndex,
        weather_stations: Optional[dict],
        num_nodes: int,
    ) -> torch.Tensor:
        """Static method for processing weather data."""
        # Weather features: wind_direction, humidity, wind_speed, precipitation
        # The df has tipo_sensor, IDSENSOR, VALORLECTURA_mean, timestamp_local

        num_features = 4  # 4 weather types
        num_timesteps = len(time_index)

        # Initialize tensor
        tensor = torch.zeros(num_nodes, num_timesteps, num_features)

        if weather_stations is None:
            return tensor

        # Convert timestamp_local to datetime
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp_local"], format="%Y%m%d_%H%M")

        # Filter to time window
        df = df[df["timestamp"].isin(time_index)]

        if len(df) == 0:
            return tensor

        # Map weather types to indices
        weather_type_map = {
            "direccionviento": 0,
            "humedad": 1,
            "velocidadviento": 2,
            "precipitacion": 3,
        }

        # Create time index mapping
        time_to_idx = {t: i for i, t in enumerate(time_index)}

        # For each road, use its assigned weather station
        # This is simplified - in practice, you'd need the station assignment
        # For now, we'll broadcast weather data to all nodes with the same station

        # Group by timestamp and sensor type
        for _, row in df.iterrows():
            sensor_type = row["tipo_sensor"]
            if sensor_type not in weather_type_map:
                continue

            feat_idx = weather_type_map[sensor_type]
            time_idx = time_to_idx.get(row["timestamp"])

            if time_idx is None:
                continue

            value = row.get("VALORLECTURA_mean", 0)
            if pd.notna(value):
                # For simplicity, broadcast to all nodes
                # In production, use weather_stations mapping
                tensor[:, time_idx, feat_idx] = float(value)

        return tensor

    def _process_pmts(
        self,
        df_puntos: pd.DataFrame,
        df_tramos: pd.DataFrame,
        time_index: pd.DatetimeIndex,
        future_time_index: pd.DatetimeIndex,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Process PMT data for both past and future windows.

        Returns:
            Tuple of (past_features, future_features)
        """
        # PMT features: coi_puntos (from puntos), coi_tramo (from tramos)
        num_features = 2
        num_timesteps_past = len(time_index)
        num_timesteps_future = len(future_time_index)

        # Initialize tensors
        tensor_past = torch.zeros(self.num_nodes, num_timesteps_past, num_features)
        tensor_future = torch.zeros(self.num_nodes, num_timesteps_future, num_features)

        # Process puntos (feature 0)
        df_puntos = df_puntos.copy()
        df_puntos["timeframe"] = pd.to_datetime(df_puntos["timeframe"])

        # Past window
        df_past = df_puntos[df_puntos["timeframe"].isin(time_index)]
        time_to_idx_past = {t: i for i, t in enumerate(time_index)}

        for _, row in df_past.iterrows():
            road_id = row["road_id"]
            if road_id not in self.node_to_id:
                continue

            node_idx = self.node_to_id[road_id]
            time_idx = time_to_idx_past.get(row["timeframe"])

            if time_idx is None:
                continue

            tensor_past[node_idx, time_idx, 0] = float(row.get("coi_puntos", 0))

        # Future window
        df_future = df_puntos[df_puntos["timeframe"].isin(future_time_index)]
        time_to_idx_future = {t: i for i, t in enumerate(future_time_index)}

        for _, row in df_future.iterrows():
            road_id = row["road_id"]
            if road_id not in self.node_to_id:
                continue

            node_idx = self.node_to_id[road_id]
            time_idx = time_to_idx_future.get(row["timeframe"])

            if time_idx is None:
                continue

            tensor_future[node_idx, time_idx, 0] = float(row.get("coi_puntos", 0))

        # Process tramos (feature 1)
        df_tramos = df_tramos.copy()
        df_tramos["timeframe"] = pd.to_datetime(df_tramos["timeframe"])

        # Past window
        df_past = df_tramos[df_tramos["timeframe"].isin(time_index)]

        for _, row in df_past.iterrows():
            road_id = row["road_id"]
            if road_id not in self.node_to_id:
                continue

            node_idx = self.node_to_id[road_id]
            time_idx = time_to_idx_past.get(row["timeframe"])

            if time_idx is None:
                continue

            tensor_past[node_idx, time_idx, 1] = float(row.get("coi_tramo", 0))

        # Future window
        df_future = df_tramos[df_tramos["timeframe"].isin(future_time_index)]

        for _, row in df_future.iterrows():
            road_id = row["road_id"]
            if road_id not in self.node_to_id:
                continue

            node_idx = self.node_to_id[road_id]
            time_idx = time_to_idx_future.get(row["timeframe"])

            if time_idx is None:
                continue

            tensor_future[node_idx, time_idx, 1] = float(row.get("coi_tramo", 0))

        return tensor_past, tensor_future

    def _create_zero_features(
        self, time_index: pd.DatetimeIndex, num_features: int
    ) -> torch.Tensor:
        """Create a zero tensor for missing features."""
        return torch.zeros(self.num_nodes, len(time_index), num_features)

    def _create_batches(
        self,
        features_dict: dict[str, torch.Tensor],
        node_seeds: Optional[list[str]] = None,
    ) -> list[Data]:
        """
        Create batches from processed features.

        Args:
            features_dict: Dictionary with processed feature tensors
            node_seeds: List of seed nodes (if None, create one batch with all nodes)

        Returns:
            List of PyTorch Geometric Data objects
        """
        batches = []

        # If no seeds provided, use all nodes (single batch)
        if node_seeds is None:
            node_seeds = [list(self.graph.nodes())[0]]  # Use first node as seed

        for seed_node in node_seeds:
            # Get neighborhood around seed
            node_indices = self._get_neighborhood(seed_node)

            # Extract features for these nodes
            x_static = self.static_features[node_indices]
            x_dynamic_past = features_dict["x_dynamic_past"][node_indices]
            x_dynamic_future = features_dict["x_dynamic_future"][node_indices]
            x_weather = features_dict["x_weather"][node_indices]

            # Get subgraph edges
            edge_index = self._get_subgraph_edges(node_indices)

            # Apply normalization if available
            if self.static_norm_tensor is not None:
                x_static = x_static / self.static_norm_tensor
                x_static = torch.clamp(x_static, -1, 1)

            # Create Data object
            data = Data(
                x_static=x_static,
                x_dynamic_past=x_dynamic_past,
                x_dynamic_future=x_dynamic_future,
                x_weather=x_weather,
                edge_index=edge_index,
                num_nodes=len(node_indices),
            )

            batches.append(data)

        return batches

    def _get_neighborhood(self, seed_node: str) -> list[int]:
        """Get node indices for neighborhood around seed node using BFS."""
        if seed_node not in self.graph.nodes():
            raise ValueError(f"Seed node {seed_node} not in graph")

        # BFS to find neighbors
        neighbors = [seed_node]
        bfs_edges = nx.bfs_edges(self.graph, source=seed_node)

        for _, v in bfs_edges:
            neighbors.append(v)
            if len(neighbors) >= self.node_batch_size:
                break

        # Convert to indices
        return [self.node_to_id[n] for n in neighbors]

    def _get_subgraph_edges(self, node_indices: list[int]) -> torch.Tensor:
        """Extract edges for subgraph containing specified nodes."""
        from torch_geometric.utils import subgraph

        node_indices_tensor = torch.tensor(node_indices, dtype=torch.long)
        edge_index, _ = subgraph(
            node_indices_tensor,
            self.edge_index,
            relabel_nodes=True,
        )

        return edge_index

    def cleanup(self):
        """Cleanup resources (shutdown Ray if initialized)."""
        if self.use_ray and ray.is_initialized():
            ray.shutdown()


if __name__ == "__main__":
    # Paths to model artifacts (from training)
    base_path = Path("data/datos_procesados")

    graph_path = str(base_path / "conectivity_graph" / "road_graph.pickle")
    static_data_path = str(base_path / "sparse_datasets" / "static_dataset.h5")
    norm_constants_path = str(
        base_path / "sparse_datasets" / "normalization_constants.pkl"
    )
    neighbor_features_path = str(base_path / "neighbor_features.parquet")
    weather_stations_path = str(base_path / "weather" / "weather_stations.pkl")

    # Paths to real-time inference data
    inference_path = Path("data/inference_tests")

    jams_path = str(inference_path / "df_jams_test.parquet")
    irregularities_path = str(inference_path / "df_irregularidades_test.parquet")
    weather_path = str(inference_path / "df_clima_test.parquet")
    pmt_puntos_path = str(inference_path / "df_pmt_puntos_test.parquet")
    pmt_tramos_path = str(inference_path / "df_pmt_tramos_test.parquet")
    events_path = str(inference_path / "df_eventos_test.parquet")
    incidents_path = str(inference_path / "df_incidentes_test.parquet")

    # Initialize the inference DataLoader
    print("Initializing InferenceDataLoader...")
    loader = InferenceDataLoader(
        graph_path=graph_path,
        static_data_path=static_data_path,
        in_timesteps=150,  # 37.5 hours of input
        out_timesteps=50,  # 12.5 hours of prediction
        node_batch_size=500,
        normalization_constants_path=norm_constants_path,
        neighbor_features_path=neighbor_features_path,
        weather_stations_path=weather_stations_path,
        use_ray=False,  # Set to True to enable Ray parallel processing
    )

    print(f"Loaded graph with {loader.num_nodes} nodes")
    print(f"Static features: {loader.num_static_features} dimensions")

    # Load and prepare batches
    print("\nLoading real-time data and preparing batches...")
    batches = loader.load_and_prepare_batch(
        jams_path=jams_path,
        irregularities_path=irregularities_path,
        weather_path=weather_path,
        pmt_puntos_path=pmt_puntos_path,
        pmt_tramos_path=pmt_tramos_path,
        events_path=events_path,
        incidents_path=incidents_path,
        node_seeds=None,  # Use all nodes (or specify specific seed nodes)
        current_timestamp=None,  # Use latest timestamp from data
    )

    print(f"\nGenerated {len(batches)} batch(es)")

    # Examine the first batch
    if len(batches) > 0:
        batch = batches[0]
        print("\nBatch structure:")
        print(f"  - x_static: {batch.x_static.shape}")
        print(f"  - x_dynamic_past: {batch.x_dynamic_past.shape}")
        print(f"  - x_dynamic_future: {batch.x_dynamic_future.shape}")
        print(f"  - x_weather: {batch.x_weather.shape}")
        print(f"  - edge_index: {batch.edge_index.shape}")
        print(f"  - num_nodes: {batch.num_nodes}")

        # Example: Make predictions with a trained model
        # Uncomment and modify when you have a trained model
        """
        print("\nMaking predictions...")
        model = load_trained_model("path/to/model.pth")
        model.eval()
        
        with torch.no_grad():
            predictions = model(
                x_static=batch.x_static.unsqueeze(0),  # Add batch dimension
                x_dynamic_past=batch.x_dynamic_past.unsqueeze(0),
                x_dynamic_future=batch.x_dynamic_future.unsqueeze(0),
                x_weather=batch.x_weather.unsqueeze(0),
                edge_index=batch.edge_index,
            )
        
        print(f"Predictions shape: {predictions.shape}")
        # predictions will be (1, num_nodes, out_timesteps) or similar
        """

    # Cleanup
    loader.cleanup()
    print("\nInference complete!")
