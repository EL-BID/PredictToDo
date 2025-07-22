import os
import pickle as pkl
from typing import Any, List

import networkx as nx
import h5py
import numpy as np
import pandas as pd
from tqdm.auto import tqdm


class SparseDatasetGenerator:
    def __init__(
        self,
        node_ids: list[Any],
        min_timeframe: pd.Timestamp,
        max_timeframe: pd.Timestamp,
        freq: str = "15min",
    ):
        self.node_ids = node_ids
        self.max_timeframe = max_timeframe
        self.min_timeframe = min_timeframe

        self.timeframes = pd.date_range(
            start=self.min_timeframe,
            end=self.max_timeframe,
            freq=freq,
        ).to_list()

        self.time_idx_map = {ts: i for i, ts in enumerate(self.timeframes)}
        self.node_idx_map = {node: i for i, node in enumerate(self.node_ids)}

    def temporal_dataframe_to_coo(
        self,
        df: pd.DataFrame,
        output_path: str,
        has_index: bool = False,
        node_col: str = "road_id",
        timeframe_col: str = "timeframe",
        return_df: bool = False,
    ) -> None | pd.DataFrame:
        """
        Generate a sparse dataset from the given DataFrame.
        The DataFrame should have columns: node_col, timeframe_col, and feature columns.
        """
        # Feature mapping: feature name -> integer index
        all_features = sorted(list(set(df.columns)))
        feature_idx_map = {feature: i for i, feature in enumerate(all_features)}

        # Reset index to turn (node_col, timeframe_col) into columns
        if not has_index:
            df = df.set_index([node_col, timeframe_col])

        # Convert the second level of the index to a datetime
        df.index = df.index.set_levels(
            [df.index.levels[0], pd.to_datetime(df.index.levels[1])],
            level=[0, 1],
        )

        # Filter dates out of range
        df = df[
            (df.index.get_level_values(timeframe_col) >= self.min_timeframe)
            & (df.index.get_level_values(timeframe_col) <= self.max_timeframe)
        ]

        feature_long_dfs = []
        # Iterate over each feature column to process it individually, avoiding `melt`
        for feature_name in tqdm(df.columns):
            # Get the series for the current feature
            series = df[feature_name]
            # Filter out zero values immediately to keep the series small
            non_zero_series = series[series != 0]

            if not non_zero_series.empty:
                # Convert the sparse series to a DataFrame
                feature_df = non_zero_series.reset_index()
                # Rename the column containing values to a generic 'value'
                feature_df.rename(columns={feature_name: "value"}, inplace=True)
                # Add a column to identify which feature these values belong to
                feature_df["feature_name"] = feature_name
                feature_long_dfs.append(feature_df)

        long_df = pd.concat(feature_long_dfs, ignore_index=True)
        # Apply the integer mappings
        long_df["node_idx"] = long_df[node_col].map(self.node_idx_map)
        long_df["time_idx"] = long_df[timeframe_col].map(self.time_idx_map)
        long_df["feature_idx"] = long_df["feature_name"].map(feature_idx_map)

        # Select and reorder columns, and drop temporary ones
        final_cols = ["node_idx", "time_idx", "feature_idx", "value"]
        long_df = long_df[final_cols]
        long_df.loc[:, "node_idx"] = long_df["node_idx"].astype(int)
        long_df.loc[:, "time_idx"] = long_df["time_idx"].astype(int)
        long_df.loc[:, "feature_idx"] = long_df["feature_idx"].astype(int)

        print(f"Found and processed {len(long_df)} non-zero entries.")
        # Save the DataFrame to an HDF5 file
        with h5py.File(output_path, "w") as f:
            f.create_dataset("data", data=long_df.to_numpy())
        metadata_path = os.path.splitext(output_path)[0] + "_metadata.pkl"
        with open(metadata_path, "wb") as f:
            pkl.dump(
                {
                    "node_idx_map": self.node_idx_map,
                    "time_idx_map": self.time_idx_map,
                    "feature_idx_map": feature_idx_map,
                },
                f,
            )
        print(f"Saved metadata to {metadata_path} and data to {output_path}")

        if return_df:
            return long_df
        else:
            return None

    def static_dataframe_to_coo(
        self,
        df: pd.DataFrame,
        output_path: str,
        has_index: bool = False,
        node_col: str = "road_id",
        return_df: bool = False,
    ) -> None | pd.DataFrame:
        """
        Generate a sparse dataset from the given static DataFrame.
        The DataFrame should have columns: node_col and feature columns.
        """
        # Feature mapping: feature name -> integer index
        all_features = sorted(list(set(df.columns)))
        feature_idx_map = {feature: i for i, feature in enumerate(all_features)}

        if not has_index:
            df = df.set_index(node_col)

        feature_long_dfs = []
        # Iterate over each feature column to process it individually, avoiding `melt`
        for feature_name in tqdm(df.columns):
            # Get the series for the current feature
            series = df[feature_name]
            # Filter out zero values immediately to keep the series small
            non_zero_series = series[series != 0]

            if not non_zero_series.empty:
                # Convert the sparse series to a DataFrame
                feature_df = non_zero_series.reset_index()
                # Rename the column containing values to a generic 'value'
                feature_df.rename(columns={feature_name: "value"}, inplace=True)
                # Add a column to identify which feature these values belong to
                feature_df["feature_name"] = feature_name
                feature_long_dfs.append(feature_df)

        long_df = pd.concat(feature_long_dfs, ignore_index=True)
        # Apply the integer mappings
        long_df["node_idx"] = long_df[node_col].map(self.node_idx_map)
        long_df["feature_idx"] = long_df["feature_name"].map(feature_idx_map)

        # Select and reorder columns, and drop temporary ones
        final_cols = ["node_idx", "feature_idx", "value"]
        long_df = long_df[final_cols]
        long_df.loc[:, "node_idx"] = long_df["node_idx"].astype(int)
        long_df.loc[:, "feature_idx"] = long_df["feature_idx"].astype(int)

        print(f"Found and processed {len(long_df)} non-zero entries.")
        # Save the DataFrame to an HDF5 file
        with h5py.File(output_path, "w") as f:
            f.create_dataset("data", data=long_df.to_numpy())
        metadata_path = os.path.splitext(output_path)[0] + "_metadata.pkl"
        with open(metadata_path, "wb") as f:
            pkl.dump(
                {"node_idx_map": self.node_idx_map, "feature_idx_map": feature_idx_map},
                f,
            )
        print(f"Saved metadata to {metadata_path} and data to {output_path}")

        if return_df:
            return long_df
        else:
            return None

    def generate_sparse_dataset(
        self,
        data_filenames: dict[str, str | List[str]],
        output_path: str,
    ) -> None:
        """
        Generate a sparse dataset from multiple DataFrames stored in files.
        """
        pass

    @staticmethod
    def _clean_node_attributes(base_graph: nx.DiGraph) -> nx.DiGraph:
        """
        Cleans the node attributes DataFrame by dropping unnecessary columns and
        ensuring the remaining columns are in the correct format.

        Args:
            base_graph (nx.DiGraph): The base graph with raw node attributes.

        Returns:
            nx.DiGraph: A new graph with cleaned node attributes.
        """
        all_nodes_df = pd.DataFrame.from_dict(
            dict(base_graph.nodes(data=True)), orient="index"
        )

        # Information about the columns with MIV can be found at:
        # https://www.ideca.gov.co/sites/default/files/2019-05/CatalogoObjetosMRV6.0.pdf

        node_attr_to_drop = [
            "id_mvi",  # It's already the node ID
            "MVICODIGO",
            "MVICCALZAD",
            "MVICIV",
            "MVICCAT",
            "MVINOMBRE",
            "MVINALTERN",
            "MVINPRINCI",
            "MVINGENERA",
            "MVINANTIGU",
            "MVIETIQUET",
            "MVISVIA",
            "OBJECTID",
            "contador",
            "cardinal",
            "sx",
            "sy",
            "ex",
            "ey",
            "buildings_yes",
            "buildings_yesmm",
        ] + [
            "bus_stop_mean_distance_meters",
            "bus_stop_median_distance_meters",
        ]

        node_attr_to_category = [
            "MVITCLA",  # It's an int but should be treated as a category
            "MVITIPO",  # It's a string, should replace "Null" and " " with the same value
            "MVIVELREG",  # Max speed, it's either [30, 50, "Null"], better to treat as category
        ]

        node_attr_num = [
            "MVINUMC",  # Number of lanes, int
            "length_m",
        ] + [
            col
            for col in all_nodes_df.columns
            if col.startswith("land_use_")
            or (
                col.startswith("bus_stop_")
                and col
                not in {
                    "bus_stop_mean_distance_meters",
                    "bus_stop_median_distance_meters",
                }
            )
            or col.startswith("population_")
            or (
                col.startswith("buildings_")
                and col not in {"buildings_yes", "buildings_yesmm"}
            )
        ]

        # assert set(all_nodes_df.drop(columns=node_attr_to_drop).columns) == set(
        #     node_attr_to_category + node_attr_num
        # )

        node_attr_to_drop = [
            col for col in node_attr_to_drop if col in all_nodes_df.columns
        ]
        node_attr_to_category = [
            col for col in node_attr_to_category if col in all_nodes_df.columns
        ]
        node_attr_num = [col for col in node_attr_num if col in all_nodes_df.columns]

        clean_node_attr_df = (
            all_nodes_df
            # Drop the columns that are not needed
            .drop(node_attr_to_drop, axis=1)
            # Convert both to the same type of Null
            .replace({"MVITIPO": {" ": "Null"}})
            # Replace the "Null" string with None
            .replace({"Null": None})
            # Convert the columns to the correct types
            .astype(
                {col: "category" for col in node_attr_to_category}
                | {col: "float32" for col in node_attr_num}
            )
            # Fill the NaN values with 0 for the numerical columns
            .fillna({col: 0 for col in node_attr_num})
            # One-hot encode the categorical columns
            .pipe(
                lambda df: pd.get_dummies(
                    df,
                    columns=node_attr_to_category,
                    prefix=node_attr_to_category,
                    prefix_sep="_",
                    drop_first=False,
                    dummy_na=True,
                )
            )
        )
        # Convert bool columns to int
        clean_node_attr_df = clean_node_attr_df.astype(
            {
                col: "int8"
                for col in clean_node_attr_df.select_dtypes(include=["bool"]).columns
            }
        )

        new_graph = nx.DiGraph()
        new_graph.add_nodes_from(base_graph.nodes)
        new_graph.add_edges_from(base_graph.edges(data=True))

        # Convert the DataFrame to the required dictionary format
        new_node_attributes = clean_node_attr_df.to_dict("index")

        # Set the new attributes on the new graph
        nx.set_node_attributes(new_graph, new_node_attributes)

        return new_graph


if __name__ == "__main__":
    processed_data_dir = "data/datos_procesados"

    with open(f"{processed_data_dir}/conectivity_graph/road_graph.pickle", "rb") as f:
        base_graph = pkl.load(f)

    # Clean the base graph node attributes
    base_graph = SparseDatasetGenerator._clean_node_attributes(base_graph)
    all_road_ids = list(base_graph.nodes())

    dataset_generator = SparseDatasetGenerator(
        node_ids=all_road_ids,
        min_timeframe=pd.to_datetime("2024-01-01 00:00:00"),
        max_timeframe=pd.to_datetime("2024-12-31 23:45:00"),
        freq="15min",
    )

    ## Generate the static dataset
    static_node_data = pd.DataFrame.from_dict(
        dict(base_graph.nodes(data=True)), orient="index"
    )
    all_road_ids = static_node_data.index.tolist()

    road_state_df = pd.read_parquet(
        f"{processed_data_dir}/road_state/road_state_dateset.parquet"
    )
    road_state_df = road_state_df.drop(
        columns=["timeframe_start", "timeframe_end"]
    ).set_index("road_id")

    static_df = static_node_data.join(road_state_df, how="left")
    static_df.index.name = "road_id"
    del base_graph, road_state_df, static_node_data  # Clean up memory

    dataset_generator.static_dataframe_to_coo(
        static_df,
        output_path=f"{processed_data_dir}/sparse_datasets/static_dataset.h5",
        has_index=True,
        return_df=False,
    )

    ## Generate the temporal datasets
    # Events
    events_df = pd.read_csv(f"{processed_data_dir}/events/events_dataset.csv")
    dataset_generator.temporal_dataframe_to_coo(
        events_df,
        output_path=f"{processed_data_dir}/sparse_datasets/events_dataset.h5",
        has_index=False,
        node_col="road_id",
        timeframe_col="timeframe",
        return_df=False,
    )
    del events_df  # Clean up memory

    # Incidents
    incidents_df = pd.read_csv(f"{processed_data_dir}/incidents/incidents_dataset.csv")
    dataset_generator.temporal_dataframe_to_coo(
        incidents_df,
        output_path=f"{processed_data_dir}/sparse_datasets/incidents_dataset.h5",
        has_index=False,
        node_col="road_id",
        timeframe_col="timeframe",
        return_df=False,
    )
    del incidents_df  # Clean up memory

    # Irregularities
    irregularities_filenames = [
        f"{processed_data_dir}/irregularidades_procesadas/" + f
        for f in os.listdir(f"{processed_data_dir}/irregularidades_procesadas/")
        if f.endswith(".parquet")
    ]
    irregularities_df = pd.concat(
        [pd.read_parquet(f) for f in irregularities_filenames],
        ignore_index=False,
    ).fillna(0)
    dataset_generator.temporal_dataframe_to_coo(
        irregularities_df,
        output_path=f"{processed_data_dir}/sparse_datasets/irregularities_dataset.h5",
        has_index=False,
        node_col="id_mvi",
        timeframe_col="timeframe",
        return_df=False,
    )

    del irregularities_df  # Clean up memory

    # COI Points
    coi_points_filenames = [
        f"{processed_data_dir}/pmts_processed/coi_puntos/" + f
        for f in os.listdir(f"{processed_data_dir}/pmts_processed/coi_puntos/")
        if f.endswith(".parquet")
    ]
    coi_points_df = pd.concat(
        [pd.read_parquet(f) for f in coi_points_filenames],
        ignore_index=False,
    ).fillna(0)
    dataset_generator.temporal_dataframe_to_coo(
        coi_points_df,
        output_path=f"{processed_data_dir}/sparse_datasets/coi_points_dataset.h5",
        has_index=False,
        node_col="road_id",
        timeframe_col="timeframe",
        return_df=False,
    )
    del coi_points_df  # Clean up memory

    # COI Tracts
    coi_tracts_filenames = [
        f"{processed_data_dir}/pmts_processed/coi_tramos/" + f
        for f in os.listdir(f"{processed_data_dir}/pmts_processed/coi_tramos/")
        if f.endswith(".parquet")
    ]
    coi_tracts_df = pd.concat(
        [pd.read_parquet(f) for f in coi_tracts_filenames],
        ignore_index=False,
    ).fillna(0)
    dataset_generator.temporal_dataframe_to_coo(
        coi_tracts_df,
        output_path=f"{processed_data_dir}/sparse_datasets/coi_tracts_dataset.h5",
        has_index=False,
        node_col="road_id",
        timeframe_col="timeframe",
        return_df=False,
    )
    del coi_tracts_df  # Clean up memory

    print("---------------------------------------------------")
    print("----- Sparse datasets generated successfully. -----")
    print("---------------------------------------------------")
