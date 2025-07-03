import os
import pickle as pkl
import shutil
from datetime import datetime, timedelta

import networkx as nx
import pandas as pd
import pytest

from src.dataloader import GraphTimeSeriesLoader


@pytest.fixture(scope="module")
def temp_data_dir():
    """Create a temporary directory for dummy data files and clean it up after tests."""
    DATA_DIR = "test_temp_data"
    os.makedirs(DATA_DIR, exist_ok=True)
    yield DATA_DIR
    shutil.rmtree(DATA_DIR)


@pytest.fixture(scope="module")
def file_paths(temp_data_dir: str) -> dict[str, str | list[str]]:
    """Creates dummy data files and returns a dictionary of their paths."""
    # Create and save Base Graph
    G = nx.DiGraph()
    G.add_node(101, static_value_1=5.2, static_value_2=1.8)
    G.add_node(102, static_value_1=2.1, static_value_2=3.4)
    G.add_node(
        103, static_value_1=8.5, static_value_2=2.3
    )  # Node with no temporal data
    graph_filename = os.path.join(temp_data_dir, "base_graph.gpickle")
    with open(graph_filename, "wb") as f:
        pkl.dump(G, f)

    # Create and save other data sources to files
    buildings_filename = os.path.join(temp_data_dir, "buildings.csv")
    buildings_df = pd.DataFrame(
        {
            "road_id": [101, 102, 103],
            "type": [1, 1, 2],
            "height": [10.5, 20.0, 15.0],
            "location": [10.0, 20.0, 30.0],
        }
    )
    buildings_df.to_csv(buildings_filename, index=False)

    bus_stops_filename = os.path.join(temp_data_dir, "bus_stops.parquet")
    bus_stops_df = pd.DataFrame(
        {
            "road_id": [101, 102],
            "stop_type_1": [0, 1],
            "stop_type_2": [1, 0],
            "location": [10.0, 20.0],
        }
    )
    bus_stops_df.to_parquet(bus_stops_filename)

    events_filename = os.path.join(temp_data_dir, "events.csv")
    events_df = pd.DataFrame(
        {
            "timeframe": pd.to_datetime(
                [
                    "2024-07-10 08:00:00",
                    "2024-07-10 08:15:00",
                    "2024-07-10 08:45:00",
                    "2024-07-10 08:45:00",
                ]
            ),
            "road_id": [101, 102, 101, 102],
            "congestion_level": [1, 2, 1, 0],
        }
    )
    events_df.to_csv(events_filename, index=False)

    incidents_filename = os.path.join(temp_data_dir, "incidents.csv")
    incidents_df = pd.DataFrame(
        {
            "timeframe": pd.to_datetime(["2024-07-10 08:15:00"]),
            "road_id": [101],
            "severity": [3],
        }
    )
    incidents_df.to_csv(incidents_filename, index=False)

    road_state_filename = os.path.join(temp_data_dir, "road_state.parquet")
    road_state_df = pd.DataFrame(
        {
            "timeframe_start": pd.to_datetime(
                ["2024-07-10 08:00:00", "2024-07-10 08:30:00"]
            ),
            "timeframe_end": pd.to_datetime(
                ["2024-07-10 08:29:59", "2024-07-10 08:59:59"]
            ),
            "road_id": [102, 102],
            "state": [5, 6],
        }
    )
    road_state_df.to_parquet(road_state_filename)

    # Create Parquet Files for list-based sources
    IRREG_DIR = os.path.join(temp_data_dir, "irregularities")
    os.makedirs(IRREG_DIR, exist_ok=True)
    irreg_df_1 = pd.DataFrame(
        {
            "timeframe": pd.to_datetime(["2024-07-10 08:15:00"]),
            "id_mvi": [101],
            "irreg_type": [0.9],
            "jamLevel": [2],
        }
    )
    irreg_f1 = os.path.join(IRREG_DIR, "irregs_proc_2024-07-01_2024-07-15.parquet")
    irreg_df_1.to_parquet(irreg_f1)

    irreg_f2 = os.path.join(IRREG_DIR, "irregs_proc_2024-08-01_2024-08-15.parquet")
    pd.DataFrame(
        {"timeframe": [], "id_mvi": [], "irreg_type": [], "jamLevel": []}
    ).to_parquet(irreg_f2)

    # COI Points
    COI_P_DIR = os.path.join(temp_data_dir, "coi_points")
    os.makedirs(COI_P_DIR, exist_ok=True)
    coi_p_df = pd.DataFrame(
        {
            "timeframe": pd.to_datetime(["2024-07-10 08:30:00"]),
            "road_id": [102],
            "coi_point_type": [1],
        }
    )
    coi_p_f1 = os.path.join(COI_P_DIR, "coi_puntos_dataset_20240701_20240731.parquet")
    coi_p_df.to_parquet(coi_p_f1)

    # COI Tracts
    COI_T_DIR = os.path.join(temp_data_dir, "coi_tracts")
    os.makedirs(COI_T_DIR, exist_ok=True)
    coi_t_df = pd.DataFrame(
        {
            "timeframe": pd.to_datetime(["2025-07-10 08:00:00"]),
            "road_id": [101],
            "coi_tract_type": [3],
        }
    )
    coi_t_f1 = os.path.join(COI_T_DIR, "coi_tramo_dataset_20250701_20250731.parquet")
    coi_t_df.to_parquet(coi_t_f1)

    return {
        "graph_pkl_filename": graph_filename,
        "buildings_csv_filename": buildings_filename,
        "bus_stops_parquet_filename": bus_stops_filename,
        "events_csv_filename": events_filename,
        "incidents_csv_filename": incidents_filename,
        "road_state_parquet_filename": road_state_filename,
        "irregularities_filenames": [irreg_f1, irreg_f2],
        "coi_points_filenames": [coi_p_f1],
        "coi_tracts_filenames": [coi_t_f1],
    }


@pytest.fixture
def data_loader(file_paths: dict[str, str | list[str]]) -> GraphTimeSeriesLoader:
    """Initializes the GraphTimeSeriesLoader with the test data file paths."""
    assert isinstance(file_paths["graph_pkl_filename"], str)
    assert isinstance(file_paths["buildings_csv_filename"], str)
    assert isinstance(file_paths["bus_stops_parquet_filename"], str)
    assert isinstance(file_paths["events_csv_filename"], str)
    assert isinstance(file_paths["incidents_csv_filename"], str)
    assert isinstance(file_paths["road_state_parquet_filename"], str)
    assert isinstance(file_paths["irregularities_filenames"], list)
    assert isinstance(file_paths["coi_points_filenames"], list)
    assert isinstance(file_paths["coi_tracts_filenames"], list)

    return GraphTimeSeriesLoader(
        graph_pkl_filename=file_paths["graph_pkl_filename"],
        buildings_csv_filename=file_paths["buildings_csv_filename"],
        bus_stops_parquet_filename=file_paths["bus_stops_parquet_filename"],
        events_csv_filename=file_paths["events_csv_filename"],
        incidents_csv_filename=file_paths["incidents_csv_filename"],
        road_state_parquet_filename=file_paths["road_state_parquet_filename"],
        irregularities_filenames=file_paths["irregularities_filenames"],
        coi_points_filenames=file_paths["coi_points_filenames"],
        coi_tracts_filenames=file_paths["coi_tracts_filenames"],
    )


def test_initialization(data_loader):
    """Test that the loader initializes and loads data from files correctly."""
    # Check that data is loaded into attributes, assuming the loader does this.
    assert isinstance(data_loader.base_graph, nx.DiGraph)
    assert len(data_loader.base_graph.nodes) == 3
    assert pd.api.types.is_datetime64_any_dtype(data_loader.events_df["timeframe"])
    assert pd.api.types.is_datetime64_any_dtype(data_loader.incidents_df["timeframe"])
    assert pd.api.types.is_datetime64_any_dtype(
        data_loader.road_state_df["timeframe_start"]
    )

    # Test file list preprocessing
    assert len(data_loader.irregularities_files) == 2
    assert data_loader.irregularities_files[0][0] == pd.to_datetime("2024-07-01")
    # Check that end date is end of day
    assert data_loader.irregularities_files[0][1] == pd.to_datetime(
        "2024-07-15 23:59:59.999999"
    )


@pytest.mark.parametrize(
    "start, end, expected_count, expected_filename_part",
    [
        # Case 1: Query window is within a file's range
        (
            pd.to_datetime("2024-07-10"),
            pd.to_datetime("2024-07-11"),
            1,
            "irregs_proc_2024-07-01_2024-07-15.parquet",
        ),
        # Case 2: Query window has no overlap
        (pd.to_datetime("2024-09-01"), pd.to_datetime("2024-09-02"), 0, None),
        # Case 3: Query window overlaps the start of a file's range
        (
            pd.to_datetime("2024-06-30"),
            pd.to_datetime("2024-07-01"),
            1,
            "irregs_proc_2024-07-01_2024-07-15.parquet",
        ),
        # Case 4: Query window overlaps the end of a file's range
        (
            pd.to_datetime("2024-07-15"),
            pd.to_datetime("2024-07-16"),
            1,
            "irregs_proc_2024-07-01_2024-07-15.parquet",
        ),
    ],
)
def test_find_relevant_files(
    data_loader, start, end, expected_count, expected_filename_part
):
    """Test the logic for finding files that overlap a time window."""
    file_list = data_loader.irregularities_files
    files = data_loader._find_relevant_files(file_list, start, end)
    assert len(files) == expected_count
    if expected_filename_part:
        assert any(expected_filename_part in f for f in files)


def test_expand_road_state_data(data_loader):
    """Test the expansion of range-based state data to discrete time points."""
    node_ids = [102]
    all_timeframes = pd.to_datetime(
        [
            "2024-07-10 08:00:00",
            "2024-07-10 08:15:00",
            "2024-07-10 08:30:00",
            "2024-07-10 08:45:00",
            "2024-07-10 09:00:00",
            "2024-07-10 09:15:00",
        ]
    )
    state_df = data_loader.road_state_df

    expanded_df = data_loader._expand_road_state_data(
        state_df, all_timeframes, node_ids
    )

    # Expected to have 4 rows, as 09:00 and 09:15 are not in the state data
    assert len(expanded_df) == 4
    assert pd.to_datetime("2024-07-10 09:00:00") not in expanded_df["timeframe"].values
    assert pd.to_datetime("2024-07-10 09:15:00") not in expanded_df["timeframe"].values
    assert set(expanded_df["road_id"].unique()) == {102}
    # States for 08:00, 08:15, 08:30, 08:45
    assert expanded_df["state"].tolist() == [5, 5, 6, 6]


def test_load_data_full(data_loader):
    """The main test case for the load_data method with data from all sources."""
    nodes_to_load = [101, 102]
    # Query for 1 hour (4 * 15 min intervals)
    time_windows_to_load = [("2024-07-10 08:30:00", 1, 2)]

    (static_df, temporal_df), target = data_loader.load_data(
        nodes_to_load, time_windows_to_load
    )

    # Check static data
    assert len(static_df) == 2
    assert static_df.index.tolist() == [101, 102]
    assert static_df.loc[101]["stop_type_1"] == 0
    assert static_df.loc[101]["stop_type_2"] == 1
    assert static_df.loc[102]["stop_type_2"] == 0
    assert static_df.loc[102]["stop_type_1"] == 1

    # Check temporal data shape and index
    assert len(temporal_df) == 8  # 2 nodes * 4 timeframes
    assert "road_id" in temporal_df.columns and "timeframe" in temporal_df.columns
    assert set(temporal_df["road_id"].unique()) == {101, 102}

    # Check specific data points
    # Set multi-index for easy lookup
    temporal_df = temporal_df.set_index(["road_id", "timeframe"])
    print(temporal_df.to_markdown())
    # All the events data should be present in the temporal_df since we queried for 4 timeframes
    assert temporal_df.loc[(101, "2024-07-10 08:00:00")]["congestion_level"] == 1
    assert temporal_df.loc[(102, "2024-07-10 08:15:00")]["congestion_level"] == 2
    assert temporal_df.loc[(101, "2024-07-10 08:45:00")]["congestion_level"] == 1
    assert temporal_df.loc[(102, "2024-07-10 08:45:00")]["congestion_level"] == 0
    # We expect severity to be 3 at 08:15 for road_id 101 and the rest to be 0
    assert temporal_df.loc[(101, "2024-07-10 08:15:00")]["severity"] == 3
    assert (temporal_df == 0)["severity"].sum() == len(
        temporal_df
    ) - 1  # All the rest should be 0
    # The road state data should be expanded to the timeframes
    assert temporal_df.loc[(102, "2024-07-10 08:00:00")]["state"] == 5
    assert temporal_df.loc[(102, "2024-07-10 08:15:00")]["state"] == 5
    assert temporal_df.loc[(102, "2024-07-10 08:30:00")]["state"] == 6
    assert temporal_df.loc[(102, "2024-07-10 08:45:00")]["state"] == 6
    # All the state values for node 101 should be 0 since it has no state data
    assert temporal_df.loc[(101, "2024-07-10 08:00:00")]["state"] == 0
    assert temporal_df.loc[(101, "2024-07-10 08:15:00")]["state"] == 0
    assert temporal_df.loc[(101, "2024-07-10 08:30:00")]["state"] == 0
    assert temporal_df.loc[(101, "2024-07-10 08:45:00")]["state"] == 0
    # Irregularities data should be present for node 101 at 08:15, and the rest 0
    assert temporal_df.loc[(101, "2024-07-10 08:15:00")]["irreg_type"] == 0.9
    assert (temporal_df == 0)["irreg_type"].sum() == len(temporal_df) - 1
    # COI Points data should be present for node 102 at 08:30, and the rest 0
    assert temporal_df.loc[(102, "2024-07-10 08:30:00")]["coi_point_type"] == 1
    assert (temporal_df == 0)["coi_point_type"].sum() == len(temporal_df) - 1
    # COI Tracts data should be 0 everywhere since the data is for 2025 (no overlap with query)
    assert (temporal_df == 0)["coi_tract_type"].sum() == len(temporal_df)

    target = target.set_index(["road_id", "timeframe"])
    # Check target values
    # The only one with data is node 101 at 08:15
    assert target.loc[(101, "2024-07-10 08:15:00")]["jam_level"] == 2
    # The rest should be 0
    assert (target == 0)["jam_level"].sum() == len(target) - 1
    # Check that the target index matches the temporal_df index
    assert set(target.index.get_level_values("road_id")) == set(
        temporal_df.index.get_level_values("road_id")
    )


def test_load_data_no_temporal_match(data_loader):
    """Test that an empty but correctly shaped DataFrame is returned when no data is found."""
    nodes_to_load = [103]  # This node has no temporal data
    time_windows_to_load = [("2024-07-10 08:30:00", 2, 1)]

    (static_df, temporal_df), target = data_loader.load_data(
        nodes_to_load, time_windows_to_load
    )
    print("STATIC DATAFRAME:")
    print(static_df.to_markdown())
    print("TEMPORAL DATAFRAME:")
    print(temporal_df.to_markdown())

    assert len(static_df) == 1
    assert static_df.index.tolist() == [103]
    assert len(temporal_df) == 4  # 1 node * 4 timeframes

    # All temporal feature columns should be present and filled with 0
    expected_cols = {
        "road_id",
        "timeframe",
        "congestion_level",
        "severity",
        "state",
        "irreg_type",
        "coi_point_type",
        "coi_tract_type",
    }
    assert expected_cols.issubset(temporal_df.columns)

    # Check if all values (except id/time) are 0
    feature_cols = temporal_df.drop(columns=["road_id", "timeframe"])
    assert (feature_cols.values == 0).all()


def test_load_data_empty_inputs(data_loader):
    """Test behavior with empty node_ids or time_windows."""
    # Empty node_ids
    (static_df, temporal_df), target = data_loader.load_data(
        [], [("2024-07-10 08:00:00", 4, 0)]
    )
    assert static_df.empty
    assert temporal_df.empty

    # Empty time_windows
    (static_df, temporal_df), target = data_loader.load_data([101], [])
    assert len(static_df) == 1
    assert temporal_df.empty


def test_load_data_symmetric_window(data_loader):
    """Test the load_data method with a symmetric window (backwards and forwards)."""
    nodes_to_load = [101, 102]
    # Query for a symmetric window around a center time
    time_windows_to_load = [("2024-07-10 08:30:00", 3, 3)]  # 3 backwards, 3 forwards

    (static_df, temporal_df), target = data_loader.load_data(
        nodes_to_load, time_windows_to_load
    )

    # Check static data
    assert len(static_df) == 2
    assert static_df.index.tolist() == [101, 102]

    # Check temporal data shape and index
    assert len(temporal_df) == 14  # 2 nodes * (3 back + 1 center + 3 forward)
    assert "road_id" in temporal_df.columns and "timeframe" in temporal_df.columns
    assert set(temporal_df["road_id"].unique()) == {101, 102}

    # Check specific data points
    temporal_df = temporal_df.set_index(["road_id", "timeframe"])
    print(temporal_df.to_markdown())
    # Check that the timeframes are correctly expanded
    expected_timeframes = pd.date_range(
        start="2024-07-10 07:45:00", end="2024-07-10 09:15:00", freq="15min"
    )
    assert set(temporal_df.index.get_level_values("timeframe")) == set(
        expected_timeframes
    )


@pytest.mark.parametrize(
    "timeframe_unmatched,timeframe_rounded",
    [
        ("2024-07-10 08:31:12", "2024-07-10 08:30:00"),  # Unmatched, should round down
        ("2024-07-10 08:29:59", "2024-07-10 08:30:00"),  # Unmatched, should round up
    ],
)
def test_load_data_non_matching_timeframe(
    data_loader, timeframe_unmatched, timeframe_rounded
):
    """Test that no data is returned when the timeframe does not match any data."""
    nodes_to_load = [101, 102]
    # Query for a timeframe that does not match any data
    time_windows_to_load = [(timeframe_unmatched, 3, 3)]
    (static_df, temporal_df), target = data_loader.load_data(
        nodes_to_load, time_windows_to_load
    )

    # No data at this exact time, but it should be treated as if
    # the timeframe was rounded to the nearest 15 minutes
    target_time_windows_to_load = [(timeframe_rounded, 3, 3)]
    (target_static_df, target_temporal_df), target_target = data_loader.load_data(
        nodes_to_load, target_time_windows_to_load
    )

    # Assert that the dataframes are the same
    pd.testing.assert_frame_equal(static_df, target_static_df)
    pd.testing.assert_frame_equal(temporal_df, target_temporal_df)
    pd.testing.assert_frame_equal(target, target_target)
