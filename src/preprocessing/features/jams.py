import gc
import glob
from pathlib import Path
from typing import List

import geopandas as gpd
import pandas as pd
from shapely import wkb, wkt
from tqdm.auto import tqdm


class JamsProcessor:
    """
    A class for processing Waze traffic jam data and assigning jams to road segments.

    This processor handles the complete pipeline from raw Waze jams data files to
    spatially-assigned, temporally-aggregated traffic metrics suitable for training
    Graph Neural Networks for traffic forecasting.

    The processor uses a chunked approach to handle large datasets efficiently, with
    automatic memory management and intermediate result saving.

    Attributes:
        jams_data_dir (str): Directory containing raw Waze jams parquet files
        road_network_path (str): Path to the road network GeoJSON file
        output_dir (str): Directory where processed files will be saved
        buffer_distance (float): Buffer distance in meters for spatial intersections (default: 4.0)
        overlap_threshold (float): Minimum overlap ratio to keep jam-road matches (default: 0.2)
        target_crs (str): Target coordinate reference system for spatial operations (default: "EPSG:3116")
        temporal_resolution (str): Temporal aggregation frequency (default: "15min")
        filter_closed_roads (bool): Whether to exclude jams with delay == -1 (default: True)
    """

    def __init__(
        self,
        jams_data_dir: str,
        road_network_path: str,
        output_dir: str,
        buffer_distance: float = 4.0,
        overlap_threshold: float = 0.2,
        target_crs: str = "EPSG:3116",
        temporal_resolution: str = "15min",
        filter_closed_roads: bool = True,
    ):
        """
        Initialize the JamsProcessor with data paths and processing parameters.

        Args:
            jams_data_dir: Directory containing raw Waze jams parquet files
            road_network_path: Path to the road network GeoJSON file (MVI format)
            output_dir: Directory where processed jam data will be saved
            buffer_distance: Buffer distance in meters around road segments for spatial intersection (default: 4.0)
            overlap_threshold: Minimum fraction of road length that must overlap with jam (default: 0.2)
            target_crs: EPSG code for projected coordinate system used in spatial operations (default: "EPSG:3116")
            temporal_resolution: Pandas frequency string for temporal aggregation (default: "15min")
            filter_closed_roads: If True, filter out jams with delay == -1 (closed roads) (default: True)
        """
        self.jams_data_dir = Path(jams_data_dir)
        self.road_network_path = road_network_path
        self.output_dir = Path(output_dir)
        self.buffer_distance = buffer_distance
        self.overlap_threshold = overlap_threshold
        self.target_crs = target_crs
        self.temporal_resolution = temporal_resolution
        self.filter_closed_roads = filter_closed_roads

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Storage for processed data
        self.road_network_buffered: gpd.GeoDataFrame | None = None
        self.jam_files: List[str] = []

    def load_road_network(self) -> gpd.GeoDataFrame:
        """
        Load and process the road network with buffered geometries for spatial joins.

        This method:
        1. Loads the road network geometry
        2. Converts to WGS84 (EPSG:4326) for consistency with Waze data
        3. Creates unique road IDs from MVI codes
        4. Transforms to projected CRS for accurate spatial operations
        5. Computes road segment lengths
        6. Creates buffers around road segments for intersection detection

        Returns:
            gpd.GeoDataFrame: Processed road network with columns:
                - road_id: Unique identifier for each road segment
                - mvi_length: Length of the road segment in meters
                - buffer_geom: Buffered polygon geometry in projected CRS
        """
        print("Loading and preparing road network...")
        mvi = gpd.read_file(self.road_network_path)

        # Convert to WGS84 for consistency with Waze data
        mvi = mvi.to_crs(epsg=4326)

        # Standardize column names
        mvi.columns = mvi.columns.str.lower()

        # Create unique road IDs from MVI codes
        mvi["id_mvi"] = (
            mvi["mviciv"].astype(str) + "_" + mvi["mvisvia"].str.replace(" ", "")
        )
        mvi["contador"] = mvi.groupby("id_mvi").cumcount() + 1
        mvi["road_id"] = mvi["id_mvi"].astype(str) + "." + mvi["contador"].astype(str)

        # Transform to projected CRS and create buffers
        mvi_buffered = mvi.to_crs(self.target_crs).copy()
        mvi_buffered["mvi_length"] = mvi_buffered.geometry.length

        # Buffer road segments for spatial intersection with jam geometries
        mvi_buffered["buffer_geom"] = mvi_buffered.geometry.buffer(
            self.buffer_distance, cap_style="square"
        )

        # Set buffer geometry as the active geometry
        mvi_buffered = mvi_buffered.set_geometry("buffer_geom")

        print(
            f"  Loaded {len(mvi_buffered)} road segments with {self.buffer_distance}m buffers"
        )
        return mvi_buffered[["road_id", "mvi_length", "buffer_geom"]]

    def discover_jam_files(self) -> List[str]:
        """
        Discover all parquet files in the jams data directory.

        Returns:
            List[str]: Sorted list of paths to jam data parquet files
        """
        pattern = str(self.jams_data_dir / "*.parquet")
        files = sorted(glob.glob(pattern))
        print(f"Discovered {len(files)} jam data files")
        return files

    def load_and_prepare_jam_file(self, file_path: str) -> gpd.GeoDataFrame:
        """
        Load a single jam data file and prepare it for processing.

        This method:
        1. Loads the parquet file
        2. Parses geometry from WKT or WKB format
        3. Converts timestamps from milliseconds to datetime
        4. Filters relevant columns
        5. Creates a GeoDataFrame in WGS84

        Args:
            file_path: Path to the jam data parquet file

        Returns:
            gpd.GeoDataFrame: Prepared jam data with columns:
                - uuid: Unique jam identifier
                - timestamp_pubMillis: Publication timestamp
                - timestamp_startTime: Jam start time
                - timestamp_endTime: Jam end time
                - speedKMH: Speed in km/h
                - level: Traffic level (congestion severity)
                - length: Jam length in meters
                - delay: Delay in seconds
                - geometry: LineString geometry of the jam
        """
        df = pd.read_parquet(file_path)

        # Parse geometry from WKT or WKB format
        df["geometry"] = df["geom_wkt"].apply(
            lambda geom: (
                wkb.loads(geom)
                if isinstance(geom, bytes)
                else wkt.loads(geom)
                if isinstance(geom, str)
                else geom
            )  # type: ignore
        )

        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame(df, geometry="geometry", crs=4326)

        # Parse timestamps
        gdf["timestamp_pubMillis"] = pd.to_datetime(gdf["pubMillis_ts_cot"], unit="ms")
        gdf["timestamp_startTime"] = pd.to_datetime(gdf["startTime_ts_cot"], unit="ms")
        gdf["timestamp_endTime"] = pd.to_datetime(gdf["endTime_ts_cot"], unit="ms")

        # Filter relevant columns
        gdf = gdf[
            [
                "uuid",
                "timestamp_pubMillis",
                "timestamp_startTime",
                "timestamp_endTime",
                "speedKMH",
                "level",
                "length",
                "delay",
                "geometry",
            ]
        ]

        return gdf

    def aggregate_jam_metadata(self, jams_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Aggregate metadata for jams with the same UUID.

        This computes the overall time range and count for each unique jam event,
        as jams may appear in multiple observations.

        Args:
            jams_gdf: GeoDataFrame with jam observations

        Returns:
            gpd.GeoDataFrame: Input dataframe with added columns:
                - general_timestamp_startTime: Earliest start time for this jam UUID
                - general_timestamp_endTime: Latest end time for this jam UUID
                - jam_id_counts: Number of observations for this jam UUID
        """
        # Add jam_id for counting
        jams_gdf["jam_id"] = jams_gdf.index

        # Group by UUID to get overall time range and count
        grouped_jams = (
            jams_gdf.groupby("uuid")
            .agg(
                {
                    "timestamp_startTime": "min",
                    "timestamp_endTime": "max",
                    "jam_id": "count",
                }
            )
            .reset_index()
        )
        grouped_jams.columns = [
            "uuid",
            "general_timestamp_startTime",
            "general_timestamp_endTime",
            "jam_id_counts",
        ]

        # Merge back to original dataframe
        jams_gdf = jams_gdf.merge(grouped_jams, how="left", on="uuid")

        return jams_gdf

    def intersect_jams_with_roads(self, jams_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Perform spatial intersection between jams and buffered road network.

        This method:
        1. Filters out closed roads if configured (delay == -1)
        2. Converts jams to projected CRS
        3. Performs spatial overlay (intersection) with road buffers
        4. Calculates intersection length and overlap ratio
        5. Filters matches by overlap threshold
        6. Adjusts delay proportionally to intersection length

        Args:
            jams_gdf: GeoDataFrame with jam geometries

        Returns:
            gpd.GeoDataFrame: Jams matched to road segments with columns:
                - All original jam columns
                - road_id: Matched road segment ID
                - mvi_length: Length of the road segment
                - intersection_length: Length of jam-road intersection
                - overlap_ratio: intersection_length / mvi_length
                - delay: Adjusted delay proportional to intersection
        """
        # Filter closed roads if configured
        if self.filter_closed_roads:
            jams_gdf = jams_gdf[jams_gdf["delay"] != -1]
            print(
                f"  Filtered closed roads (delay != -1), {len(jams_gdf)} jams remaining"
            )

        # Convert to projected CRS for accurate spatial operations
        jams_gdf = jams_gdf.to_crs(self.target_crs)

        # Perform spatial overlay (intersection)
        print("  Performing spatial intersection with road network...")
        if self.road_network_buffered is None:
            raise ValueError("Road network not loaded. Call load_road_network() first.")

        jams_gdf = gpd.overlay(
            jams_gdf,
            self.road_network_buffered,
            how="intersection",
        )

        # Calculate intersection metrics
        jams_gdf["intersection_length"] = jams_gdf.geometry.length
        jams_gdf["overlap_ratio"] = (
            jams_gdf["intersection_length"] / jams_gdf["mvi_length"]
        )

        # Filter by overlap threshold
        initial_count = len(jams_gdf)
        jams_gdf = jams_gdf[jams_gdf["overlap_ratio"] > self.overlap_threshold]
        print(
            f"  Filtered by overlap threshold ({self.overlap_threshold}): "
            f"{len(jams_gdf)}/{initial_count} matches kept"
        )

        # Adjust delay proportionally to intersection length
        jams_gdf["delay"] = jams_gdf["delay"] * (
            jams_gdf["intersection_length"] / jams_gdf["length"]
        )

        return jams_gdf

    def aggregate_to_timeframes(self, jams_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
        """
        Aggregate jam data to specified temporal resolution by road segment.

        This method:
        1. Creates timeframe bins (e.g., 15-minute intervals)
        2. Groups by road_id and timeframe
        3. Computes aggregate statistics:
           - Speed: min, mean, max
           - Level: min, mean, max
           - Delay: min, sum, max
           - Count: number of jams

        Args:
            jams_gdf: GeoDataFrame with jams matched to roads

        Returns:
            pd.DataFrame: Aggregated data with columns:
                - road_id: Road segment identifier
                - timeframe: Temporal bin (e.g., "2024-01-01 12:00:00")
                - vel_min, vel_mean, vel_max: Speed statistics
                - level_min, level_mean, level_max: Traffic level statistics
                - delay_min, delay_sum, delay_max: Delay statistics
                - count_jams: Number of jams in this timeframe
        """
        # Create timeframe bins
        jams_gdf["timeframe"] = jams_gdf["timestamp_pubMillis"].dt.floor(  # type: ignore
            self.temporal_resolution
        )

        # Group and aggregate
        print(f"  Aggregating to {self.temporal_resolution} timeframes...")
        aggregated = (
            jams_gdf.groupby(["road_id", "timeframe"])
            .agg(
                {
                    "speedKMH": ["min", "mean", "max"],
                    "level": ["min", "mean", "max"],
                    "delay": ["min", "sum", "max"],
                    "jam_id": "count",
                }
            )
            .reset_index()
        )

        # Flatten column names
        aggregated.columns = [
            "road_id",
            "timeframe",
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

        # Sort for consistency
        aggregated.sort_values(by=["road_id", "timeframe"], inplace=True)

        print(f"  Generated {len(aggregated)} aggregated records")
        return aggregated

    def process_file_chunk(
        self, file_group: List[str], chunk_index: int
    ) -> pd.DataFrame | None:
        """
        Process a chunk of jam data files.

        This method:
        1. Loads and concatenates all files in the chunk
        2. Aggregates jam metadata by UUID
        3. Performs spatial intersection with roads
        4. Aggregates to temporal resolution
        5. Returns the processed dataframe

        Args:
            file_group: List of file paths in this chunk
            chunk_index: Index of this chunk for logging

        Returns:
            pd.DataFrame | None: Aggregated jam data for this chunk, or None if empty
        """
        print(f"\n--- Processing chunk {chunk_index + 1} ({len(file_group)} files) ---")

        # Load all files in the chunk
        jams_gdf_list = []
        for file_path in tqdm(file_group, desc="Loading files"):
            gdf = self.load_and_prepare_jam_file(file_path)
            jams_gdf_list.append(gdf)

        if not jams_gdf_list:
            print("  No data in chunk, skipping...")
            return None

        # Concatenate all dataframes
        jams_gdf = gpd.GeoDataFrame(
            pd.concat(jams_gdf_list, ignore_index=True), crs=4326
        )
        print(f"  Loaded {len(jams_gdf)} jam records")

        # Free memory
        del jams_gdf_list
        gc.collect()

        # Aggregate metadata by UUID
        jams_gdf = self.aggregate_jam_metadata(jams_gdf)

        # Spatial intersection with roads
        jams_gdf = self.intersect_jams_with_roads(jams_gdf)

        # Temporal aggregation
        aggregated = self.aggregate_to_timeframes(jams_gdf)

        # Free memory
        del jams_gdf
        gc.collect()

        return aggregated

    def process_and_save(self, n_file_chunks: int = 5) -> None:
        """
        Execute the complete jam data processing pipeline with chunked processing.

        This method orchestrates the entire workflow:
        1. Discovers all jam data files
        2. Loads the road network
        3. Processes files in chunks to manage memory
        4. Saves each chunk's results to a separate parquet file
        5. Uses garbage collection to free memory between chunks

        The chunking strategy allows processing of large datasets that wouldn't fit
        in memory all at once. Each chunk is saved with a filename indicating the
        date range of the source files.

        Args:
            n_file_chunks: Number of files to process in each chunk (default: 5)
                Adjust based on available memory and file sizes

        Output files:
            {output_dir}/jams_proc_{start_date}_{end_date}.parquet: One file per chunk
        """
        print("=" * 70)
        print("Starting Waze Jams Data Processing Pipeline")
        print("=" * 70)

        # Step 1: Discover jam data files
        print("\n[1/3] Discovering jam data files...")
        self.jam_files = self.discover_jam_files()

        if not self.jam_files:
            print("No jam files found. Exiting.")
            return

        # Step 2: Load road network
        print("\n[2/3] Loading road network...")
        self.road_network_buffered = self.load_road_network()

        # Step 3: Process files in chunks
        print(f"\n[3/3] Processing jam data in chunks (chunk size: {n_file_chunks})...")

        chunk_size = len(self.jam_files) // n_file_chunks + 1
        total_chunks = chunk_size

        for i in range(chunk_size):
            # Get file group for this chunk
            start_idx = i * n_file_chunks
            end_idx = (i + 1) * n_file_chunks
            file_group = self.jam_files[start_idx:end_idx]

            if not file_group:
                continue

            # Extract date range from filenames for output naming
            min_file = min(file_group)
            max_file = max(file_group)

            # Extract dates from filenames (format: jams_data_YYYY-MM-DD.parquet)
            try:
                start_date = Path(min_file).stem.split("_")[-1]  # Gets YYYY-MM-DD
                end_date = Path(max_file).stem.split("_")[-1]
            except Exception:
                start_date = f"chunk_{i}_start"
                end_date = f"chunk_{i}_end"

            output_file_name = (
                self.output_dir / f"jams_proc_{start_date}_{end_date}.parquet"
            )

            # Process this chunk
            aggregated = self.process_file_chunk(file_group, i)

            if aggregated is None or len(aggregated) == 0:
                print(f"  Skipping empty chunk {i + 1}")
                continue

            # Save chunk results
            aggregated.to_parquet(output_file_name, index=False)
            print(f"  ✓ Saved chunk {i + 1}/{total_chunks} to: {output_file_name}")

            # Free memory
            del aggregated
            gc.collect()

        print("\n" + "=" * 70)
        print("Waze Jams Data Processing Complete!")
        print("=" * 70)
        print(f"\nAll outputs saved to: {self.output_dir}")


if __name__ == "__main__":
    # Define input data paths
    jams_data_dir = "data/4_velocidades_waze/jams/jams_originales"
    road_network_path = "data/2_Estructura_via/estructura_via_completa/MVI.geojson"
    output_dir = "data/datos_procesados/jams_procesados/"

    # Initialize processor
    processor = JamsProcessor(
        jams_data_dir=jams_data_dir,
        road_network_path=road_network_path,
        output_dir=output_dir,
        buffer_distance=4.0,
        overlap_threshold=0.2,
        target_crs="EPSG:3116",
        temporal_resolution="15min",
        filter_closed_roads=True,
    )

    # Execute processing pipeline with chunked processing
    # Adjust n_file_chunks based on available memory
    processor.process_and_save(n_file_chunks=5)

    print("\nProcessing complete!")
