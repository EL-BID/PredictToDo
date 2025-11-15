from pathlib import Path
from typing import Dict, List, Tuple

import geopandas as gpd
import pandas as pd
from shapely import wkb, wkt
from tqdm.auto import tqdm


class IrregularitiesProcessor:
    """
    A class for processing Waze traffic irregularities data with spatial-temporal aggregation.

    This processor handles the complete pipeline from raw Waze irregularities data to
    spatially and temporally aggregated features suitable for training Graph Neural Networks
    for traffic forecasting.

    The key innovation is dual weighting: features are weighted both by temporal overlap
    (how much of the 15-minute interval the irregularity covers) and spatial overlap
    (how much of the road segment the irregularity covers).

    Attributes:
        input_folder (str): Path to folder containing irregularities parquet files
        road_network_path (str): Path to the road network GeoJSON file
        output_dir (str): Directory where processed files will be saved
        n_chunks (int): Number of files to process per batch (default: 10)
        buffer_distance (float): Buffer distance in meters for spatial joins (default: 4)
        overlap_threshold (float): Minimum overlap ratio to keep a match (default: 0.2)
        temporal_resolution (str): Temporal resolution for aggregation (default: "15min")
    """

    def __init__(
        self,
        input_folder: str,
        road_network_path: str,
        output_dir: str,
        n_chunks: int = 10,
        buffer_distance: float = 4.0,
        overlap_threshold: float = 0.2,
        temporal_resolution: str = "15min",
    ):
        """
        Initialize the IrregularitiesProcessor with data paths and processing parameters.

        Args:
            input_folder: Path to folder containing irregularities parquet files
            road_network_path: Path to the road network GeoJSON file (MVI format)
            output_dir: Directory where processed irregularities data will be saved
            n_chunks: Number of files to process in each batch (default: 10)
            buffer_distance: Buffer distance in meters around road segments (default: 4)
            overlap_threshold: Minimum overlap ratio (0-1) to keep a road match (default: 0.2 = 20%)
            temporal_resolution: Temporal resolution for aggregation (default: "15min")
        """
        self.input_folder = Path(input_folder)
        self.road_network_path = road_network_path
        self.output_dir = Path(output_dir)
        self.n_chunks = n_chunks
        self.buffer_distance = buffer_distance
        self.overlap_threshold = overlap_threshold
        self.temporal_resolution = temporal_resolution

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize tqdm for pandas operations
        tqdm.pandas()

        # Storage for processed data
        self.parquet_files: List[Path] = []
        self.roads_gdf: gpd.GeoDataFrame | None = None

        # Define variable categories for aggregation
        self.non_agg_variables = [
            "type",
            "timestamp_updateDateMillis_ts_cot",
            "id",
            "timeframe",
            "timestamp_detectionDateMillis",
            "geometry",
            "overlap_ratio",
            "intersection_length",
            "irreg_id",
            "pct_of_interval",
            "timestamp_updateDateMillis",
            "timestamp_startTime_ts_cot",
            "alerts",
            "timestamp_endTime_ts_cot",
            "mvi_length",
            "id_mvi",
            "timestamp_detectionDateMillis_ts_cot",
        ]

        self.non_transformable_variables = [
            "trend",
            "speed",
            "jamLevel",
            "severity",
            "regularSpeed",
        ]

        self.countable_variables = ["causeType"]

    @staticmethod
    def quarter_slices(
        start: pd.Timestamp, end: pd.Timestamp
    ) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
        """
        Generate time slices aligned to 15-minute intervals between start and end.

        Given two pandas Timestamps (start < end), return a list of (slice_start, slice_end)
        pairs, each of which lies wholly within one 15-minute bucket (minute marks 00, 15, 30, 45).

        For example:
        - If an irregularity runs from 10:07 to 10:42:
          Returns [(10:07, 10:15), (10:15, 10:30), (10:30, 10:42)]
        - Each tuple represents a continuous period within a single 15-minute bucket

        Args:
            start: Irregularity start timestamp
            end: Irregularity end timestamp

        Returns:
            List[Tuple[pd.Timestamp, pd.Timestamp]]: List of (start, end) pairs for each slice
        """
        out = []
        current = start

        while current < end:
            # Find "floor" of current to the nearest 15 minutes, then add 15 min
            floor_q = current.floor("15min")
            next_bound = floor_q + pd.Timedelta(minutes=15)

            # If current is exactly on a quarter, jump to the next one
            if current == floor_q:
                next_bound = current + pd.Timedelta(minutes=15)

            # End of this chunk is either the next boundary or the original end
            chunk_end = min(next_bound, end)

            out.append((current, chunk_end))
            current = chunk_end

        return out

    def discover_parquet_files(self) -> List[Path]:
        """
        Discover all parquet files in the input folder.

        Returns:
            List[Path]: Sorted list of parquet file paths
        """
        print("Discovering parquet files...")
        files = sorted(self.input_folder.glob("*.parquet"))
        print(f"  Found {len(files)} parquet files")
        return files

    def load_road_network(self) -> gpd.GeoDataFrame:
        """
        Load and process the road network with buffering for spatial joins.

        This method:
        1. Loads the road network geometry
        2. Creates unique road IDs from MVI codes
        3. Transforms to projected CRS (EPSG:3116) for accurate buffering
        4. Calculates road lengths
        5. Creates buffers around road segments
        6. Transforms back to WGS84 (EPSG:4326) for consistency

        Returns:
            gpd.GeoDataFrame: Processed road network with buffered geometries
        """
        print("Loading road network...")
        mvi = gpd.read_file(self.road_network_path)
        mvi = mvi.to_crs("EPSG:4326")

        # Standardize column names
        mvi.columns = mvi.columns.str.lower()

        # Create unique road IDs
        mvi["id_mvi"] = (
            mvi["mviciv"].astype(str) + "_" + mvi["mvisvia"].str.replace(" ", "")
        )
        mvi["contador"] = mvi.groupby("id_mvi").cumcount() + 1
        mvi["id_mvi"] = mvi["id_mvi"] + "." + mvi["contador"].astype(str)

        # Transform to projected CRS for accurate buffering
        mvi_buffered = mvi.to_crs("EPSG:3116").copy()

        # Calculate road lengths
        mvi_buffered["mvi_length"] = mvi_buffered.geometry.length

        # Create buffers around road segments
        mvi_buffered["geometry"] = mvi_buffered.geometry.buffer(
            self.buffer_distance, cap_style="square"
        )

        # Transform back to WGS84
        mvi_buffered = mvi_buffered.to_crs("EPSG:4326")

        print(f"  Loaded {len(mvi_buffered)} road segments with buffers")
        return mvi_buffered[["id_mvi", "mvi_length", "geometry"]]

    def load_and_preprocess_batch(self, file_paths: List[Path]) -> gpd.GeoDataFrame:
        """
        Load and preprocess a batch of irregularities parquet files.

        This method:
        1. Loads multiple parquet files
        2. Parses geometry from WKT/WKB format
        3. Explodes alerts array into separate rows
        4. Normalizes nested JSON alerts data
        5. Aggregates alerts by type
        6. Converts timestamps from milliseconds
        7. Cleans up unnecessary columns

        Args:
            file_paths: List of parquet file paths to process

        Returns:
            gpd.GeoDataFrame: Combined and preprocessed irregularities data
        """
        print(f"  Loading {len(file_paths)} files...")
        irreg_gdf_list = []

        for archivo in tqdm(file_paths, desc="  Processing files"):
            df = pd.read_parquet(archivo)

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

            # Explode alerts array into separate rows
            df_exploded = df[["id", "alerts"]].explode("alerts")
            df_exploded = df_exploded.dropna(subset=["alerts"])

            # Normalize nested JSON alerts data
            alerts_normalized = pd.json_normalize(df_exploded["alerts"])  # type: ignore
            alerts_normalized = alerts_normalized.drop(columns=["id"], errors="ignore")

            df_exploded = df_exploded.reset_index(drop=True)
            alerts_normalized = alerts_normalized.reset_index(drop=True)

            # Combine original IDs with normalized alerts
            alerts_details = pd.concat([df_exploded[["id"]], alerts_normalized], axis=1)

            # Ensure nThumbsUp column exists
            if "nThumbsUp" not in alerts_details.columns:
                alerts_details["nThumbsUp"] = 0

            # Aggregate alerts by ID and subtype
            alerts_details["count"] = 1
            alerts_details = (
                alerts_details.groupby(["id", "subtype"])
                .aggregate({"count": "sum", "nThumbsUp": "sum"})
                .reset_index()
                .pivot_table(
                    index="id", columns="subtype", values=["count", "nThumbsUp"]
                )
                .fillna(0)
            )

            # Flatten column names
            alerts_details.columns = [
                f"{col[1]}_{col[0]}" if col[1] != "" else f"NOT_DEFINED_{col[0]}"
                for col in alerts_details.columns
            ]
            alerts_details = alerts_details.reset_index()

            # Convert timestamps from milliseconds
            df["timestamp_detectionDateMillis"] = pd.to_datetime(
                df["detectionDateMillis"], unit="ms"
            )
            df["timestamp_updateDateMillis"] = pd.to_datetime(
                df["updateDateMillis"], unit="ms"
            )
            df["timestamp_detectionDateMillis_ts_cot"] = pd.to_datetime(
                df["detectionDateMillis_ts_cot"], unit="ms"
            )
            df["timestamp_updateDateMillis_ts_cot"] = pd.to_datetime(
                df["updateDateMillis_ts_cot"], unit="ms"
            )
            df["timestamp_startTime_ts_cot"] = pd.to_datetime(
                df["startTime_ts_cot"], unit="ms"
            )
            df["timestamp_endTime_ts_cot"] = pd.to_datetime(
                df["endTime_ts_cot"], unit="ms"
            )

            # Merge alerts details back to main dataframe
            df = pd.merge(df, alerts_details, on=["id"], how="left")

            # Drop unnecessary columns
            df = df.drop(
                [
                    "country",
                    "updateDate",
                    "city",
                    "detectionDateMillis",
                    "endNode",
                    "street",
                    "highway",
                    "length",
                    "updateDateMillis",
                    "detectionDate",
                    "causeAlert",
                    "startNode",
                    "detectionDateMillis_ts_cot",
                    "updateDateMillis_ts_cot",
                    "geom_wkt",
                    "startTime_ts_cot",
                    "type",
                    "alerts",
                    "endTime_ts_cot",
                    "fecha_lectura",
                    "hora_lectura_dato",
                ],
                axis=1,
                errors="ignore",
            )

            # Convert to GeoDataFrame
            gdf = gpd.GeoDataFrame(df, geometry="geometry", crs=4326)
            irreg_gdf_list.append(gdf)

        # Combine all dataframes
        irreg_gdf = gpd.GeoDataFrame(
            pd.concat(irreg_gdf_list, ignore_index=True).fillna(0), crs=4326
        )

        print(f"  Loaded {len(irreg_gdf)} irregularity records")
        return irreg_gdf

    def temporal_slicing(self, irreg_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Slice irregularities into 15-minute temporal intervals.

        This method:
        1. Assigns unique IDs to each irregularity
        2. Fills end times by using the next start time for the same irregularity
        3. Calculates original duration
        4. Slices each irregularity into 15-minute chunks
        5. Explodes into separate rows per time slice
        6. Calculates temporal weights (proportion of interval covered)

        Args:
            irreg_gdf: GeoDataFrame of irregularities with start/end times

        Returns:
            gpd.GeoDataFrame: Irregularities sliced by 15-minute intervals with temporal weights
        """
        print("  Performing temporal slicing...")

        # Assign unique IDs
        irreg_gdf["irreg_id"] = irreg_gdf.index

        # Sort by ID and start time
        irreg_gdf = irreg_gdf.sort_values(
            ["id", "timestamp_startTime_ts_cot"]
        ).reset_index(drop=True)

        # Fill end times: use next start time for same irregularity ID
        irreg_gdf["next_start"] = irreg_gdf.groupby("id")[
            "timestamp_startTime_ts_cot"
        ].shift(-1)
        mask = irreg_gdf["next_start"].notna()
        irreg_gdf.loc[mask, "timestamp_endTime_ts_cot"] = irreg_gdf.loc[
            mask, "next_start"
        ]
        irreg_gdf = irreg_gdf.drop(columns="next_start")

        # Calculate original duration
        irreg_gdf["orig_duration"] = (
            irreg_gdf["timestamp_endTime_ts_cot"]
            - irreg_gdf["timestamp_startTime_ts_cot"]
        )

        # Generate time slices for each irregularity
        irreg_gdf["split_slices"] = irreg_gdf.apply(  # type: ignore
            lambda r: self.quarter_slices(
                r["timestamp_startTime_ts_cot"], r["timestamp_endTime_ts_cot"]
            ),
            axis=1,
        )

        # Explode into separate rows per time slice
        irreg_gdf_exploded = irreg_gdf.explode("split_slices").reset_index(drop=True)

        # Extract new start and end times
        irreg_gdf_exploded[["new_start", "new_end"]] = pd.DataFrame(
            irreg_gdf_exploded["split_slices"].tolist(),
            index=irreg_gdf_exploded.index,
        )

        # Calculate piece duration and temporal weights
        irreg_gdf_exploded["piece_duration"] = (
            irreg_gdf_exploded["new_end"] - irreg_gdf_exploded["new_start"]
        )
        irreg_gdf_exploded["pct_of_interval"] = (
            irreg_gdf_exploded["piece_duration"] / irreg_gdf_exploded["orig_duration"]
        )
        irreg_gdf_exploded["piece_duration_mins"] = (
            irreg_gdf_exploded["piece_duration"].dt.total_seconds().div(60)
        )

        # Clean up and rename columns
        irreg_gdf = irreg_gdf_exploded.drop(
            columns=[
                "timestamp_startTime_ts_cot",
                "timestamp_endTime_ts_cot",
                "split_slices",
                "orig_duration",
                "piece_duration",
            ]
        )
        irreg_gdf = irreg_gdf.rename(
            columns={
                "new_start": "timestamp_startTime_ts_cot",
                "new_end": "timestamp_endTime_ts_cot",
            }
        )
        irreg_gdf = irreg_gdf.reset_index(drop=True)

        print(f"    Generated {len(irreg_gdf)} time-sliced records")
        return irreg_gdf

    def spatial_intersection(
        self, irreg_gdf: gpd.GeoDataFrame, roads_gdf: gpd.GeoDataFrame
    ) -> gpd.GeoDataFrame:
        """
        Perform spatial intersection between irregularities and road network.

        This method:
        1. Ensures CRS consistency
        2. Performs overlay intersection
        3. Calculates intersection lengths in projected CRS
        4. Computes overlap ratios
        5. Filters low-overlap matches
        6. Creates 15-minute timeframe buckets

        Args:
            irreg_gdf: GeoDataFrame of temporally-sliced irregularities
            roads_gdf: GeoDataFrame of buffered road segments

        Returns:
            gpd.GeoDataFrame: Irregularities intersected with roads, with spatial weights
        """
        print("  Performing spatial intersection...")

        # Ensure CRS consistency
        if irreg_gdf.crs != roads_gdf.crs:
            irreg_gdf = irreg_gdf.to_crs(roads_gdf.crs)  # type: ignore

        # Perform overlay intersection
        irregs_intersections = gpd.overlay(irreg_gdf, roads_gdf, how="intersection")

        # Transform to projected CRS for accurate length calculation
        irregs_intersections = irregs_intersections.to_crs("EPSG:3116").copy()
        irregs_intersections["intersection_length"] = (
            irregs_intersections.geometry.length
        )

        # Calculate overlap ratio
        irregs_intersections["overlap_ratio"] = (
            irregs_intersections["intersection_length"]
            / irregs_intersections["mvi_length"]
        )

        # Filter low-overlap matches
        print(f"    Before filtering: {len(irregs_intersections)} intersections")
        irregs_intersections = irregs_intersections[
            irregs_intersections["overlap_ratio"] > self.overlap_threshold
        ]
        print(
            f"    After filtering (overlap > {self.overlap_threshold}): {len(irregs_intersections)} intersections"
        )

        # Transform back to WGS84
        irregs_intersections = irregs_intersections.to_crs("EPSG:4326").copy()

        # Create timeframe buckets (floor to 15-minute intervals)
        irregs_intersections["timeframe"] = irregs_intersections[
            "timestamp_startTime_ts_cot"
        ].dt.floor(self.temporal_resolution)

        return irregs_intersections

    def weighted_aggregation(
        self, irregs_intersections: gpd.GeoDataFrame
    ) -> pd.DataFrame:
        """
        Perform weighted aggregation of irregularities by road and timeframe.

        This method applies different aggregation strategies:
        1. Transformable variables (alert counts): weighted by temporal and spatial overlap
        2. Non-transformable variables (speed, severity): time-weighted averages
        3. Countable variables (cause types): one-hot encoding and summation

        The dual weighting ensures that:
        - An irregularity covering 50% of a time interval counts as 0.5 events
        - An irregularity covering 30% of a road segment is weighted by 0.3
        - Speed/severity are averaged weighted by duration

        Args:
            irregs_intersections: GeoDataFrame of intersected irregularities with weights

        Returns:
            pd.DataFrame: Aggregated features by (id_mvi, timeframe)
        """
        print("  Performing weighted aggregation...")

        # Identify transformable variables (those not in predefined categories)
        transformable_variables = sorted(
            list(
                set(irregs_intersections.columns)
                - set(self.non_transformable_variables)
                - set(self.countable_variables)
                - set(self.non_agg_variables)
            )
        )

        agg_dict: Dict[str, str] = {}

        # Apply dual weighting to transformable variables (alert counts)
        print(
            f"    Applying dual weighting to {len(transformable_variables)} variables"
        )
        for col in transformable_variables:
            if col in irregs_intersections.columns:
                irregs_intersections[col] = (
                    irregs_intersections[col]
                    * irregs_intersections["pct_of_interval"].clip(upper=1)
                    * irregs_intersections["overlap_ratio"].clip(upper=1)
                )
                agg_dict[col] = "sum"

        # Apply time weighting to non-transformable variables (speed, severity)
        print(
            f"    Applying time weighting to {len(self.non_transformable_variables)} variables"
        )
        for col in self.non_transformable_variables:
            if col in irregs_intersections.columns:
                irregs_intersections[f"{col}_weighted"] = (
                    irregs_intersections[col]
                    * irregs_intersections["piece_duration_mins"]
                )
                agg_dict[f"{col}_weighted"] = "sum"

        agg_dict["piece_duration_mins"] = "sum"

        # One-hot encode countable variables (cause types)
        print(
            f"    One-hot encoding {len(self.countable_variables)} categorical variables"
        )
        dummies = pd.get_dummies(
            irregs_intersections[self.countable_variables].astype(str),
            prefix=self.countable_variables,
        ).astype("int")

        for col in dummies.columns:
            agg_dict[col] = "sum"

        irregs_intersections = pd.concat([irregs_intersections, dummies], axis=1)  # type: ignore

        # Sort before groupby for efficiency
        irregs_intersections = irregs_intersections.sort_values(
            ["id_mvi", "timeframe"]
        ).reset_index(drop=True)

        # Perform aggregation
        print("    Grouping by road and timeframe...")
        agg_irregs_intersections = irregs_intersections.groupby(
            ["id_mvi", "timeframe"], sort=False
        ).agg(agg_dict)
        agg_irregs_intersections = agg_irregs_intersections.reset_index()

        # Calculate time-weighted averages for non-transformable variables
        for col in self.non_transformable_variables:
            if f"{col}_weighted" in agg_irregs_intersections.columns:
                agg_irregs_intersections[col] = (
                    agg_irregs_intersections[f"{col}_weighted"]
                    / agg_irregs_intersections["piece_duration_mins"]
                )
                agg_irregs_intersections = agg_irregs_intersections.drop(
                    [f"{col}_weighted"], axis=1
                )

        # Drop duration column
        agg_irregs_intersections = agg_irregs_intersections.drop(
            ["piece_duration_mins"], axis=1
        )

        print(f"    Final aggregated dataset: {len(agg_irregs_intersections)} records")
        return agg_irregs_intersections

    def process_batch(self, file_paths: List[Path], batch_name: str) -> pd.DataFrame:
        """
        Process a single batch of irregularities files.

        This method orchestrates the complete pipeline for one batch:
        1. Load and preprocess files
        2. Temporal slicing
        3. Spatial intersection
        4. Weighted aggregation

        Args:
            file_paths: List of parquet file paths to process
            batch_name: Name identifier for this batch (for output filename)

        Returns:
            pd.DataFrame: Aggregated irregularities data for this batch
        """
        print(f"\nProcessing batch: {batch_name}")

        # Step 1: Load and preprocess
        irreg_gdf = self.load_and_preprocess_batch(file_paths)

        # Step 2: Temporal slicing
        irreg_gdf = self.temporal_slicing(irreg_gdf)

        # Step 3: Spatial intersection
        irregs_intersections = self.spatial_intersection(irreg_gdf, self.roads_gdf)  # type: ignore

        # Step 4: Weighted aggregation
        agg_result = self.weighted_aggregation(irregs_intersections)

        return agg_result

    def process_and_save(self) -> None:
        """
        Execute the complete irregularities data processing pipeline and save results.

        This method orchestrates the entire workflow:
        1. Discovers all parquet files in input folder
        2. Loads the road network
        3. Divides files into batches for memory efficiency
        4. Processes each batch sequentially
        5. Saves each batch to a separate parquet file

        Output files:
            - {output_dir}/irregs_proc_{start_date}_{end_date}.parquet
              Multiple files, one per batch, containing aggregated irregularities data

        The batch processing approach ensures that even large datasets can be processed
        without running out of memory.

        These files are designed to be consumed by the SparseDatasetGenerator
        in merging.py for integration into the traffic forecasting pipeline.
        """
        print("=" * 70)
        print("Starting Irregularities Data Processing Pipeline")
        print("=" * 70)

        # Step 1: Discover parquet files
        print("\n[1/3] Discovering input files...")
        self.parquet_files = self.discover_parquet_files()

        if not self.parquet_files:
            print("ERROR: No parquet files found in input folder!")
            return

        # Step 2: Load road network
        print("\n[2/3] Loading road network...")
        self.roads_gdf = self.load_road_network()

        # Step 3: Process files in batches
        print("\n[3/3] Processing files in batches...")
        chunk_size = len(self.parquet_files) // self.n_chunks + 1
        print(f"  Total files: {len(self.parquet_files)}")
        print(f"  Batch size: {self.n_chunks} files")
        print(f"  Total batches: {chunk_size}")

        for i in tqdm(range(chunk_size), desc="Processing batches"):
            # Get files for this batch
            files_group = self.parquet_files[
                i * self.n_chunks : (i + 1) * self.n_chunks
            ]

            if not files_group:
                continue

            # Generate batch name from date range
            min_file = min(files_group)
            max_file = max(files_group)
            start_date = min_file.stem.split("_")[2]
            end_date = max_file.stem.split("_")[2]
            batch_name = f"irregs_proc_{start_date}_{end_date}"

            # Process batch
            agg_result = self.process_batch(files_group, batch_name)

            # Save batch result
            output_path = self.output_dir / f"{batch_name}.parquet"
            agg_result.to_parquet(output_path, index=False)
            print(f"  Saved batch to: {output_path}")

        print("\n" + "=" * 70)
        print("Irregularities Data Processing Complete!")
        print("=" * 70)
        print(f"\nAll outputs saved to: {self.output_dir}")


if __name__ == "__main__":
    # Define input data paths
    input_folder = "data/4_velocidades_waze/irregularidades"
    road_network_path = "data/2_Estructura_via/estructura_via_completa/MVI.geojson"
    output_dir = "data/datos_procesados/irregularidades_procesadas/"

    # Initialize processor
    processor = IrregularitiesProcessor(
        input_folder=input_folder,
        road_network_path=road_network_path,
        output_dir=output_dir,
        n_chunks=10,  # Process 10 files per batch
        buffer_distance=4.0,  # 4 meter buffer around roads
        overlap_threshold=0.2,  # Keep matches with >20% overlap
        temporal_resolution="15min",  # 15-minute aggregation
    )

    # Execute processing pipeline
    processor.process_and_save()

    print("\nProcessing complete!")
