from pathlib import Path
from typing import Dict

import geopandas as gpd
import pandas as pd
from tqdm.auto import tqdm


class RoadStateProcessor:
    """
    A class for processing road condition data and assigning it to road network segments.

    This processor handles the complete pipeline from raw road condition shapefiles
    to road-level condition attributes, suitable for training Graph Neural Networks
    for traffic forecasting.

    The road state data includes AASHTO condition ratings, state indices, and physical
    characteristics that influence traffic flow and congestion patterns.

    Attributes:
        road_network_path (str): Path to the road network GeoJSON file
        road_state_shapefile (str): Path to the road condition shapefile
        output_dir (str): Directory where processed files will be saved
        buffer_distance (float): Buffer distance in meters for spatial overlay (default: 15)
        target_crs (str): Target coordinate reference system for spatial operations (default: "EPSG:3116")
        timeframe_start (str): Start datetime for the road condition validity period
        timeframe_end (str): End datetime for the road condition validity period
    """

    # AASHTO road state classification mapping
    ROAD_STATE_MAPPING: Dict[str, int] = {
        "SATISFACTORIO": 3,  # Satisfactory
        "BUENO": 2,  # Good
        "ACEPTABLE": 1,  # Acceptable
        "SIN DATO": 0,  # No data
        "POBRE": -1,  # Poor
        "MUY POBRE": -2,  # Very poor
        "FALLADO": -3,  # Failed
        "GRAVE": -4,  # Critical
    }

    def __init__(
        self,
        road_network_path: str,
        road_state_shapefile: str,
        output_dir: str,
        buffer_distance: float = 15.0,
        target_crs: str = "EPSG:3116",
        timeframe_start: str = "2024-01-01 00:00:00",
        timeframe_end: str = "2024-06-30 23:45:00",
    ):
        """
        Initialize the RoadStateProcessor with data paths and processing parameters.

        Args:
            road_network_path: Path to the road network GeoJSON file (MVI format)
            road_state_shapefile: Path to the AASHTO road condition shapefile
            output_dir: Directory where processed road state data will be saved
            buffer_distance: Buffer distance in meters around road segments for overlay (default: 15)
            target_crs: EPSG code for projected coordinate system used in spatial operations (default: "EPSG:3116")
            timeframe_start: Start datetime for road condition validity (default: "2024-01-01 00:00:00")
            timeframe_end: End datetime for road condition validity (default: "2024-06-30 23:45:00")
        """
        self.road_network_path = road_network_path
        self.road_state_shapefile = road_state_shapefile
        self.output_dir = Path(output_dir)
        self.buffer_distance = buffer_distance
        self.target_crs = target_crs
        self.timeframe_start = timeframe_start
        self.timeframe_end = timeframe_end

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Storage for processed data
        self.mvi_buffered: gpd.GeoDataFrame | None = None
        self.road_state_gdf: gpd.GeoDataFrame | None = None
        self.road_state_dataset: pd.DataFrame | None = None

    def load_and_buffer_road_network(self, path: str) -> gpd.GeoDataFrame:
        """
        Load and process the road network with buffering for spatial overlay.

        This method:
        1. Loads the road network geometry
        2. Converts to WGS84 (EPSG:4326) for consistency
        3. Creates unique road IDs from MVI codes
        4. Dissolves split segments into unified road segments
        5. Transforms to projected CRS for accurate spatial operations
        6. Creates buffers around road segments for condition assignment
        7. Converts back to WGS84 for final operations

        Args:
            path: Path to the road network GeoJSON file

        Returns:
            gpd.GeoDataFrame: Buffered road network with columns:
                - road_id: Unique identifier for each road segment
                - mvi_length: Length of the road segment in meters
                - geometry: Buffered polygon geometry in WGS84
        """
        print("Loading and buffering road network...")
        mvi = gpd.read_file(path)

        # Convert to WGS84 for consistency
        mvi = mvi.to_crs(epsg=4326)

        # Create unique road IDs from MVI codes
        mvi["id_mvi"] = (
            mvi["MVICIV"].astype(str) + "_" + mvi["MVISVIA"].str.replace(" ", "")
        )
        mvi["contador"] = mvi.groupby("id_mvi").cumcount() + 1
        mvi["id_mvi"] = mvi["id_mvi"].astype(str) + "." + mvi["contador"].astype(str)

        # Dissolve split segments into unified roads
        mvi = (
            mvi.explode(ignore_index=True)  # Explode MultiLineStrings if present
            .dissolve(by="id_mvi")  # Merge all segments of the same road
            .reset_index()
        )

        mvi["road_id"] = mvi["id_mvi"]

        # Transform to projected CRS and create buffers
        mvi_buffered = mvi.to_crs(self.target_crs).copy()
        mvi_buffered["mvi_length"] = mvi_buffered.geometry.length

        # Buffer road segments for spatial overlay with road condition data
        mvi_buffered["geometry"] = mvi_buffered.geometry.buffer(
            self.buffer_distance, cap_style="square"
        )

        # Convert back to WGS84 for overlay operations
        mvi_buffered = mvi_buffered.set_geometry("geometry")
        mvi_buffered = mvi_buffered.to_crs("EPSG:4326")

        print(f"  Loaded and buffered {len(mvi_buffered)} road segments")
        return mvi_buffered

    def load_and_process_road_state(self, shapefile_path: str) -> gpd.GeoDataFrame:
        """
        Load and process road condition data from shapefile.

        This method:
        1. Loads the AASHTO road condition shapefile
        2. Converts to WGS84 coordinate system
        3. Maps qualitative AASHTO conditions to numeric values
        4. Extracts numeric state indices and other condition metrics

        The AASHTO classification provides standardized road condition ratings
        used for infrastructure maintenance and planning.

        Args:
            shapefile_path: Path to the road condition shapefile

        Returns:
            gpd.GeoDataFrame: Processed road condition data with columns:
                - numeric_road_state: Numeric encoding of AASHTO condition (-4 to 3)
                - numeric_state_index: State condition index (INDEST)
                - LOCEST: Local condition estimate
                - SUPERFEST: Surface condition estimate
                - KMCAINDEST: Km condition index
                - geometry: Road condition segment geometry
        """
        print("Loading road condition data...")
        road_state_gdf = gpd.read_file(shapefile_path).to_crs(epsg=4326)

        # Map AASHTO qualitative conditions to numeric values
        print("  Encoding AASHTO condition ratings...")
        road_state_gdf["numeric_road_state"] = list(
            tqdm(
                map(
                    lambda x: self.ROAD_STATE_MAPPING.get(x, 0),
                    road_state_gdf["ESTAASHTO"],
                ),
                desc="  Processing conditions",
            )
        )

        # Extract numeric state index
        road_state_gdf["numeric_state_index"] = list(
            tqdm(
                map(lambda x: int(x), road_state_gdf["INDEST"]),
                desc="  Processing indices",
            )
        )

        print(f"  Loaded {len(road_state_gdf)} road condition segments")
        return road_state_gdf

    def spatial_overlay_and_rank(
        self, road_state_gdf: gpd.GeoDataFrame, mvi_buffered: gpd.GeoDataFrame
    ) -> gpd.GeoDataFrame:
        """
        Perform spatial overlay between road conditions and network, then rank overlaps.

        This method:
        1. Performs spatial intersection between condition segments and buffered roads
        2. Calculates intersection lengths and overlap ratios
        3. Ranks overlapping segments by overlap ratio for each road
        4. Selects the best matching condition segment (highest overlap) for each road

        When multiple road condition segments overlap with a single road segment,
        the one with the highest overlap ratio is selected as the most representative.

        Args:
            road_state_gdf: GeoDataFrame with road condition data
            mvi_buffered: GeoDataFrame with buffered road network

        Returns:
            gpd.GeoDataFrame: Intersection results with overlap rankings, containing:
                - road_id: Road segment identifier
                - numeric_road_state: Assigned condition rating
                - INDEST, LOCEST, SUPERFEST, KMCAINDEST: Condition metrics
                - intersection_length: Length of intersection in meters
                - overlap_ratio: Ratio of intersection to road length
                - overlap_rank: Rank of this intersection (1 = best match)
        """
        print("Performing spatial overlay...")
        road_state_intersections = gpd.overlay(
            road_state_gdf,
            mvi_buffered[["road_id", "mvi_length", "geometry"]],
            how="intersection",
        )

        # Calculate intersection metrics
        print("  Calculating intersection lengths and overlap ratios...")
        road_state_intersections["intersection_length"] = (
            road_state_intersections.geometry.length
        )
        road_state_intersections["overlap_ratio"] = (
            road_state_intersections["intersection_length"]
            / road_state_intersections["mvi_length"]
        )

        # Rank overlaps by ratio for each road segment
        print("  Ranking overlapping segments...")
        road_state_intersections["overlap_rank"] = (
            road_state_intersections.groupby("road_id")["overlap_ratio"]
            .rank(method="dense", ascending=False)
            .astype(int)
        )

        print(f"  Found {len(road_state_intersections)} intersection segments")
        print(
            f"  Covering {road_state_intersections['road_id'].nunique()} unique roads"
        )

        return road_state_intersections

    def select_best_matches(
        self, road_state_intersections: gpd.GeoDataFrame
    ) -> pd.DataFrame:
        """
        Select the best matching road condition for each road segment.

        This method filters the intersection results to keep only the top-ranked
        (highest overlap ratio) condition segment for each road. It then prepares
        the final dataset with temporal validity information.

        Args:
            road_state_intersections: GeoDataFrame with ranked intersection results

        Returns:
            pd.DataFrame: Final road state dataset with columns:
                - road_id: Road segment identifier
                - timeframe_start: Start of validity period
                - timeframe_end: End of validity period
                - numeric_road_state: Numeric condition rating
                - INDEST: State condition index
                - LOCEST: Local condition estimate
                - SUPERFEST: Surface condition estimate
                - KMCAINDEST: Km condition index
        """
        print("Selecting best matches for each road segment...")

        # Filter to keep only rank 1 (best match) for each road
        best_matches = road_state_intersections.loc[
            road_state_intersections.overlap_rank == 1
        ].copy()

        # Add temporal validity information
        best_matches["timeframe_start"] = self.timeframe_start
        best_matches["timeframe_end"] = self.timeframe_end

        # Select relevant columns for the final dataset
        road_state_dataset = best_matches[
            [
                "road_id",
                "timeframe_start",
                "timeframe_end",
                "numeric_road_state",
                "INDEST",
                "LOCEST",
                "SUPERFEST",
                "KMCAINDEST",
            ]
        ]

        print(f"  Selected {len(road_state_dataset)} road segments with conditions")
        return road_state_dataset

    def process_and_save(self) -> pd.DataFrame:
        """
        Execute the complete road state processing pipeline and save results.

        This method orchestrates the entire workflow:
        1. Loads and buffers the road network
        2. Loads and processes road condition data
        3. Performs spatial overlay between conditions and roads
        4. Ranks overlapping segments and selects best matches
        5. Saves the final dataset to parquet

        Output file:
            - {output_dir}/road_state_dataset.parquet: Road condition assignments

        This file is designed to be consumed by the SparseDatasetGenerator
        in merging.py for integration into the traffic forecasting pipeline.

        Returns:
            pd.DataFrame: The processed road state dataset
        """
        print("=" * 70)
        print("Starting Road State Processing Pipeline")
        print("=" * 70)

        # Step 1: Load and buffer road network
        print("\n[1/5] Loading and buffering road network...")
        self.mvi_buffered = self.load_and_buffer_road_network(self.road_network_path)

        # Step 2: Load and process road condition data
        print("\n[2/5] Loading road condition data...")
        self.road_state_gdf = self.load_and_process_road_state(
            self.road_state_shapefile
        )

        # Step 3: Spatial overlay and ranking
        print("\n[3/5] Performing spatial analysis...")
        road_state_intersections = self.spatial_overlay_and_rank(
            self.road_state_gdf, self.mvi_buffered
        )

        # Step 4: Select best matches
        print("\n[4/5] Selecting best condition matches...")
        self.road_state_dataset = self.select_best_matches(road_state_intersections)

        # Step 5: Save results
        print("\n[5/5] Saving processed dataset...")
        output_path = self.output_dir / "road_state_dataset.parquet"
        self.road_state_dataset.to_parquet(output_path, index=False)
        print(f"  Saved road state data to: {output_path}")

        print("\n" + "=" * 70)
        print("Road State Processing Complete!")
        print("=" * 70)
        print(f"\nProcessed {len(self.road_state_dataset)} road segments")
        print(f"Output saved to: {output_path}")
        print("\nThis file is ready for integration with the SparseDatasetGenerator")
        print("in merging.py for traffic forecasting model training.")

        return self.road_state_dataset

    def get_road_state_summary(self) -> pd.DataFrame:
        """
        Generate a summary of road condition distributions.

        This method provides statistical insights into the road network's
        condition distribution, useful for understanding infrastructure quality.

        Returns:
            pd.DataFrame: Summary statistics showing count and percentage
                         of roads in each AASHTO condition category
        """
        if self.road_state_dataset is None:
            raise ValueError(
                "No processed data available. Run process_and_save() first."
            )

        # Reverse mapping for readable labels
        reverse_mapping = {v: k for k, v in self.ROAD_STATE_MAPPING.items()}

        summary = (
            self.road_state_dataset["numeric_road_state"]
            .value_counts()
            .sort_index(ascending=False)
        )
        total_count = int(summary.sum())
        summary_df = pd.DataFrame(
            {
                "Condition": [
                    reverse_mapping.get(idx, f"Unknown ({idx})")
                    for idx in summary.index
                ],
                "Count": summary.values,
                "Percentage": [
                    round(count / total_count * 100, 2) for count in summary.values
                ],
            }
        )

        return summary_df


if __name__ == "__main__":
    # Define input data paths
    road_network_path = "data/2_Estructura_via/estructura_via_completa/MVI.geojson"
    road_state_shapefile = "data/14_Estado_malla_vial/EstadoMallaVialOD_IDU_Sem12024/EstadoMallaVialUrbanaConsolidado.shp"
    output_dir = "data/datos_procesados/road_state/"

    # Initialize processor
    processor = RoadStateProcessor(
        road_network_path=road_network_path,
        road_state_shapefile=road_state_shapefile,
        output_dir=output_dir,
        buffer_distance=15.0,
        target_crs="EPSG:3116",
        timeframe_start="2024-01-01 00:00:00",
        timeframe_end="2024-06-30 23:45:00",
    )

    # Execute processing pipeline
    road_state_dataset = processor.process_and_save()

    # Display summary statistics
    print("\n" + "=" * 70)
    print("Road Condition Distribution Summary")
    print("=" * 70)
    summary = processor.get_road_state_summary()
    print(summary.to_string(index=False))

    print("\n✓ Processing complete! You can now use this file in merging.py")
