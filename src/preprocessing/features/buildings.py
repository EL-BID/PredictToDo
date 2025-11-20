from pathlib import Path
from typing import Tuple

import geopandas as gpd
import osmnx as ox
import pandas as pd


class BuildingsProcessor:
    """
    A class for extracting building footprints from OpenStreetMap and assigning them to road network segments.

    This processor handles the complete pipeline from OpenStreetMap building queries
    to aggregated building counts by type for each road segment, suitable for enriching
    the road network graph with land use and urban density features.

    The processor:
    1. Loads the road network and creates unique road IDs
    2. Queries OpenStreetMap for building footprints within the network's bounding box
    3. Creates 100m buffers around road segments to capture nearby buildings
    4. Performs spatial join between buildings and buffered roads
    5. Aggregates building counts by type (residential, commercial, industrial, etc.)

    Building types provide valuable features for traffic prediction:
    - Residential buildings indicate population density and commuter patterns
    - Commercial buildings suggest business activity and daytime traffic
    - Industrial buildings indicate freight and logistics activity
    - Public buildings (schools, hospitals) create specific traffic patterns

    Attributes:
        road_network_path (str): Path to the road network GeoJSON file
        output_dir (str): Directory where processed files will be saved
        buffer_distance_wgs84 (float): Initial buffer distance in WGS84 degrees for bbox calculation (default: 0.01 ≈ 1km)
        buffer_distance_meters (float): Buffer distance in meters for building assignment (default: 100)
        target_crs (str): Target coordinate reference system for accurate spatial operations (default: "EPSG:3116" for Colombia)
    """

    def __init__(
        self,
        road_network_path: str,
        output_dir: str,
        buffer_distance_wgs84: float = 0.01,
        buffer_distance_meters: float = 100.0,
        target_crs: str = "EPSG:3116",
    ):
        """
        Initialize the BuildingsProcessor with data paths and processing parameters.

        Args:
            road_network_path: Path to the road network GeoJSON file (MVI format)
            output_dir: Directory where processed building counts will be saved
            buffer_distance_wgs84: Buffer distance in WGS84 degrees for bbox calculation (default: 0.01 ≈ 1km)
            buffer_distance_meters: Buffer distance in meters around roads for building assignment (default: 100)
            target_crs: EPSG code for projected coordinate system used for accurate spatial operations (default: "EPSG:3116")
        """
        self.road_network_path = road_network_path
        self.output_dir = Path(output_dir)
        self.buffer_distance_wgs84 = buffer_distance_wgs84
        self.buffer_distance_meters = buffer_distance_meters
        self.target_crs = target_crs

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Storage for processed data
        self.roads_gdf: gpd.GeoDataFrame | None = None
        self.buildings_gdf: gpd.GeoDataFrame | None = None
        self.building_counts: pd.DataFrame | None = None

    def load_and_process_roads(self, path: str) -> gpd.GeoDataFrame:
        """
        Load and process the road network for building extraction.

        This method:
        1. Loads the road network geometry
        2. Converts to WGS84 (EPSG:4326) for OSM compatibility
        3. Creates unique road IDs from MVI codes
        4. Dissolves split segments into unified road segments
        5. Stores both original geometry and buffered geometry

        Args:
            path: Path to the road network GeoJSON file

        Returns:
            gpd.GeoDataFrame: Processed road network with columns:
                - road_id: Unique identifier for each road segment
                - geometry: Original LineString geometry in WGS84
                - buffer: Buffered geometry for spatial operations
        """
        print("Loading road network...")
        roads_gdf = gpd.read_file(path)

        # Convert to WGS84 for OSM queries
        roads_gdf = roads_gdf.to_crs(epsg=4326)

        # Create unique road IDs from MVI codes
        print("  Creating unique road identifiers...")
        roads_gdf["id_mvi"] = (
            roads_gdf["MVICIV"].astype(str)
            + "_"
            + roads_gdf["MVISVIA"].str.replace(" ", "")
        )
        roads_gdf["contador"] = roads_gdf.groupby("id_mvi").cumcount() + 1
        roads_gdf["id_mvi"] = (
            roads_gdf["id_mvi"] + "." + roads_gdf["contador"].astype(str)
        )

        # Dissolve split segments into unified roads
        print("  Dissolving road segments...")
        roads_gdf = (
            roads_gdf.explode(ignore_index=True)  # Explode MultiLineStrings if present
            .dissolve(by="id_mvi")  # Merge all segments of the same road
            .reset_index()
        )
        roads_gdf["road_id"] = roads_gdf["id_mvi"]

        print(f"  Loaded {len(roads_gdf)} road segments")

        return roads_gdf

    def calculate_bounding_box(
        self, roads_gdf: gpd.GeoDataFrame
    ) -> Tuple[float, float, float, float]:
        """
        Calculate the bounding box for OpenStreetMap building query.

        This method:
        1. Creates buffers around road segments to extend the search area
        2. Calculates the total bounds of all buffered roads
        3. Returns bbox in (west, south, east, north) format for OSM queries

        Args:
            roads_gdf: GeoDataFrame with road geometries (must have 'buffer' column)

        Returns:
            Tuple[float, float, float, float]: Bounding box coordinates (west, south, east, north)
        """
        print("Calculating bounding box for building extraction...")

        # Calculate total bounds from buffered geometries
        minx, miny, maxx, maxy = roads_gdf["buffer"].total_bounds

        # OSM format: (west, south, east, north)
        bbox = (minx, miny, maxx, maxy)

        print(
            f"  Bounding box: ({bbox[0]:.4f}, {bbox[1]:.4f}, {bbox[2]:.4f}, {bbox[3]:.4f})"
        )
        print(
            f"  Area coverage: ~{(maxx - minx) * 111:.1f}km × {(maxy - miny) * 111:.1f}km"
        )

        return bbox

    def extract_buildings_from_osm(
        self, bbox: Tuple[float, float, float, float]
    ) -> gpd.GeoDataFrame:
        """
        Extract building footprints from OpenStreetMap within the bounding box.

        This method:
        1. Queries OpenStreetMap for all features tagged with 'building'
        2. Filters for valid polygon geometries (Polygon and MultiPolygon)
        3. Removes entries without building type information
        4. Renames 'building' column to 'bld_type' for clarity

        Building types from OSM can include:
        - Residential: house, apartments, residential, detached, semidetached_house, etc.
        - Commercial: commercial, retail, office, supermarket, shop, etc.
        - Industrial: industrial, warehouse, factory, etc.
        - Public: school, hospital, university, government, civic, etc.
        - Other: yes, parking, garage, construction, etc.

        Args:
            bbox: Bounding box coordinates (west, south, east, north)

        Returns:
            gpd.GeoDataFrame: Building footprints with columns:
                - bld_type: Building type from OSM tags
                - geometry: Polygon geometry in WGS84
        """
        print("Extracting buildings from OpenStreetMap...")
        print("  This may take a few minutes depending on area size...")

        # Query OSM for all buildings within bounding box
        tags = {"building": True}
        buildings = ox.features.features_from_bbox(bbox=bbox, tags=tags)  # type: ignore

        print(f"  Downloaded {len(buildings)} building features from OSM")

        # Filter for valid building geometries
        print("  Filtering for valid building polygons...")
        buildings = buildings.loc[
            buildings.geometry.type.isin(["Polygon", "MultiPolygon"])
            & buildings["building"].notna(),
            ["building", "geometry"],
        ].rename(columns={"building": "bld_type"})

        print(f"  Retained {len(buildings)} valid building polygons")
        print(f"  Found {buildings['bld_type'].nunique()} unique building types")

        return buildings

    def assign_buildings_to_roads(
        self, roads_gdf: gpd.GeoDataFrame, buildings_gdf: gpd.GeoDataFrame
    ) -> gpd.GeoDataFrame:
        """
        Assign buildings to road segments using spatial intersection.

        This method:
        1. Projects both datasets to target CRS for accurate spatial operations
        2. Creates buffers around road segments (default: 100 meters)
        3. Performs spatial join between buffered roads and building footprints
        4. Returns buildings that intersect with buffered road segments

        The buffer distance determines which buildings are considered "near" a road.
        A larger buffer captures more buildings but may include irrelevant ones.

        Args:
            roads_gdf: GeoDataFrame with road geometries
            buildings_gdf: GeoDataFrame with building footprints

        Returns:
            gpd.GeoDataFrame: Buildings assigned to roads with columns:
                - road_id: Identifier of the intersecting road
                - bld_type: Building type from OSM
                - Other columns from spatial join
        """
        print("Assigning buildings to road segments...")

        # Project to target CRS for accurate metric operations
        print(f"  Converting to {self.target_crs} for metric calculations...")
        roads_proj = roads_gdf.to_crs(self.target_crs)
        buildings_proj = buildings_gdf.to_crs(self.target_crs)

        # Create buffers around roads
        print(f"  Creating {self.buffer_distance_meters}m buffers around roads...")
        roads_proj["buffer"] = roads_proj.geometry.buffer(self.buffer_distance_meters)

        # Perform spatial join using buffered roads
        print("  Performing spatial join (this may take a while)...")
        roads_buf = roads_proj.set_geometry("buffer")[["road_id", "buffer"]]
        joined = gpd.sjoin(
            roads_buf, buildings_proj, how="left", predicate="intersects"
        )

        print(f"  Spatial join complete: {len(joined)} road-building intersections")
        print(
            f"  Roads with buildings: {joined['bld_type'].notna().sum()} / {len(roads_proj)}"
        )

        return joined

    def aggregate_building_counts(self, joined_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
        """
        Aggregate building counts by type for each road segment.

        This method:
        1. Filters out roads without any buildings
        2. Groups by road_id and building type
        3. Counts occurrences of each building type per road
        4. Pivots to create a wide format with building types as columns
        5. Fills missing values with 0 (roads without certain building types)

        The resulting dataset provides a feature vector for each road segment
        describing the surrounding urban environment through building type distribution.

        Args:
            joined_gdf: GeoDataFrame from spatial join with road-building assignments

        Returns:
            pd.DataFrame: Building counts per road with columns:
                - road_id: Road segment identifier
                - [building_type_1]: Count of buildings of type 1
                - [building_type_2]: Count of buildings of type 2
                - ... (one column per unique building type found)
        """
        print("Aggregating building counts by type...")

        # Count buildings by type for each road
        counts = (
            joined_gdf.dropna(subset=["bld_type"])
            .groupby(["road_id", "bld_type"], sort=False)
            .size()
            .reset_index(name="count")
        )

        print(f"  Generated {len(counts)} road-building type combinations")

        # Pivot to wide format: one column per building type
        df_counts = (
            counts.pivot(index="road_id", columns="bld_type", values="count")
            .fillna(0)
            .astype(int)
            .reset_index()
        )

        print(
            f"  Final dataset shape: {len(df_counts)} roads × {len(df_counts.columns) - 1} building types"
        )
        print(
            f"  Building type columns: {', '.join(df_counts.columns[1:10].tolist())}..."
        )

        return df_counts

    def process_and_save(self) -> pd.DataFrame:
        """
        Execute the complete buildings processing pipeline and save results.

        This method orchestrates the entire workflow:
        1. Loads and processes the road network
        2. Creates buffers and calculates bounding box
        3. Extracts building footprints from OpenStreetMap
        4. Assigns buildings to road segments via spatial join
        5. Aggregates building counts by type per road
        6. Saves the final dataset to CSV

        Output file:
            - {output_dir}/roads_buildings.csv: Building counts by type per road

        This file is designed to be consumed by the RoadGraphConstructor
        in graph.py for enriching the road network graph with urban context features.

        Returns:
            pd.DataFrame: The processed building aggregations
        """
        print("=" * 70)
        print("Starting Buildings Processing Pipeline")
        print("=" * 70)

        # Step 1: Load road network
        print("\n[1/6] Loading road network...")
        self.roads_gdf = self.load_and_process_roads(self.road_network_path)

        # Step 2: Create initial buffers for bbox calculation (in WGS84)
        print("\n[2/6] Creating buffers for bounding box calculation...")
        self.roads_gdf["buffer"] = self.roads_gdf.geometry.buffer(
            self.buffer_distance_wgs84
        )
        bbox = self.calculate_bounding_box(self.roads_gdf)

        # Step 3: Extract buildings from OSM
        print("\n[3/6] Extracting buildings from OpenStreetMap...")
        self.buildings_gdf = self.extract_buildings_from_osm(bbox)

        # Step 4: Assign buildings to roads
        print("\n[4/6] Assigning buildings to road segments...")
        joined_gdf = self.assign_buildings_to_roads(self.roads_gdf, self.buildings_gdf)

        # Step 5: Aggregate counts
        print("\n[5/6] Aggregating building counts by type...")
        self.building_counts = self.aggregate_building_counts(joined_gdf)

        # Step 6: Save results
        print("\n[6/6] Saving results...")
        output_path = self.output_dir / "roads_buildings.csv"
        self.building_counts.to_csv(output_path, index=False)
        print(f"  Saved building data to: {output_path}")

        print("\n" + "=" * 70)
        print("Buildings Processing Complete!")
        print("=" * 70)
        print(f"\nProcessed {len(self.building_counts)} road segments")
        print(f"Total building types: {len(self.building_counts.columns) - 1}")
        print(f"Output saved to: {output_path}")
        print("\nThis file is ready for use by RoadGraphConstructor in graph.py")

        return self.building_counts

    def get_building_distribution_summary(self) -> pd.DataFrame:
        """
        Generate a summary of building distribution across the road network.

        This method provides statistical insights into how buildings are
        distributed across road segments and which building types are most common.

        Returns:
            pd.DataFrame: Summary statistics showing:
                - Total road segments processed
                - Roads with buildings
                - Total buildings counted
                - Most common building types
                - Average buildings per road
        """
        if self.building_counts is None:
            raise ValueError(
                "No processed data available. Run process_and_save() first."
            )

        # Calculate total buildings per road
        building_cols = [
            col for col in self.building_counts.columns if col != "road_id"
        ]
        total_buildings_per_road = self.building_counts[building_cols].sum(axis=1)

        # Sum buildings by type across all roads
        building_type_totals = (
            self.building_counts[building_cols].sum().sort_values(ascending=False)
        )

        summary_data = {
            "Metric": [
                "Total Road Segments",
                "Roads with Buildings",
                "Total Buildings Counted",
                "Unique Building Types",
                "Avg Buildings per Road (all)",
                "Avg Buildings per Road (with buildings)",
                "Most Common Building Type",
                "Second Most Common Type",
                "Third Most Common Type",
            ],
            "Value": [
                len(self.building_counts),
                (total_buildings_per_road > 0).sum(),
                int(total_buildings_per_road.sum()),
                len(building_cols),
                round(total_buildings_per_road.mean(), 2),
                round(total_buildings_per_road[total_buildings_per_road > 0].mean(), 2),
                f"{building_type_totals.index[0]} ({int(building_type_totals.iloc[0])})",
                f"{building_type_totals.index[1]} ({int(building_type_totals.iloc[1])})"
                if len(building_type_totals) > 1
                else "N/A",
                f"{building_type_totals.index[2]} ({int(building_type_totals.iloc[2])})"
                if len(building_type_totals) > 2
                else "N/A",
            ],
        }

        return pd.DataFrame(summary_data)


if __name__ == "__main__":
    # Define input data paths
    road_network_path = "data/2_Estructura_via/estructura_via_completa/MVI.geojson"
    output_dir = "data/datos_procesados/buildings/"

    # Initialize processor
    processor = BuildingsProcessor(
        road_network_path=road_network_path,
        output_dir=output_dir,
        buffer_distance_wgs84=0.01,  # ~1km for bbox calculation
        buffer_distance_meters=100.0,  # 100m for building assignment
        target_crs="EPSG:3116",  # Colombia projected CRS
    )

    # Execute processing pipeline
    buildings_dataset = processor.process_and_save()

    print("\nProcessing complete! You can now use this file in graph.py")
