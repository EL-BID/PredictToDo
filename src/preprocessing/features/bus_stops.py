from pathlib import Path

import geopandas as gpd
import pandas as pd


class BusStopsProcessor:
    """
    A class for processing bus stop data and assigning stops to road network segments.

    This processor handles the complete pipeline from raw SITP bus stop shapefiles
    to aggregated bus stop counts per road segment, suitable for enriching the road
    network graph with public transit infrastructure features.

    The processor:
    1. Loads bus stop point geometries from shapefile
    2. Loads the road network and creates unique road IDs
    3. Assigns each bus stop to the nearest road segment using spatial joins
    4. Aggregates stop counts and statistics (distance, type) per road segment

    Attributes:
        bus_stops_shapefile (str): Path to the SITP bus stops shapefile
        road_network_path (str): Path to the road network GeoJSON file
        output_dir (str): Directory where processed files will be saved
        max_distance_degrees (float): Maximum search distance for nearest road (default: 0.005 ≈ 500m)
        target_crs (str): Target coordinate reference system (default: "EPSG:21899" for Colombia)
    """

    def __init__(
        self,
        bus_stops_shapefile: str,
        road_network_path: str,
        output_dir: str,
        max_distance_degrees: float = 0.005,
        target_crs: str = "EPSG:21899",
    ):
        """
        Initialize the BusStopsProcessor with data paths and processing parameters.

        Args:
            bus_stops_shapefile: Path to the SITP bus stops shapefile (e.g., Paraderos_Zonales_del_SITP.shp)
            road_network_path: Path to the road network GeoJSON file (MVI format)
            output_dir: Directory where processed bus stop assignments will be saved
            max_distance_degrees: Maximum distance in degrees for nearest road search (default: 0.005 ≈ 500m)
            target_crs: EPSG code for projected coordinate system used for accurate distance calculation (default: "EPSG:21899")
        """
        self.bus_stops_shapefile = bus_stops_shapefile
        self.road_network_path = road_network_path
        self.output_dir = Path(output_dir)
        self.max_distance_degrees = max_distance_degrees
        self.target_crs = target_crs

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Storage for processed data
        self.stops_gdf: gpd.GeoDataFrame | None = None
        self.roads_gdf: gpd.GeoDataFrame | None = None
        self.assigned_stops: gpd.GeoDataFrame | None = None
        self.final_dataset: pd.DataFrame | None = None

    def load_and_process_bus_stops(self, shapefile_path: str) -> gpd.GeoDataFrame:
        """
        Load and process bus stop point data from shapefile.

        This method:
        1. Loads the SITP bus stops shapefile
        2. Selects relevant columns (objectid, latitud, longitud, tipo_m_s)
        3. Creates point geometries from lat/lon coordinates
        4. Converts to WGS84 (EPSG:4326) coordinate system

        The tipo_m_s field indicates the bus stop type:
        - 'M': Main stop (Parada Principal)
        - 'S': Secondary stop (Parada Secundaria)

        Args:
            shapefile_path: Path to the bus stops shapefile

        Returns:
            gpd.GeoDataFrame: Processed bus stops with columns:
                - objectid: Unique stop identifier
                - latitud: Latitude coordinate
                - longitud: Longitude coordinate
                - tipo_m_s: Stop type (M/S)
                - geometry: Point geometry in WGS84
        """
        print("Loading bus stop data...")
        stops_gdf = gpd.read_file(shapefile_path)

        # Select relevant columns
        stops_gdf = stops_gdf[["objectid", "latitud", "longitud", "tipo_m_s"]]

        # Create point geometries from coordinates
        stops_gdf = gpd.GeoDataFrame(
            stops_gdf,
            geometry=gpd.points_from_xy(stops_gdf.longitud, stops_gdf.latitud),
            crs="EPSG:4326",
        )

        # Ensure WGS84 coordinate system
        stops_gdf = stops_gdf.to_crs(epsg=4326)

        print(f"  Loaded {len(stops_gdf)} bus stops")
        print(f"    - Main stops (M): {(stops_gdf['tipo_m_s'] == 'M').sum()}")
        print(f"    - Secondary stops (S): {(stops_gdf['tipo_m_s'] == 'S').sum()}")

        return stops_gdf

    def load_and_process_roads(self, path: str) -> gpd.GeoDataFrame:
        """
        Load and process the road network for bus stop assignment.

        This method:
        1. Loads the road network geometry
        2. Converts to WGS84 (EPSG:4326) for consistency
        3. Creates unique road IDs from MVI codes
        4. Dissolves split segments into unified road segments

        Args:
            path: Path to the road network GeoJSON file

        Returns:
            gpd.GeoDataFrame: Processed road network with columns:
                - road_id: Unique identifier for each road segment
                - geometry: LineString geometry in WGS84
        """
        print("Loading road network...")
        roads_gdf = gpd.read_file(path)

        # Convert to WGS84 for consistency
        roads_gdf = roads_gdf.to_crs(epsg=4326)

        # Create unique road IDs from MVI codes
        roads_gdf["id_mvi"] = (
            roads_gdf["MVICIV"].astype(str)
            + "_"
            + roads_gdf["MVISVIA"].str.replace(" ", "")
        )
        roads_gdf["contador"] = roads_gdf.groupby("id_mvi").cumcount() + 1
        roads_gdf["id_mvi"] = (
            roads_gdf["id_mvi"].astype(str) + "." + roads_gdf["contador"].astype(str)
        )

        # Dissolve split segments into unified roads
        roads_gdf = (
            roads_gdf.explode(ignore_index=True)  # Explode MultiLineStrings if present
            .dissolve(by="id_mvi")  # Merge all segments of the same road
            .reset_index()
        )
        roads_gdf["road_id"] = roads_gdf["id_mvi"]

        print(f"  Loaded {len(roads_gdf)} road segments")

        return roads_gdf

    def assign_stops_to_streets(
        self, bus_stops_gdf: gpd.GeoDataFrame, streets_gdf: gpd.GeoDataFrame
    ) -> gpd.GeoDataFrame:
        """
        Assign bus stops to the nearest street segment with accurate distance calculation.

        This method uses a two-step process:
        1. Fast spatial join in WGS84 to find candidate nearest roads
        2. Accurate distance calculation in projected CRS (meters)

        The spatial join uses a maximum search distance to limit the search area.
        When multiple roads are within the search distance, the one with the
        highest overlap is selected.

        Args:
            bus_stops_gdf: GeoDataFrame with bus stop point geometries
            streets_gdf: GeoDataFrame with road segment line geometries

        Returns:
            gpd.GeoDataFrame: Bus stops assigned to roads with columns:
                - objectid: Bus stop identifier
                - tipo_m_s: Stop type (M/S)
                - road_id: Assigned road identifier
                - distance_meters: Distance from stop to road centerline in meters
                - index_right: Internal index of the matched road
        """
        print("Assigning bus stops to nearest road segments...")

        # Step 1: Fast join in WGS84 to get candidate matches
        # This uses approximate degrees-based distance, but is very fast
        print("  Performing spatial join to find candidate roads...")
        joined_gdf = gpd.sjoin_nearest(
            bus_stops_gdf,
            streets_gdf,
            how="left",
            max_distance=self.max_distance_degrees,  # Approx 500 meters in degrees
        )

        # Drop geometry from joined result to avoid conflicts
        joined_gdf = joined_gdf.drop(columns="geometry")

        # Step 2: Calculate accurate distances in projected CRS
        print("  Calculating accurate distances in meters...")

        # Project both datasets to target CRS for metric calculations
        bus_stops_gdf_proj = bus_stops_gdf.to_crs(self.target_crs)
        streets_gdf_proj = streets_gdf.to_crs(self.target_crs)

        # Get geometries using original indices
        stop_geoms = bus_stops_gdf_proj.loc[joined_gdf.index].geometry
        street_geoms = streets_gdf_proj.loc[joined_gdf["index_right"]].geometry

        # Calculate distances (align=False to use positional alignment)
        joined_gdf["distance_meters"] = stop_geoms.distance(street_geoms, align=False)

        print(f"  Assigned {len(joined_gdf)} bus stops to roads")
        print(f"  Average distance: {joined_gdf['distance_meters'].mean():.2f} meters")
        print(f"  Median distance: {joined_gdf['distance_meters'].median():.2f} meters")

        return joined_gdf

    def aggregate_stops_per_road(
        self, assigned_stops: gpd.GeoDataFrame
    ) -> pd.DataFrame:
        """
        Aggregate bus stop statistics per road segment.

        This method creates summary statistics for each road segment:
        - Total number of stops
        - Number of main stops (tipo_m_s == 'M')
        - Number of secondary stops (tipo_m_s == 'S')
        - Mean distance from stops to road centerline
        - Median distance from stops to road centerline

        Args:
            assigned_stops: GeoDataFrame with bus stops assigned to roads

        Returns:
            pd.DataFrame: Aggregated statistics per road with columns:
                - road_id: Road segment identifier
                - num_stops: Total number of bus stops
                - num_stops_m: Number of main stops
                - num_stops_s: Number of secondary stops
                - mean_distance_meters: Mean distance from stops to road
                - median_distance_meters: Median distance from stops to road
        """
        print("Aggregating bus stop statistics per road segment...")

        # Create indicator variables for stop types
        assigned_stops["is_stop_m"] = assigned_stops["tipo_m_s"] == "M"
        assigned_stops["is_stop_s"] = assigned_stops["tipo_m_s"] == "S"

        # Aggregate by road_id
        final_dataset = assigned_stops.groupby("road_id").agg(
            {
                "objectid": "count",  # Total number of stops
                "is_stop_m": "sum",  # Number of main stops
                "is_stop_s": "sum",  # Number of secondary stops
                "distance_meters": ["mean", "median"],  # Distance statistics
            }
        )

        # Flatten multi-level column names
        final_dataset.columns = [
            "num_stops",
            "num_stops_m",
            "num_stops_s",
            "mean_distance_meters",
            "median_distance_meters",
        ]

        # Reset index to make road_id a column
        final_dataset = final_dataset.reset_index()

        print(f"  Generated statistics for {len(final_dataset)} road segments")
        print(f"  Total stops assigned: {final_dataset['num_stops'].sum()}")
        print(f"  Roads with main stops: {(final_dataset['num_stops_m'] > 0).sum()}")
        print(
            f"  Roads with secondary stops: {(final_dataset['num_stops_s'] > 0).sum()}"
        )

        return final_dataset

    def process_and_save(self) -> pd.DataFrame:
        """
        Execute the complete bus stops processing pipeline and save results.

        This method orchestrates the entire workflow:
        1. Loads and processes bus stop point data
        2. Loads and processes the road network
        3. Assigns each bus stop to the nearest road segment
        4. Aggregates stop counts and statistics per road
        5. Saves the final dataset to parquet

        Output file:
            - {output_dir}/bus_stops_per_road.parquet: Aggregated bus stop statistics

        This file is designed to be consumed by the RoadGraphConstructor
        in graph.py for enriching the road network graph with transit features.

        Returns:
            pd.DataFrame: The processed bus stop aggregations
        """
        print("=" * 70)
        print("Starting Bus Stops Processing Pipeline")
        print("=" * 70)

        # Step 1: Load bus stop data
        print("\n[1/4] Loading bus stop data...")
        self.stops_gdf = self.load_and_process_bus_stops(self.bus_stops_shapefile)

        # Step 2: Load road network
        print("\n[2/4] Loading road network...")
        self.roads_gdf = self.load_and_process_roads(self.road_network_path)

        # Step 3: Assign stops to roads
        print("\n[3/4] Assigning stops to road segments...")
        self.assigned_stops = self.assign_stops_to_streets(
            self.stops_gdf[["objectid", "tipo_m_s", "geometry"]],
            self.roads_gdf[["road_id", "geometry"]],
        )

        # Step 4: Aggregate statistics
        print("\n[4/4] Aggregating statistics and saving...")
        self.final_dataset = self.aggregate_stops_per_road(self.assigned_stops)

        # Save results
        output_path = self.output_dir / "bus_stops_per_road.parquet"
        self.final_dataset.to_parquet(output_path, index=False)
        print(f"  Saved bus stop data to: {output_path}")

        print("\n" + "=" * 70)
        print("Bus Stops Processing Complete!")
        print("=" * 70)
        print(f"\nProcessed {len(self.final_dataset)} road segments")
        print(f"Total bus stops assigned: {self.final_dataset['num_stops'].sum()}")
        print(f"Output saved to: {output_path}")
        print("\nThis file is ready for use by RoadGraphConstructor in graph.py")

        return self.final_dataset

    def get_stop_distribution_summary(self) -> pd.DataFrame:
        """
        Generate a summary of bus stop distribution across the road network.

        This method provides statistical insights into how bus stops are
        distributed across road segments, useful for understanding transit
        infrastructure coverage.

        Returns:
            pd.DataFrame: Summary statistics showing distribution of stops per road
        """
        if self.final_dataset is None:
            raise ValueError(
                "No processed data available. Run process_and_save() first."
            )

        summary_data = {
            "Metric": [
                "Total Road Segments",
                "Road Segments with Stops",
                "Total Bus Stops",
                "Main Stops (M)",
                "Secondary Stops (S)",
                "Avg Stops per Road (all)",
                "Avg Stops per Road (with stops)",
                "Max Stops on Single Road",
                "Avg Distance to Road (meters)",
            ],
            "Value": [
                len(self.final_dataset),
                (self.final_dataset["num_stops"] > 0).sum(),
                self.final_dataset["num_stops"].sum(),
                self.final_dataset["num_stops_m"].sum(),
                self.final_dataset["num_stops_s"].sum(),
                round(self.final_dataset["num_stops"].mean(), 2),
                round(
                    self.final_dataset[self.final_dataset["num_stops"] > 0][
                        "num_stops"
                    ].mean(),
                    2,
                ),
                self.final_dataset["num_stops"].max(),
                round(self.final_dataset["mean_distance_meters"].mean(), 2),
            ],
        }

        return pd.DataFrame(summary_data)


if __name__ == "__main__":
    # Define input data paths
    bus_stops_shapefile = (
        "data/15_Paraderos_transp_publico_Zonales_SITP/Paraderos_Zonales_del_SITP.shp"
    )
    road_network_path = "data/2_Estructura_via/estructura_via_completa/MVI.geojson"
    output_dir = "data/datos_procesados/bus_stops/"

    # Initialize processor
    processor = BusStopsProcessor(
        bus_stops_shapefile=bus_stops_shapefile,
        road_network_path=road_network_path,
        output_dir=output_dir,
        max_distance_degrees=0.005,  # Approx 500 meters
        target_crs="EPSG:21899",  # Projected CRS for Colombia
    )

    # Execute processing pipeline
    bus_stops_dataset = processor.process_and_save()

    print("\nProcessing complete! You can now use this file in graph.py")
