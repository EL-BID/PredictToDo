import datetime as dt
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point


class IncidentsProcessor:
    """
    A class for processing traffic incidents data and assigning them to road segments.

    This processor handles the complete pipeline from raw incidents data to spatially-assigned
    and temporally-aligned incident features suitable for training Graph Neural Networks
    for traffic forecasting.

    Unlike events (which spread through the network), incidents are localized disruptions
    that affect only the specific road segment where they occur. The processor extracts
    both incident types and case-specific information to provide rich feature representations.

    Attributes:
        incidents_data_path (str): Path to the incidents CSV file
        road_network_path (str): Path to the road network GeoJSON file
        output_dir (str): Directory where processed files will be saved
        buffer_distance (float): Buffer distance in decimal degrees for spatial joins (default: 0.001)
        temporal_resolution (str): Temporal aggregation frequency (default: "15min")
    """

    def __init__(
        self,
        incidents_data_path: str,
        road_network_path: str,
        output_dir: str,
        buffer_distance: float = 0.001,
        temporal_resolution: str = "15min",
    ):
        """
        Initialize the IncidentsProcessor with data paths and processing parameters.

        Args:
            incidents_data_path: Path to the incidents CSV file with columns:
                - Coordenada X: Incident longitude
                - Coordenada Y: Incident latitude
                - Fecha Cierre: Incident closing date/time (format: "YYYY-MM-DD HH:MM:SS")
                - Tipo Incidente: Incident type category
                - Especificación del Caso: Case-specific incident details
            road_network_path: Path to the road network GeoJSON file (MVI format)
            output_dir: Directory where processed incidents data will be saved
            buffer_distance: Buffer distance in decimal degrees for spatial joins (default: 0.001 ≈ 100m)
            temporal_resolution: Pandas frequency string for temporal aggregation (default: "15min")
        """
        self.incidents_data_path = incidents_data_path
        self.road_network_path = road_network_path
        self.output_dir = Path(output_dir)
        self.buffer_distance = buffer_distance
        self.temporal_resolution = temporal_resolution

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Storage for processed data
        self.incidents_gdf: gpd.GeoDataFrame | None = None
        self.roads_gdf: gpd.GeoDataFrame | None = None

    def load_road_network(self) -> gpd.GeoDataFrame:
        """
        Load and process the road network from GeoJSON.

        This method:
        1. Loads the road network geometry
        2. Creates unique road IDs from MVI codes
        3. Dissolves split segments into unified roads
        4. Creates buffers for spatial joins with incidents

        Returns:
            gpd.GeoDataFrame: Processed road network with buffer geometries
        """
        print("Loading road network...")
        roads_gdf = gpd.read_file(self.road_network_path)
        roads_gdf = roads_gdf.to_crs(epsg=4326)

        # Create unique road IDs
        roads_gdf["id_mvi"] = (
            roads_gdf["MVICIV"].astype(str)
            + "_"
            + roads_gdf["MVISVIA"].str.replace(" ", "")
        )
        roads_gdf["contador"] = roads_gdf.groupby("id_mvi").cumcount() + 1
        roads_gdf["id_mvi"] = (
            roads_gdf["id_mvi"] + "." + roads_gdf["contador"].astype(str)
        )

        # Dissolve split segments
        roads_gdf = (
            roads_gdf.explode(ignore_index=True)  # Explode MultiLineStrings
            .dissolve(by="id_mvi")  # Merge all segments of same road
            .reset_index()
        )

        # Create road_id and buffer
        roads_gdf["road_id"] = roads_gdf["id_mvi"]
        roads_gdf["buffer"] = roads_gdf.geometry.buffer(self.buffer_distance)

        # Return only necessary columns with buffer geometry
        roads_buf = roads_gdf.set_geometry("buffer")[["road_id", "buffer"]]

        print(f"  Loaded {len(roads_buf)} road segments")
        return roads_buf

    def load_incidents_data(self) -> gpd.GeoDataFrame:
        """
        Load and preprocess incidents data from CSV.

        This method:
        1. Loads the incidents CSV file
        2. Creates Point geometries from coordinates
        3. Renames columns to standardized format
        4. Converts to GeoDataFrame

        Returns:
            gpd.GeoDataFrame: Processed incidents with geometry
        """
        print("Loading incidents data...")
        incidents = pd.read_csv(self.incidents_data_path, encoding="latin-1")

        # Create Point geometries from coordinates
        incidents["geometry"] = incidents.apply(  # type: ignore[call-overload]
            lambda row: Point(row["Coordenada X"], row["Coordenada Y"]), axis=1
        )

        # Convert to GeoDataFrame
        incidents_gdf = gpd.GeoDataFrame(incidents, crs="EPSG:4326")

        # Rename columns to English for consistency
        incidents_gdf = incidents_gdf.rename(
            columns={
                "Tipo Incidente": "incident_type",
                "Especificación del Caso": "case_specifics",
            }
        )

        print(f"  Loaded {len(incidents_gdf)} incidents")
        return incidents_gdf

    def assign_incidents_to_roads(
        self, incidents_gdf: gpd.GeoDataFrame, roads_gdf: gpd.GeoDataFrame
    ) -> gpd.GeoDataFrame:
        """
        Assign incidents to road segments using spatial join.

        This method:
        1. Performs spatial join between buffered roads and incident points
        2. Filters out incidents that don't intersect with any road
        3. Parses closing dates and aligns to temporal resolution
        4. Adds count column for aggregation

        Args:
            incidents_gdf: GeoDataFrame of incidents with Point geometries
            roads_gdf: GeoDataFrame of roads with buffer geometries

        Returns:
            gpd.GeoDataFrame: Incidents assigned to road segments with temporal alignment
        """
        print("Assigning incidents to road segments...")

        # Spatial join: find roads that intersect with each incident
        joined = gpd.sjoin(roads_gdf, incidents_gdf, how="left", predicate="intersects")

        # Keep only incidents that matched with roads
        joined = joined.loc[pd.notnull(joined.index_right)].reset_index()

        # Add count column for aggregation
        joined["totals"] = 1

        # Parse closing dates with progress bar
        print("  Parsing closing dates...")
        joined["closing_date"] = joined["Fecha Cierre"].progress_apply(
            lambda x: dt.datetime.strptime(x, "%Y-%m-%d %H:%M:%S")
        )

        # Align to temporal resolution (default: 15-minute intervals)
        joined["timeframe"] = joined["closing_date"].dt.floor(self.temporal_resolution)

        print(f"  Assigned {len(joined)} incidents to road segments")
        return joined

    def aggregate_by_incident_type(self, joined_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
        """
        Aggregate incidents by road, timeframe, and incident type.

        This creates a feature matrix where each row represents a (road_id, timeframe) pair
        and columns represent counts of different incident types.

        Args:
            joined_gdf: GeoDataFrame with incidents assigned to roads

        Returns:
            pd.DataFrame: Pivot table with incident type counts per road-timeframe
        """
        print("Aggregating by incident type...")

        incident_type_df = (
            joined_gdf.groupby(["road_id", "timeframe", "incident_type"])
            .agg({"totals": "sum"})
            .reset_index()
            .pivot_table(
                index=["road_id", "timeframe"],
                columns="incident_type",
                values="totals",
            )
            .fillna(0)
            .reset_index()
        )

        # Prefix column names to avoid conflicts
        incident_type_df.columns = [
            f"incident_type_{col}" if col not in ["road_id", "timeframe"] else col
            for col in incident_type_df.columns
        ]

        print(f"  Created {len(incident_type_df.columns) - 2} incident type features")
        return incident_type_df

    def aggregate_by_case_specifics(self, joined_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
        """
        Aggregate incidents by road, timeframe, and case specifics.

        This creates a feature matrix where each row represents a (road_id, timeframe) pair
        and columns represent counts of different case-specific categories.

        Args:
            joined_gdf: GeoDataFrame with incidents assigned to roads

        Returns:
            pd.DataFrame: Pivot table with case-specific counts per road-timeframe
        """
        print("Aggregating by case specifics...")

        case_specifics_df = (
            joined_gdf.groupby(["road_id", "timeframe", "case_specifics"])
            .agg({"totals": "sum"})
            .reset_index()
            .pivot_table(
                index=["road_id", "timeframe"],
                columns="case_specifics",
                values="totals",
            )
            .fillna(0)
            .reset_index()
        )

        # Prefix column names to avoid conflicts
        case_specifics_df.columns = [
            f"incident_case_specifics_{col}"
            if col not in ["road_id", "timeframe"]
            else col
            for col in case_specifics_df.columns
        ]

        print(f"  Created {len(case_specifics_df.columns) - 2} case-specific features")
        return case_specifics_df

    def merge_incident_features(
        self, incident_type_df: pd.DataFrame, case_specifics_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Merge incident type and case-specific features into a single dataset.

        This combines both feature sets to create a comprehensive incident feature matrix
        with all available information.

        Args:
            incident_type_df: DataFrame with incident type features
            case_specifics_df: DataFrame with case-specific features

        Returns:
            pd.DataFrame: Merged dataframe with all incident features
        """
        print("Merging incident features...")

        incidents_data = pd.merge(
            incident_type_df,
            case_specifics_df,
            on=["road_id", "timeframe"],
            how="left",
        )

        print(f"  Final dataset: {len(incidents_data)} road-timeframe records")
        print(f"  Total features: {len(incidents_data.columns) - 2}")
        return incidents_data

    def process_and_save(self) -> None:
        """
        Execute the complete incidents data processing pipeline and save results.

        This method orchestrates the entire workflow:
        1. Loads road network from GeoJSON
        2. Loads incidents data from CSV
        3. Assigns incidents to nearest road segments via spatial join
        4. Aligns incidents temporally to 15-minute intervals
        5. Aggregates by incident type
        6. Aggregates by case specifics
        7. Merges both feature sets
        8. Saves the final dataset to CSV

        Output file:
            - {output_dir}/incidents_dataset.csv: Processed incidents dataset with columns:
                - road_id: Road segment identifier
                - timeframe: 15-minute time bucket
                - incident_type_*: Count of incidents by type (one column per type)
                - incident_case_specifics_*: Count of incidents by case (one column per case)

        This file is designed to be consumed by the SparseDatasetGenerator
        in merging.py for integration into the traffic forecasting pipeline.
        """
        print("=" * 70)
        print("Starting Incidents Data Processing Pipeline")
        print("=" * 70)

        # Step 1: Load road network
        print("\n[1/7] Loading road network...")
        self.roads_gdf = self.load_road_network()

        # Step 2: Load incidents data
        print("\n[2/7] Loading incidents data...")
        self.incidents_gdf = self.load_incidents_data()

        # Step 3: Assign incidents to roads
        print("\n[3/7] Assigning incidents to roads...")
        assigned_incidents = self.assign_incidents_to_roads(
            self.incidents_gdf, self.roads_gdf
        )

        # Step 4: Aggregate by incident type
        print("\n[4/7] Aggregating by incident type...")
        incident_type_df = self.aggregate_by_incident_type(assigned_incidents)

        # Step 5: Aggregate by case specifics
        print("\n[5/7] Aggregating by case specifics...")
        case_specifics_df = self.aggregate_by_case_specifics(assigned_incidents)

        # Step 6: Merge features
        print("\n[6/7] Merging incident features...")
        incidents_dataset = self.merge_incident_features(
            incident_type_df, case_specifics_df
        )

        # Step 7: Save results
        print("\n[7/7] Saving processed dataset...")
        output_path = self.output_dir / "incidents_dataset.csv"
        incidents_dataset.to_csv(output_path, index=False)
        print(f"  Saved incidents dataset to: {output_path}")

        print("\n" + "=" * 70)
        print("Incidents Data Processing Complete!")
        print("=" * 70)
        print(f"\nOutput saved to: {output_path}")
        print("\nThis file is ready for integration with the SparseDatasetGenerator")
        print("in merging.py for traffic forecasting model training.")


if __name__ == "__main__":
    # Define input data paths
    incidents_data_path = "data/7_incidentes/incidents_dataset_prev.csv"
    road_network_path = "data/2_Estructura_via/estructura_via_completa/MVI.geojson"
    output_dir = "data/datos_procesados/incidents/"

    # Initialize processor
    processor = IncidentsProcessor(
        incidents_data_path=incidents_data_path,
        road_network_path=road_network_path,
        output_dir=output_dir,
        buffer_distance=0.001,  # Approximately 100 meters in degrees
        temporal_resolution="15min",
    )

    # Execute processing pipeline
    processor.process_and_save()

    print("\n✓ Processing complete! You can now use this file in merging.py")
