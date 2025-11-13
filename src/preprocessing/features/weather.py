from pathlib import Path
from typing import Dict, List

import geopandas as gpd
import pandas as pd


class WeatherProcessor:
    """
    A class for processing weather data and assigning weather stations to road segments.

    This processor handles the complete pipeline from raw weather data to spatially-assigned
    station-to-road mappings, suitable for training Graph Neural Networks for traffic forecasting.

    Attributes:
        weather_data_paths (Dict[str, str]): Dictionary mapping weather variable names to file paths
        road_network_path (str): Path to the road network GeoJSON file
        output_dir (str): Directory where processed files will be saved
        target_year (int): Year to filter weather data (default: 2024)
        temporal_resolution (str): Temporal aggregation frequency (default: "15min")
        buffer_distance (float): Buffer distance in meters for spatial joins (default: 15)
        target_crs (str): Target coordinate reference system for spatial operations (default: "EPSG:3116")
    """

    def __init__(
        self,
        weather_data_paths: Dict[str, str],
        road_network_path: str,
        output_dir: str,
        target_year: int = 2024,
        temporal_resolution: str = "15min",
        buffer_distance: float = 15.0,
        target_crs: str = "EPSG:3116",
    ):
        """
        Initialize the WeatherProcessor with data paths and processing parameters.

        Args:
            weather_data_paths: Dictionary mapping weather variable names to their parquet file paths.
                Expected keys: "direccion_viento", "humedad", "precipitacion", "velocidad_viento"
            road_network_path: Path to the road network GeoJSON file (MVI format)
            output_dir: Directory where processed weather data will be saved
            target_year: Year to filter weather observations (default: 2024)
            temporal_resolution: Pandas frequency string for temporal aggregation (default: "15min")
            buffer_distance: Buffer distance in meters around road segments for station assignment (default: 15)
            target_crs: EPSG code for projected coordinate system used in spatial operations (default: "EPSG:3116")
        """
        self.weather_data_paths = weather_data_paths
        self.road_network_path = road_network_path
        self.output_dir = Path(output_dir)
        self.target_year = target_year
        self.temporal_resolution = temporal_resolution
        self.buffer_distance = buffer_distance
        self.target_crs = target_crs

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Storage for processed data
        self.weather_dfs: List[pd.DataFrame] = []
        self.roads_gdf: gpd.GeoDataFrame | None = None
        self.assigned_stations_dfs: List[gpd.GeoDataFrame] = []

    def process_weather_dataframe(self, df_path: str, target_col: str) -> pd.DataFrame:
        """
        Load and process a single weather variable dataset.

        This method:
        1. Loads the parquet file
        2. Renames columns to standardized format
        3. Converts timestamps to datetime
        4. Filters data for the target year
        5. Aggregates to specified temporal resolution (default: 15-minute intervals)
        6. Removes missing values

        Args:
            df_path: Path to the weather data parquet file
            target_col: Name of the weather variable (e.g., "direccion_viento", "humedad")

        Returns:
            pd.DataFrame: Processed weather dataframe with columns:
                - timestamp (index): DateTime index at specified temporal resolution
                - estacion_{target_col}: Station name
                - lat: Station latitude
                - long: Station longitude
                - {target_col}: Weather measurement values
        """
        print(f"Loading {target_col} data...")
        df = pd.read_parquet(df_path)

        # Standardize column names
        df = df[
            ["FECHALECTURA", "VALORLECTURA", "LATITUD", "LONGITUD", "ESTACION"]
        ].rename(
            columns={
                "VALORLECTURA": target_col,
                "FECHALECTURA": "timestamp",
                "LONGITUD": "long",
                "LATITUD": "lat",
                "ESTACION": "estacion",
            }
        )

        # Parse timestamps
        df["timestamp"] = pd.to_datetime(df["timestamp"], format="%d-%m-%Y %H:%M:%S")

        # Filter for target year
        time_filter = df.timestamp.dt.year == self.target_year
        df = df[time_filter]

        # Aggregate to specified temporal resolution using groupby
        df = (
            df.set_index("timestamp")
            .groupby(
                [
                    pd.Grouper(freq=self.temporal_resolution),
                    "estacion",
                    "lat",
                    "long",
                ]
            )
            .mean()
            .reset_index()
            .set_index("timestamp")
        )

        # Remove missing values
        df = df.dropna(subset=[target_col])

        # Rename station column to be variable-specific
        df = df.rename(columns={"estacion": f"estacion_{target_col}"})

        print(f"  Processed {len(df)} records for {target_col}")
        return df

    def load_road_network(self, path: str) -> gpd.GeoDataFrame:
        """
        Load and process the road network from a GeoJSON file.

        This method:
        1. Loads the road network geometry
        2. Converts to WGS84 (EPSG:4326) for consistency
        3. Creates unique road IDs from MVI codes
        4. Dissolves split segments into unified road segments
        5. Transforms to projected CRS for accurate spatial operations
        6. Creates buffers around road segments for station assignment

        Args:
            path: Path to the road network GeoJSON file

        Returns:
            gpd.GeoDataFrame: Processed road network with columns:
                - road_id: Unique identifier for each road segment
                - geometry: Buffered polygon geometry in projected CRS
                - mvi_length: Length of the road segment in meters
        """
        print("Loading road network...")
        mvi = gpd.read_file(path)

        # Convert to WGS84 for consistency
        mvi = mvi.to_crs(epsg=4326)

        # Create unique road IDs from MVI codes
        mvi["id_mvi"] = (
            mvi["MVICIV"].astype(str) + "_" + mvi["MVISVIA"].str.replace(" ", "")
        )
        mvi["contador"] = mvi.groupby("id_mvi").cumcount() + 1
        mvi["id_mvi"] = mvi["id_mvi"] + "." + mvi["contador"].astype(str)

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

        # Buffer road segments for spatial join with weather stations
        mvi_buffered["geometry"] = mvi_buffered.geometry.buffer(
            self.buffer_distance, cap_style="square"
        )

        print(f"  Loaded {len(mvi_buffered)} road segments")
        return mvi_buffered[["road_id", "geometry", "mvi_length"]]

    def assign_stations_to_roads(
        self,
        weather_df: pd.DataFrame,
        roads_gdf: gpd.GeoDataFrame,
        station_prefix: str = "estacion_",
    ) -> gpd.GeoDataFrame:
        """
        Assign weather stations to the nearest road segments using spatial joins.

        This method performs efficient spatial assignment by:
        1. Extracting unique station locations from weather data
        2. Creating a GeoDataFrame of station points
        3. Using GeoPandas sjoin_nearest to find the closest road for each station
        4. Returning a mapping of road_id to assigned weather stations

        The assignment is done in a projected CRS (EPSG:3116) to ensure accurate
        distance calculations in meters.

        Args:
            weather_df: Weather dataframe containing station locations and measurements
            roads_gdf: GeoDataFrame of buffered road segments
            station_prefix: Prefix used in station column names (default: "estacion_")

        Returns:
            gpd.GeoDataFrame: Assignment results with columns:
                - road_id: Road segment identifier
                - estacion_{variable}: Assigned weather station name
                - road_geometry: Road segment geometry
                - station_geometry: Weather station point geometry
        """
        # Identify the station column for this weather variable
        station_col = [
            col for col in weather_df.columns if col.startswith(station_prefix)
        ][0]

        # Extract unique stations with their coordinates
        stations_df = weather_df.reset_index()[
            [station_col, "lat", "long"]
        ].drop_duplicates()

        # Create GeoDataFrame of station points
        stations_gdf = gpd.GeoDataFrame(
            stations_df,
            geometry=gpd.points_from_xy(stations_df.long, stations_df.lat),
            crs="EPSG:4326",
        )

        # Ensure roads are in projected CRS
        if roads_gdf.crs and roads_gdf.crs.to_epsg() != 3116:
            roads_gdf_proj = roads_gdf.to_crs(epsg=3116)
        else:
            roads_gdf_proj = roads_gdf

        # Convert stations to projected CRS for accurate distance calculations
        stations_gdf_proj = stations_gdf.to_crs(epsg=3116)

        # Perform spatial join to find nearest station for each road
        joined_gdf = gpd.sjoin_nearest(
            roads_gdf_proj,
            stations_gdf_proj,
            how="left",
            lsuffix="_road",
            rsuffix="_station",
        )

        # Extract relevant columns
        result = joined_gdf[["road_id", station_col]].copy()
        result["road_geometry"] = joined_gdf["geometry"]

        # Add station geometry by merging
        result = result.merge(
            stations_gdf_proj[[station_col, "geometry"]].rename(
                columns={"geometry": "station_geometry"}
            ),
            on=station_col,
            how="left",
        )

        print(
            f"  Assigned stations to {len(result)} road segments for {station_col.replace('estacion_', '')}"
        )
        return result

    def merge_station_assignments(
        self, assigned_stations_dfs: List[gpd.GeoDataFrame]
    ) -> pd.DataFrame:
        """
        Merge station assignments from multiple weather variables into a single dataframe.

        This creates a comprehensive mapping where each road segment has assigned stations
        for all weather variables (wind direction, humidity, precipitation, wind speed).

        Args:
            assigned_stations_dfs: List of GeoDataFrames with station assignments per variable

        Returns:
            pd.DataFrame: Merged dataframe indexed by road_id with columns:
                - estacion_direccion_viento: Wind direction station
                - estacion_humedad: Humidity station
                - estacion_precipitacion: Precipitation station
                - estacion_velocidad_viento: Wind speed station
        """
        print("Merging station assignments...")

        # Start with the first assignment
        merged_df = (
            assigned_stations_dfs[0]
            .set_index("road_id")
            .drop(columns=["station_geometry", "road_geometry"])
        )

        # Join with remaining assignments
        for df in assigned_stations_dfs[1:]:
            merged_df = merged_df.join(
                df.set_index("road_id").drop(
                    columns=["station_geometry", "road_geometry"]
                ),
                how="outer",
            )

        print(f"  Merged assignments for {len(merged_df)} road segments")
        return merged_df

    def process_and_save(self) -> None:
        """
        Execute the complete weather data processing pipeline and save results.

        This method orchestrates the entire workflow:
        1. Loads and processes each weather variable dataset
        2. Loads the road network
        3. Assigns weather stations to road segments for each variable
        4. Merges all station assignments
        5. Saves station assignments to parquet
        6. Saves individual weather dataframes to parquet

        Output files:
            - {output_dir}/stations.parquet: Road-to-station assignments
            - {output_dir}/direccion_viento.parquet: Processed wind direction data
            - {output_dir}/humedad.parquet: Processed humidity data
            - {output_dir}/precipitacion.parquet: Processed precipitation data
            - {output_dir}/velocidad_viento.parquet: Processed wind speed data

        These files are designed to be consumed by the SparseDatasetGenerator
        in merging.py for integration into the traffic forecasting pipeline.
        """
        print("=" * 70)
        print("Starting Weather Data Processing Pipeline")
        print("=" * 70)

        # Step 1: Process all weather dataframes
        print("\n[1/5] Processing weather dataframes...")
        for var_name, file_path in self.weather_data_paths.items():
            weather_df = self.process_weather_dataframe(file_path, var_name)
            self.weather_dfs.append(weather_df)

        # Step 2: Load road network
        print("\n[2/5] Loading road network...")
        self.roads_gdf = self.load_road_network(self.road_network_path)

        # Step 3: Assign stations to roads for each weather variable
        print("\n[3/5] Assigning weather stations to road segments...")
        for weather_df in self.weather_dfs:
            assigned_df = self.assign_stations_to_roads(weather_df, self.roads_gdf)
            self.assigned_stations_dfs.append(assigned_df)

        # Step 4: Merge all station assignments
        print("\n[4/5] Merging station assignments...")
        merged_stations = self.merge_station_assignments(self.assigned_stations_dfs)

        # Save merged station assignments
        stations_output_path = self.output_dir / "stations.parquet"
        merged_stations.to_parquet(stations_output_path, index=True)
        print(f"  Saved station assignments to: {stations_output_path}")

        # Step 5: Save individual weather dataframes
        print("\n[5/5] Saving processed weather dataframes...")
        for var_name, weather_df in zip(
            self.weather_data_paths.keys(), self.weather_dfs
        ):
            # Drop coordinate columns (not needed in time series data)
            weather_df_clean = weather_df.drop(columns=["long", "lat"])

            output_path = self.output_dir / f"{var_name}.parquet"
            weather_df_clean.to_parquet(output_path, index=True)
            print(f"  Saved {var_name} data to: {output_path}")

        print("\n" + "=" * 70)
        print("Weather Data Processing Complete!")
        print("=" * 70)
        print(f"\nAll outputs saved to: {self.output_dir}")
        print("\nThese files are ready for integration with the SparseDatasetGenerator")
        print("in merging.py for traffic forecasting model training.")


if __name__ == "__main__":
    # Define input data paths
    weather_data_paths = {
        "direccion_viento": "data/3_clima/tratados/direccion_viento.parquet",
        "humedad": "data/3_clima/tratados/humedad.parquet",
        "precipitacion": "data/3_clima/tratados/precipitacion.parquet",
        "velocidad_viento": "data/3_clima/tratados/velocidad_viento.parquet",
    }

    road_network_path = "data/2_Estructura_via/estructura_via_completa/MVI.geojson"
    output_dir = "data/datos_procesados/weather/"

    # Initialize processor
    processor = WeatherProcessor(
        weather_data_paths=weather_data_paths,
        road_network_path=road_network_path,
        output_dir=output_dir,
        target_year=2024,
        temporal_resolution="15min",
        buffer_distance=15.0,
        target_crs="EPSG:3116",
    )

    # Execute processing pipeline
    processor.process_and_save()

    print("\nProcessing complete! You can now use these files in merging.py")
