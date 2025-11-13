import datetime as dt
import re
from datetime import time
from pathlib import Path
from typing import Dict, List, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

# Enable progress bars for pandas operations
tqdm.pandas()


class PMTSProcessor:
    """
    A class for processing PMTS (Traffic Management Plans) data and assigning
    their impact to road segments.

    This processor handles the complete pipeline from raw COI data to temporally-expanded
    traffic management indicators suitable for training Graph Neural Networks.

    Attributes:
        coi_parques_path (str): Path to COI parks GeoJSON file
        coi_puntos_path (str): Path to COI points GeoJSON file
        coi_tramos_path (str | None): Path to COI road segments GeoJSON file (optional)
        road_network_path (str): Path to the road network GeoJSON file
        output_dir (str): Directory where processed files will be saved
        buffer_distance (float): Buffer distance in meters for spatial joins (default: 4)
        overlap_threshold (float): Minimum overlap ratio for park intersections (default: 0.15)
        temporal_resolution (str): Temporal aggregation frequency (default: "15T" = 15 minutes)
        target_crs (str): Target coordinate reference system for spatial operations (default: "EPSG:3116")
    """

    # Weekday mapping for schedule parsing
    WEEKDAY_MAP = {
        "LUNES": 0,
        "MARTES": 1,
        "MIERCOLES": 2,
        "MIÉRCOLES": 2,
        "JUEVES": 3,
        "VIERNES": 4,
        "SABADO": 5,
        "SÁBADO": 5,
        "DOMINGO": 6,
    }

    # Simple HCIE (schedule) mappings for parks
    HCIE_SIMPLE_MAPPING = {
        "24 HORAS": (time(0, 0), time(23, 59)),
        "06:00-18:00": (time(6, 0), time(18, 0)),
        "07:00-18:00": (time(7, 0), time(18, 0)),
    }

    def __init__(
        self,
        coi_parques_path: str,
        coi_puntos_path: str,
        road_network_path: str,
        output_dir: str,
        coi_tramos_path: str | None = None,
        buffer_distance: float = 4.0,
        overlap_threshold: float = 0.15,
        temporal_resolution: str = "15T",
        target_crs: str = "EPSG:3116",
    ):
        """
        Initialize the PMTSProcessor with data paths and processing parameters.

        Args:
            coi_parques_path: Path to the COI parks GeoJSON file
            coi_puntos_path: Path to the COI points GeoJSON file
            road_network_path: Path to the road network GeoJSON file (MVI format)
            output_dir: Directory where processed PMTS data will be saved
            coi_tramos_path: Optional path to COI road segments GeoJSON file
            buffer_distance: Buffer distance in meters around road segments for spatial joins (default: 4)
            overlap_threshold: Minimum overlap ratio to consider park-road intersection (default: 0.15)
            temporal_resolution: Pandas frequency string for temporal expansion (default: "15T")
            target_crs: EPSG code for projected coordinate system (default: "EPSG:3116")
        """
        self.coi_parques_path = coi_parques_path
        self.coi_puntos_path = coi_puntos_path
        self.coi_tramos_path = coi_tramos_path
        self.road_network_path = road_network_path
        self.output_dir = Path(output_dir)
        self.buffer_distance = buffer_distance
        self.overlap_threshold = overlap_threshold
        self.temporal_resolution = temporal_resolution
        self.target_crs = target_crs

        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "coi_puntos").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "coi_tramos").mkdir(parents=True, exist_ok=True)

        # Storage for processed road network
        self.road_network_buffered: gpd.GeoDataFrame | None = None

    def load_and_buffer_road_network(self) -> gpd.GeoDataFrame:
        """
        Load and process the road network with buffering for spatial joins.

        This method:
        1. Loads the road network geometry
        2. Converts to WGS84 (EPSG:4326) for consistency
        3. Creates unique road IDs from MVI codes
        4. Transforms to projected CRS for accurate spatial operations
        5. Creates small buffers around road segments for point/polygon intersections

        Returns:
            gpd.GeoDataFrame: Buffered road network with columns:
                - id_mvi: Unique identifier for each road segment
                - mvi_length: Length of the road segment in meters
                - geometry: Buffered line geometry in WGS84
        """
        print("Loading and buffering road network...")
        mvi = gpd.read_file(self.road_network_path)
        mvi = mvi.to_crs("EPSG:4326")
        mvi.columns = mvi.columns.str.lower()

        # Create unique road IDs
        mvi["id_mvi"] = (
            mvi["mviciv"].astype(str) + "_" + mvi["mvisvia"].str.replace(" ", "")
        )
        mvi["contador"] = mvi.groupby("id_mvi").cumcount() + 1
        mvi["id_mvi"] = mvi["id_mvi"] + "." + mvi["contador"].astype(str)

        # Buffer roads in projected CRS for accurate distance
        mvi_buffered = mvi.to_crs(self.target_crs).copy()
        mvi_buffered["mvi_length"] = mvi_buffered.geometry.length
        mvi_buffered["geometry"] = mvi_buffered.geometry.buffer(
            self.buffer_distance, cap_style="square"
        )

        # Convert back to WGS84 for consistency with other data
        mvi_buffered = mvi_buffered.set_geometry("geometry").to_crs("EPSG:4326")

        print(f"  Loaded {len(mvi_buffered)} road segments")
        return mvi_buffered

    def generate_simple_timeframes(self, row: pd.Series) -> pd.DatetimeIndex:
        """
        Generate 15-minute timeframes for a simple schedule (parks).

        This method expands date ranges into 15-minute intervals, filtering by
        start and end times of day. Handles overnight schedules (e.g., 22:00-06:00).

        Args:
            row: DataFrame row with columns:
                - FINI: Start datetime
                - FFIN: End datetime
                - hcie_start: Start time of day
                - hcie_end: End time of day

        Returns:
            pd.DatetimeIndex: Array of 15-minute timestamps within the schedule
        """
        # Generate full range for the entire interval
        all_times = pd.date_range(
            start=row["FINI"], end=row["FFIN"], freq=self.temporal_resolution
        )

        # Convert to array of times for filtering
        times_of_day = all_times.time

        # Handle normal vs overnight HCIE schedules
        if row["hcie_start"] <= row["hcie_end"]:
            # Normal schedule: 06:00-18:00
            mask = (times_of_day >= row["hcie_start"]) & (
                times_of_day <= row["hcie_end"]
            )
        else:
            # Overnight schedule: 22:00-06:00
            mask = (times_of_day >= row["hcie_start"]) | (
                times_of_day <= row["hcie_end"]
            )

        return all_times[mask]

    @staticmethod
    def normalize_hcie_text(text: str) -> str:
        """
        Normalize HCIE (schedule) text for consistent parsing.

        Performs multiple transformations:
        - Converts to uppercase
        - Standardizes weekday abbreviations
        - Removes accents
        - Removes parenthetical notes
        - Normalizes whitespace

        Args:
            text: Raw schedule text (e.g., "L-V 08:00-18:00")

        Returns:
            str: Normalized schedule text
        """
        text = str(text).upper()
        text = re.sub(r"\s+", " ", text)

        # Standardize weekday abbreviations
        text = text.replace("L. A V.", "LUNES A VIERNES")
        text = text.replace("L A V", "LUNES A VIERNES")
        text = text.replace("L-VIERNES", "LUNES A VIERNES")
        text = text.replace("L-V", "LUNES A VIERNES")
        text = text.replace("L-VI", "LUNES A VIERNES")
        text = text.replace("SAB", "SABADO")
        text = text.replace("DOM", "DOMINGO")
        text = text.replace("FESTIVOS", "DOMINGO")

        # Remove accents
        text = (
            text.replace("Á", "A")
            .replace("É", "E")
            .replace("Í", "I")
            .replace("Ó", "O")
            .replace("Ú", "U")
        )

        # Remove parenthetical notes
        text = re.sub(r"\([^)]*\)", "", text)
        text = text.strip()

        return text

    @classmethod
    def extract_weekday_ranges(cls, text: str) -> set:
        """
        Extract weekday numbers from schedule text.

        Supports:
        - Weekday ranges: "LUNES A VIERNES" -> {0, 1, 2, 3, 4}
        - Individual days: "SABADO" -> {5}
        - Wrapping ranges: "VIERNES A LUNES" -> {5, 6, 0}

        Args:
            text: Normalized schedule text

        Returns:
            set: Set of weekday integers (0=Monday, 6=Sunday)
        """
        weekdays = set()

        # Extract ranges like "LUNES A VIERNES"
        for match in re.finditer(
            r"(LUNES|MARTES|MIERCOLES|JUEVES|VIERNES|SABADO|DOMINGO)\s+A\s+(LUNES|MARTES|MIERCOLES|JUEVES|VIERNES|SABADO|DOMINGO)",
            text,
        ):
            start = cls.WEEKDAY_MAP[match.group(1)]
            end = cls.WEEKDAY_MAP[match.group(2)]

            if start <= end:
                # Normal range: Monday to Friday
                weekdays.update(range(start, end + 1))
            else:
                # Wrapping range: Friday to Monday
                weekdays.update(list(range(start, 7)) + list(range(0, end + 1)))

        # Extract individual weekday mentions
        for word in cls.WEEKDAY_MAP.keys():
            if word in text:
                weekdays.add(cls.WEEKDAY_MAP[word])

        # If no days specified, assume all days
        if not weekdays:
            weekdays.update(range(7))

        return weekdays

    @staticmethod
    def extract_timeframes(text: str) -> List[Tuple[time, time]]:
        """
        Extract time ranges from schedule text.

        Supports formats:
        - "08:00-18:00"
        - "8.00-18.00"
        - "08:00 A 18:00"

        Args:
            text: Normalized schedule text

        Returns:
            List[Tuple[time, time]]: List of (start_time, end_time) tuples
        """
        timeframes = []

        # Match time patterns with various separators
        for match in re.finditer(
            r"(\d{1,2}[:.]\d{2})\s*[-–A]\s*(\d{1,2}[:.]\d{2})", text
        ):
            start = dt.datetime.strptime(
                match.group(1).replace(".", ":"), "%H:%M"
            ).time()
            end = dt.datetime.strptime(match.group(2).replace(".", ":"), "%H:%M").time()
            timeframes.append((start, end))

        return timeframes

    @classmethod
    def parse_schedule(cls, schedule_text: str) -> Dict[int, Tuple[time, time]]:
        """
        Parse complex schedule text into a weekday-to-timerange mapping.

        This method handles complex schedules like:
        - "24 HORAS" -> All day, every day
        - "LUNES A VIERNES 08:00-18:00" -> Weekdays, 8am-6pm
        - "LUNES 07:00-19:00 SABADO 08:00-14:00" -> Different times per day

        Args:
            schedule_text: Raw schedule text from HCIE field

        Returns:
            Dict[int, Tuple[time, time]]: Mapping of weekday (0-6) to (start_time, end_time)
        """
        text = cls.normalize_hcie_text(schedule_text)

        # Handle 24-hour schedules
        if "24 HORAS" in text:
            return {day: (dt.time(0, 0), dt.time(23, 59)) for day in range(7)}

        schedules = {}

        # Split by weekday names to isolate blocks
        blocks = re.split(
            r"(LUNES|MARTES|MIERCOLES|JUEVES|VIERNES|SABADO|DOMINGO)", text
        )

        # Reconstruct blocks with their weekdays
        if len(blocks) > 1:
            reconstructed = []
            i = 0
            while i < len(blocks):
                if blocks[i] in cls.WEEKDAY_MAP:
                    # Combine weekday with its following content
                    reconstructed.append(blocks[i] + " " + blocks[i + 1])
                    i += 2
                else:
                    reconstructed.append(blocks[i])
                    i += 1
        else:
            reconstructed = [text]

        # Parse each block
        for block in reconstructed:
            days = cls.extract_weekday_ranges(block)
            timeframes = cls.extract_timeframes(block)

            for day in days:
                for tf in timeframes:
                    schedules[day] = (tf[0], tf[1])

        return schedules

    @staticmethod
    def time_to_seconds(t: time) -> int:
        """Convert time object to seconds since midnight."""
        return t.hour * 3600 + t.minute * 60 + t.second

    @classmethod
    def generate_complex_timeframes(cls, row: pd.Series) -> pd.DatetimeIndex:
        """
        Generate 15-minute timeframes for complex schedules with weekday-specific times.

        This vectorized method efficiently filters timestamps based on:
        - Weekday (Monday-Sunday)
        - Time of day (per weekday)
        - Overnight schedules

        Args:
            row: DataFrame row with columns:
                - FINI: Start datetime
                - FFIN: End datetime
                - parsed_schedule: Dict mapping weekday to (start_time, end_time)

        Returns:
            pd.DatetimeIndex: Array of 15-minute timestamps within the schedule
        """
        if pd.isna(row["FINI"]) or pd.isna(row["FFIN"]):
            return pd.DatetimeIndex([])

        # Generate all 15-min timestamps for the interval
        all_times = pd.date_range(start=row["FINI"], end=row["FFIN"], freq="15T")
        if all_times.empty:
            return pd.DatetimeIndex([])

        weekdays = all_times.weekday
        times_of_day = np.array([cls.time_to_seconds(t) for t in all_times.time])

        # Build schedule arrays for all weekdays
        schedule = row["parsed_schedule"]
        start_seconds = np.array(
            [
                cls.time_to_seconds(schedule.get(d, (None, None))[0])
                if schedule.get(d, (None, None))[0]
                else None
                for d in weekdays
            ]
        )
        end_seconds = np.array(
            [
                cls.time_to_seconds(schedule.get(d, (None, None))[1])
                if schedule.get(d, (None, None))[1]
                else None
                for d in weekdays
            ]
        )

        valid_mask = ~pd.isna(start_seconds)

        # Replace None with dummy values for comparison
        start_seconds = np.where(valid_mask, start_seconds, 0)
        end_seconds = np.where(valid_mask, end_seconds, 0)

        # Calculate time masks
        normal_mask = (
            (start_seconds <= end_seconds)
            & (times_of_day >= start_seconds)
            & (times_of_day <= end_seconds)
        )
        overnight_mask = (start_seconds > end_seconds) & (
            (times_of_day >= start_seconds) | (times_of_day <= end_seconds)
        )

        final_mask = valid_mask & (normal_mask | overnight_mask)

        return all_times[final_mask]

    @staticmethod
    def generate_monthly_splits(df: pd.DataFrame) -> pd.DataFrame:
        """
        Split long-duration records into monthly chunks for efficient processing.

        This is crucial for handling records that span multiple months, as it allows
        batch processing by month and reduces memory usage.

        Args:
            df: DataFrame with FINI and FFIN datetime columns

        Returns:
            pd.DataFrame: Expanded dataframe where long records are split by month boundaries
        """

        def get_month_starts(row):
            if pd.isna(row["FINI"]) or pd.isna(row["FFIN"]):
                return []
            start_month = row["FINI"].replace(day=1, hour=0, minute=0, second=0)
            end_month = row["FFIN"].replace(day=1, hour=0, minute=0, second=0)
            return pd.date_range(start=start_month, end=end_month, freq="MS").to_list()

        df["month_starts"] = df.progress_apply(get_month_starts, axis=1)
        df_exploded = df.explode("month_starts").reset_index(drop=True)

        # Adjust chunk boundaries
        df_exploded["chunk_start"] = df_exploded["month_starts"]
        df_exploded.loc[
            df_exploded["chunk_start"] < df_exploded["FINI"], "chunk_start"
        ] = df_exploded["FINI"]

        df_exploded["chunk_end"] = df_exploded["month_starts"] + pd.offsets.MonthEnd(1)
        df_exploded["chunk_end"] = df_exploded[["chunk_end", "FFIN"]].min(axis=1)

        df_exploded = df_exploded.drop(columns=["month_starts", "FINI", "FFIN"])
        df_exploded = df_exploded.rename(
            columns={"chunk_start": "FINI", "chunk_end": "FFIN"}
        )

        return df_exploded

    def process_coi_parques(self) -> None:
        """
        Process COI parks data and save to parquet.

        This method:
        1. Loads park polygons
        2. Intersects with buffered road network
        3. Filters by overlap ratio threshold
        4. Maps simple HCIE schedules
        5. Expands to 15-minute timeframes
        6. Aggregates by road and timeframe

        Output:
            Single parquet file: {output_dir}/coi_parques_dataset.parquet
            Columns: road_id, timeframe, coi_parques (binary indicator)
        """
        print("\n" + "=" * 70)
        print("Processing COI Parques (Parks)")
        print("=" * 70)

        # Load and intersect with road network
        coi_parques = gpd.read_file(self.coi_parques_path)
        coi_parques = coi_parques.to_crs("EPSG:4326")

        coi_parques_intersections = gpd.overlay(
            coi_parques,
            self.road_network_buffered[["id_mvi", "mvi_length", "geometry"]],
            how="intersection",
        )

        # Calculate overlap ratios in projected CRS
        coi_parques_intersections = coi_parques_intersections.to_crs(
            self.target_crs
        ).copy()
        coi_parques_intersections["intersection_length"] = (
            coi_parques_intersections.geometry.length
        )
        coi_parques_intersections["overlap_ratio"] = (
            coi_parques_intersections["intersection_length"]
            / coi_parques_intersections["mvi_length"]
        )

        # Filter by overlap threshold
        coi_parques_intersections = coi_parques_intersections[
            coi_parques_intersections["overlap_ratio"] > self.overlap_threshold
        ]
        print(
            f"  Found {len(coi_parques_intersections)} park-road intersections above {self.overlap_threshold} overlap"
        )

        coi_parques_intersections = coi_parques_intersections.to_crs("EPSG:4326").copy()
        coi_parques_intersections["road_id"] = coi_parques_intersections["id_mvi"]
        coi_parques_intersections["coi_parques"] = 1

        # Prepare for temporal expansion
        coi_parques_work = coi_parques_intersections[
            ["road_id", "FINI", "FFIN", "HCIE", "coi_parques"]
        ].copy()

        # Map simple HCIE schedules
        coi_parques_work["hcie_start"], coi_parques_work["hcie_end"] = zip(
            *coi_parques_work["HCIE"].map(self.HCIE_SIMPLE_MAPPING)
        )

        # Generate timeframes
        print("  Generating 15-minute timeframes...")
        coi_parques_work["time_slots"] = coi_parques_work.progress_apply(
            self.generate_simple_timeframes, axis=1
        )

        # Explode and aggregate
        coi_parques_work = coi_parques_work.explode("time_slots").reset_index(drop=True)
        coi_parques_work["timeframe"] = coi_parques_work["time_slots"].dt.strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        coi_parques_dataset = (
            coi_parques_work.groupby(["road_id", "timeframe"])
            .aggregate({"coi_parques": "max"})
            .reset_index()
        )

        # Save
        output_path = self.output_dir / "coi_parques_dataset.parquet"
        coi_parques_dataset.to_parquet(output_path, index=False)
        print(f"  Saved {len(coi_parques_dataset)} records to: {output_path}")

    def process_coi_puntos(self) -> None:
        """
        Process COI points data and save to monthly parquet files.

        This method handles complex schedules with weekday-specific times:
        1. Loads point geometries
        2. Intersects with buffered road network
        3. Splits records into monthly chunks for efficient processing
        4. Parses complex HCIE schedules (weekday + time ranges)
        5. Generates timeframes respecting weekday constraints
        6. Saves monthly batches to separate parquet files

        Output:
            Multiple parquet files: {output_dir}/coi_puntos/coi_puntos_dataset_{start}_{end}.parquet
            Columns: road_id, timeframe, coi_puntos (binary indicator)
        """
        print("\n" + "=" * 70)
        print("Processing COI Puntos (Points)")
        print("=" * 70)

        # Load and intersect with road network
        coi_puntos = gpd.read_file(self.coi_puntos_path)
        coi_puntos = coi_puntos.to_crs("EPSG:4326")
        coi_puntos["id_coi_punto"] = coi_puntos.index

        coi_puntos_intersections = gpd.overlay(
            coi_puntos,
            self.road_network_buffered[["id_mvi", "mvi_length", "geometry"]],
            how="intersection",
        )

        coi_puntos_intersections = coi_puntos_intersections.to_crs("EPSG:4326").copy()
        coi_puntos_intersections["road_id"] = coi_puntos_intersections["id_mvi"]
        coi_puntos_intersections["coi_puntos"] = 1

        print(f"  Found {len(coi_puntos_intersections)} point-road intersections")

        # Prepare for temporal expansion
        coi_puntos_work = coi_puntos_intersections[
            ["id_coi_punto", "road_id", "FINI", "FFIN", "HCIE", "coi_puntos"]
        ].copy()

        # Split into monthly chunks
        print("  Splitting records into monthly chunks...")
        coi_puntos_work = self.generate_monthly_splits(coi_puntos_work)

        # Generate list of months to process
        min_date = coi_puntos_work["FINI"].min().normalize()
        max_date = coi_puntos_work["FFIN"].max().normalize()

        start_date = min_date.replace(day=1)
        end_date = (max_date + pd.DateOffset(months=1)).replace(day=1)

        monthly_dates = pd.date_range(start=start_date, end=end_date, freq="MS")
        monthly_dates_list = [pd.Timestamp(d.date(), tz="UTC") for d in monthly_dates]

        print(f"  Processing {len(monthly_dates_list) - 1} monthly batches...")

        # Process each month
        for i in tqdm(range(len(monthly_dates_list) - 1)):
            date_file = dt.datetime.strftime(monthly_dates_list[i], "%Y%m%d")
            date_file_end = dt.datetime.strftime(monthly_dates_list[i + 1], "%Y%m%d")

            # Filter for current month
            mask = (coi_puntos_work["FFIN"] >= monthly_dates_list[i]) & (
                coi_puntos_work["FFIN"] < monthly_dates_list[i + 1]
            )
            coi_puntos_work_current = coi_puntos_work.loc[mask].copy()

            if coi_puntos_work_current.empty:
                continue

            # Parse complex schedules
            coi_puntos_work_current["parsed_schedule"] = coi_puntos_work_current[
                "HCIE"
            ].apply(self.parse_schedule)

            # Generate timeframes with weekday constraints
            coi_puntos_work_current["time_slots"] = (
                coi_puntos_work_current.progress_apply(
                    self.generate_complex_timeframes, axis=1
                )
            )

            # Explode and aggregate
            coi_puntos_work_current = coi_puntos_work_current.explode(
                "time_slots"
            ).reset_index(drop=True)
            coi_puntos_work_current["timeframe"] = coi_puntos_work_current[
                "time_slots"
            ].dt.strftime("%Y-%m-%d %H:%M:%S")

            coi_puntos_dataset = (
                coi_puntos_work_current.groupby(["road_id", "timeframe"])
                .aggregate({"coi_puntos": "max"})
                .reset_index()
            )

            # Save monthly batch
            output_path = (
                self.output_dir
                / "coi_puntos"
                / f"coi_puntos_dataset_{date_file}_{date_file_end}.parquet"
            )
            coi_puntos_dataset.to_parquet(output_path, index=False)

        print(
            f"  Saved {len(monthly_dates_list) - 1} monthly files to: {self.output_dir / 'coi_puntos'}"
        )

    def process_and_save(self) -> None:
        """
        Execute the complete PMTS data processing pipeline and save results.

        This method orchestrates the entire workflow:
        1. Loads and buffers the road network
        2. Processes COI parks data
        3. Processes COI points data (monthly batches)
        4. Optionally processes COI road segments data (if provided)

        Output files are designed to be consumed by the SparseDatasetGenerator
        in merging.py for integration into the traffic forecasting pipeline.
        """
        print("=" * 70)
        print("Starting PMTS Data Processing Pipeline")
        print("=" * 70)

        # Step 1: Load road network
        print("\n[1/3] Loading road network...")
        self.road_network_buffered = self.load_and_buffer_road_network()

        # Step 2: Process COI Parques
        print("\n[2/3] Processing COI Parques...")
        self.process_coi_parques()

        # Step 3: Process COI Puntos
        print("\n[3/3] Processing COI Puntos...")
        self.process_coi_puntos()

        # Optional: Process COI Tramos (if path provided)
        if self.coi_tramos_path:
            print("\n[OPTIONAL] Processing COI Tramos...")
            # To be implemented based on specific requirements
            print("  COI Tramos processing not yet implemented")

        print("\n" + "=" * 70)
        print("PMTS Data Processing Complete!")
        print("=" * 70)
        print(f"\nAll outputs saved to: {self.output_dir}")


if __name__ == "__main__":
    # Define input data paths
    coi_parques_path = "data/9_pmts/COI_PARQUES_202504071925.geojson"
    coi_puntos_path = "data/9_pmts/COI_PUNTO_202504071914.geojson"
    road_network_path = "data/2_Estructura_via/estructura_via_completa/MVI.geojson"
    output_dir = "data/datos_procesados/pmts_processed/"

    # Initialize processor
    processor = PMTSProcessor(
        coi_parques_path=coi_parques_path,
        coi_puntos_path=coi_puntos_path,
        road_network_path=road_network_path,
        output_dir=output_dir,
        buffer_distance=4.0,
        overlap_threshold=0.15,
        temporal_resolution="15T",
        target_crs="EPSG:3116",
    )

    # Execute processing pipeline
    processor.process_and_save()

    print("\n✓ Processing complete! You can now use these files in merging.py")
