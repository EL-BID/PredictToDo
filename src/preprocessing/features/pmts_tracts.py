import datetime
import datetime as dt
import re
from pathlib import Path
from typing import Dict, List, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

# Enable progress bars for pandas operations
tqdm.pandas()


class PMTSTractsProcessor:
    """
    A class for processing PMTS (Traffic Management Plan) data and assigning interventions to road segments.

    This processor handles the complete pipeline from raw PMTS GeoJSON files to temporally-expanded
    datasets suitable for training Graph Neural Networks for traffic forecasting.

    Attributes:
        coi_tramo_path (str): Path to COI Tramo (tract interventions) GeoJSON file
        coos_punto_path (str): Path to COOS Punto (point interventions) GeoJSON file
        road_network_path (str): Path to the road network GeoJSON file
        output_dir (str): Directory where processed files will be saved
        buffer_distance (float): Buffer distance in meters for spatial joins (default: 4)
        overlap_threshold (float): Minimum overlap ratio to consider intersection valid (default: 0.15)
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

    def __init__(
        self,
        coi_tramo_path: str,
        coos_punto_path: str,
        road_network_path: str,
        output_dir: str,
        buffer_distance: float = 4.0,
        overlap_threshold: float = 0.15,
        target_crs: str = "EPSG:3116",
    ):
        """
        Initialize the PMTSProcessor with data paths and processing parameters.

        Args:
            coi_tramo_path: Path to COI Tramo GeoJSON file (tract-based interventions)
            coos_punto_path: Path to COOS Punto GeoJSON file (point-based interventions)
            road_network_path: Path to the road network GeoJSON file (MVI format)
            output_dir: Directory where processed PMTS data will be saved
            buffer_distance: Buffer distance in meters around road segments for spatial joins (default: 4)
            overlap_threshold: Minimum ratio of overlap to consider valid intersection (default: 0.15)
            target_crs: EPSG code for projected coordinate system used in spatial operations (default: "EPSG:3116")
        """
        self.coi_tramo_path = coi_tramo_path
        self.coos_punto_path = coos_punto_path
        self.road_network_path = road_network_path
        self.output_dir = Path(output_dir)
        self.buffer_distance = buffer_distance
        self.overlap_threshold = overlap_threshold
        self.target_crs = target_crs

        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "coi_tramos").mkdir(exist_ok=True)
        (self.output_dir / "coi_puntos").mkdir(exist_ok=True)

        # Storage for processed data
        self.mvi_buffered: gpd.GeoDataFrame | None = None
        self.coi_tramo_gdf: gpd.GeoDataFrame | None = None
        self.coos_punto_gdf: gpd.GeoDataFrame | None = None

    # ==================== Road Network Processing ====================

    def load_and_prepare_road_network(self) -> gpd.GeoDataFrame:
        """
        Load and process the road network for spatial joins with PMTS data.

        This method:
        1. Loads the MVI road network geometry
        2. Converts to WGS84 (EPSG:4326) for consistency
        3. Creates unique road IDs from MVI codes
        4. Transforms to projected CRS for accurate spatial operations
        5. Creates buffers around road segments for spatial joins

        Returns:
            gpd.GeoDataFrame: Processed and buffered road network with columns:
                - id_mvi: Unique identifier for each road segment
                - geometry: Buffered polygon geometry in EPSG:4326
                - mvi_length: Length of the road segment in meters
        """
        print("Loading road network...")
        mvi = gpd.read_file(self.road_network_path)
        mvi = mvi.to_crs("EPSG:4326")
        mvi.columns = mvi.columns.str.lower()

        # Create unique road IDs
        mvi["id_mvi"] = (
            mvi["mviciv"].astype(str) + "_" + mvi["mvisvia"].str.replace(" ", "")
        )
        mvi["contador"] = mvi.groupby("id_mvi").cumcount() + 1
        mvi["id_mvi"] = mvi["id_mvi"] + "." + mvi["contador"].astype(str)

        # Transform to projected CRS and create buffers
        mvi_buffered = mvi.to_crs(self.target_crs).copy()
        mvi_buffered["mvi_length"] = mvi_buffered.geometry.length
        mvi_buffered["geometry"] = mvi_buffered.geometry.buffer(
            self.buffer_distance, cap_style="square"
        )

        # Convert back to WGS84 for consistency
        mvi_buffered = mvi_buffered.to_crs("EPSG:4326")

        print(f"  Loaded and buffered {len(mvi_buffered)} road segments")
        return mvi_buffered[["id_mvi", "geometry", "mvi_length"]]

    # ==================== Schedule Parsing ====================

    @staticmethod
    def normalize_hcie_text(text: str) -> str:
        """
        Normalize and clean HCIE (schedule) text for parsing.

        This function standardizes various schedule format variations found in PMTS data:
        - Removes extra whitespace
        - Expands common abbreviations (L-V, SAB, DOM, etc.)
        - Removes accents from Spanish text
        - Removes parenthetical notes

        Args:
            text: Raw HCIE schedule string

        Returns:
            str: Normalized schedule text ready for parsing
        """
        text = str(text).upper()
        text = re.sub(r"\s+", " ", text)

        # Expand common abbreviations
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
        Extract weekday ranges from normalized schedule text.

        Handles patterns like:
        - "LUNES A VIERNES" (range)
        - "SABADO" (single day)
        - Overnight ranges (e.g., "VIERNES A LUNES")

        Args:
            text: Normalized HCIE schedule text

        Returns:
            set: Set of weekday integers (0=Monday, 6=Sunday)
        """
        weekdays = set()

        # Find weekday ranges (e.g., "LUNES A VIERNES")
        for match in re.finditer(
            r"(LUNES|MARTES|MIERCOLES|JUEVES|VIERNES|SABADO|DOMINGO)\s+A\s+(LUNES|MARTES|MIERCOLES|JUEVES|VIERNES|SABADO|DOMINGO)",
            text,
        ):
            start, end = (
                cls.WEEKDAY_MAP[match.group(1)],
                cls.WEEKDAY_MAP[match.group(2)],
            )
            if start <= end:
                weekdays.update(range(start, end + 1))
            else:
                # Handle overnight ranges
                weekdays.update(list(range(start, 7)) + list(range(0, end + 1)))

        # Find individual weekdays
        for word in cls.WEEKDAY_MAP.keys():
            if word in text:
                weekdays.add(cls.WEEKDAY_MAP[word])

        # Default to all days if no specific days found
        if not weekdays:
            weekdays.update(range(7))

        return weekdays

    @staticmethod
    def extract_timeframes(text: str) -> List[Tuple[datetime.time, datetime.time]]:
        """
        Extract time ranges from schedule text.

        Handles patterns like:
        - "06:00-18:00"
        - "6:00 A 18:00"
        - "06.00-18.00" (dots instead of colons)

        Args:
            text: Schedule text containing time ranges

        Returns:
            List[Tuple[datetime.time, datetime.time]]: List of (start_time, end_time) tuples
        """
        timeframes = []

        for match in re.finditer(
            r"(\d{1,2}[:.]\d{2})\s*[-–A]\s*(\d{1,2}[:.]\d{2})", text
        ):
            start_str = match.group(1).replace(".", ":")
            end_str = match.group(2).replace(".", ":")

            # Handle 24:00 edge case
            if start_str == "24:00":
                start_str = "23:59"
            if end_str == "24:00":
                end_str = "23:59"

            start = datetime.datetime.strptime(start_str, "%H:%M").time()
            end = datetime.datetime.strptime(end_str, "%H:%M").time()
            timeframes.append((start, end))

        return timeframes

    @classmethod
    def parse_schedule_row(
        cls, row: str
    ) -> Dict[int, Tuple[datetime.time, datetime.time]]:
        """
        Parse a complete HCIE schedule string into a structured schedule dictionary.

        This is the main schedule parsing function that combines normalization,
        weekday extraction, and timeframe extraction to create a complete schedule.

        Handles complex patterns like:
        - "LUNES A VIERNES 06:00-20:00, SABADO 08:00-14:00"
        - "24 HORAS"
        - Multiple time blocks per day

        Args:
            row: Raw HCIE schedule string

        Returns:
            Dict[int, Tuple[datetime.time, datetime.time]]: Dictionary mapping weekday integers
                to (start_time, end_time) tuples. Keys are 0-6 (Monday-Sunday).
        """
        text = cls.normalize_hcie_text(row)

        # Handle 24-hour schedules
        if "24 HORAS" in text:
            return {
                day: (datetime.time(0, 0), datetime.time(23, 59)) for day in range(7)
            }

        schedules = {}

        # Split by weekday names to identify blocks
        blocks = re.split(
            r"(LUNES|MARTES|MIERCOLES|JUEVES|VIERNES|SABADO|DOMINGO)", text
        )

        if len(blocks) > 1:
            # Reconstruct blocks by pairing weekday names with their time info
            reconstructed = []
            i = 0
            while i < len(blocks):
                if blocks[i] in cls.WEEKDAY_MAP:
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

    # ==================== Temporal Expansion ====================

    @staticmethod
    def time_to_seconds(t: datetime.time) -> int:
        """
        Convert a time object to seconds since midnight.

        Args:
            t: Time object

        Returns:
            int: Number of seconds since midnight
        """
        return t.hour * 3600 + t.minute * 60 + t.second

    @classmethod
    def generate_timeframes_vectorized_per_row(cls, row: pd.Series) -> pd.DatetimeIndex:
        """
        Generate all valid 15-minute timeframes for a PMTS intervention based on its schedule.

        This vectorized function efficiently expands date ranges into 15-minute intervals
        while respecting the parsed schedule constraints (weekdays and time ranges).

        Handles:
        - Normal time ranges (e.g., 06:00-18:00)
        - Overnight ranges (e.g., 22:00-06:00)
        - Different schedules for different weekdays

        Args:
            row: DataFrame row containing:
                - FINI: Start date of intervention
                - FFIN: End date of intervention
                - parsed_schedule: Dict mapping weekdays to (start_time, end_time)

        Returns:
            pd.DatetimeIndex: All valid 15-minute timestamps for this intervention
        """
        if pd.isna(row["FINI"]) or pd.isna(row["FFIN"]):
            return pd.DatetimeIndex([])

        # Generate all 15-minute timestamps for this date range
        all_times = pd.date_range(start=row["FINI"], end=row["FFIN"], freq="15min")
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

        # Check for normal and overnight time ranges
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

    @classmethod
    def generate_timeframes_vectorized_fully(cls, row: pd.Series) -> pd.DatetimeIndex:
        """
        Alternative implementation of timeframe generation using numpy vectorization.

        This version is used for COOS Punto data and uses a slightly different
        vectorization approach for better performance with point-based data.

        Args:
            row: DataFrame row containing FINI, FFIN, and parsed_schedule

        Returns:
            pd.DatetimeIndex: All valid 15-minute timestamps for this intervention
        """
        if pd.isna(row["FINI"]) or pd.isna(row["FFIN"]):
            return pd.DatetimeIndex([])

        all_times = pd.date_range(start=row["FINI"], end=row["FFIN"], freq="15min")
        if all_times.empty:
            return pd.DatetimeIndex([])

        weekdays = all_times.weekday
        times_of_day = np.array(all_times.time)

        # Use numpy vectorize to retrieve schedules
        def get_schedule(day):
            return row["parsed_schedule"].get(day, (None, None))

        vfunc = np.vectorize(get_schedule)
        hcie_start, hcie_end = list(vfunc(weekdays))

        hcie_start = np.array(hcie_start)
        hcie_end = np.array(hcie_end)

        # Build valid_mask
        valid_mask = hcie_start != None

        # Convert None to dummy time for comparison
        dummy_time = datetime.time(0, 0)
        hcie_start = np.array([t if t is not None else dummy_time for t in hcie_start])
        hcie_end = np.array([t if t is not None else dummy_time for t in hcie_end])

        # Calculate masks
        normal_mask = (
            (hcie_start <= hcie_end)
            & (times_of_day >= hcie_start)
            & (times_of_day <= hcie_end)
        )
        overnight_mask = (hcie_start > hcie_end) & (
            (times_of_day >= hcie_start) | (times_of_day <= hcie_end)
        )

        final_mask = (normal_mask | overnight_mask) & valid_mask

        return all_times[final_mask]

    # ==================== Monthly Splitting ====================

    @staticmethod
    def generate_monthly_splits(df: pd.DataFrame) -> pd.DataFrame:
        """
        Split PMTS interventions into monthly chunks for efficient processing.

        This is crucial for handling large datasets: instead of processing all timeframes
        at once, we split by month and process incrementally to manage memory usage.

        Args:
            df: DataFrame with FINI and FFIN date columns

        Returns:
            pd.DataFrame: Exploded dataframe where each row represents one month chunk
                of the original intervention
        """
        print("  Generating monthly splits...")

        # Step 1: Build list of month start dates per row
        def get_month_starts(row):
            start_month = row["FINI"].replace(day=1, hour=0, minute=0, second=0)
            end_month = row["FFIN"].replace(day=1, hour=0, minute=0, second=0)
            return pd.date_range(start=start_month, end=end_month, freq="MS").to_list()

        df["month_starts"] = df.progress_apply(get_month_starts, axis=1)  # type: ignore

        # Step 2: Explode to create one row per month
        df_exploded = df.explode("month_starts").reset_index(drop=True)

        # Step 3: Calculate chunk_start (max of month start and original FINI)
        df_exploded["chunk_start"] = df_exploded["month_starts"]
        df_exploded.loc[
            df_exploded["chunk_start"] < df_exploded["FINI"], "chunk_start"
        ] = df_exploded["FINI"]

        # Step 4: Calculate chunk_end (min of month end and original FFIN)
        df_exploded["chunk_end"] = df_exploded["month_starts"] + pd.offsets.MonthEnd(1)  # type: ignore
        df_exploded["chunk_end"] = df_exploded[["chunk_end", "FFIN"]].min(axis=1)

        # Clean up and rename
        df_exploded = df_exploded.drop(columns=["month_starts", "FINI", "FFIN"])
        df_exploded = df_exploded.rename(
            {"chunk_start": "FINI", "chunk_end": "FFIN"}, axis=1
        )

        return df_exploded

    # ==================== COI Tramo Processing ====================

    def process_coi_tramo(self) -> None:
        """
        Process COI Tramo (tract-based interventions) data.

        This method:
        1. Loads COI Tramo GeoJSON data
        2. Performs spatial intersection with buffered road network
        3. Filters intersections by overlap ratio
        4. Splits data into monthly chunks
        5. Expands each chunk into 15-minute timeframes
        6. Aggregates by road and timeframe
        7. Saves monthly parquet files

        Output files are saved to: {output_dir}/coi_tramos/coi_tramo_dataset_{start}_{end}.parquet
        """
        print("\n" + "=" * 70)
        print("Processing COI Tramo (Tract Interventions)")
        print("=" * 70)

        # Load COI Tramo data
        print("\n[1/6] Loading COI Tramo data...")
        coi_tramo = gpd.read_file(self.coi_tramo_path)
        coi_tramo = coi_tramo.to_crs("EPSG:4326")
        coi_tramo["id_coi_tramo"] = coi_tramo.index
        print(f"  Loaded {len(coi_tramo)} COI Tramo interventions")

        # Spatial intersection with road network
        print("\n[2/6] Performing spatial intersection with road network...")
        assert self.mvi_buffered is not None, (
            "Road network must be loaded and buffered before processing COI Tramo"
        )
        coi_tramo_intersections = gpd.overlay(
            coi_tramo,
            self.mvi_buffered[["id_mvi", "mvi_length", "geometry"]],
            how="intersection",
        )

        # Calculate overlap ratios
        coi_tramo_intersections = coi_tramo_intersections.to_crs(self.target_crs).copy()
        coi_tramo_intersections["intersection_length"] = (
            coi_tramo_intersections.geometry.length
        )
        coi_tramo_intersections["overlap_ratio"] = (
            coi_tramo_intersections["intersection_length"]
            / coi_tramo_intersections["mvi_length"]
        )

        # Filter by overlap threshold
        coi_tramo_intersections = coi_tramo_intersections[
            coi_tramo_intersections["overlap_ratio"] > self.overlap_threshold
        ]
        coi_tramo_intersections = coi_tramo_intersections.to_crs("EPSG:4326").copy()

        print(
            f"  Found {len(coi_tramo_intersections)} valid road-intervention intersections"
        )

        # Prepare working dataframe
        coi_tramo_intersections["road_id"] = coi_tramo_intersections["id_mvi"]
        coi_tramo_intersections["coi_tramo"] = 1
        coi_tramo_work = coi_tramo_intersections[
            ["id_coi_tramo", "road_id", "FINI", "FFIN", "HCIE", "coi_tramo"]
        ].copy()

        # Generate monthly splits
        print("\n[3/6] Splitting into monthly chunks...")
        coi_tramo_work = self.generate_monthly_splits(coi_tramo_work)

        # Calculate date range for monthly processing
        print("\n[4/6] Calculating date range...")
        min_date = coi_tramo_work["FINI"].min().normalize()
        max_date = coi_tramo_work["FFIN"].max().normalize()

        # Add 1 month buffer on each side
        start_date = min_date.replace(day=1)
        end_date = (max_date + pd.DateOffset(months=1)).replace(day=1)

        # Generate full list of months
        monthly_dates = pd.date_range(start=start_date, end=end_date, freq="MS")
        monthly_dates_list = [pd.Timestamp(d) for d in monthly_dates]

        print(f"  Processing {len(monthly_dates_list) - 1} monthly chunks")

        # Process each month
        print("\n[5/6] Processing monthly chunks and expanding timeframes...")
        for i in tqdm(range(len(monthly_dates_list) - 1)):
            date_file = dt.datetime.strftime(monthly_dates_list[i], "%Y%m%d")
            date_file_end = dt.datetime.strftime(monthly_dates_list[i + 1], "%Y%m%d")

            # Filter data for current month
            mask = (coi_tramo_work["FFIN"] >= monthly_dates_list[i]) & (
                coi_tramo_work["FFIN"] < monthly_dates_list[i + 1]
            )
            coi_tramo_work_current = coi_tramo_work.loc[mask].copy()

            if len(coi_tramo_work_current) == 0:
                continue

            # Parse schedules and expand timeframes
            coi_tramo_work_current["parsed_schedule"] = coi_tramo_work_current[
                "HCIE"
            ].apply(self.parse_schedule_row)
            coi_tramo_work_current["time_slots"] = (
                coi_tramo_work_current.progress_apply(
                    self.generate_timeframes_vectorized_per_row, axis=1
                )
            )  # type: ignore

            # Explode time slots
            coi_tramo_work_current = coi_tramo_work_current.explode(
                "time_slots"
            ).reset_index(drop=True)
            coi_tramo_work_current["timeframe"] = coi_tramo_work_current[
                "time_slots"
            ].dt.strftime("%Y-%m-%d %H:%M:%S")

            # Aggregate by road and timeframe
            coi_tramo_dataset = (
                coi_tramo_work_current.groupby(["road_id", "timeframe"])
                .aggregate({"coi_tramo": "max"})
                .reset_index()
            )

            # Save to parquet
            output_path = (
                self.output_dir
                / f"coi_tramos/coi_tramo_dataset_{date_file}_{date_file_end}.parquet"
            )
            coi_tramo_dataset.to_parquet(output_path, index=False)

        print(f"\n[6/6] Saved COI Tramo datasets to: {self.output_dir / 'coi_tramos'}")

    # ==================== COOS Punto Processing ====================

    def process_coos_punto(self) -> None:
        """
        Process COOS Punto (point-based interventions) data.

        This method:
        1. Loads COOS Punto GeoJSON data
        2. Performs spatial intersection with buffered road network
        3. Parses schedules and expands to 15-minute timeframes
        4. Aggregates by road and timeframe
        5. Saves single parquet file

        Output file is saved to: {output_dir}/coi_puntos/coos_punto_dataset.parquet

        Note: Unlike COI Tramo, COOS Punto is processed as a single batch since
        point-based interventions typically have smaller data volumes.
        """
        print("\n" + "=" * 70)
        print("Processing COOS Punto (Point Interventions)")
        print("=" * 70)

        # Load COOS Punto data
        print("\n[1/5] Loading COOS Punto data...")
        coos_punto = gpd.read_file(self.coos_punto_path)
        coos_punto = coos_punto.to_crs("EPSG:4326")
        coos_punto["id_coos_puntos"] = coos_punto.index
        print(f"  Loaded {len(coos_punto)} COOS Punto interventions")

        # Spatial intersection with road network
        print("\n[2/5] Performing spatial intersection with road network...")
        assert self.mvi_buffered is not None, (
            "Road network must be loaded and buffered before processing COOS Punto"
        )
        coos_punto_intersections = gpd.overlay(
            coos_punto,
            self.mvi_buffered[["id_mvi", "mvi_length", "geometry"]],
            how="intersection",
        )
        coos_punto_intersections = coos_punto_intersections.to_crs("EPSG:4326").copy()
        coos_punto_intersections["road_id"] = coos_punto_intersections["id_mvi"]
        coos_punto_intersections["coos_punto"] = 1

        print(
            f"  Found {len(coos_punto_intersections)} valid road-intervention intersections"
        )

        # Prepare working dataframe
        coos_punto_work = coos_punto_intersections[
            ["id_coos_puntos", "road_id", "FINI", "FFIN", "HCIE", "coos_punto"]
        ].copy()

        # Parse schedules and expand timeframes
        print("\n[3/5] Parsing schedules and expanding timeframes...")
        coos_punto_work["parsed_schedule"] = coos_punto_work["HCIE"].apply(
            self.parse_schedule_row
        )
        coos_punto_work["time_slots"] = coos_punto_work.progress_apply(
            self.generate_timeframes_vectorized_fully, axis=1
        )  # type: ignore

        # Explode time slots
        print("\n[4/5] Exploding time slots...")
        coos_punto_work = coos_punto_work.explode("time_slots").reset_index(drop=True)
        coos_punto_work["timeframe"] = coos_punto_work["time_slots"].dt.strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        # Aggregate by road and timeframe
        coos_punto_dataset = (
            coos_punto_work.groupby(["road_id", "timeframe"])
            .aggregate({"coos_punto": "max"})
            .reset_index()
        )

        # Save to parquet
        output_path = self.output_dir / "coi_puntos/coos_punto_dataset.parquet"
        coos_punto_dataset.to_parquet(output_path, index=False)

        print(f"\n[5/5] Saved COOS Punto dataset to: {output_path}")

    # ==================== Main Processing Pipeline ====================

    def process_and_save(self) -> None:
        """
        Execute the complete PMTS data processing pipeline and save results.

        This method orchestrates the entire workflow:
        1. Loads and prepares the road network
        2. Processes COI Tramo (tract interventions)
        3. Processes COOS Punto (point interventions)

        Output files:
            - {output_dir}/coi_tramos/coi_tramo_dataset_{start}_{end}.parquet: Monthly COI Tramo datasets
            - {output_dir}/coi_puntos/coos_punto_dataset.parquet: Complete COOS Punto dataset

        These files are designed to be consumed by the SparseDatasetGenerator
        in merging.py for integration into the traffic forecasting pipeline.
        """
        print("=" * 70)
        print("Starting PMTS Data Processing Pipeline")
        print("=" * 70)

        # Step 1: Load road network
        self.mvi_buffered = self.load_and_prepare_road_network()

        # Step 2: Process COI Tramo
        self.process_coi_tramo()

        # Step 3: Process COOS Punto
        self.process_coos_punto()

        print("\n" + "=" * 70)
        print("PMTS Data Processing Complete!")
        print("=" * 70)
        print(f"\nAll outputs saved to: {self.output_dir}")


if __name__ == "__main__":
    # Define input data paths
    coi_tramo_path = "data/9_pmts/COI_TRAMO_202504071919.geojson"
    coos_punto_path = "data/9_pmts/COOS_PUNTO_202504071922.geojson"
    road_network_path = "data/2_Estructura_via/estructura_via_completa/MVI.geojson"
    output_dir = "data/datos_procesados/pmts_processed/"

    # Initialize processor
    processor = PMTSTractsProcessor(
        coi_tramo_path=coi_tramo_path,
        coos_punto_path=coos_punto_path,
        road_network_path=road_network_path,
        output_dir=output_dir,
        buffer_distance=4.0,
        overlap_threshold=0.15,
        target_crs="EPSG:3116",
    )

    # Execute processing pipeline
    processor.process_and_save()

    print("\nProcessing complete!")
