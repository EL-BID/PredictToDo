import datetime as dt
import pickle
from itertools import product
from pathlib import Path
from typing import Dict, List

import geopandas as gpd
import networkx as nx
import pandas as pd
from shapely.geometry import Point
from tqdm.auto import tqdm


class EventsProcessor:
    """
    A class for processing public events data and propagating their impact through the road network.

    This processor handles the complete pipeline from raw events data to spatially and temporally
    distributed event features suitable for training Graph Neural Networks for traffic forecasting.

    The key innovation is network-based impact propagation: events don't just affect the immediate
    road segment, but also nearby roads within a distance determined by event complexity.

    Attributes:
        events_data_path (str): Path to the events CSV file
        road_network_path (str): Path to the road network GeoJSON file
        connectivity_graph_path (str): Path to the pickled NetworkX graph with road connectivity
        output_dir (str): Directory where processed files will be saved
        buffer_distance (float): Buffer distance in decimal degrees for spatial joins (default: 0.0027)
        complexity_distances (Dict[str, float]): Mapping of complexity levels to propagation distances in meters
        temporal_resolution (str): Temporal resolution for time slicing (default: "15min")
    """

    def __init__(
        self,
        events_data_path: str,
        road_network_path: str,
        connectivity_graph_path: str,
        output_dir: str,
        buffer_distance: float = 0.0027,
        complexity_distances: Dict[str, float] | None = None,
        temporal_resolution: str = "15min",
    ):
        """
        Initialize the EventsProcessor with data paths and processing parameters.

        Args:
            events_data_path: Path to the events CSV file with columns:
                - fecha_inicio: Event start date/time
                - fecha_final: Event end date/time
                - Longitude: Event longitude
                - Latutude: Event latitude (note the typo in the original data)
                - Complejidad: Event complexity level (ALTA, MEDIA, BAJA, SIN DEFINIR)
                - tipo_actividad: Type of activity/event
            road_network_path: Path to the road network GeoJSON file (MVI format)
            connectivity_graph_path: Path to pickled NetworkX graph with road connectivity
            output_dir: Directory where processed event data will be saved
            buffer_distance: Buffer distance in decimal degrees for spatial joins (default: 0.0027 ≈ 300m)
            complexity_distances: Custom mapping of complexity levels to propagation distances in meters.
                Default: {"ALTA": 500, "MEDIA": 300, "BAJA": 200, "SIN DEFINIR": 500}
            temporal_resolution: Temporal resolution for time slicing (default: "15min")
        """
        self.events_data_path = events_data_path
        self.road_network_path = road_network_path
        self.connectivity_graph_path = connectivity_graph_path
        self.output_dir = Path(output_dir)
        self.buffer_distance = buffer_distance
        self.temporal_resolution = temporal_resolution

        # Default complexity-to-distance mapping
        if complexity_distances is None:
            self.complexity_distances = {
                "ALTA": 500,
                "MEDIA": 300,
                "BAJA": 200,
                "SIN DEFINIR": 500,
            }
        else:
            self.complexity_distances = complexity_distances

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize tqdm for pandas operations
        tqdm.pandas()

        # Storage for processed data
        self.events_gdf: gpd.GeoDataFrame | None = None
        self.roads_gdf: gpd.GeoDataFrame | None = None
        self.connectivity_graph: nx.Graph | None = None

    @staticmethod
    def parse_date(date_string: str) -> dt.datetime:
        """
        Parse date string with multiple possible formats.

        The input data has inconsistent date formats, so this method tries multiple parsers.

        Args:
            date_string: Date string in format "YYYY-MM-DD HH:MM:SS" or "DD/MM//YYYY"

        Returns:
            dt.datetime: Parsed datetime object
        """
        try:
            return dt.datetime.strptime(date_string, "%Y-%m-%d %H:%M:%S")
        except (ValueError, TypeError):
            try:
                return dt.datetime.strptime(date_string, "%d/%m//%Y")
            except (ValueError, TypeError):
                # If all parsing fails, return a default or raise
                raise ValueError(f"Unable to parse date string: {date_string}")

    @staticmethod
    def quarter_slices(start: pd.Timestamp, end: pd.Timestamp) -> List[pd.Timestamp]:
        """
        Generate time slices aligned to 15-minute intervals between start and end.

        Given two pandas Timestamps (start < end), return a list of timestamps
        representing the start of each 15-minute bucket that overlaps with the event period.
        The 15-minute buckets are aligned to minute marks 00, 15, 30, 45.

        For example:
        - If an event runs from 10:07 to 10:42, the slices would be [10:00, 10:15, 10:30]
        - If an event runs from 10:15 to 10:30, the slice would be [10:15]

        Args:
            start: Event start timestamp
            end: Event end timestamp

        Returns:
            List[pd.Timestamp]: List of 15-minute bucket start times that overlap with the event
        """
        out = []
        current = start

        while current < end:
            # Find "floor" of current to the nearest 15 minutes
            floor_q = current.floor("15T")
            next_bound = floor_q + pd.Timedelta(minutes=15)

            # If current is exactly on a quarter, jump to the next one
            if current == floor_q:
                next_bound = current + pd.Timedelta(minutes=15)

            # End of this chunk is either the next boundary or the original end
            chunk_end = min(next_bound, end)

            out.append(current.floor("15T"))
            current = chunk_end

        return list(set(out))  # Remove duplicates

    @staticmethod
    def get_roads_within_distance(
        graph: nx.Graph, seed_road: str, max_distance_m: float
    ) -> List[str]:
        """
        Find all roads reachable from a seed road within a specified distance.

        This method uses Dijkstra's algorithm to traverse the road network graph,
        using road lengths as edge weights. This allows events to propagate their
        impact through the network based on actual road connectivity rather than
        simple Euclidean distance.

        Args:
            graph: NetworkX graph where nodes represent roads and have 'length_m' attribute
            seed_road: Node ID (road_id) from which to start the search
            max_distance_m: Maximum cumulative distance in meters to propagate

        Returns:
            List[str]: List of road IDs reachable within the distance threshold
        """
        # Ensure each edge has a 'weight' attribute: average of node lengths
        for u, v, data in graph.edges(data=True):
            if "weight" not in data:
                # Use average length of the two roads as edge weight
                data["weight"] = (
                    graph.nodes[u]["length_m"] + graph.nodes[v]["length_m"]
                ) / 2.0

        # Run Dijkstra's algorithm up to the distance cutoff
        try:
            distances = nx.single_source_dijkstra_path_length(
                graph, seed_road, cutoff=max_distance_m, weight="weight"
            )
            reachable = list(distances.keys())
        except nx.NodeNotFound:
            # If the seed road is not in the graph, return empty list
            reachable = []

        return reachable

    def load_events_data(self) -> gpd.GeoDataFrame:
        """
        Load and preprocess events data from CSV.

        This method:
        1. Loads the events CSV file
        2. Filters out events without location data
        3. Creates Point geometries from coordinates
        4. Parses start and end dates
        5. Assigns unique event IDs

        Returns:
            gpd.GeoDataFrame: Processed events with geometry and temporal information
        """
        print("Loading events data...")
        events = pd.read_csv(self.events_data_path, encoding="latin-1")

        # Filter out events without coordinates
        events = events.loc[pd.notnull(events.Longitude)]

        # Create Point geometries
        events["geometry"] = events.apply(  # type: ignore
            lambda row: Point(row["Longitude"], row["Latutude"]),  # type: ignore
            axis=1,  # type: ignore
        )

        # Convert to GeoDataFrame
        events_gdf = gpd.GeoDataFrame(events, crs="EPSG:4326")

        # Assign unique IDs
        events_gdf["id_event"] = events_gdf.index

        # Parse dates with progress bar
        print("  Parsing dates...")
        events_gdf["fecha_inicio"] = events_gdf["fecha_inicio"].progress_apply(
            self.parse_date
        )
        events_gdf["fecha_final"] = events_gdf["fecha_final"].progress_apply(
            self.parse_date
        )

        print(f"  Loaded {len(events_gdf)} events")
        return events_gdf

    def load_road_network(self) -> gpd.GeoDataFrame:
        """
        Load and process the road network from GeoJSON.

        This method:
        1. Loads the road network geometry
        2. Creates unique road IDs from MVI codes
        3. Dissolves split segments into unified roads
        4. Creates centroids for distance calculations
        5. Creates buffers for spatial joins with events

        Returns:
            gpd.GeoDataFrame: Processed road network with buffers and centroids
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
            roads_gdf["id_mvi"].astype(str) + "." + roads_gdf["contador"].astype(str)
        )

        # Dissolve split segments
        roads_gdf = (
            roads_gdf.explode(ignore_index=True)  # Explode MultiLineStrings
            .dissolve(by="id_mvi")  # Merge all segments of same road
            .reset_index()
        )

        # Create road_id and spatial features
        roads_gdf["road_id"] = roads_gdf["id_mvi"]
        roads_gdf["centroid"] = roads_gdf.geometry.centroid
        roads_gdf["buffer"] = roads_gdf.geometry.buffer(self.buffer_distance)

        # Return only necessary columns with buffer geometry
        roads_buf = roads_gdf.set_geometry("buffer")[["road_id", "buffer", "centroid"]]

        print(f"  Loaded {len(roads_buf)} road segments")
        return roads_buf

    def load_connectivity_graph(self) -> nx.Graph:
        """
        Load the road connectivity graph from pickle file.

        This graph represents the topological connectivity of the road network,
        where nodes are roads and edges connect adjacent roads. It's used for
        propagating event impacts through the network.

        Returns:
            nx.Graph: NetworkX graph with road connectivity
        """
        print("Loading connectivity graph...")
        with open(self.connectivity_graph_path, "rb") as f:
            graph = pickle.load(f)

        print(
            f"  Loaded graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges"
        )
        return graph

    def assign_events_to_roads(
        self, events_gdf: gpd.GeoDataFrame, roads_gdf: gpd.GeoDataFrame
    ) -> gpd.GeoDataFrame:
        """
        Assign events to the nearest road segment using spatial join and distance ranking.

        This method:
        1. Performs spatial join between events and buffered roads
        2. Calculates distance from event to road centroid
        3. Ranks roads by distance for each event
        4. Selects the closest road for each event

        Args:
            events_gdf: GeoDataFrame of events with Point geometries
            roads_gdf: GeoDataFrame of roads with buffer geometries and centroids

        Returns:
            gpd.GeoDataFrame: Events assigned to their nearest road segment
        """
        print("Assigning events to road segments...")

        # Spatial join: find all roads that intersect with each event's location
        joined = gpd.sjoin(events_gdf, roads_gdf, how="left", predicate="intersects")

        # Drop duplicate index column
        joined = joined.drop(columns=["index_right"], errors="ignore")

        # Ensure geometries are Point objects
        joined["geometry"] = joined["geometry"].apply(  # type: ignore
            lambda x: Point(x) if not isinstance(x, Point) else x  # type: ignore
        )
        joined["centroid"] = joined["centroid"].apply(  # type: ignore
            lambda x: Point(x) if not isinstance(x, Point) else x  # type: ignore
        )

        # Calculate distance from event to road centroid
        joined["distance"] = joined.apply(
            lambda row: row["geometry"].distance(row["centroid"]), axis=1
        )

        # Rank roads by distance for each event
        joined["rank_distance"] = joined.groupby("id_event")["distance"].rank(
            method="dense"
        )

        # Keep only the closest road for each event
        joined = joined.loc[joined.rank_distance == 1].reset_index(drop=True)

        print(f"  Assigned {len(joined)} events to road segments")
        return joined

    def propagate_events_through_network(
        self, events_gdf: gpd.GeoDataFrame, graph: nx.Graph
    ) -> gpd.GeoDataFrame:
        """
        Propagate events through the road network based on complexity levels.

        This method:
        1. Maps complexity levels to propagation distances
        2. For each event, finds all reachable roads within the distance threshold
        3. Expands events temporally across 15-minute intervals
        4. Creates (road, timeframe) pairs for each event

        Args:
            events_gdf: GeoDataFrame of events assigned to road segments
            graph: NetworkX graph of road connectivity

        Returns:
            gpd.GeoDataFrame: Events with reachable roads and timeframes
        """
        print("Propagating events through road network...")

        # Map complexity to propagation distance
        print("  Determining propagation distances...")
        events_gdf["hop_distance"] = events_gdf["Complejidad"].progress_apply(
            lambda x: self.complexity_distances.get(x, 500)
        )

        # Find reachable roads for each event
        print("  Finding reachable roads...")
        events_gdf["reachable_nodes"] = events_gdf.progress_apply(  # type: ignore
            lambda row: self.get_roads_within_distance(
                graph, row["road_id"], row["hop_distance"]
            )
            if pd.notnull(row["road_id"])
            else [],
            axis=1,
        )

        # Drop the original road_id since we now have reachable_nodes
        events_gdf = events_gdf.drop(columns=["road_id"])

        # Generate time slices for each event
        print("  Generating time slices...")
        events_gdf["timeframe_set"] = events_gdf.progress_apply(  # type: ignore
            lambda row: self.quarter_slices(row["fecha_inicio"], row["fecha_final"]),
            axis=1,
        )

        # Create (road, timeframe) pairs
        print("  Creating road-timeframe pairs...")
        events_gdf["reachable_pairs"] = events_gdf.progress_apply(  # type: ignore
            lambda row: list(product(row["reachable_nodes"], row["timeframe_set"])),
            axis=1,
        )

        print(
            f"  Generated {events_gdf['reachable_pairs'].apply(len).sum()} road-timeframe pairs"
        )
        return events_gdf

    def explode_and_aggregate(self, events_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
        """
        Explode events by (road, timeframe) pairs and aggregate by event type.

        This method:
        1. Explodes each event into multiple rows (one per road-timeframe pair)
        2. Aggregates by road_id, timeframe, and event type
        3. Creates a pivot table with event types as columns
        4. Produces the final dataset ready for model training

        Args:
            events_gdf: GeoDataFrame with reachable_pairs for each event

        Returns:
            pd.DataFrame: Aggregated events dataset indexed by (road_id, timeframe)
                with columns for each event type
        """
        print("Exploding and aggregating events...")

        # Explode reachable_pairs into separate rows
        exploded_df = events_gdf.explode("reachable_pairs").reset_index(drop=True)

        # Split pairs into road_id and timeframe columns
        exploded_df[["road_id", "timeframe"]] = pd.DataFrame(
            exploded_df["reachable_pairs"].tolist(), index=exploded_df.index
        )

        # Drop unnecessary columns
        exploded_df = exploded_df.drop(
            columns=[
                "reachable_pairs",
                "reachable_nodes",
                "timeframe_set",
                "centroid",
                "geometry",
            ]
        )

        # Create count column
        exploded_df["total"] = 1

        # Aggregate by road, timeframe, and event type
        print("  Aggregating by event type...")
        events_dataset = (
            exploded_df.groupby(["road_id", "timeframe", "tipo_actividad"])
            .aggregate({"total": "sum"})
            .reset_index()
            .pivot_table(
                index=["road_id", "timeframe"],
                columns=["tipo_actividad"],
                values="total",
            )
            .reset_index()
            .fillna(0)
        )

        print(f"  Final dataset: {len(events_dataset)} road-timeframe records")
        print(
            f"  Event types: {[col for col in events_dataset.columns if col not in ['road_id', 'timeframe']]}"
        )
        return events_dataset

    def process_and_save(self) -> None:
        """
        Execute the complete events data processing pipeline and save results.

        This method orchestrates the entire workflow:
        1. Loads events data from CSV
        2. Loads road network from GeoJSON
        3. Loads connectivity graph from pickle
        4. Assigns events to nearest road segments
        5. Propagates events through the network based on complexity
        6. Expands events temporally across 15-minute intervals
        7. Aggregates by event type
        8. Saves the final dataset to CSV

        Output file:
            - {output_dir}/events_dataset.csv: Processed events dataset with columns:
                - road_id: Road segment identifier
                - timeframe: 15-minute time bucket
                - {event_type}: Count of events of each type (one column per type)

        This file is designed to be consumed by the SparseDatasetGenerator
        in merging.py for integration into the traffic forecasting pipeline.
        """
        print("=" * 70)
        print("Starting Events Data Processing Pipeline")
        print("=" * 70)

        # Step 1: Load events data
        print("\n[1/7] Loading events data...")
        self.events_gdf = self.load_events_data()

        # Step 2: Load road network
        print("\n[2/7] Loading road network...")
        self.roads_gdf = self.load_road_network()

        # Step 3: Load connectivity graph
        print("\n[3/7] Loading connectivity graph...")
        self.connectivity_graph = self.load_connectivity_graph()

        # Step 4: Assign events to roads
        print("\n[4/7] Assigning events to roads...")
        assigned_events = self.assign_events_to_roads(self.events_gdf, self.roads_gdf)

        # Step 5: Propagate through network
        print("\n[5/7] Propagating events through network...")
        propagated_events = self.propagate_events_through_network(
            assigned_events, self.connectivity_graph
        )

        # Step 6: Explode and aggregate
        print("\n[6/7] Exploding and aggregating...")
        events_dataset = self.explode_and_aggregate(propagated_events)

        # Step 7: Save results
        print("\n[7/7] Saving processed dataset...")
        output_path = self.output_dir / "events_dataset.csv"
        events_dataset.to_csv(output_path, index=False)
        print(f"  Saved events dataset to: {output_path}")

        print("\n" + "=" * 70)
        print("Events Data Processing Complete!")
        print("=" * 70)
        print(f"\nOutput saved to: {output_path}")


if __name__ == "__main__":
    # Define input data paths
    events_data_path = "data/8_eventos/events_dataset_preprocessing.csv"
    road_network_path = "data/2_Estructura_via/estructura_via_completa/MVI.geojson"
    connectivity_graph_path = (
        "data/datos_procesados/conectivity_graph/road_graph.pickle"
    )
    output_dir = "data/datos_procesados/events/"

    # Initialize processor with custom complexity distances
    processor = EventsProcessor(
        events_data_path=events_data_path,
        road_network_path=road_network_path,
        connectivity_graph_path=connectivity_graph_path,
        output_dir=output_dir,
        buffer_distance=0.0027,  # Approximately 300 meters in degrees
        complexity_distances={
            "ALTA": 500,  # High complexity: 500m radius
            "MEDIA": 300,  # Medium complexity: 300m radius
            "BAJA": 200,  # Low complexity: 200m radius
            "SIN DEFINIR": 500,  # Undefined: 500m radius (default)
        },
        temporal_resolution="15min",
    )

    # Execute processing pipeline
    processor.process_and_save()

    print("\nProcessing complete!")
