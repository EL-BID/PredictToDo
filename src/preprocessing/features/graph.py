import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from networkx.readwrite import json_graph as nx_json_graph
from shapely.geometry import Point
from tqdm.auto import tqdm


class RoadGraphConstructor:
    """
    A class for constructing directed road network graphs with rich node and edge attributes.

    This constructor builds a spatial graph where:
    - Nodes represent road segments with static features (length, land use, population, buildings, bus stops)
    - Edges represent intersections with directionality rules and influence weights
    - Legal edges (influence=1.0) follow traffic flow rules
    - Illegal edges (influence=0.3) represent spatial proximity but not legal traffic flow

    Attributes:
        road_network_path (str): Path to MVI road network GeoJSON file
        buildings_path (str): Path to buildings per road CSV file
        bus_stops_path (str): Path to bus stops per road parquet file
        zat_geometry_path (str): Path to ZAT geometry GDB file
        zat_population_path (str): Path to ZAT population and land use CSV file
        output_dir (str): Directory where graph files will be saved
        target_crs (str): Target coordinate reference system (default: "EPSG:3116")
        spatial_tolerance (float): Tolerance in meters for spatial joins (default: 1.0)
        legal_influence_weight (float): Edge weight for legal connections (default: 1.0)
        illegal_influence_weight (float): Edge weight for illegal connections (default: 0.3)
    """

    # Road type and direction mappings
    EAST_WEST_ROADS = ["AC", "CL"]  # Calles (Streets)
    NORTH_SOUTH_ROADS = ["KR", "AK"]  # Carreras (Avenues)

    # ZAT feature columns
    ZAT_FEATURES = [
        "land_use_km2_comercial",
        "land_use_km2_industrial",
        "land_use_km2_equipamientos",
        "land_use_km2_educacion",
        "land_use_km2_comser",
        "land_use_km2_recreacional",
        "land_use_km2_cuposedu",
        "land_use_km2_residencial",
        "population_5_24 años",
        "population_25_64 años",
        "population_more_65 años",
        "population_Total",
    ]

    def __init__(
        self,
        road_network_path: str,
        buildings_path: str,
        bus_stops_path: str,
        zat_geometry_path: str,
        zat_population_path: str,
        output_dir: str,
        target_crs: str = "EPSG:3116",
        spatial_tolerance: float = 1.0,
        legal_influence_weight: float = 1.0,
        illegal_influence_weight: float = 0.3,
    ):
        """
        Initialize the RoadGraphConstructor with data paths and processing parameters.

        Args:
            road_network_path: Path to the MVI road network GeoJSON file
            buildings_path: Path to CSV file with building counts per road
            bus_stops_path: Path to parquet file with bus stop counts per road
            zat_geometry_path: Path to GDB file with ZAT (Traffic Analysis Zone) geometries
            zat_population_path: Path to CSV file with population and land use per ZAT
            output_dir: Directory where graph files will be saved
            target_crs: EPSG code for projected coordinate system (default: "EPSG:3116" for Bogotá)
            spatial_tolerance: Tolerance in meters for spatial intersection detection (default: 1.0)
            legal_influence_weight: Edge weight for legal traffic connections (default: 1.0)
            illegal_influence_weight: Edge weight for illegal but spatially close connections (default: 0.3)
        """
        self.road_network_path = road_network_path
        self.buildings_path = buildings_path
        self.bus_stops_path = bus_stops_path
        self.zat_geometry_path = zat_geometry_path
        self.zat_population_path = zat_population_path
        self.output_dir = Path(output_dir)
        self.target_crs = target_crs
        self.spatial_tolerance = spatial_tolerance
        self.legal_influence_weight = legal_influence_weight
        self.illegal_influence_weight = illegal_influence_weight

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def calculate_cardinal_direction(road_type: str, flow_direction: str) -> str:
        """
        Calculate the cardinal direction of traffic flow based on road type and direction code.

        Bogotá's road system uses:
        - Calles (AC, CL): East-West streets
        - Carreras (KR, AK): North-South avenues
        - Direction codes: B (bidirectional), TF/FT (one-way indicators)

        Args:
            road_type: Road type code (AC, CL, KR, AK, etc.)
            flow_direction: Flow direction code (B, TF, FT)

        Returns:
            str: Cardinal direction label:
                - "bidirectional": Two-way traffic
                - "left-right" / "right-left": East-West one-way
                - "south-north" / "north-south": North-South one-way
        """
        if flow_direction == "B":  # Bidirectional
            return "bidirectional"

        if road_type in RoadGraphConstructor.EAST_WEST_ROADS:  # Streets (East-West)
            return "left-right" if flow_direction == "TF" else "right-left"
        elif (
            road_type in RoadGraphConstructor.NORTH_SOUTH_ROADS
        ):  # Avenues (North-South)
            return "south-north" if flow_direction == "FT" else "north-south"

        return "bidirectional"  # Default to bidirectional for unknown types

    @staticmethod
    def extract_flat_coordinates(geometry) -> List[Tuple[float, float]]:
        """
        Extract a flat list of coordinates from LineString or MultiLineString geometry.

        Args:
            geometry: Shapely LineString or MultiLineString geometry

        Returns:
            List[Tuple[float, float]]: Flat list of (x, y) coordinate tuples
        """
        if geometry.geom_type == "MultiLineString":
            coords = []
            for part in geometry.geoms:
                coords.extend(part.coords)
            return coords
        return list(geometry.coords)

    @staticmethod
    def compute_legal_connections(df: pd.DataFrame, tol: float = 1e-6) -> pd.Series:
        """
        Vectorized computation of legal traffic connections between road pairs.

        This method determines if traffic can legally flow from left road to right road
        based on:
        1. Exit legality: Can vehicles exit the left road at the connection point?
        2. Entry legality: Can vehicles enter the right road at the connection point?

        A connection is legal only if both exit and entry are allowed based on
        road directionality and geometry.

        Args:
            df: DataFrame with columns:
                - cardinal_left, cardinal_right: Directional labels
                - sx_left, sy_left, ex_left, ey_left: Start/end coords of left road
                - sx_right, sy_right, ex_right, ey_right: Start/end coords of right road
            tol: Numerical tolerance for coordinate comparison (default: 1e-6)

        Returns:
            pd.Series: Boolean mask indicating legal connections
        """
        # Extract directional and coordinate columns
        cl = df["cardinal_left"]
        cr = df["cardinal_right"]
        sxl, syl = df["sx_left"], df["sy_left"]
        exl, eyl = df["ex_left"], df["ey_left"]
        sxr, syr = df["sx_right"], df["sy_right"]
        exr, eyr = df["ex_right"], df["ey_right"]

        # 1) Check if exit is allowed from left segment
        exit_allowed = (
            (cl == "bidirectional")  # Can exit from either end
            | ((cl == "left-right") & np.isclose(exl, exl, atol=tol))  # Exit at end
            | ((cl == "right-left") & np.isclose(sxl, exl, atol=tol))  # Exit at start
            | ((cl == "south-north") & np.isclose(exl, exl, atol=tol))  # Exit at end
            | ((cl == "north-south") & np.isclose(sxl, exl, atol=tol))  # Exit at start
        )

        # 2) Check if entry is allowed on right segment
        entry_allowed = (
            (cr == "bidirectional")  # Can enter from either end
            | (  # Entry at start for left-right
                (cr == "left-right")
                & np.isclose(sxr, exl, atol=tol)
                & np.isclose(syr, eyl, atol=tol)
            )
            | (  # Entry at end for right-left
                (cr == "right-left")
                & np.isclose(exr, exl, atol=tol)
                & np.isclose(eyr, eyl, atol=tol)
            )
            | (  # Entry at start for south-north
                (cr == "south-north")
                & np.isclose(sxr, exl, atol=tol)
                & np.isclose(syr, eyl, atol=tol)
            )
            | (  # Entry at end for north-south
                (cr == "north-south")
                & np.isclose(exr, exl, atol=tol)
                & np.isclose(eyr, eyl, atol=tol)
            )
        )

        # 3) Legal connection requires both exit and entry to be allowed
        return exit_allowed & entry_allowed

    def load_auxiliary_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load auxiliary datasets: buildings and bus stops per road.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Buildings and bus stops dataframes with prefixed columns
        """
        print("Loading auxiliary data...")

        # Load buildings data
        buildings = pd.read_csv(self.buildings_path)
        buildings.columns = [
            f"buildings_{col}" if col != "road_id" else col for col in buildings.columns
        ]

        # Load bus stops data
        bus_stops = pd.read_parquet(self.bus_stops_path)
        bus_stops.columns = [
            f"bus_stop_{col}" if col != "road_id" else col for col in bus_stops.columns
        ]

        print(f"  Loaded {len(buildings)} building records")
        print(f"  Loaded {len(bus_stops)} bus stop records")

        return buildings, bus_stops

    def load_and_process_roads(
        self, buildings: pd.DataFrame, bus_stops: pd.DataFrame
    ) -> gpd.GeoDataFrame:
        """
        Load and process the road network with auxiliary features.

        This method:
        1. Loads the MVI road network geometry
        2. Creates unique road IDs and dissolves split segments
        3. Merges building and bus stop data
        4. Computes cardinal directions for traffic flow
        5. Extracts start and end points for each road segment

        Args:
            buildings: DataFrame with building counts per road
            bus_stops: DataFrame with bus stop counts per road

        Returns:
            gpd.GeoDataFrame: Processed road network with geometry and attributes
        """
        print("Loading and processing road network...")

        # Load road network
        road_df = gpd.read_file(self.road_network_path)

        # Create unique road IDs from MVI codes
        road_df["id_mvi"] = (
            road_df["MVICIV"].astype(str)
            + "_"
            + road_df["MVISVIA"].str.replace(" ", "")
        )
        road_df["contador"] = road_df.groupby("id_mvi").cumcount() + 1  # type: ignore
        road_df["id_mvi"] = road_df["id_mvi"] + "." + road_df["contador"].astype(str)

        # Dissolve split segments into unified roads
        road_df = (
            road_df.explode(ignore_index=True)  # Explode MultiLineStrings
            .dissolve(by="id_mvi")  # Merge segments with same ID
            .reset_index()
        )
        road_df["road_id"] = road_df["id_mvi"]

        # Merge auxiliary data and convert to target CRS
        road_df = gpd.GeoDataFrame(
            pd.merge(road_df, buildings, on="road_id", how="left")
        ).to_crs(self.target_crs)

        road_df = gpd.GeoDataFrame(
            pd.merge(road_df, bus_stops, on="road_id", how="left")
        ).to_crs(self.target_crs)

        # Calculate cardinal directions for traffic flow
        road_df["cardinal"] = road_df.apply(
            lambda row: self.calculate_cardinal_direction(
                row["MVITIPO"], row["MVISVIA"]
            ),
            axis=1,
        )

        # Extract start and end points
        road_df[["start_pt", "end_pt"]] = road_df.geometry.apply(
            lambda g: pd.Series(
                [
                    Point(self.extract_flat_coordinates(g)[0]),
                    Point(self.extract_flat_coordinates(g)[-1]),
                ]
            )
        )

        # Extract coordinates as separate columns
        road_df[["sx", "sy"]] = pd.DataFrame(
            [(p.x, p.y) for p in road_df.start_pt], index=road_df.index
        )
        road_df[["ex", "ey"]] = pd.DataFrame(
            [(p.x, p.y) for p in road_df.end_pt], index=road_df.index
        )

        # Calculate road length in meters
        road_df["length_m"] = road_df.geometry.length

        print(f"  Processed {len(road_df)} road segments")

        return road_df

    def load_and_process_zat_data(self) -> gpd.GeoDataFrame:
        """
        Load and process ZAT (Traffic Analysis Zone) population and land use data.

        This method:
        1. Loads ZAT geometries and population/land use attributes
        2. Merges geometry with demographic data
        3. Converts to target CRS

        Returns:
            gpd.GeoDataFrame: ZAT zones with population and land use attributes
        """
        print("Loading and processing ZAT data...")

        # Load ZAT geometries
        zats = gpd.read_file(self.zat_geometry_path, layer=0)

        # Load population and land use data
        pop_land = pd.read_csv(self.zat_population_path, encoding="latin-1")

        # Merge and convert to target CRS
        zat_pop_land = gpd.GeoDataFrame(pd.merge(pop_land, zats, on="ZAT"))
        zat_pop_land = zat_pop_land.to_crs(self.target_crs)

        print(f"  Loaded {len(zat_pop_land)} ZAT zones")

        return zat_pop_land

    def assign_zat_features_to_roads(
        self, road_df: gpd.GeoDataFrame, zat_pop_land: gpd.GeoDataFrame
    ) -> Dict[str, Dict[str, float]]:
        """
        Assign ZAT features to road segments based on spatial overlay.

        This method performs a sophisticated spatial assignment:
        1. Overlays road segments with ZAT zones
        2. Calculates the length of each road-ZAT intersection
        3. Identifies the dominant ZAT for each road (by intersection length)
        4. Distributes ZAT attributes proportionally based on road's share of the ZAT

        Args:
            road_df: GeoDataFrame of road segments
            zat_pop_land: GeoDataFrame of ZAT zones with attributes

        Returns:
            Dict[str, Dict[str, float]]: Mapping from road_id to ZAT feature dictionary
        """
        print("Assigning ZAT features to roads...")

        # Spatial overlay: intersect roads with ZAT zones
        road_zats_data = gpd.overlay(
            road_df[["road_id", "geometry"]], zat_pop_land, how="intersection"
        )

        # Calculate intersection segment lengths
        road_zats_data["seg_len"] = road_zats_data.geometry.length

        # Aggregate by road and ZAT
        agg = road_zats_data.groupby(["road_id", "ZAT"])["seg_len"].sum().reset_index()

        # Calculate total length per road
        total = agg.groupby("road_id")["seg_len"].sum().rename("total_len")
        agg = agg.join(total, on="road_id")
        agg["pct"] = agg["seg_len"] / agg["total_len"]

        # Find dominant ZAT for each road (longest intersection)
        dominant = agg.loc[agg.groupby("road_id")["seg_len"].idxmax()]

        # Calculate ZAT-level totals for proportional distribution
        dominant_reformed = pd.merge(
            dominant,
            dominant.groupby(["ZAT"]).aggregate({"total_len": "sum"}).reset_index(),
            on="ZAT",
            how="left",
            suffixes=("", "_zat"),
        )
        dominant_reformed["zat_perc"] = (
            dominant_reformed.total_len / dominant_reformed.total_len_zat
        )

        # Merge ZAT attributes
        id_road_pop_land = pd.merge(
            dominant_reformed[["road_id", "ZAT", "zat_perc"]],
            zat_pop_land[["ZAT", *self.ZAT_FEATURES]],
            on="ZAT",
            how="left",
        )

        # Distribute ZAT attributes proportionally
        for col in self.ZAT_FEATURES:
            id_road_pop_land[col] = id_road_pop_land[col] * id_road_pop_land.zat_perc

        # Convert to dictionary format
        zat_road_vars_dict = {
            item["road_id"]: {feature: item[feature] for feature in self.ZAT_FEATURES}
            for item in id_road_pop_land.to_dict("records")
        }

        print(f"  Assigned ZAT features to {len(zat_road_vars_dict)} roads")

        return zat_road_vars_dict

    def build_node_attributes(
        self, road_df: gpd.GeoDataFrame, zat_features: Dict[str, Dict[str, float]]
    ) -> Dict[str, Dict]:
        """
        Build comprehensive node attribute dictionary for the graph.

        Each node (road segment) gets:
        - Road geometry attributes (length, direction, type)
        - Building counts
        - Bus stop counts
        - ZAT-derived features (population, land use)

        Args:
            road_df: GeoDataFrame of road segments with attributes
            zat_features: Dictionary mapping road_id to ZAT features

        Returns:
            Dict[str, Dict]: Mapping from road_id to complete attribute dictionary
        """
        print("Building node attributes...")

        # Fill missing values
        road_df = road_df.fillna("Null")

        # Select static columns (exclude geometry and temporary columns)
        static_cols = [
            c
            for c in road_df.columns
            if c not in ("road_id", "geometry", "start_pt", "end_pt")
        ]

        # Build initial attribute dictionary from road data
        attr_dict = {}
        for _, row in tqdm(
            road_df.iterrows(), total=len(road_df), desc="Processing nodes"
        ):
            road_id = row.road_id
            # Start with road attributes
            attrs = {col: row[col] for col in static_cols}

            # Add ZAT features (with defaults for roads without ZAT assignment)
            zat_attrs = zat_features.get(
                road_id, {feature: 0 for feature in self.ZAT_FEATURES}
            )
            attrs.update(zat_attrs)

            attr_dict[road_id] = attrs

        print(f"  Built attributes for {len(attr_dict)} nodes")

        return attr_dict

    def find_intersecting_roads(self, road_df: gpd.GeoDataFrame) -> pd.DataFrame:
        """
        Find pairs of intersecting roads using spatial join with tolerance buffer.

        This method:
        1. Buffers road geometries by spatial tolerance
        2. Performs spatial join to find intersecting pairs
        3. Filters out self-intersections
        4. Removes duplicate pairs

        Args:
            road_df: GeoDataFrame of road segments

        Returns:
            pd.DataFrame: Pairs of intersecting roads with columns road_id_left, road_id_right
        """
        print("Finding intersecting roads...")

        # Create buffered copy for spatial join
        road_df_buf = road_df[["road_id", "geometry"]].copy()
        road_df_buf["geometry"] = road_df_buf.geometry.buffer(self.spatial_tolerance)

        # Spatial join to find intersecting pairs
        pairs = gpd.sjoin(
            road_df_buf[["road_id", "geometry"]],
            road_df[["road_id", "geometry"]],
            how="inner",
            predicate="intersects",
        )[["road_id_left", "road_id_right"]]

        # Remove self-intersections
        pairs = pairs[pairs.road_id_left != pairs.road_id_right]

        # Sort and remove duplicates
        pairs = pairs.sort_values(["road_id_left", "road_id_right"])
        pairs = pairs.drop_duplicates(subset=["road_id_left", "road_id_right"])

        print(f"  Found {len(pairs)} intersecting road pairs")

        return pairs

    def compute_edge_legality(
        self, pairs: pd.DataFrame, road_df: gpd.GeoDataFrame
    ) -> pd.DataFrame:
        """
        Compute edge legality and influence weights for all intersecting road pairs.

        This method:
        1. Merges directional and coordinate data for both roads in each pair
        2. Applies vectorized legality computation
        3. Assigns influence weights (1.0 for legal, 0.3 for illegal)

        Args:
            pairs: DataFrame of intersecting road pairs
            road_df: GeoDataFrame with road attributes

        Returns:
            pd.DataFrame: Pairs with legality flags and influence weights
        """
        print("Computing edge legality...")

        # Merge left road data
        pairs_data = pd.merge(
            pairs,
            road_df[["road_id", "cardinal", "sx", "sy", "ex", "ey"]],
            how="left",
            left_on="road_id_left",
            right_on="road_id",
        ).drop(["road_id"], axis=1)

        # Merge right road data
        pairs_data = pd.merge(
            pairs_data,
            road_df[["road_id", "cardinal", "sx", "sy", "ex", "ey"]],
            how="left",
            left_on="road_id_right",
            right_on="road_id",
            suffixes=("_left", "_right"),
        ).drop(["road_id"], axis=1)

        # Compute legal connections
        pairs_data["legal"] = self.compute_legal_connections(pairs_data)

        # Assign influence weights
        pairs_data["influence"] = np.where(
            pairs_data["legal"],
            self.legal_influence_weight,
            self.illegal_influence_weight,
        )

        # Convert boolean to int for clarity
        pairs_data["legal"] = np.where(pairs_data["legal"], 1, 0)

        print(f"  Legal edges: {pairs_data['legal'].sum()}")
        print(f"  Illegal edges: {(pairs_data['legal'] == 0).sum()}")

        return pairs_data

    def build_graph(self) -> nx.DiGraph:
        """
        Execute the complete graph construction pipeline.

        This method orchestrates the entire workflow:
        1. Loads auxiliary data (buildings, bus stops)
        2. Loads and processes road network
        3. Loads and assigns ZAT features
        4. Builds node attributes
        5. Finds intersecting roads
        6. Computes edge legality
        7. Constructs directed graph with nodes and edges

        Returns:
            nx.DiGraph: Directed road network graph with rich attributes
        """
        print("=" * 70)
        print("Starting Road Network Graph Construction")
        print("=" * 70)

        # Step 1: Load auxiliary data
        print("\n[1/7] Loading auxiliary data...")
        buildings, bus_stops = self.load_auxiliary_data()

        # Step 2: Load and process roads
        print("\n[2/7] Loading and processing road network...")
        road_df = self.load_and_process_roads(buildings, bus_stops)

        # Step 3: Load ZAT data
        print("\n[3/7] Loading ZAT data...")
        zat_pop_land = self.load_and_process_zat_data()

        # Step 4: Assign ZAT features to roads
        print("\n[4/7] Assigning ZAT features to roads...")
        zat_features = self.assign_zat_features_to_roads(road_df, zat_pop_land)

        # Step 5: Build node attributes
        print("\n[5/7] Building node attributes...")
        node_attributes = self.build_node_attributes(road_df, zat_features)

        # Step 6: Find intersecting roads
        print("\n[6/7] Finding intersecting roads...")
        pairs = self.find_intersecting_roads(road_df)

        # Step 7: Compute edge legality
        print("\n[7/7] Computing edge legality and building graph...")
        pairs_data = self.compute_edge_legality(pairs, road_df)

        # Build the directed graph
        G = nx.DiGraph()

        # Add nodes
        G.add_nodes_from(road_df["road_id"])

        # Set node attributes
        nx.set_node_attributes(G, node_attributes)

        # Add legal edges
        legal_edges = pairs_data[pairs_data.legal == 1]
        G.add_edges_from(
            zip(legal_edges["road_id_left"], legal_edges["road_id_right"]),
            legal=1,
            influence_weight=self.legal_influence_weight,
        )

        # Add illegal edges (for spatial influence in GNN)
        illegal_edges = pairs_data[pairs_data.legal == 0]
        G.add_edges_from(
            zip(illegal_edges["road_id_left"], illegal_edges["road_id_right"]),
            legal=0,
            influence_weight=self.illegal_influence_weight,
        )

        print("\n" + "=" * 70)
        print("Graph Construction Complete!")
        print("=" * 70)
        print(f"Nodes (road segments): {G.number_of_nodes()}")
        print(f"Edges (intersections): {G.number_of_edges()}")
        print(f"  - Legal edges: {len(legal_edges)}")
        print(f"  - Illegal edges: {len(illegal_edges)}")

        # Show example node attributes
        example_node = next(iter(G.nodes()))
        print(f"\nExample node attributes for '{example_node}':")
        for key, value in list(G.nodes[example_node].items())[:5]:
            print(f"  {key}: {value}")

        return G

    def save_graph(
        self, graph: nx.DiGraph, formats: List[str] = ["pickle", "graphml", "json"]
    ) -> None:
        """
        Save the constructed graph in multiple formats.

        Supported formats:
        - pickle: Python pickle format (preserves all data types, fastest)
        - graphml: GraphML XML format (interoperable with other tools)
        - json: JSON format (human-readable, web-compatible)

        Args:
            graph: NetworkX directed graph to save
            formats: List of output formats (default: all three)
        """
        print("\nSaving graph files...")

        if "pickle" in formats:
            pickle_path = self.output_dir / "road_graph.pickle"
            with open(pickle_path, "wb") as f:
                pickle.dump(graph, f, pickle.HIGHEST_PROTOCOL)
            print(f"  Saved pickle format: {pickle_path}")

        if "graphml" in formats:
            graphml_path = self.output_dir / "road_graph.graphml"
            nx.write_graphml(graph, graphml_path)
            print(f"  Saved GraphML format: {graphml_path}")

        if "json" in formats:
            json_path = self.output_dir / "road_graph.json"
            json_data = nx_json_graph.node_link_data(graph)
            with open(json_path, "w") as f:
                json.dump(json_data, f)
            print(f"  Saved JSON format: {json_path}")

        print(f"\nAll graph files saved to: {self.output_dir}")

    @staticmethod
    def get_roads_within_hops(
        graph: nx.DiGraph, seed_road: str, k: int
    ) -> Dict[str, int]:
        """
        Get all roads reachable from a seed road within k hops.

        This is useful for:
        - Visualizing local neighborhoods
        - Debugging graph connectivity
        - Understanding spatial structure

        Args:
            graph: NetworkX directed graph
            seed_road: Road ID to start from
            k: Maximum number of hops (edges) to traverse

        Returns:
            Dict[str, int]: Mapping from reachable road_id to distance in hops
        """
        return nx.single_source_shortest_path_length(graph, seed_road, cutoff=k)

    @staticmethod
    def get_roads_within_distance(
        graph: nx.DiGraph, seed_road: str, max_distance_m: float
    ) -> Dict[str, float]:
        """
        Get all roads reachable from a seed road within a distance threshold.

        This method uses Dijkstra's algorithm with road length as edge weight
        to find all roads within a specified distance in meters.

        Args:
            graph: NetworkX directed graph with 'length_m' node attribute
            seed_road: Road ID to start from
            max_distance_m: Maximum cumulative distance in meters

        Returns:
            Dict[str, float]: Mapping from reachable road_id to cumulative distance
        """
        # Ensure each edge has a weight attribute (average of node lengths)
        for u, v, data in graph.edges(data=True):
            if "weight" not in data:
                data["weight"] = (
                    graph.nodes[u]["length_m"] + graph.nodes[v]["length_m"]
                ) / 2.0

        # Run Dijkstra with distance cutoff
        distances = nx.single_source_dijkstra_path_length(
            graph, seed_road, cutoff=max_distance_m, weight="weight"
        )

        return distances


if __name__ == "__main__":
    # Define input data paths
    road_network_path = "data/2_Estructura_via/estructura_via_completa/MVI.geojson"
    buildings_path = "data/datos_procesados/buildings/roads_buildings.csv"
    bus_stops_path = "data/datos_procesados/bus_stops/bus_stops_per_road.parquet"
    zat_geometry_path = "data/10_uso_del_suelo/Zonificacion_2023_GDB/ZAT_2023.gdb"
    zat_population_path = "data/13_poblacion_ZAT/zat_population_land_use.csv"
    output_dir = "data/datos_procesados/conectivity_graph/"

    # Initialize constructor
    constructor = RoadGraphConstructor(
        road_network_path=road_network_path,
        buildings_path=buildings_path,
        bus_stops_path=bus_stops_path,
        zat_geometry_path=zat_geometry_path,
        zat_population_path=zat_population_path,
        output_dir=output_dir,
        target_crs="EPSG:3116",
        spatial_tolerance=1.0,
        legal_influence_weight=1.0,
        illegal_influence_weight=0.3,
    )

    # Build the graph
    graph = constructor.build_graph()

    # Save in all formats
    constructor.save_graph(graph, formats=["pickle", "graphml", "json"])

    print(
        "\nGraph construction complete! You can now use this graph with sparse_dataloader.py"
    )

    # Example: Get roads within 2 hops
    example_road = list(graph.nodes())[0]
    nearby_roads = RoadGraphConstructor.get_roads_within_hops(graph, example_road, k=2)
    print(f"\nExample: {len(nearby_roads)} roads within 2 hops of '{example_road}'")
