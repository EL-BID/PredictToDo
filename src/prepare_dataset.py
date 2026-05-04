import os
import pickle
import sys
from pathlib import Path

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent / "src"))


def run_feature_generation():
    """
    Step 1: Generate all feature datasets from raw data.

    This step processes raw data files and creates cleaned, spatially-assigned,
    and temporally-aligned feature datasets ready for merging.
    """
    print("\n" + "=" * 80)
    print("STEP 1: FEATURE GENERATION")
    print("=" * 80)

    # Import processors
    from preprocessing.features.buildings import BuildingsProcessor
    from preprocessing.features.bus_stops import BusStopsProcessor
    from preprocessing.features.events import EventsProcessor
    from preprocessing.features.graph import RoadGraphConstructor
    from preprocessing.features.incidents import IncidentsProcessor
    from preprocessing.features.irregularities import IrregularitiesProcessor
    from preprocessing.features.jams import JamsProcessor
    from preprocessing.features.pmts_points import PMTSPointsProcessor
    from preprocessing.features.pmts_tracts import PMTSTractsProcessor
    from preprocessing.features.road_state import RoadStateProcessor
    from preprocessing.features.weather import WeatherProcessor

    # 1.0: Buildings (must run first - graph constructor depends on this)
    print("\n[1.0] Processing Buildings...")
    print("-" * 80)
    buildings_processor = BuildingsProcessor(
        road_network_path="data/2_Estructura_via/estructura_via_completa/MVI.geojson",
        output_dir="data/datos_procesados/buildings/",
        buffer_distance_wgs84=0.01,  # ~1km for bbox calculation
        buffer_distance_meters=100.0,  # 100m for building assignment
        target_crs="EPSG:3116",  # Colombia projected CRS
    )
    buildings_processor.process_and_save()

    # 1.1: Bus Stops (must run before graph constructor)
    print("\n[1.1] Processing Bus Stops...")
    print("-" * 80)
    bus_stops_processor = BusStopsProcessor(
        bus_stops_shapefile="data/15_Paraderos_transp_publico_Zonales_SITP/Paraderos_Zonales_del_SITP.shp",
        road_network_path="data/2_Estructura_via/estructura_via_completa/MVI.geojson",
        output_dir="data/datos_procesados/bus_stops/",
        max_distance_degrees=0.005,
        target_crs="EPSG:21899",
    )
    bus_stops_processor.process_and_save()

    # 1.2: Road Network Graph (depends on buildings and bus stops)
    print("\n[1.2] Building Road Network Graph...")
    print("-" * 80)
    graph_constructor = RoadGraphConstructor(
        road_network_path="data/2_Estructura_via/estructura_via_completa/MVI.geojson",
        buildings_path="data/datos_procesados/buildings/roads_buildings.csv",
        bus_stops_path="data/datos_procesados/bus_stops/bus_stops_per_road.parquet",
        zat_geometry_path="data/10_uso_del_suelo/Zonificacion_2023_GDB/ZAT_2023.gdb",
        zat_population_path="data/13_poblacion_ZAT/zat_population_land_use.csv",
        output_dir="data/datos_procesados/conectivity_graph/",
        target_crs="EPSG:3116",
        spatial_tolerance=1.0,
        legal_influence_weight=1.0,
        illegal_influence_weight=0.3,
    )
    graph = graph_constructor.build_graph()
    graph_constructor.save_graph(graph, formats=["pickle", "graphml", "json"])

    # 1.3: Events
    print("\n[1.3] Processing Events...")
    print("-" * 80)
    events_processor = EventsProcessor(
        events_data_path="data/8_eventos/events_dataset_preprocessing.csv",
        road_network_path="data/2_Estructura_via/estructura_via_completa/MVI.geojson",
        connectivity_graph_path="data/datos_procesados/conectivity_graph/road_graph.pickle",
        output_dir="data/datos_procesados/events/",
        buffer_distance=0.0027,
        complexity_distances={
            "ALTA": 500,
            "MEDIA": 300,
            "BAJA": 200,
            "SIN DEFINIR": 500,
        },
        temporal_resolution="15min",
    )
    events_processor.process_and_save()

    # 1.4: Incidents
    print("\n[1.4] Processing Incidents...")
    print("-" * 80)
    incidents_processor = IncidentsProcessor(
        incidents_data_path="data/7_incidentes/incidents_dataset_prev.csv",
        road_network_path="data/2_Estructura_via/estructura_via_completa/MVI.geojson",
        output_dir="data/datos_procesados/incidents/",
        buffer_distance=0.001,
        temporal_resolution="15min",
    )
    incidents_processor.process_and_save()

    # 1.5: Irregularities
    print("\n[1.5] Processing Irregularities...")
    print("-" * 80)
    irregularities_processor = IrregularitiesProcessor(
        input_folder="data/4_velocidades_waze/irregularidades",
        road_network_path="data/2_Estructura_via/estructura_via_completa/MVI.geojson",
        output_dir="data/datos_procesados/irregularidades_procesadas/",
        n_chunks=10,
        buffer_distance=4.0,
        overlap_threshold=0.2,
        temporal_resolution="15min",
    )
    irregularities_processor.process_and_save()

    # 1.6: Jams
    print("\n[1.6] Processing Jams...")
    print("-" * 80)
    jams_processor = JamsProcessor(
        jams_data_dir="data/4_velocidades_waze/jams/",
        road_network_path="data/2_Estructura_via/estructura_via_completa/MVI.geojson",
        output_dir="data/datos_procesados/jams_procesados/",
        buffer_distance=4.0,
        overlap_threshold=0.2,
        target_crs="EPSG:3116",
        temporal_resolution="15min",
        filter_closed_roads=True,
    )
    jams_processor.process_and_save()

    # 1.7: PMTs Points (COI Points)
    print("\n[1.7] Processing PMTs Points...")
    print("-" * 80)
    pmts_points_processor = PMTSPointsProcessor(
        coi_parques_path="data/9_pmts/COI_PARQUES_202504071925.geojson",
        coi_puntos_path="data/9_pmts/COI_PUNTO_202504071914.geojson",
        road_network_path="data/2_Estructura_via/estructura_via_completa/MVI.geojson",
        output_dir="data/datos_procesados/pmts_processed/",
        buffer_distance=4.0,
        overlap_threshold=0.15,
        temporal_resolution="15T",
        target_crs="EPSG:3116",
    )
    pmts_points_processor.process_and_save()

    # 1.8: PMTs Tracts (COI Tracts)
    print("\n[1.8] Processing PMTs Tracts...")
    print("-" * 80)
    pmts_tracts_processor = PMTSTractsProcessor(
        coi_tramo_path="data/9_pmts/COI_TRAMO_202504071919.geojson",
        coos_punto_path="data/9_pmts/COOS_PUNTO_202504071922.geojson",
        road_network_path="data/2_Estructura_via/estructura_via_completa/MVI.geojson",
        output_dir="data/datos_procesados/pmts_processed/",
        buffer_distance=4.0,
        overlap_threshold=0.15,
        target_crs="EPSG:3116",
    )
    pmts_tracts_processor.process_and_save()

    # 1.9: Road State
    print("\n[1.9] Processing Road State...")
    print("-" * 80)
    road_state_processor = RoadStateProcessor(
        road_network_path="data/2_Estructura_via/estructura_via_completa/MVI.geojson",
        road_state_shapefile="data/14_Estado_malla_vial/EstadoMallaVialOD_IDU_Sem12024/EstadoMallaVialUrbanaConsolidado.shp",
        output_dir="data/datos_procesados/road_state/",
        buffer_distance=15.0,
        target_crs="EPSG:3116",
        timeframe_start="2024-01-01 00:00:00",
        timeframe_end="2024-06-30 23:45:00",
    )
    road_state_processor.process_and_save()

    # 1.10: Weather
    print("\n[1.10] Processing Weather...")
    print("-" * 80)
    weather_processor = WeatherProcessor(
        weather_data_paths={
            "direccion_viento": "data/3_clima/tratados/direccion_viento.parquet",
            "humedad": "data/3_clima/tratados/humedad.parquet",
            "precipitacion": "data/3_clima/tratados/precipitacion.parquet",
            "velocidad_viento": "data/3_clima/tratados/velocidad_viento.parquet",
        },
        road_network_path="data/2_Estructura_via/estructura_via_completa/MVI.geojson",
        output_dir="data/datos_procesados/weather/",
        target_year=2024,
        temporal_resolution="15min",
        buffer_distance=15.0,
        target_crs="EPSG:3116",
    )
    weather_processor.process_and_save()

    print("\n" + "=" * 80)
    print("STEP 1 COMPLETE: All feature datasets generated successfully")
    print("=" * 80)


def run_imputation_generation():
    """
    Step 2: Generate imputation data for missing features.

    This step uses the connectivity graph and irregularities data to compute
    neighbor-aggregated features that can be used to impute missing values
    during training.

    IMPORTANT: This must run AFTER feature generation (needs graph and irregularities)
    and BEFORE dataset instantiation (dataset needs these features).
    """
    print("\n" + "=" * 80)
    print("STEP 2: IMPUTATION DATA GENERATION")
    print("=" * 80)

    from preprocessing.imputation import generate_neighbor_features_from_files

    processed_data_dir = "data/datos_procesados"
    graph_path = f"{processed_data_dir}/conectivity_graph/road_graph.pickle"
    irregularities_dir = f"{processed_data_dir}/irregularidades_procesadas/"
    output_path = f"{processed_data_dir}/neighbor_features.parquet"
    k_hops = 10

    print("\nGenerating neighbor features for imputation...")
    print(f"  Graph: {graph_path}")
    print(f"  Irregularities: {irregularities_dir}")
    print(f"  Output: {output_path}")
    print(f"  K-hops: {k_hops}")
    print()

    neighbor_features = generate_neighbor_features_from_files(
        graph_path=graph_path,
        irregularities_dir=irregularities_dir,
        k_hops=k_hops,
    )

    # Save the results
    print(f"\nSaving neighbor features to {output_path}...")
    neighbor_features.to_parquet(output_path, index=True)

    print("\n" + "=" * 80)
    print("STEP 2 COMPLETE: Imputation data generated successfully")
    print("=" * 80)


def run_dataset_merging():
    """
    Step 3: Merge all datasets into sparse HDF5 format.

    This step converts all feature datasets into efficient HDF5 files with
    COO (Coordinate) sparse matrix format. This dramatically reduces storage
    and improves loading times during training.

    IMPORTANT: This must run AFTER feature generation and imputation.
    """
    print("\n" + "=" * 80)
    print("STEP 3: DATASET MERGING")
    print("=" * 80)

    # The merging script is designed to be run as a module
    # We'll import and execute its main logic
    print("\nExecuting merging pipeline from src/preprocessing/merging.py...")
    print("This will convert all feature datasets to HDF5 sparse format.")
    print()

    # Import after adding to path
    import pandas as pd

    from preprocessing.merging import SparseDatasetGenerator

    processed_data_dir = "data/datos_procesados"

    # Load the base graph
    print("Loading road network graph...")
    with open(f"{processed_data_dir}/conectivity_graph/road_graph.pickle", "rb") as f:
        base_graph = pickle.load(f)

    # Clean the base graph node attributes
    base_graph = SparseDatasetGenerator._clean_node_attributes(base_graph)
    all_road_ids = list(base_graph.nodes())

    # Initialize dataset generator
    dataset_generator = SparseDatasetGenerator(
        node_ids=all_road_ids,
        min_timeframe=pd.to_datetime("2024-01-01 00:00:00"),
        max_timeframe=pd.to_datetime("2024-12-31 23:45:00"),
        freq="15min",
    )

    # Generate static dataset
    print("\nProcessing static features...")
    static_node_data = pd.DataFrame.from_dict(
        dict(base_graph.nodes(data=True)), orient="index"
    )

    road_state_df = pd.read_parquet(
        f"{processed_data_dir}/road_state/road_state_dateset.parquet"
    )
    road_state_df = road_state_df.drop(
        columns=["timeframe_start", "timeframe_end"]
    ).set_index("road_id")

    static_df = static_node_data.join(road_state_df, how="left")
    static_df.index.name = "road_id"

    # Fill NaN values
    static_df["numeric_road_state"] = static_df["numeric_road_state"].fillna(0)
    feature_medians = static_df[
        ["SUPERFEST", "INDEST", "LOCEST", "KMCAINDEST"]
    ].median()
    static_df[["SUPERFEST", "INDEST", "LOCEST", "KMCAINDEST"]] = static_df[
        ["SUPERFEST", "INDEST", "LOCEST", "KMCAINDEST"]
    ].fillna(feature_medians)

    del base_graph, road_state_df, static_node_data

    dataset_generator.static_dataframe_to_coo(
        static_df,
        output_path=f"{processed_data_dir}/sparse_datasets/static_dataset.h5",
        has_index=True,
        return_df=False,
    )

    # Generate temporal datasets
    print("\nProcessing temporal features...")

    # Events
    print("  - Events...")
    events_df = pd.read_csv(f"{processed_data_dir}/events/events_dataset.csv")
    dataset_generator.temporal_dataframe_to_coo(
        events_df,
        output_path=f"{processed_data_dir}/sparse_datasets/events_dataset.h5",
        has_index=False,
        node_col="road_id",
        timeframe_col="timeframe",
        return_df=False,
    )
    del events_df

    # Incidents
    print("  - Incidents...")
    incidents_df = pd.read_csv(f"{processed_data_dir}/incidents/incidents_dataset.csv")
    dataset_generator.temporal_dataframe_to_coo(
        incidents_df,
        output_path=f"{processed_data_dir}/sparse_datasets/incidents_dataset.h5",
        has_index=False,
        node_col="road_id",
        timeframe_col="timeframe",
        return_df=False,
    )
    del incidents_df

    # Irregularities
    print("  - Irregularities...")
    irregularities_filenames = [
        f"{processed_data_dir}/irregularidades_procesadas/" + f
        for f in os.listdir(f"{processed_data_dir}/irregularidades_procesadas/")
        if f.endswith(".parquet")
    ]
    irregularities_df = pd.concat(
        [pd.read_parquet(f) for f in irregularities_filenames],
        ignore_index=False,
    ).fillna(0)
    dataset_generator.temporal_dataframe_to_coo(
        irregularities_df,
        output_path=f"{processed_data_dir}/sparse_datasets/irregularities_dataset.h5",
        has_index=False,
        node_col="id_mvi",
        timeframe_col="timeframe",
        return_df=False,
    )
    del irregularities_df

    # COI Points
    print("  - COI Points...")
    coi_points_filenames = [
        f"{processed_data_dir}/pmts_processed/coi_puntos/" + f
        for f in os.listdir(f"{processed_data_dir}/pmts_processed/coi_puntos/")
        if f.endswith(".parquet")
    ]
    coi_points_df = pd.concat(
        [pd.read_parquet(f) for f in coi_points_filenames],
        ignore_index=False,
    ).fillna(0)
    dataset_generator.temporal_dataframe_to_coo(
        coi_points_df,
        output_path=f"{processed_data_dir}/sparse_datasets/coi_points_dataset.h5",
        has_index=False,
        node_col="road_id",
        timeframe_col="timeframe",
        return_df=False,
    )
    del coi_points_df

    # COI Tracts
    print("  - COI Tracts...")
    coi_tracts_filenames = [
        f"{processed_data_dir}/pmts_processed/coi_tramos/" + f
        for f in os.listdir(f"{processed_data_dir}/pmts_processed/coi_tramos/")
        if f.endswith(".parquet")
    ]
    coi_tracts_df = pd.concat(
        [pd.read_parquet(f) for f in coi_tracts_filenames],
        ignore_index=False,
    ).fillna(0)
    dataset_generator.temporal_dataframe_to_coo(
        coi_tracts_df,
        output_path=f"{processed_data_dir}/sparse_datasets/coi_tracts_dataset.h5",
        has_index=False,
        node_col="road_id",
        timeframe_col="timeframe",
        return_df=False,
    )
    del coi_tracts_df

    # Jams - split into feature groups
    print("  - Jams...")
    jams_filenames = [
        f"{processed_data_dir}/jams_procesados/" + f
        for f in os.listdir(f"{processed_data_dir}/jams_procesados/")
        if f.endswith(".parquet")
    ]

    # Main jams dataset (for target variable)
    jams_df = pd.concat(
        [pd.read_parquet(f) for f in jams_filenames],
        ignore_index=False,
    ).fillna(0)[["road_id", "timeframe", "level_mean"]]

    dataset_generator.temporal_dataframe_to_coo(
        jams_df.rename(columns={"level_mean": "mean_jam_level"}),
        output_path=f"{processed_data_dir}/sparse_datasets/jams_dataset.h5",
        has_index=False,
        node_col="road_id",
        timeframe_col="timeframe",
        return_df=False,
    )
    del jams_df

    # Process jam features in groups
    jam_feature_groups = [
        {
            "name": "velocity",
            "columns": ["road_id", "timeframe", "vel_min", "vel_mean", "vel_max"],
            "output": f"{processed_data_dir}/sparse_datasets/jam_velocity_dataset.h5",
        },
        {
            "name": "delay",
            "columns": ["road_id", "timeframe", "delay_min", "delay_sum", "delay_max"],
            "output": f"{processed_data_dir}/sparse_datasets/jam_delay_dataset.h5",
        },
        {
            "name": "level_and_count",
            "columns": [
                "road_id",
                "timeframe",
                "level_min",
                "level_mean",
                "level_max",
                "count_jams",
            ],
            "output": f"{processed_data_dir}/sparse_datasets/jam_level_count_dataset.h5",
        },
    ]

    for group in jam_feature_groups:
        print(f"    - Jam {group['name']}...")
        jam_group_df = pd.concat(
            [pd.read_parquet(f)[group["columns"]] for f in jams_filenames],
            ignore_index=False,
        ).fillna(0)

        dataset_generator.temporal_dataframe_to_coo(
            jam_group_df,
            output_path=group["output"],
            has_index=False,
            node_col="road_id",
            timeframe_col="timeframe",
            return_df=False,
        )
        del jam_group_df

    # Weather data
    print("  - Weather...")
    stations_df = pd.read_parquet(f"{processed_data_dir}/weather/stations.parquet")
    wind_direction_df = pd.read_parquet(
        f"{processed_data_dir}/weather/direccion_viento.parquet"
    )
    humidity_df = pd.read_parquet(f"{processed_data_dir}/weather/humedad.parquet")
    wind_speed_df = pd.read_parquet(
        f"{processed_data_dir}/weather/velocidad_viento.parquet"
    )
    precipitation_df = pd.read_parquet(
        f"{processed_data_dir}/weather/precipitacion.parquet"
    )

    dataset_generator._process_weather_data(
        stations_df=stations_df,
        wind_direction_df=wind_direction_df,
        humidity_df=humidity_df,
        wind_speed_df=wind_speed_df,
        precipitation_df=precipitation_df,
        output_dir=f"{processed_data_dir}/sparse_datasets/",
    )

    del stations_df, wind_direction_df, humidity_df, wind_speed_df, precipitation_df

    print("\n" + "=" * 80)
    print("STEP 3 COMPLETE: All datasets merged into HDF5 format")
    print("=" * 80)


def run_dataset_instantiation():
    """
    Step 4: Instantiate and pickle the TrafficDataset.

    This step creates a TrafficDataset object with all the processed data
    and saves it as a pickle file for quick loading during training.

    IMPORTANT: This must run AFTER all previous steps.
    """
    print("\n" + "=" * 80)
    print("STEP 4: DATASET INSTANTIATION")
    print("=" * 80)

    from src.sparse_dataloader import TrafficDataset

    processed_data_dir = "data/datos_procesados"
    sparse_data_dir = f"{processed_data_dir}/sparse_datasets"

    # Configuration
    graph_filename = f"{processed_data_dir}/conectivity_graph/road_graph.pickle"
    static_data_filename = f"{sparse_data_dir}/static_dataset.h5"
    dynamic_data_filenames = [
        f"{sparse_data_dir}/coi_points_dataset.h5",
        f"{sparse_data_dir}/coi_tracts_dataset.h5",
        f"{sparse_data_dir}/irregularities_dataset.h5",
        f"{sparse_data_dir}/events_dataset.h5",
        f"{sparse_data_dir}/incidents_dataset.h5",
        f"{sparse_data_dir}/jam_velocity_dataset.h5",
        f"{sparse_data_dir}/jam_level_count_dataset.h5",
        f"{sparse_data_dir}/jam_delay_dataset.h5",
    ]
    metadata_filenames = [
        f"{sparse_data_dir}/coi_points_dataset_metadata.pkl",
        f"{sparse_data_dir}/coi_tracts_dataset_metadata.pkl",
        f"{sparse_data_dir}/irregularities_dataset_metadata.pkl",
        f"{sparse_data_dir}/events_dataset_metadata.pkl",
        f"{sparse_data_dir}/incidents_dataset_metadata.pkl",
        f"{sparse_data_dir}/jam_velocity_dataset_metadata.pkl",
        f"{sparse_data_dir}/jam_level_count_dataset_metadata.pkl",
        f"{sparse_data_dir}/jam_delay_dataset_metadata.pkl",
    ]
    target_data_filename = f"{sparse_data_dir}/jams_dataset.h5"

    # Parameters
    in_timesteps = 150
    out_timesteps = 50
    node_batch_size = 500
    future_features = [
        "coi_puntos",
        "coi_tramo",
        "AGLOMERACION",
        "EMERGENCIA",
        "MANIFESTACION",
    ]

    print("\nInitializing TrafficDataset...")
    print(f"  Input timesteps: {in_timesteps}")
    print(f"  Output timesteps: {out_timesteps}")
    print(f"  Node batch size: {node_batch_size}")
    print(f"  Future features: {future_features}")
    print(f"  Including weather: True")
    print(f"  Including neighbor features for imputation: True")
    print()

    traffic_dataset = TrafficDataset(
        graph_path=graph_filename,
        static_data_path=static_data_filename,
        dynamic_data_paths=dynamic_data_filenames,
        metadata_paths=metadata_filenames,
        target_data_path=target_data_filename,
        in_timesteps=in_timesteps,
        out_timesteps=out_timesteps,
        future_features=future_features,
        node_batch_size=node_batch_size,
        weather_data_dir=sparse_data_dir,
        include_weather=True,
        neighbor_features_path=f"{processed_data_dir}/neighbor_features.parquet",
    )

    print(f"Dataset initialized successfully!")
    print(f"  Total nodes: {traffic_dataset.num_nodes}")
    print(f"  Static features: {traffic_dataset.num_static_features}")
    print(f"  Dynamic features: {traffic_dataset.num_dynamic_features}")
    print(f"  Weather features: {traffic_dataset.num_weather_features}")
    print(f"  Total samples per seed: {len(traffic_dataset)}")
    print(f"  Total node batches needed: {traffic_dataset.total_possible_node_batches}")

    # Save to pickle
    output_path = f"{processed_data_dir}/traffic_dataset.pkl"
    print(f"\nSaving dataset to {output_path}...")
    with open(output_path, "wb") as f:
        pickle.dump(traffic_dataset, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Dataset saved successfully!")
    print(f"  File size: {os.path.getsize(output_path) / (1024**2):.2f} MB")

    print("\n" + "=" * 80)
    print("STEP 4 COMPLETE: Dataset instantiated and pickled")
    print("=" * 80)


def main():
    """
    Execute the complete dataset preparation pipeline.

    This orchestrates all four steps in the correct order:
    1. Feature Generation (creates all feature datasets)
    2. Imputation Data (generates neighbor features)
    3. Dataset Merging (converts to HDF5 sparse format)
    4. Dataset Instantiation (creates and pickles TrafficDataset)
    """
    print("\n" + "#" * 80)
    print("#" + " " * 78 + "#")
    print("#" + " " * 20 + "TRAFFIC DATASET PREPARATION PIPELINE" + " " * 22 + "#")
    print("#" + " " * 78 + "#")
    print("#" * 80)
    print("\nThis pipeline will execute all preprocessing steps to prepare the dataset")
    print("for traffic forecasting model training.")
    print("\nEstimated time: 30-60 minutes (depending on data size)")
    print("\nPipeline Steps:")
    print("  1. Feature Generation (Graph, Events, Incidents, Irregularities, etc.)")
    print("  2. Imputation Data Generation (Neighbor features)")
    print("  3. Dataset Merging (Convert to HDF5 sparse format)")
    print("  4. Dataset Instantiation (Create and pickle TrafficDataset)")

    try:
        # Step 1: Feature Generation
        run_feature_generation()

        # Step 2: Imputation Data
        run_imputation_generation()

        # Step 3: Dataset Merging
        run_dataset_merging()

        # Step 4: Dataset Instantiation
        run_dataset_instantiation()

        # Success!
        print("\n" + "#" * 80)
        print("#" + " " * 78 + "#")
        print("#" + " " * 20 + "PIPELINE COMPLETED SUCCESSFULLY!" + " " * 27 + "#")
        print("#" + " " * 78 + "#")
        print("#" * 80)
        print("\nAll datasets have been prepared and are ready for training.")
        print("\nNext steps:")
        print(
            "  1. Load the dataset: traffic_dataset = pickle.load(open('data/datos_procesados/traffic_dataset.pkl', 'rb'))"
        )
        print(
            "  2. Create a DataLoader: dataloader = DataLoader(traffic_dataset, batch_size=16, ...)"
        )
        print("  3. Start training: for batch in dataloader: ...")
        print("\nFor more information, see the documentation in each processor class.")

    except Exception as e:
        print("\n" + "!" * 80)
        print("ERROR: Pipeline failed!")
        print("!" * 80)
        print(f"\nError message: {str(e)}")
        print(
            "\nPlease check the error message above and ensure all data files are present."
        )
        raise


if __name__ == "__main__":
    main()
