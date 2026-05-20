# Code Reference

Generated from Python AST on 2026-05-20.

This file inventories Python classes, functions, and class methods. It is intended as a map for maintainers and agents; behavioral details still live in the adjacent runbooks and module docs.

## `src/calibration_entrypoint.py`

| Type | Name | Line | Purpose |
|---|---|---:|---|
| function | `parse_args` | 39 |  |
| function | `_copy_or_extract_model_artifact` | 64 |  |
| function | `_load_checkpoint_model` | 102 |  |
| class | `ReservoirPairCollector` | 114 |  |
| method | `ReservoirPairCollector.__init__` | 115 |  |
| method | `ReservoirPairCollector.add` | 124 |  |
| method | `ReservoirPairCollector.arrays` | 146 |  |
| function | `_apply_qmap` | 154 |  |
| function | `_fit_quantile_map` | 158 |  |
| function | `_fit_zero_gate` | 168 |  |
| function | `_apply_hurdle_qmap` | 184 |  |
| function | `_apply_hybrid` | 199 |  |
| function | `_compute_basic_metrics` | 206 |  |
| function | `_summarize_overall` | 218 |  |
| function | `_summarize_by_level` | 226 |  |
| function | `_summarize_by_timestep_and_level` | 237 |  |
| function | `_log_metric_group` | 262 |  |
| function | `_make_seeded_loader` | 272 |  |
| function | `_run_collection_pass` | 302 |  |
| function | `main` | 336 |  |

## `src/inference_dataloader.py`

| Type | Name | Line | Purpose |
|---|---|---:|---|
| class | `InferenceDataLoader` | 39 | DataLoader for inference that loads real-time data and generates batches |
| method | `InferenceDataLoader.__init__` | 51 | Initialize the inference DataLoader. |
| method | `InferenceDataLoader._columns_for_dataset` | 147 | Return minimal columns required for each dataset. |
| method | `InferenceDataLoader._load_graph` | 175 | Load the road network graph. |
| method | `InferenceDataLoader._load_static_features` | 187 | Load static features from HDF5. |
| method | `InferenceDataLoader._load_normalization_constants` | 206 | Load normalization constants for feature scaling. |
| method | `InferenceDataLoader._get_expected_cols` | 231 | Return expected feature columns for a dataset key, if provided. |
| method | `InferenceDataLoader.load_and_prepare_batch` | 238 | Load real-time data and prepare batches for inference. |
| method | `InferenceDataLoader._load_data_parallel` | 302 | Load data in parallel using Ray. |
| method | `InferenceDataLoader._load_data_sequential` | 383 | Load data sequentially (fallback when Ray is not available). |
| method | `InferenceDataLoader._filter_existing_columns` | 424 |  |
| method | `InferenceDataLoader._ensure_expected_columns` | 432 |  |
| method | `InferenceDataLoader._get_latest_timestamp` | 448 | Get the latest timestamp across all data sources. |
| method | `InferenceDataLoader._process_features` | 498 | Process raw data into feature tensors aligned with graph nodes and time. |
| method | `InferenceDataLoader._process_features_sequential` | 548 | Process features sequentially. |
| method | `InferenceDataLoader._process_features_parallel` | 626 | Process features in parallel using Ray. |
| method | `InferenceDataLoader._process_jams` | 729 | Process jams data into tensor format. |
| method | `InferenceDataLoader._process_jams_static` | 741 | Static method for processing jams (can be called by Ray). |
| method | `InferenceDataLoader._process_irregularities` | 802 | Process irregularities data into tensor format. |
| method | `InferenceDataLoader._process_irregularities_static` | 814 | Static method for processing irregularities. |
| method | `InferenceDataLoader._process_events` | 868 | Process events data into tensor format. |
| method | `InferenceDataLoader._process_events_static` | 880 | Static method for processing events data. |
| method | `InferenceDataLoader._process_incidents` | 952 | Process incidents data into tensor format. |
| method | `InferenceDataLoader._process_incidents_static` | 964 | Static method for processing incidents data. |
| method | `InferenceDataLoader._process_weather` | 1049 | Process weather data into tensor format. |
| method | `InferenceDataLoader._process_weather_static` | 1058 | Static method for processing weather data. |
| method | `InferenceDataLoader._process_pmts` | 1122 | Process PMT data for both past and future windows. |
| method | `InferenceDataLoader._create_zero_features` | 1220 | Create a zero tensor for missing features. |
| method | `InferenceDataLoader._create_batches` | 1226 | Create batches from processed features. |
| method | `InferenceDataLoader._get_neighborhood` | 1282 | Get node indices for neighborhood around seed node using BFS. |
| method | `InferenceDataLoader._get_subgraph_edges` | 1299 | Extract edges for subgraph containing specified nodes. |
| method | `InferenceDataLoader.cleanup` | 1312 | Cleanup resources (shutdown Ray if initialized). |

## `src/inference_production.py`

| Type | Name | Line | Purpose |
|---|---|---:|---|
| class | `InferencePipeline` | 37 | Production inference pipeline with monitoring and error handling. |
| method | `InferencePipeline.__init__` | 40 | Initialize the inference pipeline. |
| method | `InferencePipeline._load_target_norm_constants` | 71 | Load target normalization constants for de-normalization. |
| method | `InferencePipeline._denormalize_predictions` | 86 | De-normalize predictions to actual values. |
| method | `InferencePipeline._load_model` | 148 | Load trained model. |
| method | `InferencePipeline.predict` | 161 | Run inference on provided data. |
| method | `InferencePipeline._save_predictions` | 337 | Save predictions to file. |
| method | `InferencePipeline._compute_hop_counts` | 357 |  |
| method | `InferencePipeline._filter_predictions_by_hops` | 370 |  |
| method | `InferencePipeline.batch_predict` | 404 | Run inference on multiple data files in a directory. |
| method | `InferencePipeline.cleanup` | 482 | Cleanup resources. |
| function | `main` | 488 | Main entry point. |

## `src/merge_seed_time_artifacts.py`

| Type | Name | Line | Purpose |
|---|---|---:|---|
| function | `parse_args` | 10 |  |
| function | `_parse_bucket_quantiles` | 19 |  |
| function | `main` | 29 |  |

## `src/models/gnn_temporal.py`

| Type | Name | Line | Purpose |
|---|---|---:|---|
| function | `make_batched_edge_index` | 6 |  |
| class | `StaticEncoder` | 16 |  |
| method | `StaticEncoder.__init__` | 17 |  |
| method | `StaticEncoder.forward` | 25 |  |
| class | `GNN_Temporal_Model` | 29 |  |
| method | `GNN_Temporal_Model.__init__` | 30 |  |
| method | `GNN_Temporal_Model.forward` | 71 |  |

## `src/prediction_reconstruction.py`

| Type | Name | Line | Purpose |
|---|---|---:|---|
| function | `reconstruct_predictions` | 10 | Reconstruct predictions from a batch into a DataFrame with MultiIndex (NodeID, Timestamp). |

## `src/prepare_dataset.py`

| Type | Name | Line | Purpose |
|---|---|---:|---|
| function | `run_feature_generation` | 10 | Step 1: Generate all feature datasets from raw data. |
| function | `run_imputation_generation` | 203 | Step 2: Generate imputation data for missing features. |
| function | `run_dataset_merging` | 248 | Step 3: Merge all datasets into sparse HDF5 format. |
| function | `run_dataset_instantiation` | 515 | Step 4: Instantiate and pickle the TrafficDataset. |
| function | `main` | 616 | Execute the complete dataset preparation pipeline. |

## `src/preprocessing/features/buildings.py`

| Type | Name | Line | Purpose |
|---|---|---:|---|
| class | `BuildingsProcessor` | 9 | A class for extracting building footprints from OpenStreetMap and assigning them to road network segments. |
| method | `BuildingsProcessor.__init__` | 38 | Initialize the BuildingsProcessor with data paths and processing parameters. |
| method | `BuildingsProcessor.load_and_process_roads` | 70 | Load and process the road network for building extraction. |
| method | `BuildingsProcessor.calculate_bounding_box` | 121 | Calculate the bounding box for OpenStreetMap building query. |
| method | `BuildingsProcessor.extract_buildings_from_osm` | 155 | Extract building footprints from OpenStreetMap within the bounding box. |
| method | `BuildingsProcessor.assign_buildings_to_roads` | 204 | Assign buildings to road segments using spatial intersection. |
| method | `BuildingsProcessor.aggregate_building_counts` | 254 | Aggregate building counts by type for each road segment. |
| method | `BuildingsProcessor.process_and_save` | 307 | Execute the complete buildings processing pipeline and save results. |
| method | `BuildingsProcessor.get_building_distribution_summary` | 371 | Generate a summary of building distribution across the road network. |

## `src/preprocessing/features/bus_stops.py`

| Type | Name | Line | Purpose |
|---|---|---:|---|
| class | `BusStopsProcessor` | 7 | A class for processing bus stop data and assigning stops to road network segments. |
| method | `BusStopsProcessor.__init__` | 29 | Initialize the BusStopsProcessor with data paths and processing parameters. |
| method | `BusStopsProcessor.load_and_process_bus_stops` | 62 | Load and process bus stop point data from shapefile. |
| method | `BusStopsProcessor.load_and_process_roads` | 109 | Load and process the road network for bus stop assignment. |
| method | `BusStopsProcessor.assign_stops_to_streets` | 156 | Assign bus stops to the nearest street segment with accurate distance calculation. |
| method | `BusStopsProcessor.aggregate_stops_per_road` | 217 | Aggregate bus stop statistics per road segment. |
| method | `BusStopsProcessor.process_and_save` | 279 | Execute the complete bus stops processing pipeline and save results. |
| method | `BusStopsProcessor.get_stop_distribution_summary` | 337 | Generate a summary of bus stop distribution across the road network. |

## `src/preprocessing/features/events.py`

| Type | Name | Line | Purpose |
|---|---|---:|---|
| class | `EventsProcessor` | 14 | A class for processing public events data and propagating their impact through the road network. |
| method | `EventsProcessor.__init__` | 34 | Initialize the EventsProcessor with data paths and processing parameters. |
| method | `EventsProcessor.parse_date` | 93 | Parse date string with multiple possible formats. |
| method | `EventsProcessor.quarter_slices` | 115 | Generate time slices aligned to 15-minute intervals between start and end. |
| method | `EventsProcessor.get_roads_within_distance` | 155 | Find all roads reachable from a seed road within a specified distance. |
| method | `EventsProcessor.load_events_data` | 194 | Load and preprocess events data from CSV. |
| method | `EventsProcessor.load_road_network` | 238 | Load and process the road network from GeoJSON. |
| method | `EventsProcessor.load_connectivity_graph` | 285 | Load the road connectivity graph from pickle file. |
| method | `EventsProcessor.assign_events_to_roads` | 305 | Assign events to the nearest road segment using spatial join and distance ranking. |
| method | `EventsProcessor.propagate_events_through_network` | 356 | Propagate events through the road network based on complexity levels. |
| method | `EventsProcessor.explode_and_aggregate` | 416 | Explode events by (road, timeframe) pairs and aggregate by event type. |
| method | `EventsProcessor.process_and_save` | 478 | Execute the complete events data processing pipeline and save results. |

## `src/preprocessing/features/graph.py`

| Type | Name | Line | Purpose |
|---|---|---:|---|
| class | `RoadGraphConstructor` | 15 | A class for constructing directed road network graphs with rich node and edge attributes. |
| method | `RoadGraphConstructor.__init__` | 66 | Initialize the RoadGraphConstructor with data paths and processing parameters. |
| method | `RoadGraphConstructor.calculate_cardinal_direction` | 117 | Calculate the cardinal direction of traffic flow based on road type and direction code. |
| method | `RoadGraphConstructor.extract_flat_coordinates` | 149 | Extract a flat list of coordinates from LineString or MultiLineString geometry. |
| method | `RoadGraphConstructor.compute_legal_connections` | 167 | Vectorized computation of legal traffic connections between road pairs. |
| method | `RoadGraphConstructor.load_auxiliary_data` | 234 | Load auxiliary datasets: buildings and bus stops per road. |
| method | `RoadGraphConstructor.load_and_process_roads` | 264 | Load and process the road network with auxiliary features. |
| method | `RoadGraphConstructor.load_and_process_zat_data` | 350 | Load and process ZAT (Traffic Analysis Zone) population and land use data. |
| method | `RoadGraphConstructor.assign_zat_features_to_roads` | 378 | Assign ZAT features to road segments based on spatial overlay. |
| method | `RoadGraphConstructor.build_node_attributes` | 452 | Build comprehensive node attribute dictionary for the graph. |
| method | `RoadGraphConstructor.find_intersecting_roads` | 504 | Find pairs of intersecting roads using spatial join with tolerance buffer. |
| method | `RoadGraphConstructor.compute_edge_legality` | 545 | Compute edge legality and influence weights for all intersecting road pairs. |
| method | `RoadGraphConstructor.build_graph` | 602 | Execute the complete graph construction pipeline. |
| method | `RoadGraphConstructor.save_graph` | 691 | Save the constructed graph in multiple formats. |
| method | `RoadGraphConstructor.get_roads_within_hops` | 729 | Get all roads reachable from a seed road within k hops. |
| method | `RoadGraphConstructor.get_roads_within_distance` | 751 | Get all roads reachable from a seed road within a distance threshold. |

## `src/preprocessing/features/incidents.py`

| Type | Name | Line | Purpose |
|---|---|---:|---|
| class | `IncidentsProcessor` | 9 | A class for processing traffic incidents data and assigning them to road segments. |
| method | `IncidentsProcessor.__init__` | 29 | Initialize the IncidentsProcessor with data paths and processing parameters. |
| method | `IncidentsProcessor.load_road_network` | 65 | Load and process the road network from GeoJSON. |
| method | `IncidentsProcessor.load_incidents_data` | 110 | Load and preprocess incidents data from CSV. |
| method | `IncidentsProcessor.assign_incidents_to_roads` | 146 | Assign incidents to road segments using spatial join. |
| method | `IncidentsProcessor.aggregate_by_incident_type` | 188 | Aggregate incidents by road, timeframe, and incident type. |
| method | `IncidentsProcessor.aggregate_by_case_specifics` | 225 | Aggregate incidents by road, timeframe, and case specifics. |
| method | `IncidentsProcessor.merge_incident_features` | 264 | Merge incident type and case-specific features into a single dataset. |
| method | `IncidentsProcessor.process_and_save` | 293 | Execute the complete incidents data processing pipeline and save results. |

## `src/preprocessing/features/irregularities.py`

| Type | Name | Line | Purpose |
|---|---|---:|---|
| class | `IrregularitiesProcessor` | 10 | A class for processing Waze traffic irregularities data with spatial-temporal aggregation. |
| method | `IrregularitiesProcessor.__init__` | 32 | Initialize the IrregularitiesProcessor with data paths and processing parameters. |
| method | `IrregularitiesProcessor.quarter_slices` | 104 | Generate time slices aligned to 15-minute intervals between start and end. |
| method | `IrregularitiesProcessor.discover_parquet_files` | 145 | Discover all parquet files in the input folder. |
| method | `IrregularitiesProcessor.load_road_network` | 157 | Load and process the road network with buffering for spatial joins. |
| method | `IrregularitiesProcessor.load_and_preprocess_batch` | 203 | Load and preprocess a batch of irregularities parquet files. |
| method | `IrregularitiesProcessor.temporal_slicing` | 340 | Slice irregularities into 15-minute temporal intervals. |
| method | `IrregularitiesProcessor.spatial_intersection` | 433 | Perform spatial intersection between irregularities and road network. |
| method | `IrregularitiesProcessor.weighted_aggregation` | 494 | Perform weighted aggregation of irregularities by road and timeframe. |
| method | `IrregularitiesProcessor.process_batch` | 602 | Process a single batch of irregularities files. |
| method | `IrregularitiesProcessor.process_and_save` | 635 | Execute the complete irregularities data processing pipeline and save results. |

## `src/preprocessing/features/jams.py`

| Type | Name | Line | Purpose |
|---|---|---:|---|
| class | `JamsProcessor` | 12 | A class for processing Waze traffic jam data and assigning jams to road segments. |
| method | `JamsProcessor.__init__` | 34 | Initialize the JamsProcessor with data paths and processing parameters. |
| method | `JamsProcessor.load_road_network` | 74 | Load and process the road network with buffered geometries for spatial joins. |
| method | `JamsProcessor.discover_jam_files` | 125 | Discover all parquet files in the jams data directory. |
| method | `JamsProcessor.load_and_prepare_jam_file` | 137 | Load a single jam data file and prepare it for processing. |
| method | `JamsProcessor.aggregate_jam_metadata` | 201 | Aggregate metadata for jams with the same UUID. |
| method | `JamsProcessor.intersect_jams_with_roads` | 244 | Perform spatial intersection between jams and buffered road network. |
| method | `JamsProcessor.aggregate_to_timeframes` | 310 | Aggregate jam data to specified temporal resolution by road segment. |
| method | `JamsProcessor.process_file_chunk` | 377 | Process a chunk of jam data files. |
| method | `JamsProcessor.process_and_save` | 434 | Execute the complete jam data processing pipeline with chunked processing. |

## `src/preprocessing/features/pmts_points.py`

| Type | Name | Line | Purpose |
|---|---|---:|---|
| class | `PMTSPointsProcessor` | 16 | A class for processing PMTS (Traffic Management Plans) data and assigning |
| method | `PMTSPointsProcessor.__init__` | 56 | Initialize the PMTSProcessor with data paths and processing parameters. |
| method | `PMTSPointsProcessor.load_and_buffer_road_network` | 100 | Load and process the road network with buffering for spatial joins. |
| method | `PMTSPointsProcessor.generate_simple_timeframes` | 142 | Generate 15-minute timeframes for a simple schedule (parks). |
| method | `PMTSPointsProcessor.normalize_hcie_text` | 182 | Normalize HCIE (schedule) text for consistent parsing. |
| method | `PMTSPointsProcessor.extract_weekday_ranges` | 228 | Extract weekday numbers from schedule text. |
| method | `PMTSPointsProcessor.extract_timeframes` | 272 | Extract time ranges from schedule text. |
| method | `PMTSPointsProcessor.parse_schedule` | 302 | Parse complex schedule text into a weekday-to-timerange mapping. |
| method | `PMTSPointsProcessor.time_to_seconds` | 357 | Convert time object to seconds since midnight. |
| method | `PMTSPointsProcessor.generate_complex_timeframes` | 362 | Generate 15-minute timeframes for complex schedules with weekday-specific times. |
| method | `PMTSPointsProcessor.generate_monthly_splits` | 431 | Split long-duration records into monthly chunks for efficient processing. |
| method | `PMTSPointsProcessor.process_coi_parques` | 471 | Process COI parks data and save to parquet. |
| method | `PMTSPointsProcessor.process_coi_puntos` | 562 | Process COI points data and save to monthly parquet files. |
| method | `PMTSPointsProcessor.process_and_save` | 676 | Execute the complete PMTS data processing pipeline and save results. |

## `src/preprocessing/features/pmts_tracts.py`

| Type | Name | Line | Purpose |
|---|---|---:|---|
| class | `PMTSTractsProcessor` | 16 | A class for processing PMTS (Traffic Management Plan) data and assigning interventions to road segments. |
| method | `PMTSTractsProcessor.__init__` | 46 | Initialize the PMTSProcessor with data paths and processing parameters. |
| method | `PMTSTractsProcessor.load_and_prepare_road_network` | 88 | Load and process the road network for spatial joins with PMTS data. |
| method | `PMTSTractsProcessor.normalize_hcie_text` | 133 | Normalize and clean HCIE (schedule) text for parsing. |
| method | `PMTSTractsProcessor.extract_weekday_ranges` | 178 | Extract weekday ranges from normalized schedule text. |
| method | `PMTSTractsProcessor.extract_timeframes` | 222 | Extract time ranges from schedule text. |
| method | `PMTSTractsProcessor.parse_schedule_row` | 258 | Parse a complete HCIE schedule string into a structured schedule dictionary. |
| method | `PMTSTractsProcessor.time_to_seconds` | 321 | Convert a time object to seconds since midnight. |
| method | `PMTSTractsProcessor.generate_timeframes_vectorized_per_row` | 334 | Generate all valid 15-minute timeframes for a PMTS intervention based on its schedule. |
| method | `PMTSTractsProcessor.generate_timeframes_vectorized_fully` | 406 | Alternative implementation of timeframe generation using numpy vectorization. |
| method | `PMTSTractsProcessor.generate_monthly_splits` | 464 | Split PMTS interventions into monthly chunks for efficient processing. |
| method | `PMTSTractsProcessor.process_coi_tramo` | 511 | Process COI Tramo (tract-based interventions) data. |
| method | `PMTSTractsProcessor.process_coos_punto` | 645 | Process COOS Punto (point-based interventions) data. |
| method | `PMTSTractsProcessor.process_and_save` | 726 | Execute the complete PMTS data processing pipeline and save results. |

## `src/preprocessing/features/road_state.py`

| Type | Name | Line | Purpose |
|---|---|---:|---|
| class | `RoadStateProcessor` | 9 | A class for processing road condition data and assigning it to road network segments. |
| method | `RoadStateProcessor.__init__` | 42 | Initialize the RoadStateProcessor with data paths and processing parameters. |
| method | `RoadStateProcessor.load_and_buffer_road_network` | 80 | Load and process the road network with buffering for spatial overlay. |
| method | `RoadStateProcessor.load_and_process_road_state` | 140 | Load and process road condition data from shapefile. |
| method | `RoadStateProcessor.spatial_overlay_and_rank` | 191 | Perform spatial overlay between road conditions and network, then rank overlaps. |
| method | `RoadStateProcessor.select_best_matches` | 251 | Select the best matching road condition for each road segment. |
| method | `RoadStateProcessor.process_and_save` | 303 | Execute the complete road state processing pipeline and save results. |
| method | `RoadStateProcessor.get_road_state_summary` | 363 | Generate a summary of road condition distributions. |

## `src/preprocessing/features/weather.py`

| Type | Name | Line | Purpose |
|---|---|---:|---|
| class | `WeatherProcessor` | 8 | A class for processing weather data and assigning weather stations to road segments. |
| method | `WeatherProcessor.__init__` | 25 | Initialize the WeatherProcessor with data paths and processing parameters. |
| method | `WeatherProcessor.process_weather_dataframe` | 64 | Load and process a single weather variable dataset. |
| method | `WeatherProcessor.load_road_network` | 136 | Load and process the road network from a GeoJSON file. |
| method | `WeatherProcessor.assign_stations_to_roads` | 191 | Assign weather stations to the nearest road segments using spatial joins. |
| method | `WeatherProcessor.merge_station_assignments` | 274 | Merge station assignments from multiple weather variables into a single dataframe. |
| method | `WeatherProcessor.process_and_save` | 314 | Execute the complete weather data processing pipeline and save results. |

## `src/preprocessing/imputation.py`

| Type | Name | Line | Purpose |
|---|---|---:|---|
| function | `load_graph` | 11 | Loads a graph from a pickle file. |
| function | `load_and_process_irregularities` | 19 | Loads all parquet files from a directory, concatenates them, |
| function | `get_disconnected_nodes` | 47 | Identifies all nodes not in the largest connected component. |
| function | `get_street_types` | 66 | Extracts the 'MVITIPO' attribute for each node into a DataFrame. |
| function | `calculate_mean_features_by_street_type` | 79 | Computes the mean 'trend' and 'regularSpeed' for each street type. |
| function | `compute_neighbor_features` | 95 | For each node, calculates the mean features of its neighbors within k_hops |
| function | `impute_missing_features` | 147 | Fills NaN values in the feature DataFrame using the mean values |
| function | `generate_neighbor_features_from_files` | 178 | Full pipeline to load data and generate the neighbor features DataFrame. |

## `src/preprocessing/merging.py`

| Type | Name | Line | Purpose |
|---|---|---:|---|
| class | `SparseDatasetGenerator` | 12 |  |
| method | `SparseDatasetGenerator.__init__` | 13 |  |
| method | `SparseDatasetGenerator.temporal_dataframe_to_coo` | 33 | Generate a sparse dataset from the given DataFrame. |
| method | `SparseDatasetGenerator.static_dataframe_to_coo` | 121 | Generate a sparse dataset from the given static DataFrame. |
| method | `SparseDatasetGenerator.generate_sparse_dataset` | 185 | Generate a sparse dataset from multiple DataFrames stored in files. |
| method | `SparseDatasetGenerator._clean_node_attributes` | 196 | Cleans the node attributes DataFrame by dropping unnecessary columns and |
| method | `SparseDatasetGenerator._process_weather_data` | 328 | Process weather data into efficient format for the TrafficDataset. |

## `src/preprocessing/normalize.py`

| Type | Name | Line | Purpose |
|---|---|---:|---|
| function | `compute_normalization_constants` | 9 | Compute robust normalization constants for each feature in a sparse HDF5 dataset. |
| function | `compute_all_normalization_constants` | 86 | Compute normalization constants for all datasets in a directory. |
| function | `create_flat_normalization_dict` | 223 | Create a flat dictionary of normalization constants for easy lookup. |

## `src/rebalance_sampling.py`

| Type | Name | Line | Purpose |
|---|---|---:|---|
| class | `SeedTimeBucketArtifact` | 23 |  |
| method | `SeedTimeBucketArtifact.to_parquet` | 27 |  |
| method | `SeedTimeBucketArtifact.write_metadata` | 32 |  |
| function | `_log_metrics` | 38 |  |
| class | `PrebucketedSeedTimeBatchSampler` | 43 | Batch sampler for precomputed `(seed, time_idx)` rows. |
| method | `PrebucketedSeedTimeBatchSampler.__init__` | 56 |  |
| method | `PrebucketedSeedTimeBatchSampler.__len__` | 90 |  |
| method | `PrebucketedSeedTimeBatchSampler.__iter__` | 93 |  |
| function | `_parse_bucket_quantiles` | 112 |  |
| function | `_ensure_output_mirror_paths` | 122 |  |
| function | `_bucketize_by_quantiles` | 136 |  |
| function | `_load_seed_candidates` | 148 |  |
| function | `_default_seed_candidates` | 170 |  |
| function | `_load_time_indices` | 176 |  |
| function | `_materialize_input_uri` | 201 |  |
| function | `_upload_file_to_s3` | 215 |  |
| function | `_append_suffix_to_path` | 225 |  |
| function | `_append_suffix_to_s3_uri` | 230 |  |
| function | `_compute_future_high_fraction_for_seed` | 242 | Score a `(seed_id, time_idx)` pair using only the seed neighborhood. |
| function | `build_seed_time_bucket_artifact` | 301 |  |
| function | `_build_partial_artifact` | 405 |  |
| function | `parse_args` | 443 |  |
| function | `_slice_for_shard` | 476 |  |
| function | `main` | 484 |  |

## `src/seed_sharding.py`

| Type | Name | Line | Purpose |
|---|---|---:|---|
| function | `parse_args` | 10 |  |
| function | `main` | 18 |  |

## `src/seed_strategy.py`

| Type | Name | Line | Purpose |
|---|---|---:|---|
| class | `SeedStrategyArtifact` | 13 |  |
| method | `SeedStrategyArtifact.to_csv` | 17 |  |
| method | `SeedStrategyArtifact.write_metadata` | 22 |  |
| function | `load_optimal_seeds` | 28 |  |
| function | `load_top_congestion_nodes` | 36 |  |
| function | `build_seed_strategy_artifact` | 57 |  |
| function | `parse_args` | 99 |  |
| function | `main` | 126 |  |

## `src/seeds_optimizer.py`

| Type | Name | Line | Purpose |
|---|---|---:|---|
| class | `SeedsOptimizer` | 9 | Optimizer to find the minimum set of seed nodes for traffic dataset batching. |
| method | `SeedsOptimizer.__init__` | 25 | Initialize the SeedsOptimizer. |
| method | `SeedsOptimizer._load_graph` | 55 | Load a NetworkX graph from a pickle file. |
| method | `SeedsOptimizer._compute_bfs_neighborhood` | 61 | Compute the BFS neighborhood of a seed node up to `node_batch_size` nodes. |
| method | `SeedsOptimizer._compute_all_neighborhoods` | 84 | Precompute BFS neighborhoods for all nodes in the graph. |
| method | `SeedsOptimizer.find_optimal_seeds` | 96 | Find the minimum set of seed nodes whose BFS neighborhoods cover all graph nodes. |
| method | `SeedsOptimizer.verify_coverage` | 177 | Verify that the selected seeds cover all nodes in the graph. |
| method | `SeedsOptimizer.save_seeds` | 210 | Save the list of seed nodes to a pickle file. |
| method | `SeedsOptimizer.print_summary` | 234 | Print a summary of the optimization results. |

## `src/sparse_dataloader.py`

| Type | Name | Line | Purpose |
|---|---|---:|---|
| class | `TrafficDataset` | 16 | Custom Dataset class for the traffic forecasting problem with memory-efficient |
| method | `TrafficDataset.__init__` | 36 | Args: |
| method | `TrafficDataset._load_normalization_constants` | 266 | Load normalization constants and build tensors for efficient normalization. |
| method | `TrafficDataset.get_first_n_reachable` | 346 | For every node in a graph, finds the first N reachable neighboring nodes |
| method | `TrafficDataset._get_future_dynamic_feature_idxs` | 383 | Maps future feature names to their corresponding indices in each dynamic |
| method | `TrafficDataset.reasign_seed` | 424 | Reassigns the node seed for batch selection. |
| method | `TrafficDataset.set_seed_list` | 434 | Sets a stateless seed list for multi-worker data loading. |
| method | `TrafficDataset._load_graph` | 448 | Loads the graph from a pickle file. |
| method | `TrafficDataset._load_static_features` | 465 | Loads static features from HDF5 and returns a dense tensor. |
| method | `TrafficDataset._load_weather_data` | 482 | Load weather data for efficient access during training. |
| method | `TrafficDataset._load_time_index_map` | 502 | Load canonical timestamps from target metadata. |
| method | `TrafficDataset._build_temporal_features` | 517 | Build deterministic temporal covariates from the timestamp index. |
| method | `TrafficDataset._resolve_timestamp` | 557 | Resolve a timestamp for a given time index, tolerating small metadata gaps. |
| method | `TrafficDataset._identify_imputation_features` | 569 | Identify which HDF5 files and feature indices correspond to the imputation features. |
| method | `TrafficDataset._precompute_time_indices` | 590 | Pre-computes the start and end row indices for every possible sample |
| method | `TrafficDataset.__len__` | 630 | Returns the number of time windows available for the CURRENT seed node only. |
| method | `TrafficDataset.total_training_samples` | 653 | Returns the total number of training samples across all node batches and time windows. |
| method | `TrafficDataset._get_time_and_node_indices` | 666 | Given a global index, computes the corresponding time index and |
| method | `TrafficDataset._get_dynamic_past_features` | 697 | Fetches and populates the past dynamic features for the given time window |
| method | `TrafficDataset._get_dynamic_future_features` | 809 | Fetches and populates the future dynamic features for the given time window |
| method | `TrafficDataset._get_weather_features` | 894 | Fetches and populates weather features for the given time window and node batch. |
| method | `TrafficDataset._apply_normalization` | 998 | Apply normalization to all feature tensors and targets. |
| method | `TrafficDataset.__getitem__` | 1107 | Returns a single sample, which may be a subset of nodes for a specific |

## `src/temporal_split.py`

| Type | Name | Line | Purpose |
|---|---|---:|---|
| class | `TemporalSubset` | 6 |  |
| method | `TemporalSubset.__init__` | 7 | A temporal subset of a TrafficDataset, defined by a range of time indices. |
| method | `TemporalSubset.__len__` | 38 | Returns the number of time windows in this temporal subset. |
| method | `TemporalSubset.total_training_samples` | 43 | Total samples across all node batches for this temporal subset. |
| method | `TemporalSubset.node_seed` | 48 | Returns the current seed node from the underlying dataset. |
| method | `TemporalSubset.total_possible_node_batches` | 53 | Returns the number of node batches from the underlying dataset. |
| method | `TemporalSubset.num_nodes` | 58 | Returns the number of nodes from the underlying dataset. |
| method | `TemporalSubset.node_batch_size` | 63 | Returns the node batch size from the underlying dataset. |
| method | `TemporalSubset.__getitem__` | 67 | Fetches a sample from the subset. |
| method | `TemporalSubset.reasign_seed` | 87 | Reassigns the seed node in the underlying dataset. |
| function | `temporal_train_test_split` | 96 | Splits a TrafficDataset into train and test sets based on time. |
| function | `temporal_train_val_test_split` | 154 | Splits a TrafficDataset into train, validation, and test sets based on time. |

## `src/time_index_strategy.py`

| Type | Name | Line | Purpose |
|---|---|---:|---|
| function | `build_time_index_strategy` | 11 |  |
| function | `parse_args` | 27 |  |
| function | `main` | 37 |  |

## `src/training/loop.py`

| Type | Name | Line | Purpose |
|---|---|---:|---|
| function | `_sync_if_cuda` | 14 |  |
| function | `train_one_epoch` | 19 |  |

## `src/training/losses.py`

| Type | Name | Line | Purpose |
|---|---|---:|---|
| function | `_horizon_weights_like` | 6 | Build per-forecast-step weights broadcastable to ``pred``. |
| function | `_weighted_nodewise_pearson_loss` | 33 | Weighted node-wise Pearson shape loss over the forecast horizon. |
| function | `masked_mse` | 114 | Composite loss for congestion level prediction. |

## `src/training_entrypoint.py`

| Type | Name | Line | Purpose |
|---|---|---:|---|
| function | `_str2bool` | 51 |  |
| function | `_tensor_stats` | 62 |  |
| function | `_sanitize_` | 70 |  |
| function | `_assert_edge_index_ok` | 76 |  |
| function | `_assert_finite` | 89 |  |
| function | `_log_metrics` | 95 |  |
| function | `_broadcast_node_mask` | 106 |  |
| function | `_apply_node_mask_to_target` | 119 |  |
| function | `_node_valid_mask_for_target` | 129 |  |
| function | `_masked_tensor_mse` | 141 |  |
| function | `_maybe_split_losses` | 155 |  |
| function | `_extract_level_tensors` | 191 |  |
| function | `_get_level_scale` | 206 |  |
| function | `_to_real_level_space` | 212 |  |
| function | `_compute_operational_metrics` | 223 |  |
| function | `_init_operational_accumulator` | 291 |  |
| function | `_update_operational_accumulator` | 311 |  |
| function | `_finalize_operational_accumulator` | 366 |  |
| function | `_num_batches` | 410 |  |
| function | `_build_seed_list` | 420 |  |
| function | `_ensure_seed_list_length` | 432 |  |
| function | `_format_bucket_ratios` | 442 |  |
| function | `parse_args` | 449 |  |
| function | `check_data_channels` | 808 | Check if the data channels for training and validation exist. |
| function | `_select_dynamic_files` | 829 |  |
| function | `_required_weather_files` | 856 |  |
| function | `_copy_if_exists` | 870 |  |
| function | `_copy_file` | 877 |  |
| function | `_resolve_seed_time_artifact_path` | 882 | Resolve a seed-time artifact path with the following precedence: |
| function | `_resolve_optional_artifact_path` | 943 |  |
| function | `_series_to_bool` | 981 |  |
| function | `_build_region_node_mask` | 995 |  |
| function | `_batch_region_node_mask` | 1033 |  |
| function | `cache_training_data` | 1046 |  |
| function | `setup_sagemaker_paths` | 1087 | Set up all required file paths from the SageMaker training data channel. |
| function | `train_model` | 1167 | Full training pipeline: |

## `src/training_utils.py`

| Type | Name | Line | Purpose |
|---|---|---:|---|
| function | `pad_collate` | 14 |  |
| class | `SeededBatchSampler` | 118 | Batch sampler that keeps all items in a batch tied to the same seed index. |
| method | `SeededBatchSampler.__init__` | 126 |  |
| method | `SeededBatchSampler.__iter__` | 162 |  |
| method | `SeededBatchSampler.__len__` | 176 |  |
| class | `SeedTimeEpochPlan` | 181 |  |
| method | `SeedTimeEpochPlan.num_batches` | 188 |  |
| method | `SeedTimeEpochPlan.num_samples` | 192 |  |
| function | `_materialize_parquet_path` | 196 |  |
| function | `_normalize_bucket_ratios` | 210 |  |
| function | `_allocate_bucket_counts` | 231 |  |
| function | `_sample_time_indices_for_seed` | 245 |  |
| class | `SeedTimeEpochPlanner` | 275 | Builds epoch plans where each batch uses a single seed and multiple time_idx. |
| method | `SeedTimeEpochPlanner.__init__` | 283 |  |
| method | `SeedTimeEpochPlanner.build_epoch_plan` | 322 |  |
| class | `PlannedSeedTimeBatchSampler` | 356 |  |
| method | `PlannedSeedTimeBatchSampler.__init__` | 357 |  |
| method | `PlannedSeedTimeBatchSampler.__iter__` | 365 |  |
| method | `PlannedSeedTimeBatchSampler.__len__` | 370 |  |
| class | `Node50Picker` | 375 |  |
| method | `Node50Picker.pick` | 380 |  |
| function | `build_node_50_picker` | 392 |  |
| function | `_default_node_key` | 413 | Extracts a node id from a seed like '50008544_TF.1' -> 50008544. |
| class | `EpochUniquePicker` | 426 |  |
| method | `EpochUniquePicker.__post_init__` | 439 |  |
| method | `EpochUniquePicker.start_epoch` | 445 |  |
| method | `EpochUniquePicker._draw_from` | 452 |  |
| method | `EpochUniquePicker.pick` | 465 |  |
| method | `EpochUniquePicker.remaining` | 495 |  |

## `src/utils/logging.py`

| Type | Name | Line | Purpose |
|---|---|---:|---|
| function | `_now_str` | 5 |  |
| function | `log` | 12 |  |

## `src/utils/pickers.py`

| Type | Name | Line | Purpose |
|---|---|---:|---|
| class | `Node50Picker` | 9 |  |
| method | `Node50Picker.pick` | 14 |  |
| function | `build_node_50_picker` | 29 |  |
| class | `EpochUniquePicker` | 45 | Wraps a base picker and prevents selecting the same *node id* twice within an epoch. |
| method | `EpochUniquePicker.__init__` | 52 |  |
| method | `EpochUniquePicker.start_epoch` | 63 |  |
| method | `EpochUniquePicker.pick` | 66 |  |

## `src/utils/tensor_stats.py`

| Type | Name | Line | Purpose |
|---|---|---:|---|
| function | `tensor_stats` | 4 |  |

## `scripts/build_corridor_region_map.py`

| Type | Name | Line | Purpose |
|---|---|---:|---|
| function | `parse_args` | 15 |  |
| function | `extract_lpkx` | 57 |  |
| function | `load_graph_nodes` | 76 |  |
| function | `build_mvi_road_ids` | 82 |  |
| function | `main` | 94 |  |

## `scripts/build_region_seed_strategies.py`

| Type | Name | Line | Purpose |
|---|---|---:|---|
| function | `parse_args` | 12 |  |
| function | `load_coverage_seeds` | 41 |  |
| function | `build_region_strategy` | 56 |  |
| function | `main` | 107 |  |

## `scripts/compute_regional_seed_cover.py`

| Type | Name | Line | Purpose |
|---|---|---:|---|
| function | `load_graph` | 27 |  |
| function | `bfs_k_hop_targets` | 32 |  |
| function | `greedy_max_cover` | 50 |  |
| function | `write_seed_json` | 128 |  |
| function | `main` | 139 |  |

## `scripts/compute_seed_cover.py`

| Type | Name | Line | Purpose |
|---|---|---:|---|
| function | `bfs_k_hop_nodes` | 20 |  |
| function | `compute_seed_cover` | 35 |  |
| function | `main` | 55 |  |

## `scripts/export_calibration_artifacts.py`

| Type | Name | Line | Purpose |
|---|---|---:|---|
| function | `load_frame` | 24 |  |
| function | `main` | 32 |  |

## `scripts/finalize_region_seed_time_artifacts.py`

| Type | Name | Line | Purpose |
|---|---|---:|---|
| function | `parse_args` | 13 |  |
| function | `parse_quantiles` | 38 |  |
| function | `download_region_outputs` | 48 |  |
| function | `select_region_input_parts` | 64 |  |
| function | `rebucket` | 128 |  |
| function | `upload_file` | 152 |  |
| function | `main` | 157 |  |

## `scripts/launch_region_training_jobs.py`

| Type | Name | Line | Purpose |
|---|---|---:|---|
| function | `parse_args` | 21 |  |
| function | `parse_model_configs` | 75 |  |
| function | `upload_file` | 90 |  |
| function | `tar_and_upload_source` | 95 |  |
| function | `metric_definitions` | 115 |  |
| function | `base_hyperparameters` | 136 |  |
| function | `build_request` | 184 |  |
| function | `main` | 258 |  |

## `scripts/prepare_region_seed_time_jobs.py`

| Type | Name | Line | Purpose |
|---|---|---:|---|
| function | `parse_args` | 44 |  |
| function | `split_strategy` | 107 |  |
| function | `upload_file` | 121 |  |
| function | `tar_and_upload_source` | 126 |  |
| function | `build_request` | 143 |  |
| function | `count_active_seedtime_jobs` | 221 |  |
| function | `count_active_seedtime_jobs_by_instance_type` | 237 |  |
| function | `main` | 253 |  |

## `scripts/qmap_output_transform.py`

| Type | Name | Line | Purpose |
|---|---|---:|---|
| class | `QuantileMapCalibrator` | 22 |  |
| method | `QuantileMapCalibrator.__init__` | 23 |  |
| method | `QuantileMapCalibrator.transform` | 27 |  |
| class | `HurdleQMapCalibrator` | 38 |  |
| method | `HurdleQMapCalibrator.__init__` | 39 |  |
| method | `HurdleQMapCalibrator.transform` | 51 |  |
| function | `load_transform_config` | 67 |  |
| function | `load_qmap_calibrator` | 71 |  |
| function | `load_hurdle_calibrator` | 81 |  |
| function | `apply_qmap_global` | 93 |  |
| function | `apply_hybrid_transform` | 101 |  |
| function | `apply_hurdle_qmap` | 115 |  |
| function | `apply_selected_transform` | 123 |  |

## `scripts/realtime_eval_backfill.py`

| Type | Name | Line | Purpose |
|---|---|---:|---|
| class | `S3Object` | 34 |  |
| method | `S3Object.name` | 40 |  |
| function | `run_aws` | 44 |  |
| function | `parse_s3_ls` | 54 |  |
| function | `list_objects` | 76 |  |
| function | `parse_prediction_file_dt` | 81 |  |
| function | `parse_jams_file_dt` | 89 |  |
| function | `prediction_start_utc` | 97 |  |
| function | `latest_observed_utc_from_jams_name` | 102 |  |
| function | `overlap_steps` | 107 |  |
| function | `select_prediction_files` | 115 |  |
| function | `select_jams_files` | 134 |  |
| function | `download` | 151 |  |
| function | `load_jams` | 159 |  |
| function | `metric_bundle` | 170 |  |
| function | `fit_linear_calibration` | 196 |  |
| function | `apply_calibration` | 203 |  |
| function | `build_report` | 207 |  |
| function | `main` | 244 |  |

## `scripts/realtime_eval_last7d_dense.py`

| Type | Name | Line | Purpose |
|---|---|---:|---|
| function | `metric_bundle` | 12 |  |
| function | `main` | 36 |  |

## `scripts/realtime_eval_metrics.py`

| Type | Name | Line | Purpose |
|---|---|---:|---|
| class | `QuantileMapCalibrator` | 24 |  |
| method | `QuantileMapCalibrator.transform` | 28 |  |
| method | `QuantileMapCalibrator.to_dict` | 38 |  |
| method | `QuantileMapCalibrator.from_dict` | 45 |  |
| class | `HurdleQMapCalibrator` | 53 |  |
| method | `HurdleQMapCalibrator.transform` | 59 |  |
| method | `HurdleQMapCalibrator.to_dict` | 74 |  |
| method | `HurdleQMapCalibrator.from_dict` | 83 |  |
| function | `discretize_level` | 92 |  |
| function | `fit_quantile_map` | 96 |  |
| function | `fit_zero_gate` | 112 |  |
| function | `fit_hurdle_qmap` | 134 |  |
| function | `apply_hybrid_calibration` | 159 |  |
| function | `metric_row` | 173 |  |
| function | `build_model_predictions` | 186 |  |
| function | `summarize_overall` | 204 |  |
| function | `summarize_by_level` | 213 |  |
| function | `summarize_by_timestep_and_level` | 225 |  |
| function | `confusion_matrix_table` | 247 |  |
| function | `fit_from_train_df` | 261 |  |
| function | `save_calibrator_artifact` | 265 |  |
| function | `load_calibrator_artifact` | 275 |  |
| function | `export_calibration_bundle` | 280 |  |

## `scripts/realtime_eval_sample.py`

| Type | Name | Line | Purpose |
|---|---|---:|---|
| class | `S3Object` | 31 |  |
| function | `run_aws` | 37 |  |
| function | `parse_s3_ls` | 47 |  |
| function | `list_objects` | 69 |  |
| function | `download_object` | 74 |  |
| function | `load_jams` | 81 |  |
| function | `load_predictions` | 91 |  |
| function | `build_eval_dataset` | 100 |  |
| function | `summarize_dataset` | 130 |  |
| function | `main` | 155 |  |

## `scripts/retry_failed_seed_time_jobs.py`

| Type | Name | Line | Purpose |
|---|---|---:|---|
| function | `parse_args` | 14 |  |
| function | `active_training_jobs` | 32 |  |
| function | `latest_failed_jobs` | 44 |  |
| function | `main` | 53 |  |

## `scripts/reuse_existing_seed_time_by_region.py`

| Type | Name | Line | Purpose |
|---|---|---:|---|
| function | `parse_args` | 11 |  |
| function | `main` | 36 |  |

## `tests/preprocessing/test_imputation.py`

| Type | Name | Line | Purpose |
|---|---|---:|---|
| function | `sample_graph` | 13 | Creates a sample graph with distinct components and node attributes. |
| function | `sample_mean_features` | 32 | Creates a sample DataFrame of mean features for specific nodes. |
| function | `temp_dir` | 40 | Creates a temporary directory for file-based tests. |
| function | `test_load_graph` | 45 | Tests loading a graph from a pickle file. |
| function | `test_load_and_process_irregularities` | 58 | Tests loading and aggregating data from multiple parquet files. |
| function | `test_load_irregularities_no_files` | 97 | Tests that a FileNotFoundError is raised if no parquet files exist. |
| function | `test_get_disconnected_nodes` | 103 | Tests the identification of nodes outside the largest component. |
| function | `test_get_street_types` | 111 | Tests the extraction of street type attributes from the graph. |
| function | `test_calculate_mean_features_by_street_type` | 123 | Tests the aggregation of features by street type for imputation. |
| function | `test_compute_neighbor_features` | 149 | Tests the core logic of computing features from neighbors. |
| function | `test_impute_missing_features` | 180 | Tests the filling of NaN values using mean values by street type. |
| function | `test_generate_neighbor_features_integration` | 216 | Tests the main orchestrator function by mocking the helper functions. |

## `tests/preprocessing/test_normalize.py`

| Type | Name | Line | Purpose |
|---|---|---:|---|
| class | `TestComputeNormalizationConstants` | 13 | Tests for compute_normalization_constants function. |
| method | `TestComputeNormalizationConstants.test_temporal_dataset_basic` | 16 | Test normalization constants for a temporal dataset. |
| method | `TestComputeNormalizationConstants.test_static_dataset_basic` | 54 | Test normalization constants for a static dataset. |
| method | `TestComputeNormalizationConstants.test_percentile_handling` | 87 | Test that percentile-based normalization handles outliers. |
| method | `TestComputeNormalizationConstants.test_negative_values` | 113 | Test that negative values are handled correctly via absolute value. |
| method | `TestComputeNormalizationConstants.test_min_scale_enforcement` | 139 | Test that min_scale prevents division by very small numbers. |
| method | `TestComputeNormalizationConstants.test_empty_feature` | 166 | Test handling of features with no data. |
| class | `TestComputeAllNormalizationConstants` | 195 | Tests for compute_all_normalization_constants function. |
| method | `TestComputeAllNormalizationConstants.test_processes_multiple_datasets` | 198 | Test that function processes static and dynamic datasets. |
| method | `TestComputeAllNormalizationConstants.test_saves_output_files` | 249 | Test that output pickle and summary files are created. |
| method | `TestComputeAllNormalizationConstants.test_skips_missing_datasets` | 277 | Test that missing datasets are gracefully skipped. |
| class | `TestCreateFlatNormalizationDict` | 300 | Tests for create_flat_normalization_dict function. |
| method | `TestCreateFlatNormalizationDict.test_flattens_nested_dict` | 303 | Test that nested dictionary is properly flattened. |
| method | `TestCreateFlatNormalizationDict.test_handles_duplicate_feature_names` | 318 | Test that later values override earlier ones for duplicates. |
| method | `TestCreateFlatNormalizationDict.test_empty_input` | 330 | Test handling of empty input. |
| class | `TestNormalizationIntegration` | 336 | Integration tests for the normalization workflow. |
| method | `TestNormalizationIntegration.test_normalized_values_in_range` | 339 | Test that applying normalization constants keeps values in expected range. |

## `tests/test_prediction_reconstruction.py`

| Type | Name | Line | Purpose |
|---|---|---:|---|
| function | `mock_dataset` | 15 | Create a mock dataset with minimal required attributes. |
| function | `mock_metadata` | 51 | Create mock metadata file with time mappings. |
| function | `test_reconstruct_single_batch` | 72 | Test reconstruction of a single batch of predictions. |
| function | `test_reconstruct_single_sample` | 112 | Test reconstruction of a single sample (no batch dimension). |
| function | `test_reconstruction_temporal_alignment` | 130 | Test that predictions align correctly with future timestamps. |
| function | `test_reconstruction_preserves_prediction_values` | 161 | Test that prediction values are correctly preserved. |
| function | `test_reconstruction_with_custom_column_name` | 182 | Test that custom column names work correctly. |
| function | `test_reconstruction_multiple_samples` | 200 | Test reconstruction with multiple samples in a batch. |
| function | `test_reconstruction_node_ordering` | 228 | Test that node ordering is preserved correctly. |

## `tests/test_regional_training_masks.py`

| Type | Name | Line | Purpose |
|---|---|---:|---|
| function | `test_build_region_node_mask_selects_central_nodes` | 13 |  |
| function | `test_apply_node_mask_to_target_sets_non_region_nodes_to_nan` | 33 |  |
| function | `test_regional_masked_mse_ignores_non_region_nodes` | 42 |  |
| function | `test_node_valid_mask_broadcasts_to_full_target_shape` | 52 |  |
| function | `test_composite_masked_mse_uses_explicit_region_mask` | 63 |  |
| function | `test_masked_mse_handles_empty_finite_target` | 75 |  |

## `tests/test_sparse_dataloader.py`

| Type | Name | Line | Purpose |
|---|---|---:|---|
| class | `TestDataGenerator` | 21 | Helper class to generate dummy test data in the correct format. |
| method | `TestDataGenerator.__init__` | 24 |  |
| method | `TestDataGenerator.create_graph` | 40 | Create a simple test graph. |
| method | `TestDataGenerator.create_static_features` | 58 | Create static features HDF5 file. |
| method | `TestDataGenerator.create_dynamic_features` | 76 | Create dynamic features HDF5 file and corresponding metadata. |
| method | `TestDataGenerator.create_target_data` | 113 | Create target data (jams) HDF5 file with metadata. |
| method | `TestDataGenerator.create_weather_stations` | 162 | Create weather stations mapping for testing. |
| method | `TestDataGenerator.create_weather_data` | 187 | Create deterministic weather data for testing. |
| class | `TestTrafficDataset` | 247 | Test suite for TrafficDataset class. |
| method | `TestTrafficDataset.test_data_setup` | 251 | Create test data files in a temporary directory. |
| method | `TestTrafficDataset.test_initialization_basic` | 312 | Test basic dataset initialization. |
| method | `TestTrafficDataset.test_initialization_invalid_future_features` | 345 | Test that an exception is raised with invalid future features. |
| method | `TestTrafficDataset.test_invalid_node_seed` | 367 | Test that initialization raises error with invalid default node_seed. |
| method | `TestTrafficDataset.test_dataset_length` | 388 | Test dataset length calculation. |
| method | `TestTrafficDataset.test_dataset_length_insufficient_data` | 410 | Test dataset length when there's insufficient data. |
| method | `TestTrafficDataset.test_get_item_structure` | 429 | Test that __getitem__ returns correctly structured Data objects. |
| method | `TestTrafficDataset.test_node_batch_mapping` | 504 | Test that node batch mapping works correctly. |
| method | `TestTrafficDataset.test_seed_node_neighborhood` | 535 | Tests that the neighborhood of the seed node is computed correctly. |
| method | `TestTrafficDataset.test_time_and_node_indices` | 560 | Test the _get_time_and_node_indices method. |
| method | `TestTrafficDataset.test_static_features_loading` | 592 | Test static features are loaded correctly with exact numerical validation. |
| method | `TestTrafficDataset.test_node_batching_logic` | 678 | Tests the node batching logic to ensure different batches have different nodes. |
| method | `TestTrafficDataset.test_dynamic_past_features_loading` | 738 | Test that dynamic past features are loaded with correct values. |
| method | `TestTrafficDataset.test_dynamic_future_features_loading` | 840 | Test that dynamic future features are loaded with correct values. |
| method | `TestTrafficDataset.test_target_data_loading` | 959 | Test that target data (y) is loaded with correct values. |
| method | `TestTrafficDataset.test_future_feature_indices` | 1053 | Test future feature index mapping. |
| method | `TestTrafficDataset._create_deterministic_dynamic_features` | 1092 | Create dynamic features with deterministic, predictable values for testing. |
| method | `TestTrafficDataset._create_deterministic_target_data` | 1144 | Create target data with deterministic values for testing. |
| method | `TestTrafficDataset._create_deterministic_static_features` | 1189 | Create static features with deterministic, predictable values for testing. |
| method | `TestTrafficDataset.test_weather_features_initialization` | 1212 | Test that weather features are properly initialized when include_weather=True. |
| method | `TestTrafficDataset.test_get_weather_features_method` | 1256 | Test the _get_weather_features method in isolation. |
| method | `TestTrafficDataset.test_getitem_with_weather_enabled` | 1342 | Test __getitem__ method when weather is enabled. |
| method | `TestTrafficDataset.test_getitem_without_weather` | 1390 | Test __getitem__ method when weather is disabled. |
| method | `TestTrafficDataset.test_weather_data_consistency` | 1421 | Test that weather data loaded in dataset matches the generated test data. |
| method | `TestTrafficDataset.test_weather_features_with_different_time_ranges` | 1473 | Test weather features with different time windows. |
| method | `TestTrafficDataset.test_weather_features_edge_cases` | 1509 | Test weather features with edge cases. |
| method | `TestTrafficDataset.test_weather_loading_without_weather_dir` | 1542 | Test that dataset works correctly when weather is disabled. |
| method | `TestTrafficDataset.test_node_index_safeguard` | 1577 | Test that the node index safeguard properly raises an exception when node indices are out of bounds. |
| method | `TestTrafficDataset.test_imputation_logic` | 1618 | Test that imputation logic correctly fills missing values for trend, speed, and regularSpeed. |
| method | `TestTrafficDataset._create_irregularities_with_missing_values` | 1741 | Create irregularities dataset with missing values for imputation testing. |
| method | `TestTrafficDataset._create_neighbor_features_parquet` | 1790 | Create neighbor features parquet file for imputation testing. |
| class | `TestVelocityTarget` | 1816 | Test suite for the include_velocity_target feature. |
| method | `TestVelocityTarget.velocity_target_setup` | 1820 | Set up test data with velocity target enabled. |
| method | `TestVelocityTarget.single_target_setup` | 1856 | Set up test data with only jam level target (no velocity). |
| method | `TestVelocityTarget.test_velocity_target_disabled_default` | 1890 | Test that velocity target is disabled by default. |
| method | `TestVelocityTarget.test_velocity_target_enabled` | 1916 | Test velocity target when explicitly enabled. |
| method | `TestVelocityTarget.test_velocity_target_values_populated` | 1944 | Test that both jam level and velocity values are correctly populated. |
| method | `TestVelocityTarget.test_velocity_target_error_when_missing` | 1982 | Test that enabling velocity target without velocity data raises an error. |
| method | `TestVelocityTarget.test_target_data_with_velocity_can_still_use_level_only` | 2001 | Test that target data with velocity can still be used with velocity disabled. |

## `tests/test_temporal_split.py`

| Type | Name | Line | Purpose |
|---|---|---:|---|
| class | `TestTemporalSplit` | 19 | Test suite for temporal splitting functionality. |
| method | `TestTemporalSplit.test_temporal_train_test_split_basic` | 22 | Test basic train-test split functionality. |
| method | `TestTemporalSplit.test_temporal_subset_indices` | 54 | Test that temporal subsets access correct time indices. |
| method | `TestTemporalSplit.test_temporal_split_no_overlap_in_indices` | 90 | Test that train and test sets have non-overlapping time indices. |
| method | `TestTemporalSplit.test_seed_reassignment_with_subset` | 116 | Test that seed reassignment works through temporal subsets. |
| method | `TestTemporalSplit.test_train_val_test_split` | 157 | Test three-way split into train, val, and test. |
| method | `TestTemporalSplit.test_temporal_subset_properties` | 198 | Test that temporal subset exposes necessary properties. |
| method | `TestTemporalSplit.test_temporal_subset_index_bounds` | 233 | Test that temporal subset raises errors for out-of-bounds indices. |
| method | `TestTemporalSplit.test_invalid_split_ratios` | 265 | Test that invalid split ratios raise errors. |
| method | `TestTemporalSplit.test_data_consistency_across_split` | 298 | Test that the same absolute time index produces the same data. |

## `tests/test_training_losses.py`

| Type | Name | Line | Purpose |
|---|---|---:|---|
| function | `test_zero_overprediction_penalty_increases_loss_for_false_positive_zero` | 6 |  |
| function | `test_high_underprediction_penalty_increases_loss_for_missed_high_levels` | 39 |  |
| function | `test_horizon_weights_emphasize_short_horizon_errors` | 72 |  |
