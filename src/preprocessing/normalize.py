import os
import pickle as pkl

import h5py
import numpy as np
from tqdm.auto import tqdm


def compute_normalization_constants(
    h5_path: str,
    metadata_path: str,
    percentile: float = 99.0,
    min_scale: float = 1e-8,
) -> dict[str, float]:
    """
    Compute robust normalization constants for each feature in a sparse HDF5 dataset.

    Uses percentile-based normalization to handle outliers. For each feature,
    computes the specified percentile of absolute values as the normalization
    constant. This ensures that:
    - Most values will be in [-1, 1] after normalization
    - Outliers beyond the percentile will be clipped
    - Data resolution is preserved for the majority of values

    Args:
        h5_path: Path to the HDF5 file containing the sparse dataset.
        metadata_path: Path to the pickle file containing feature_idx_map.
        percentile: The percentile of absolute values to use as the normalization
                    constant. Default is 99.0 (99th percentile).
        min_scale: Minimum normalization constant to avoid division by very small
                   numbers. Default is 1e-8.

    Returns:
        Dictionary mapping feature names to their normalization constants.
    """
    # Load metadata to get feature index mapping
    with open(metadata_path, "rb") as f:
        metadata = pkl.load(f)

    feature_idx_map: dict[str, int] = metadata.get("feature_idx_map", {})
    idx_to_feature = {idx: name for name, idx in feature_idx_map.items()}

    # Load the sparse data
    with h5py.File(h5_path, "r") as f:
        data = f["data"][:]  # type: ignore
        assert isinstance(data, np.ndarray)

    # Determine if this is a temporal or static dataset based on number of columns
    # Temporal: [node_idx, time_idx, feature_idx, value] -> 4 columns
    # Static: [node_idx, feature_idx, value] -> 3 columns
    is_temporal = data.shape[1] == 4

    if is_temporal:
        feature_indices = data[:, 2].astype(int)
        values = data[:, 3]
    else:
        feature_indices = data[:, 1].astype(int)
        values = data[:, 2]

    # Compute normalization constants per feature
    norm_constants: dict[str, float] = {}

    for feature_idx in np.unique(feature_indices):
        feature_name = idx_to_feature.get(feature_idx)
        if feature_name is None:
            continue

        # Get all values for this feature
        feature_mask = feature_indices == feature_idx
        feature_values = np.abs(values[feature_mask])

        if len(feature_values) == 0:
            norm_constants[feature_name] = 1.0
            continue

        # Compute the percentile of absolute values
        percentile_value = np.percentile(feature_values, percentile)

        # Ensure minimum scale to avoid numerical issues
        norm_constant = max(percentile_value, min_scale)
        norm_constants[feature_name] = float(norm_constant)

    return norm_constants


def compute_all_normalization_constants(
    sparse_data_dir: str,
    output_path: str,
    static_filename: str = "static_dataset.h5",
    dynamic_filenames: list[str] | None = None,
    target_filename: str = "jams_dataset.h5",
    weather_filenames: list[str] | None = None,
    percentile: float = 99.0,
    min_scale: float = 1e-8,
) -> dict[str, dict[str, float]]:
    """
    Compute normalization constants for all datasets in a directory.

    Args:
        sparse_data_dir: Directory containing the sparse HDF5 datasets.
        output_path: Path where the normalization constants will be saved.
        static_filename: Name of the static dataset file.
        dynamic_filenames: List of dynamic dataset filenames. If None, uses defaults.
        target_filename: Name of the target dataset file.
        weather_filenames: List of weather data filenames. If None, uses defaults.
        percentile: The percentile to use for normalization.
        min_scale: Minimum normalization constant.

    Returns:
        Dictionary mapping dataset names to their feature normalization constants.
    """
    if dynamic_filenames is None:
        dynamic_filenames = [
            "coi_points_dataset.h5",
            "coi_tracts_dataset.h5",
            "irregularities_dataset.h5",
            "events_dataset.h5",
            "incidents_dataset.h5",
            "jam_velocity_dataset.h5",
            "jam_level_count_dataset.h5",
            "jam_delay_dataset.h5",
        ]

    if weather_filenames is None:
        weather_filenames = [
            "weather_wind_direction.parquet",
            "weather_humidity.parquet",
            "weather_wind_speed.parquet",
            "weather_precipitation.parquet",
        ]

    all_norm_constants: dict[str, dict[str, float]] = {}

    # Process static dataset
    static_path = os.path.join(sparse_data_dir, static_filename)
    static_metadata_path = static_path.replace(".h5", "_metadata.pkl")
    if os.path.exists(static_path):
        print(f"Processing static dataset: {static_filename}")
        all_norm_constants["static"] = compute_normalization_constants(
            static_path, static_metadata_path, percentile, min_scale
        )
        print(f"  Found {len(all_norm_constants['static'])} features")

    # Process dynamic datasets
    for filename in tqdm(dynamic_filenames, desc="Processing dynamic datasets"):
        h5_path = os.path.join(sparse_data_dir, filename)
        metadata_path = h5_path.replace(".h5", "_metadata.pkl")

        if not os.path.exists(h5_path):
            print(f"  Skipping {filename} (not found)")
            continue

        dataset_name = filename.replace(".h5", "")
        all_norm_constants[dataset_name] = compute_normalization_constants(
            h5_path, metadata_path, percentile, min_scale
        )
        print(f"  {dataset_name}: {len(all_norm_constants[dataset_name])} features")

    # Process target dataset
    target_path = os.path.join(sparse_data_dir, target_filename)
    target_metadata_path = target_path.replace(".h5", "_metadata.pkl")
    if os.path.exists(target_path):
        print(f"Processing target dataset: {target_filename}")
        all_norm_constants["target"] = compute_normalization_constants(
            target_path, target_metadata_path, percentile, min_scale
        )
        print(f"  Found {len(all_norm_constants['target'])} features")

    # Process weather datasets (these are parquet files, not HDF5)
    weather_norm_constants: dict[str, float] = {}
    for filename in weather_filenames:
        parquet_path = os.path.join(sparse_data_dir, filename)
        if not os.path.exists(parquet_path):
            continue

        print(f"Processing weather data: {filename}")
        try:
            import pandas as pd

            df = pd.read_parquet(parquet_path)
            # Weather data typically has a 'value' column or similar
            # Find numeric columns that contain the actual weather values
            for col in df.select_dtypes(include=[np.number]).columns:
                if col in ["time_idx", "node_idx"]:
                    continue
                values = np.abs(df[col].dropna().values)
                if len(values) > 0:
                    percentile_value = np.percentile(values, percentile)
                    weather_norm_constants[col] = float(
                        max(percentile_value, min_scale)
                    )
        except Exception as e:
            print(f"  Warning: Could not process {filename}: {e}")

    if weather_norm_constants:
        all_norm_constants["weather"] = weather_norm_constants
        print(f"  Weather features: {len(weather_norm_constants)}")

    # Save the normalization constants
    with open(output_path, "wb") as f:
        pkl.dump(all_norm_constants, f)
    print(f"\nSaved normalization constants to {output_path}")

    # Also save a human-readable summary
    summary_path = output_path.replace(".pkl", "_summary.txt")
    with open(summary_path, "w") as f:
        f.write("Normalization Constants Summary\n")
        f.write("=" * 50 + "\n")
        f.write(f"Percentile used: {percentile}\n")
        f.write(f"Minimum scale: {min_scale}\n\n")

        for dataset_name, features in all_norm_constants.items():
            f.write(f"\n{dataset_name.upper()}\n")
            f.write("-" * 30 + "\n")
            for feature_name, constant in sorted(features.items()):
                f.write(f"  {feature_name}: {constant:.6g}\n")

    print(f"Saved summary to {summary_path}")

    return all_norm_constants


def create_flat_normalization_dict(
    all_norm_constants: dict[str, dict[str, float]],
) -> dict[str, float]:
    """
    Create a flat dictionary of normalization constants for easy lookup.

    This merges all dataset-specific constants into a single dictionary.
    If there are duplicate feature names across datasets, the later one wins.

    Args:
        all_norm_constants: Nested dictionary from compute_all_normalization_constants.

    Returns:
        Flat dictionary mapping feature_name -> normalization_constant.
    """
    flat_dict: dict[str, float] = {}
    for dataset_name, features in all_norm_constants.items():
        for feature_name, constant in features.items():
            flat_dict[feature_name] = constant
    return flat_dict


if __name__ == "__main__":
    processed_data_dir = "data/datos_procesados"
    sparse_data_dir = f"{processed_data_dir}/sparse_datasets"
    output_path = f"{sparse_data_dir}/normalization_constants.pkl"

    print("Computing normalization constants for all datasets...")
    print("=" * 60)

    all_constants = compute_all_normalization_constants(
        sparse_data_dir=sparse_data_dir,
        output_path=output_path,
        percentile=99.0,
        min_scale=1e-8,
    )

    print("\n" + "=" * 60)
    print("Summary of normalization constants:")
    print("=" * 60)

    total_features = 0
    for dataset_name, features in all_constants.items():
        print(f"\n{dataset_name}: {len(features)} features")
        total_features += len(features)

    print(f"\nTotal features: {total_features}")
    print("=" * 60)
    print("Done!")
