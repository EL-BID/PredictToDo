import argparse
import pickle as pkl
from pathlib import Path
from typing import Iterable, List, Sequence

import h5py
import numpy as np
import pandas as pd

try:
    import pyarrow.parquet as pq
except ImportError:
    pq = None

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

try:
    from .merging import SparseDatasetGenerator
except ImportError:  # Permite ejecutar el script directamente
    from merging import SparseDatasetGenerator  # type: ignore


DEFAULT_BATCH_SIZE = 200_000


def iterate_parquet_batches(
    path: Path,
    columns: Sequence[str],
    batch_size: int,
) -> Iterable[pd.DataFrame]:
    if pq is None:
        yield pd.read_parquet(path, columns=columns)
        return
    parquet_file = pq.ParquetFile(path)
    for batch in parquet_file.iter_batches(columns=columns, batch_size=batch_size):
        yield batch.to_pandas()


def discover_schema_columns(parquet_path: Path) -> Sequence[str]:
    if pq is not None:
        return pq.read_schema(parquet_path).names
    return pd.read_parquet(parquet_path, nrows=0).columns.tolist()


def load_base_graph(processed_data_dir: Path) -> List[str]:
    graph_path = processed_data_dir / "conectivity_graph" / "road_graph.pickle"
    print(f"[1/6] Cargando grafo base: {graph_path}")
    with open(graph_path, "rb") as f:
        base_graph = pkl.load(f)
    road_ids = list(base_graph.nodes())
    print(f"      Tramos en grafo: {len(road_ids):,}")
    return road_ids


def discover_jams_files(processed_data_dir: Path) -> tuple[List[Path], Sequence[str]]:
    jams_dir = processed_data_dir / "jams_procesados"
    jams_files = sorted(jams_dir.glob("*.parquet"))
    if not jams_files:
        raise FileNotFoundError(
            f"No se encontraron archivos parquet en {jams_dir}. "
            "Verifica que la ruta sea correcta."
        )
    print(f"[2/6] Archivos de jams detectados: {len(jams_files)}")
    sample_list = "\n".join(f"      - {p.name}" for p in jams_files[:5])
    print(f"      (Primeros archivos)\n{sample_list}")

    schema_cols = discover_schema_columns(jams_files[0])
    if "timeframe" not in schema_cols:
        raise KeyError(
            "La columna 'timeframe' no existe en los archivos de jams. "
            "Revisa el proceso de generación."
        )
    if "road_id" not in schema_cols:
        raise KeyError(
            "La columna 'road_id' no existe en los archivos de jams. "
            "Revisa el proceso de generación."
        )
    return jams_files, schema_cols


def load_jams_time_range(
    jams_files: Sequence[Path],
    batch_size: int,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    print("[3/6] Escaneando rango temporal de los jams...")
    min_time = None
    max_time = None
    for parquet_path in tqdm(jams_files, desc="  Archivos"):
        for chunk in iterate_parquet_batches(
            parquet_path, columns=["timeframe"], batch_size=batch_size
        ):
            times = pd.to_datetime(
                chunk["timeframe"], errors="coerce", utc=True
            ).dt.tz_localize(None)
            if times.empty:
                continue
            current_min = times.min()
            current_max = times.max()
            if min_time is None or current_min < min_time:
                min_time = current_min
            if max_time is None or current_max > max_time:
                max_time = current_max
    if min_time is None or max_time is None:
        raise ValueError(
            "No se pudieron determinar los timestamps mínimos y máximos en los datos de jams."
        )
    min_time = min_time.floor("15min")
    max_time = max_time.ceil("15min")
    print(f"      Rango detectado: {min_time} -> {max_time}")
    return min_time, max_time


def prepare_sparse_writer(output_path: Path) -> tuple[h5py.File, h5py.Dataset]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    h5_file = h5py.File(output_path, "w")
    dataset = h5_file.create_dataset(
        "data",
        shape=(0, 4),
        maxshape=(None, 4),
        dtype=np.float64,
        chunks=True,
    )
    return h5_file, dataset


def flush_entries(dataset: h5py.Dataset, entries: np.ndarray) -> None:
    if entries.size == 0:
        return
    current_size = dataset.shape[0]
    dataset.resize(current_size + entries.shape[0], axis=0)
    dataset[current_size : current_size + entries.shape[0]] = entries


def build_sparse_entries(
    chunk: pd.DataFrame,
    feature_idx_map: dict[str, int],
    node_idx_map: dict[str, int],
    time_idx_map: dict[pd.Timestamp, int],
) -> np.ndarray:
    chunk = chunk.copy()
    chunk["road_id"] = chunk["road_id"].astype(str)
    chunk["timeframe"] = (
        pd.to_datetime(chunk["timeframe"], errors="coerce", utc=True)
        .dt.tz_localize(None)
        .dt.floor("15min")
    )
    chunk.dropna(subset=["road_id", "timeframe"], inplace=True)

    chunk = chunk[
        chunk["road_id"].isin(node_idx_map)
        & chunk["timeframe"].isin(time_idx_map)
    ]
    if chunk.empty:
        return np.empty((0, 4), dtype=np.float64)

    chunk["node_idx"] = chunk["road_id"].map(node_idx_map)
    chunk["time_idx"] = chunk["timeframe"].map(time_idx_map)
    chunk.dropna(subset=["node_idx", "time_idx"], inplace=True)
    if chunk.empty:
        return np.empty((0, 4), dtype=np.float64)

    node_idx_array = chunk["node_idx"].to_numpy(dtype=np.int64, copy=False)
    time_idx_array = chunk["time_idx"].to_numpy(dtype=np.int64, copy=False)

    entries_list: list[np.ndarray] = []

    for feature_name, feature_idx in feature_idx_map.items():
        if feature_name not in chunk.columns:
            continue
        values = pd.to_numeric(chunk[feature_name], errors="coerce").to_numpy(
            dtype=np.float64, copy=False
        )
        mask = np.isfinite(values) & (values != 0)
        if not mask.any():
            continue

        feature_entries = np.column_stack(
            [
                node_idx_array[mask],
                time_idx_array[mask],
                np.full(mask.sum(), feature_idx, dtype=np.int64),
                values[mask],
            ]
        )
        entries_list.append(feature_entries)

    if not entries_list:
        return np.empty((0, 4), dtype=np.float64)

    entries = np.vstack(entries_list)
    # Remove exact duplicates and enforce ordering by node, time, feature
    entries = np.unique(entries, axis=0)
    order = np.lexsort((entries[:, 2], entries[:, 1], entries[:, 0]))
    return entries[order]


def main(processed_data_dir: Path, batch_size: int) -> None:
    processed_data_dir = processed_data_dir.resolve()
    print("== Generación de dataset esparso de variables de jams ==")
    print(f"Ruta base utilizada: {processed_data_dir}")

    road_ids = load_base_graph(processed_data_dir)
    jams_files, schema_cols = discover_jams_files(processed_data_dir)
    time_min, time_max = load_jams_time_range(jams_files, batch_size=batch_size)

    print("[4/6] Creando generador de índices (nodos y timeframes)")
    dataset_generator = SparseDatasetGenerator(
        node_ids=road_ids,
        min_timeframe=time_min,
        max_timeframe=time_max,
        freq="15min",
    )
    node_idx_map = dataset_generator.node_idx_map
    time_idx_map = dataset_generator.time_idx_map

    jams_variables_columns = [
        "vel_min",
        "vel_mean",
        "vel_max",
        "level_min",
        "level_mean",
        "level_max",
        "delay_min",
        "delay_sum",
        "delay_max",
        "count_jams",
    ]
    available_columns = [col for col in jams_variables_columns if col in schema_cols]
    if not available_columns:
        raise ValueError(
            "No se encontraron columnas de variables de jams (vel_*, level_*, delay_* o count_jams). "
            "Revisa el contenido de los archivos de jams."
        )
    print("[5/6] Columnas incluidas en 'jams_variables_dataset.h5':")
    print("      " + ", ".join(available_columns))

    feature_idx_map = {
        feature_name: idx for idx, feature_name in enumerate(available_columns)
    }
    sparse_dir = processed_data_dir / "sparse_datasets"
    output_path = sparse_dir / "jams_variables_dataset.h5"
    metadata_path = sparse_dir / "jams_variables_dataset_metadata.pkl"
    print(f"[5/6] Escribiendo dataset esparso en {output_path}")

    total_rows = 0
    total_non_zero = 0

    h5_file, dataset = prepare_sparse_writer(output_path)
    try:
        needed_columns = ["road_id", "timeframe"] + available_columns
        for parquet_path in tqdm(jams_files, desc="  Procesando archivos de jams"):
            for chunk in iterate_parquet_batches(
                parquet_path, columns=needed_columns, batch_size=batch_size
            ):
                total_rows += len(chunk)
                entries = build_sparse_entries(
                    chunk,
                    feature_idx_map=feature_idx_map,
                    node_idx_map=node_idx_map,
                    time_idx_map=time_idx_map,
                )
                total_non_zero += entries.shape[0]
                flush_entries(dataset, entries)
    finally:
        h5_file.close()

    print(f"      Filas leídas: {total_rows:,}")
    print(f"      Entradas no-cero almacenadas: {total_non_zero:,}")

    metadata = {
        "node_idx_map": node_idx_map,
        "time_idx_map": time_idx_map,
        "feature_idx_map": feature_idx_map,
    }
    with open(metadata_path, "wb") as f:
        pkl.dump(metadata, f)
    print(f"[6/6] Metadata guardada en {metadata_path}")
    print("== Proceso finalizado correctamente ==")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Genera dataset esparso con variables de jams."
    )
    parser.add_argument(
        "--processed-data-dir",
        default="data/datos_procesados",
        help="Ruta al directorio de datos procesados (por defecto: data/datos_procesados).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Número aproximado de filas por lote al leer los parquet (default 200k).",
    )
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError("batch_size debe ser un entero positivo.")

    main(Path(args.processed_data_dir), batch_size=args.batch_size)
