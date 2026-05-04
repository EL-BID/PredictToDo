from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import boto3
import h5py
import numpy as np
import pandas as pd

from src.sparse_dataloader import TrafficDataset
from src.training_entrypoint import check_data_channels, setup_sagemaker_paths


@dataclass
class SeedTimeBucketArtifact:
    rows: pd.DataFrame
    metadata: dict

    def to_parquet(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.rows.to_parquet(path, index=False)

    def write_metadata(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.metadata, indent=2))


def _log_metrics(prefix: str, metrics: dict[str, float | int]) -> None:
    payload = " ".join(f"{k}={v}" for k, v in metrics.items())
    print(f"[METRIC] {prefix} {payload}", flush=True)


class PrebucketedSeedTimeBatchSampler:
    """
    Batch sampler for precomputed `(seed, time_idx)` rows.

    It yields dataset-global indices computed as:
        dataset_idx = seed_slot * num_time_windows + time_idx

    Assumptions:
    - the training entrypoint sets `dataset.seed_list` to the sampler's `seed_list`
    - `seed_list` order matches `seed_to_slot`
    - `time_idx` values are valid for the dataset's current temporal configuration
    """

    def __init__(
        self,
        rows: pd.DataFrame,
        num_time_windows: int,
        batch_size: int,
        drop_last: bool = False,
        shuffle: bool = True,
        rng_seed: int = 42,
    ) -> None:
        required = {"seed_id", "time_idx", "bucket", "score"}
        missing = required - set(rows.columns)
        if missing:
            raise ValueError(f"rows is missing required columns: {sorted(missing)}")
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if num_time_windows <= 0:
            raise ValueError("num_time_windows must be > 0")
        if rows.empty:
            raise ValueError("rows must not be empty")

        self.rows = rows.reset_index(drop=True).copy()
        self.num_time_windows = int(num_time_windows)
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.shuffle = bool(shuffle)
        self.rng = np.random.default_rng(rng_seed)
        self.seed_list = list(dict.fromkeys(self.rows["seed_id"].astype(str).tolist()))
        self.seed_to_slot = {seed: i for i, seed in enumerate(self.seed_list)}
        self._num_batches = (
            len(self.rows) // self.batch_size
            if self.drop_last
            else int(math.ceil(len(self.rows) / self.batch_size))
        )

    def __len__(self) -> int:
        return self._num_batches

    def __iter__(self):
        order = np.arange(len(self.rows))
        if self.shuffle:
            self.rng.shuffle(order)

        for start in range(0, len(order), self.batch_size):
            batch_order = order[start : start + self.batch_size]
            if self.drop_last and len(batch_order) < self.batch_size:
                break

            batch_indices: list[int] = []
            for row_idx in batch_order.tolist():
                row = self.rows.iloc[row_idx]
                seed_slot = self.seed_to_slot[str(row["seed_id"])]
                time_idx = int(row["time_idx"])
                batch_indices.append(seed_slot * self.num_time_windows + time_idx)
            yield batch_indices


def _parse_bucket_quantiles(raw: str) -> tuple[float, float]:
    values = [float(v.strip()) for v in raw.split(",") if v.strip()]
    if len(values) != 2:
        raise ValueError("bucket_quantiles must contain exactly two comma-separated values")
    low_q, high_q = values
    if not (0.0 < low_q < high_q < 1.0):
        raise ValueError("bucket_quantiles must satisfy 0 < low_q < high_q < 1")
    return low_q, high_q


def _ensure_output_mirror_paths(
    parquet_path: str | Path,
    metadata_path: str | Path,
) -> tuple[Path, Path, Path | None, Path | None]:
    parquet_path = Path(parquet_path)
    metadata_path = Path(metadata_path)
    output_data_dir = Path("/opt/ml/output/data")
    if not output_data_dir.exists():
        return parquet_path, metadata_path, None, None
    parquet_mirror = output_data_dir / parquet_path.name
    metadata_mirror = output_data_dir / metadata_path.name
    return parquet_path, metadata_path, parquet_mirror, metadata_mirror


def _bucketize_by_quantiles(scores: np.ndarray, low_q: float, high_q: float) -> tuple[np.ndarray, dict]:
    low_thr = float(np.quantile(scores, low_q))
    high_thr = float(np.quantile(scores, high_q))
    buckets = np.full(scores.shape[0], "medium", dtype=object)
    buckets[scores <= low_thr] = "low"
    buckets[scores >= high_thr] = "high"
    return buckets, {
        "bucket_quantiles": [low_q, high_q],
        "bucket_score_thresholds": [low_thr, high_thr],
    }


def _load_seed_candidates(seed_path: str | Path) -> list[str]:
    seed_path = _materialize_input_uri(seed_path)
    seed_path = Path(seed_path)
    if seed_path.suffix == ".pkl":
        with seed_path.open("rb") as f:
            payload = pickle.load(f)
        if isinstance(payload, list):
            return [str(x) for x in payload]
        if isinstance(payload, dict):
            for key in ("seed_list", "seeds", "nodes"):
                if key in payload:
                    return [str(x) for x in payload[key]]
        raise ValueError(f"Unsupported pickle format in {seed_path}")

    df = pd.read_csv(seed_path)
    if "seed_id" in df.columns:
        return df["seed_id"].astype(str).tolist()
    if "node" in df.columns:
        return df["node"].astype(str).tolist()
    return df.iloc[:, 0].astype(str).tolist()


def _default_seed_candidates(dataset: TrafficDataset) -> list[str]:
    if getattr(dataset, "batch_mapping", None):
        return [str(seed) for seed in dataset.batch_mapping.keys()]
    return [str(node) for node in dataset.graph.nodes()]


def _load_time_indices(path: str | Path) -> list[int]:
    path = _materialize_input_uri(path)
    path = Path(path)
    if path.suffix == ".pkl":
        with path.open("rb") as f:
            payload = pickle.load(f)
        if isinstance(payload, list):
            return [int(x) for x in payload]
        if isinstance(payload, dict):
            for key in ("time_indices", "times", "time_idx"):
                if key in payload:
                    return [int(x) for x in payload[key]]
        raise ValueError(f"Unsupported pickle format in {path}")

    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    for col in ("time_idx", "time_indices"):
        if col in df.columns:
            return df[col].astype(int).tolist()
    return df.iloc[:, 0].astype(int).tolist()


def _materialize_input_uri(path_or_uri: str | Path) -> Path:
    raw = str(path_or_uri)
    if not raw.startswith("s3://"):
        return Path(raw)

    bucket_and_key = raw[len("s3://") :]
    bucket, key = bucket_and_key.split("/", 1)
    local_dir = Path(tempfile.gettempdir()) / "congestion_bogota_sampling_inputs"
    local_dir.mkdir(parents=True, exist_ok=True)
    local_path = local_dir / Path(key).name
    boto3.client("s3").download_file(bucket, key, str(local_path))
    return local_path


def _upload_file_to_s3(local_path: str | Path, s3_uri: str) -> None:
    if not s3_uri:
        return
    if not s3_uri.startswith("s3://"):
        raise ValueError(f"Expected S3 URI, got: {s3_uri}")
    bucket_and_key = s3_uri[len("s3://") :]
    bucket, key = bucket_and_key.split("/", 1)
    boto3.client("s3").upload_file(str(local_path), bucket, key)


def _append_suffix_to_path(path: str | Path, suffix: str) -> Path:
    path = Path(path)
    return path.with_name(f"{path.stem}{suffix}{path.suffix}")


def _append_suffix_to_s3_uri(s3_uri: str, suffix: str) -> str:
    if not s3_uri:
        return s3_uri
    if not s3_uri.startswith("s3://"):
        raise ValueError(f"Expected S3 URI, got: {s3_uri}")
    bucket_and_key = s3_uri[len("s3://") :]
    bucket, key = bucket_and_key.split("/", 1)
    key_path = Path(key)
    new_key = key_path.with_name(f"{key_path.stem}{suffix}{key_path.suffix}").as_posix()
    return f"s3://{bucket}/{new_key}"


def _compute_future_high_fraction_for_seed(
    dataset: TrafficDataset,
    seed_id: str,
    time_idx: int,
    high_level_threshold: float,
) -> float:
    """
    Score a `(seed_id, time_idx)` pair using only the seed neighborhood.

    Missing sparse target rows are interpreted as 0, matching dataset semantics.
    """
    neighbors = dataset.batch_mapping.get(seed_id, [])
    if not neighbors:
        return 0.0

    node_ids = [dataset.node_to_id[n] for n in neighbors if n in dataset.node_to_id]
    if not node_ids:
        return 0.0

    input_start = int(time_idx)
    input_end = input_start + dataset.in_timesteps - 1
    output_start = input_end + 1
    output_end = output_start + dataset.out_timesteps - 1

    high_count = 0
    total_slots = len(node_ids) * dataset.out_timesteps

    file_path = dataset.target_data_path
    with h5py.File(file_path, "r") as f:
        start_index = dataset.target_data_indices[file_path][output_start, 0]
        end_index = dataset.target_data_indices[file_path][output_end, 1]
        relevant = f["data"][start_index:end_index]
        if relevant.shape[0] == 0:
            relevant = np.empty((0, 4))
        if relevant.shape[0] > 0:
            node_idx = relevant[:, 0].astype(int)
            time_idx_abs = relevant[:, 1].astype(int)
            feature_idx = relevant[:, 2].astype(int)
            values = relevant[:, 3]
        else:
            node_idx = np.empty((0,), dtype=int)
            time_idx_abs = np.empty((0,), dtype=int)
            feature_idx = np.empty((0,), dtype=int)
            values = np.empty((0,), dtype=float)

    mask = (
        (time_idx_abs >= output_start)
        & (time_idx_abs <= output_end)
        & np.isin(node_idx, node_ids)
        & (feature_idx == dataset.target_level_idx)
        & (values >= high_level_threshold)
    )
    high_count += int(mask.sum())

    if total_slots <= 0:
        return 0.0
    return float(high_count / total_slots)


def build_seed_time_bucket_artifact(
    dataset: TrafficDataset,
    seed_candidates: Iterable[str],
    train_time_indices: Iterable[int],
    high_level_threshold: float = 3.0,
    bucket_quantiles: tuple[float, float] = (0.5, 0.75),
    max_seed_candidates: int | None = None,
    max_time_indices: int | None = None,
    progress_log_every: int = 0,
    progress_callback=None,
) -> SeedTimeBucketArtifact:
    seed_list = list(dict.fromkeys(str(x) for x in seed_candidates))
    time_indices = list(dict.fromkeys(int(x) for x in train_time_indices))

    if max_seed_candidates is not None:
        seed_list = seed_list[: max(0, int(max_seed_candidates))]
    if max_time_indices is not None:
        time_indices = time_indices[: max(0, int(max_time_indices))]

    rows: list[dict] = []
    processed = 0
    score_sum = 0.0
    score_min = float("inf")
    score_max = float("-inf")
    seed_rows_done = 0
    partial_scores: list[float] = []
    for seed_id in seed_list:
        if seed_id not in dataset.graph.nodes():
            continue
        for time_idx in time_indices:
            score = _compute_future_high_fraction_for_seed(
                dataset=dataset,
                seed_id=seed_id,
                time_idx=time_idx,
                high_level_threshold=high_level_threshold,
            )
            rows.append(
                {
                    "seed_id": str(seed_id),
                    "time_idx": int(time_idx),
                    "score": float(score),
                }
            )
            processed += 1
            score_sum += float(score)
            score_min = min(score_min, float(score))
            score_max = max(score_max, float(score))
            partial_scores.append(float(score))

            if progress_log_every > 0 and processed % progress_log_every == 0:
                score_arr = np.asarray(partial_scores, dtype=float)
                partial_buckets, _ = _bucketize_by_quantiles(
                    score_arr,
                    low_q=bucket_quantiles[0],
                    high_q=bucket_quantiles[1],
                )
                _log_metrics(
                    "seed_time_build_progress",
                    {
                        "pairs_evaluated": processed,
                        "seeds_completed": seed_rows_done,
                        "partial_score_mean": score_sum / processed,
                        "partial_score_min": score_min,
                        "partial_score_max": score_max,
                        "partial_low_ratio": float((partial_buckets == "low").mean()),
                        "partial_medium_ratio": float((partial_buckets == "medium").mean()),
                        "partial_high_ratio": float((partial_buckets == "high").mean()),
                    },
                )
                if progress_callback is not None:
                    progress_callback(rows, processed)
        seed_rows_done += 1

    if not rows:
        raise ValueError("No rows were generated for the seed-time artifact")

    df = pd.DataFrame(rows)
    buckets, bucket_meta = _bucketize_by_quantiles(
        df["score"].to_numpy(dtype=float),
        low_q=bucket_quantiles[0],
        high_q=bucket_quantiles[1],
    )
    df["bucket"] = buckets

    metadata = {
        "score_method": "future_high_fraction_seed_local",
        "high_level_threshold": float(high_level_threshold),
        "node_batch_size": int(dataset.node_batch_size),
        "input_seq_len": int(dataset.in_timesteps),
        "output_seq_len": int(dataset.out_timesteps),
        "num_rows": int(len(df)),
        "num_seeds": int(df["seed_id"].nunique()),
        "num_time_indices": int(df["time_idx"].nunique()),
        "score_min": float(df["score"].min()),
        "score_mean": float(df["score"].mean()),
        "score_max": float(df["score"].max()),
        "bucket_counts": {
            str(k): int(v) for k, v in df["bucket"].value_counts().sort_index().items()
        },
        **bucket_meta,
    }
    return SeedTimeBucketArtifact(rows=df, metadata=metadata)


def _build_partial_artifact(
    rows: list[dict],
    dataset: TrafficDataset,
    high_level_threshold: float,
    bucket_quantiles: tuple[float, float],
    shard_id: int,
    num_shards: int,
) -> SeedTimeBucketArtifact:
    df = pd.DataFrame(rows)
    buckets, bucket_meta = _bucketize_by_quantiles(
        df["score"].to_numpy(dtype=float),
        low_q=bucket_quantiles[0],
        high_q=bucket_quantiles[1],
    )
    df["bucket"] = buckets
    metadata = {
        "score_method": "future_high_fraction_seed_local",
        "high_level_threshold": float(high_level_threshold),
        "node_batch_size": int(dataset.node_batch_size),
        "input_seq_len": int(dataset.in_timesteps),
        "output_seq_len": int(dataset.out_timesteps),
        "num_rows": int(len(df)),
        "num_seeds": int(df["seed_id"].nunique()),
        "num_time_indices": int(df["time_idx"].nunique()),
        "score_min": float(df["score"].min()),
        "score_mean": float(df["score"].mean()),
        "score_max": float(df["score"].max()),
        "bucket_counts": {
            str(k): int(v) for k, v in df["bucket"].value_counts().sort_index().items()
        },
        "shard_id": int(shard_id),
        "num_shards": int(num_shards),
        "is_partial_checkpoint": True,
        **bucket_meta,
    }
    return SeedTimeBucketArtifact(rows=df, metadata=metadata)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build prebucketed seed-time sampling artifacts.")
    parser.add_argument("--seed-candidates-path", default="", help="CSV or PKL with candidate seeds")
    parser.add_argument("--time-indices-path", default="", help="CSV/Parquet/PKL with explicit time_idx list")
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument(
        "--output-parquet",
        default="/opt/ml/model/seed_time_bucket_rows.parquet",
        help="Where to write the row artifact parquet",
    )
    parser.add_argument(
        "--output-metadata",
        default="/opt/ml/model/seed_time_bucket_metadata.json",
        help="Where to write the metadata JSON",
    )
    parser.add_argument("--output-parquet-s3-uri", default="", help="Optional direct S3 upload for parquet output")
    parser.add_argument("--output-metadata-s3-uri", default="", help="Optional direct S3 upload for metadata output")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--input-seq-len", type=int, default=150)
    parser.add_argument("--output-seq-len", type=int, default=50)
    parser.add_argument("--node-batch-size", type=int, default=500)
    parser.add_argument("--high-level-threshold", type=float, default=3.0)
    parser.add_argument("--bucket-quantiles", type=str, default="0.5,0.75")
    parser.add_argument("--max-seed-candidates", type=int, default=None)
    parser.add_argument("--max-time-indices", type=int, default=None)
    parser.add_argument("--progress-log-every", type=int, default=5000)
    parser.add_argument("--include-weather", type=str, default="True")
    parser.add_argument("--include-velocity-target", type=str, default="True")
    parser.add_argument("--use-minimal-datasets", type=str, default="True")
    return parser.parse_args()


def _slice_for_shard(seed_candidates: list[str], shard_id: int, num_shards: int) -> list[str]:
    if num_shards <= 1:
        return seed_candidates
    if shard_id < 0 or shard_id >= num_shards:
        raise ValueError(f"Invalid shard configuration: shard_id={shard_id}, num_shards={num_shards}")
    return seed_candidates[shard_id::num_shards]


def main() -> None:
    args = parse_args()
    bucket_quantiles = _parse_bucket_quantiles(args.bucket_quantiles)
    include_weather = str(args.include_weather).lower() == "true"
    include_velocity_target = str(args.include_velocity_target).lower() == "true"
    use_minimal_datasets = str(args.use_minimal_datasets).lower() == "true"

    train_path = check_data_channels()
    paths = setup_sagemaker_paths(
        train_path=train_path,
        include_weather=include_weather,
        use_minimal_datasets=use_minimal_datasets,
    )

    dataset = TrafficDataset(
        graph_path=str(paths["graph"]),
        static_data_path=str(paths["static"]),
        dynamic_data_paths=[str(p) for p in paths["dynamic"]],
        metadata_paths=[str(p) for p in paths["metadata"]],
        target_data_path=str(paths["target"]),
        in_timesteps=args.input_seq_len,
        out_timesteps=args.output_seq_len,
        future_features=[
            "coi_puntos",
            "coi_tramo",
            "AGLOMERACION",
            "EMERGENCIA",
            "MANIFESTACION",
        ],
        node_batch_size=args.node_batch_size,
        weather_data_dir=str(paths["weather_dir"]) if paths.get("weather_dir") else None,
        include_weather=include_weather,
        neighbor_features_path=str(paths["neighbor_features"]) if paths.get("neighbor_features") else None,
        include_velocity_target=include_velocity_target,
        normalization_constants_path=str(paths["normalization"]) if paths.get("normalization") else None,
    )
    num_time_windows = dataset.effective_total_timesteps - dataset.total_timesteps_window + 1
    if num_time_windows <= 0:
        raise ValueError("No valid time windows found for the current dataset configuration")

    train_end = int(num_time_windows * float(args.train_ratio))
    train_time_indices = list(range(train_end))
    if args.time_indices_path:
        requested_time_indices = _load_time_indices(args.time_indices_path)
        train_time_indices = [
            int(t) for t in requested_time_indices if 0 <= int(t) < train_end
        ]
        if not train_time_indices:
            raise ValueError("No valid time_idx entries remained after filtering against train range")
    if args.seed_candidates_path:
        seed_candidates = _load_seed_candidates(args.seed_candidates_path)
    else:
        seed_candidates = _default_seed_candidates(dataset)
    seed_candidates = _slice_for_shard(
        seed_candidates,
        shard_id=int(args.shard_id),
        num_shards=int(args.num_shards),
    )

    _log_metrics(
        "seed_time_build_config",
        {
            "num_time_windows": num_time_windows,
            "train_time_indices": len(train_time_indices),
            "seed_candidates": len(seed_candidates),
            "shard_id": int(args.shard_id),
            "num_shards": int(args.num_shards),
            "node_batch_size": dataset.node_batch_size,
            "input_seq_len": dataset.in_timesteps,
            "output_seq_len": dataset.out_timesteps,
        },
    )

    parquet_path, metadata_path, parquet_mirror, metadata_mirror = _ensure_output_mirror_paths(
        args.output_parquet,
        args.output_metadata,
    )

    def _checkpoint(rows: list[dict], processed: int) -> None:
        partial_artifact = _build_partial_artifact(
            rows=rows,
            dataset=dataset,
            high_level_threshold=float(args.high_level_threshold),
            bucket_quantiles=bucket_quantiles,
            shard_id=int(args.shard_id),
            num_shards=int(args.num_shards),
        )
        partial_parquet = _append_suffix_to_path(parquet_path, ".partial")
        partial_metadata = _append_suffix_to_path(metadata_path, ".partial")
        partial_artifact.to_parquet(partial_parquet)
        partial_artifact.write_metadata(partial_metadata)

        if parquet_mirror is not None:
            partial_artifact.to_parquet(_append_suffix_to_path(parquet_mirror, ".partial"))
        if metadata_mirror is not None:
            partial_artifact.write_metadata(_append_suffix_to_path(metadata_mirror, ".partial"))

        if args.output_parquet_s3_uri:
            _upload_file_to_s3(partial_parquet, _append_suffix_to_s3_uri(args.output_parquet_s3_uri, ".partial"))
            _upload_file_to_s3(
                partial_parquet,
                _append_suffix_to_s3_uri(args.output_parquet_s3_uri, f".checkpoint_{processed}"),
            )
        if args.output_metadata_s3_uri:
            _upload_file_to_s3(partial_metadata, _append_suffix_to_s3_uri(args.output_metadata_s3_uri, ".partial"))
            _upload_file_to_s3(
                partial_metadata,
                _append_suffix_to_s3_uri(args.output_metadata_s3_uri, f".checkpoint_{processed}"),
            )

    artifact = build_seed_time_bucket_artifact(
        dataset=dataset,
        seed_candidates=seed_candidates,
        train_time_indices=train_time_indices,
        high_level_threshold=float(args.high_level_threshold),
        bucket_quantiles=bucket_quantiles,
        max_seed_candidates=args.max_seed_candidates,
        max_time_indices=args.max_time_indices,
        progress_log_every=args.progress_log_every,
        progress_callback=_checkpoint,
    )
    artifact.metadata["shard_id"] = int(args.shard_id)
    artifact.metadata["num_shards"] = int(args.num_shards)
    artifact.to_parquet(parquet_path)
    artifact.write_metadata(metadata_path)
    if parquet_mirror is not None:
        artifact.to_parquet(parquet_mirror)
    if metadata_mirror is not None:
        artifact.write_metadata(metadata_mirror)
    if args.output_parquet_s3_uri:
        _upload_file_to_s3(parquet_path, args.output_parquet_s3_uri)
    if args.output_metadata_s3_uri:
        _upload_file_to_s3(metadata_path, args.output_metadata_s3_uri)
    _log_metrics(
        "seed_time_build_final",
        {
            "pairs_evaluated": artifact.metadata["num_rows"],
            "num_seeds": artifact.metadata["num_seeds"],
            "num_time_indices": artifact.metadata["num_time_indices"],
            "score_min": artifact.metadata["score_min"],
            "score_mean": artifact.metadata["score_mean"],
            "score_max": artifact.metadata["score_max"],
            "bucket_low_count": artifact.metadata["bucket_counts"].get("low", 0),
            "bucket_medium_count": artifact.metadata["bucket_counts"].get("medium", 0),
            "bucket_high_count": artifact.metadata["bucket_counts"].get("high", 0),
        },
    )
    print(json.dumps(artifact.metadata, indent=2))


if __name__ == "__main__":
    main()
