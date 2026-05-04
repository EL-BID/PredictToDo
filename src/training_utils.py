import math
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Optional

import boto3
import numpy as np
import pandas as pd
import torch


def pad_collate(batch):
    assert len(batch) > 0, "Empty batch."

    # --- Sanity: all samples must share same node set and edges ---
    N0 = batch[0].x_static.size(0)
    E0 = batch[0].edge_index.size(1)
    edge0 = batch[0].edge_index

    for i, d in enumerate(batch[1:], start=1):
        if d.x_static.size(0) != N0:
            raise ValueError(
                f"Item {i} has N={d.x_static.size(0)} but first has N={N0}."
            )
        if d.edge_index.size(1) != E0:
            raise ValueError(
                f"Item {i} has E={d.edge_index.size(1)} but first has E={E0}."
            )
        # Require identical edges (same ordering). If you only need same *set* of edges,
        # customize this check.
        if not torch.equal(d.edge_index, edge0):
            raise ValueError(f"Item {i} edge_index differs from first item.")

    # --- Always present ---
    x_static = torch.stack([d.x_static for d in batch], dim=0)  # (B, N, F_static)
    y = torch.stack([d.y for d in batch], dim=0)  # (B, N, Tout)

    # --- Dynamic features: support either split past/future or a single x_dynamic ---
    out = {
        "x_static": x_static,
        "y": y,
        "edge_index": edge0,  # (2, E) shared across batch
        "num_nodes": N0,
        "num_edges": E0,
    }

    has_past = hasattr(batch[0], "x_dynamic_past")
    has_future = hasattr(batch[0], "x_dynamic_future")
    has_single = hasattr(batch[0], "x_dynamic")

    if has_past and has_future:
        out["x_dynamic_past"] = torch.stack(
            [d.x_dynamic_past for d in batch], dim=0
        )  # (B, N, Tin, F_dyn)
        out["x_dynamic_future"] = torch.stack(
            [d.x_dynamic_future for d in batch], dim=0
        )  # (B, N, Tout, F_future)
        if hasattr(batch[0], "x_temporal_past"):
            out["x_temporal_past"] = torch.stack(
                [d.x_temporal_past for d in batch], dim=0
            )
        if hasattr(batch[0], "x_temporal_future"):
            out["x_temporal_future"] = torch.stack(
                [d.x_temporal_future for d in batch], dim=0
            )
    elif has_single:
        out["x_dynamic"] = torch.stack(
            [d.x_dynamic for d in batch], dim=0
        )  # (B, N, Tin, F_dyn)
    else:
        # If your dataset always has dynamics, you can raise instead:
        # raise AttributeError("Expected x_dynamic_past/x_dynamic_future or x_dynamic in Data.")
        pass

    # Optional: if your Data carries a node id tensor you want to pass through
    if hasattr(batch[0], "node_indices"):
        out["node_indices"] = batch[0].node_indices  # (N,) shared across items

    # Optional: weather features (concat into dynamic past features)
    if has_past and hasattr(batch[0], "x_weather"):
        x_weather = torch.stack(
            [d.x_weather for d in batch], dim=0
        )  # (B, N, Tin, F_weather)
        if x_weather.dim() != out["x_dynamic_past"].dim():
            raise ValueError(
                "x_weather rank does not match x_dynamic_past rank: "
                f"{tuple(x_weather.shape)} vs {tuple(out['x_dynamic_past'].shape)}"
            )
        if x_weather.shape[:3] != out["x_dynamic_past"].shape[:3]:
            # Some paths may produce a singleton node dimension for weather. Expand it
            # if and only if the mismatch is exactly N=1 -> N=target.
            if (
                x_weather.shape[0] == out["x_dynamic_past"].shape[0]
                and x_weather.shape[1] == 1
                and x_weather.shape[2] == out["x_dynamic_past"].shape[2]
            ):
                x_weather = x_weather.expand(
                    x_weather.shape[0],
                    out["x_dynamic_past"].shape[1],
                    x_weather.shape[2],
                    x_weather.shape[3],
                )
            else:
                raise ValueError(
                    "x_weather shape is incompatible with x_dynamic_past before concat: "
                    f"{tuple(x_weather.shape)} vs {tuple(out['x_dynamic_past'].shape)}"
                )
        out["x_dynamic_past"] = torch.cat(
            [out["x_dynamic_past"], x_weather], dim=-1
        )
        out["x_weather"] = x_weather

    return out


class SeededBatchSampler:
    """
    Batch sampler that keeps all items in a batch tied to the same seed index.

    Each batch consists of time indices for a single seed. This allows
    multi-worker loading without mutating dataset state between batches.
    """

    def __init__(
        self,
        seed_list: list[str],
        time_indices: Iterable[int],
        num_time_windows: int,
        batch_size: int,
        drop_last: bool = False,
        shuffle_time: bool = True,
        rng_seed: int = 42,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if num_time_windows <= 0:
            raise ValueError("num_time_windows must be > 0")
        if not seed_list:
            raise ValueError("seed_list must contain at least one seed")

        self.seed_list = list(seed_list)
        self.time_indices = list(time_indices)
        self.num_time_windows = num_time_windows
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle_time = shuffle_time
        self.rng = np.random.default_rng(rng_seed)

        if self.drop_last:
            self._num_batches = len(self.time_indices) // self.batch_size
        else:
            self._num_batches = math.ceil(len(self.time_indices) / self.batch_size)

        if len(self.seed_list) < self._num_batches:
            raise ValueError(
                "seed_list must have at least one entry per batch. "
                f"Got {len(self.seed_list)} seeds for {self._num_batches} batches."
            )

    def __iter__(self):
        time_indices = list(self.time_indices)
        if self.shuffle_time:
            self.rng.shuffle(time_indices)

        for batch_idx in range(self._num_batches):
            start = batch_idx * self.batch_size
            end = start + self.batch_size
            batch_times = time_indices[start:end]
            if self.drop_last and len(batch_times) < self.batch_size:
                break
            seed_offset = batch_idx * self.num_time_windows
            yield [seed_offset + time_idx for time_idx in batch_times]

    def __len__(self) -> int:
        return self._num_batches


@dataclass
class SeedTimeEpochPlan:
    seed_list: list[str]
    batch_time_indices: list[list[int]]
    requested_bucket_counts: dict[str, int]
    realized_bucket_counts: dict[str, int]

    @property
    def num_batches(self) -> int:
        return len(self.batch_time_indices)

    @property
    def num_samples(self) -> int:
        return sum(len(batch) for batch in self.batch_time_indices)


def _materialize_parquet_path(path_or_uri: str | Path) -> Path:
    raw = str(path_or_uri)
    if not raw.startswith("s3://"):
        return Path(raw)

    bucket_and_key = raw[len("s3://") :]
    bucket, key = bucket_and_key.split("/", 1)
    local_dir = Path(tempfile.gettempdir()) / "congestion_bogota_training_artifacts"
    local_dir.mkdir(parents=True, exist_ok=True)
    local_path = local_dir / Path(key).name
    boto3.client("s3").download_file(bucket, key, str(local_path))
    return local_path


def _normalize_bucket_ratios(raw: str | dict[str, float] | tuple[float, float, float]) -> dict[str, float]:
    if isinstance(raw, dict):
        ratios = {
            "high": float(raw["high"]),
            "medium": float(raw["medium"]),
            "low": float(raw["low"]),
        }
    elif isinstance(raw, tuple):
        ratios = {"high": float(raw[0]), "medium": float(raw[1]), "low": float(raw[2])}
    else:
        values = [float(v.strip()) for v in str(raw).split(",") if v.strip()]
        if len(values) != 3:
            raise ValueError("seed_time_bucket_target_ratios must contain exactly 3 comma-separated values")
        ratios = {"high": values[0], "medium": values[1], "low": values[2]}

    total = sum(ratios.values())
    if total <= 0:
        raise ValueError("Bucket ratios must sum to a positive value")
    return {k: v / total for k, v in ratios.items()}


def _allocate_bucket_counts(batch_size: int, ratios: dict[str, float]) -> dict[str, int]:
    exact = {bucket: ratios[bucket] * batch_size for bucket in ("high", "medium", "low")}
    counts = {bucket: int(math.floor(value)) for bucket, value in exact.items()}
    remainder = batch_size - sum(counts.values())
    ranked = sorted(
        ((bucket, exact[bucket] - counts[bucket]) for bucket in counts),
        key=lambda item: item[1],
        reverse=True,
    )
    for bucket, _ in ranked[:remainder]:
        counts[bucket] += 1
    return counts


def _sample_time_indices_for_seed(
    rng: np.random.Generator,
    per_bucket: dict[str, np.ndarray],
    requested_counts: dict[str, int],
) -> tuple[list[int], dict[str, int]]:
    realized_counts = {"high": 0, "medium": 0, "low": 0}
    selected: list[int] = []

    all_times = np.concatenate([arr for arr in per_bucket.values() if len(arr) > 0])
    if len(all_times) == 0:
        raise ValueError("Seed has no available time_idx rows in artifact")

    for bucket in ("high", "medium", "low"):
        need = int(requested_counts[bucket])
        if need <= 0:
            continue
        arr = per_bucket.get(bucket)
        if arr is not None and len(arr) > 0:
            replace = len(arr) < need
            picks = rng.choice(arr, size=need, replace=replace).tolist()
            selected.extend(int(x) for x in picks)
            realized_counts[bucket] += need
        else:
            fallback = rng.choice(all_times, size=need, replace=len(all_times) < need).tolist()
            selected.extend(int(x) for x in fallback)

    rng.shuffle(selected)
    return selected, realized_counts


class SeedTimeEpochPlanner:
    """
    Builds epoch plans where each batch uses a single seed and multiple time_idx.

    The heavy work is done once when loading the precomputed artifact. Per epoch we only
    shuffle seeds and sample time_idx from cached numpy arrays, which keeps loader overhead low.
    """

    def __init__(
        self,
        artifact_path: str | Path,
        batch_size: int,
        bucket_target_ratios: str | dict[str, float] | tuple[float, float, float] = "0.45,0.30,0.25",
        rng_seed: int = 42,
        allowed_seeds: Optional[Iterable[str]] = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        self.batch_size = int(batch_size)
        self.rng = np.random.default_rng(rng_seed)
        self.bucket_target_ratios = _normalize_bucket_ratios(bucket_target_ratios)
        self.batch_bucket_counts = _allocate_bucket_counts(self.batch_size, self.bucket_target_ratios)

        rows = pd.read_parquet(_materialize_parquet_path(artifact_path))
        required = {"seed_id", "time_idx", "bucket"}
        missing = required - set(rows.columns)
        if missing:
            raise ValueError(f"Seed-time artifact is missing columns: {sorted(missing)}")

        rows = rows[["seed_id", "time_idx", "bucket"]].copy()
        rows["seed_id"] = rows["seed_id"].astype(str)
        rows["time_idx"] = rows["time_idx"].astype(int)
        rows["bucket"] = rows["bucket"].astype(str)
        if allowed_seeds is not None:
            allowed = {str(seed) for seed in allowed_seeds}
            rows = rows[rows["seed_id"].isin(allowed)].copy()
        if rows.empty:
            raise ValueError("Seed-time artifact is empty after filtering")

        self.seed_ids = list(dict.fromkeys(rows["seed_id"].tolist()))
        self.by_seed_bucket: dict[str, dict[str, np.ndarray]] = {}
        for seed_id, seed_df in rows.groupby("seed_id"):
            self.by_seed_bucket[seed_id] = {}
            for bucket in ("high", "medium", "low"):
                arr = seed_df.loc[seed_df["bucket"] == bucket, "time_idx"].to_numpy(dtype=np.int64)
                self.by_seed_bucket[seed_id][bucket] = arr

    def build_epoch_plan(self, seed_cycles: int = 1, shuffle_seeds: bool = True) -> SeedTimeEpochPlan:
        if seed_cycles <= 0:
            raise ValueError("seed_cycles must be > 0")
        epoch_seeds: list[str] = []
        for _ in range(int(seed_cycles)):
            cycle = list(self.seed_ids)
            if shuffle_seeds:
                self.rng.shuffle(cycle)
            epoch_seeds.extend(cycle)

        batch_time_indices: list[list[int]] = []
        realized = {"high": 0, "medium": 0, "low": 0}
        for seed_id in epoch_seeds:
            batch_times, realized_counts = _sample_time_indices_for_seed(
                rng=self.rng,
                per_bucket=self.by_seed_bucket[seed_id],
                requested_counts=self.batch_bucket_counts,
            )
            batch_time_indices.append(batch_times)
            for bucket, value in realized_counts.items():
                realized[bucket] += int(value)

        requested = {
            bucket: int(count * len(epoch_seeds))
            for bucket, count in self.batch_bucket_counts.items()
        }
        return SeedTimeEpochPlan(
            seed_list=epoch_seeds,
            batch_time_indices=batch_time_indices,
            requested_bucket_counts=requested,
            realized_bucket_counts=realized,
        )


class PlannedSeedTimeBatchSampler:
    def __init__(self, batch_time_indices: list[list[int]], num_time_windows: int) -> None:
        if num_time_windows <= 0:
            raise ValueError("num_time_windows must be > 0")
        if not batch_time_indices:
            raise ValueError("batch_time_indices must not be empty")
        self.batch_time_indices = [[int(x) for x in batch] for batch in batch_time_indices]
        self.num_time_windows = int(num_time_windows)

    def __iter__(self):
        for batch_idx, batch_times in enumerate(self.batch_time_indices):
            seed_offset = batch_idx * self.num_time_windows
            yield [seed_offset + int(time_idx) for time_idx in batch_times]

    def __len__(self) -> int:
        return len(self.batch_time_indices)


@dataclass
class Node50Picker:
    top_nodes: np.ndarray
    other_nodes: np.ndarray
    rng: np.random.Generator

    def pick(self):
        if len(self.top_nodes) == 0:
            return self.rng.choice(self.other_nodes)
        if len(self.other_nodes) == 0:
            return self.rng.choice(self.top_nodes)
        return (
            self.rng.choice(self.top_nodes)
            if self.rng.random() < 0.5
            else self.rng.choice(self.other_nodes)
        )


def build_node_50_picker(
    csv_path: str, top_frac: float = 0.2, seed: int = 0
) -> Node50Picker:
    df = pd.read_csv(csv_path)  # assumes: col0=node (literal), col1=hours
    df = df.iloc[:, :2].copy()
    df.columns = ["node", "hours"]
    df["hours"] = pd.to_numeric(df["hours"], errors="coerce").fillna(0)

    # keep nodes exactly as strings to preserve literals like "11013660_B.1"
    d = df.sort_values("hours", ascending=False, ignore_index=True)
    cutoff = max(1, int(np.ceil(len(d) * top_frac)))
    top_nodes = d["node"].astype(str).to_numpy()
    other_nodes = d["node"].astype(str).to_numpy()[cutoff:]

    return Node50Picker(
        top_nodes=top_nodes,
        other_nodes=other_nodes,
        rng=np.random.default_rng(seed),
    )


def _default_node_key(seed_str: str) -> int:
    """
    Extracts a node id from a seed like '50008544_TF.1' -> 50008544.
    If no '_' present, tries to parse the whole string as an int.
    """
    m = re.match(r"^(\d+)", str(seed_str))
    if not m:
        raise ValueError(f"Cannot parse node id from seed '{seed_str}'")
    return int(m.group(1))


# --- epoch-unique wrapper ----------------------------------------------------
@dataclass
class EpochUniquePicker:
    base: Node50Picker
    p_top: float = 0.5
    on_exhaust: Literal["reset", "allow_repeat"] = "allow_repeat"
    key_fn: Optional[Callable[[Any], Any]] = None  # map literal -> uniqueness key

    _rng: np.random.Generator = field(init=False, repr=False)
    _seen: set = field(default_factory=set, init=False, repr=False)
    _top_perm: np.ndarray | None = field(default=None, init=False, repr=False)
    _rest_perm: np.ndarray | None = field(default=None, init=False, repr=False)
    _i_top: int = field(default=0, init=False, repr=False)
    _i_rest: int = field(default=0, init=False, repr=False)

    def __post_init__(self):
        self._rng = self.base.rng
        if self.key_fn is None:
            # default: uniqueness by the full literal
            self.key_fn = lambda x: x

    def start_epoch(self):
        self._seen.clear()
        self._top_perm = self._rng.permutation(self.base.top_nodes)
        self._rest_perm = self._rng.permutation(self.base.other_nodes)
        self._i_top = 0
        self._i_rest = 0

    def _draw_from(self, which: str):
        if which == "top" and self._top_perm is not None:
            if self._i_top < len(self._top_perm):
                v = self._top_perm[self._i_top]
                self._i_top += 1
                return v
        else:
            if self._rest_perm is not None and self._i_rest < len(self._rest_perm):
                v = self._rest_perm[self._i_rest]
                self._i_rest += 1
                return v
        return None

    def pick(self):
        prefer_top = self._rng.random() < self.p_top
        node = self._draw_from("top" if prefer_top else "rest")
        if node is None:
            node = self._draw_from("rest" if prefer_top else "top")

        if node is None:
            if self.on_exhaust == "reset":
                self.start_epoch()
                return self.pick()
            # fallback: allow repeats using base picker (with replacement)
            return self.base.pick()

        if self.key_fn is None:
            raise ValueError("key_fn should have been set in __post_init__")
        key = self.key_fn(node)
        if key in self._seen:
            # try the other pool once
            alt = self._draw_from("rest" if prefer_top else "top")
            if alt is None:
                if self.on_exhaust == "reset":
                    self.start_epoch()
                    return self.pick()
                return self.base.pick()
            node = alt
            key = self.key_fn(node)

        self._seen.add(key)
        return node  # <-- return the literal as-is

    def remaining(self):
        if self._top_perm is None or self._rest_perm is None:
            raise ValueError("Call start_epoch() before remaining()")
        return (len(self._top_perm) - self._i_top, len(self._rest_perm) - self._i_rest)
