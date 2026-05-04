import argparse
import json
import logging
import os
import random
import shutil
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.models.gnn_temporal import GNN_Temporal_Model
from src.sparse_dataloader import TrafficDataset
from src.training_entrypoint import (
    _extract_level_tensors,
    _get_level_scale,
    _str2bool,
    _to_real_level_space,
    cache_training_data,
    check_data_channels,
    setup_sagemaker_paths,
)
from src.training_utils import SeededBatchSampler, pad_collate

LOG_LEVEL = int(os.getenv("SM_LOG_LEVEL", logging.INFO))
MODEL_DIR = Path(os.getenv("SM_MODEL_DIR", "/opt/ml/model"))

logging.basicConfig(
    level=LOG_LEVEL,
    format="[%(asctime)s] %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-artifact-path", required=True)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--input-seq-len", type=int, default=150)
    parser.add_argument("--output-seq-len", type=int, default=50)
    parser.add_argument("--node-batch-size", type=int, default=500)
    parser.add_argument("--include-weather", type=_str2bool, default=True)
    parser.add_argument("--include-velocity-target", type=_str2bool, default=True)
    parser.add_argument("--use-minimal-datasets", type=_str2bool, default=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--pin-memory", type=_str2bool, default=True)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--persistent-workers", type=_str2bool, default=True)
    parser.add_argument("--drop-last", type=_str2bool, default=False)
    parser.add_argument("--cache-local", type=_str2bool, default=True)
    parser.add_argument("--cache-dir", type=str, default="/opt/ml/input/data/cache")
    parser.add_argument("--calibration-max-pairs", type=int, default=200000)
    parser.add_argument("--qmap-quantiles", type=int, default=201)
    parser.add_argument("--calibration-thresholds", type=str, default="0.6,0.8")
    parser.add_argument("--selection-metric", type=str, default="accuracy_exact")
    return parser.parse_args()


def _copy_or_extract_model_artifact(artifact_path: str, work_dir: Path) -> Path:
    work_dir.mkdir(parents=True, exist_ok=True)
    if artifact_path.startswith("s3://"):
        import boto3

        bucket_key = artifact_path[len("s3://") :]
        bucket, key = bucket_key.split("/", 1)
        local_tgz = work_dir / Path(key).name
        boto3.client("s3").download_file(bucket, key, str(local_tgz))
    else:
        local_candidate = Path(artifact_path)
        if local_candidate.is_dir():
            return local_candidate
        local_tgz = work_dir / local_candidate.name
        shutil.copy2(local_candidate, local_tgz)

    extract_dir = work_dir / "model_artifact"
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(local_tgz, "r:gz") as tar:
        tar.extractall(extract_dir)
    model_path = extract_dir / "model.pth"
    if model_path.exists():
        return extract_dir
    nested_model_paths = list(extract_dir.rglob("model.pth"))
    if not nested_model_paths:
        logger.warning("model.pth not found after extracting artifact %s into %s", artifact_path, extract_dir)
        return extract_dir
    if len(nested_model_paths) > 1:
        logger.warning(
            "Multiple model.pth files found after extracting %s; using %s",
            artifact_path,
            nested_model_paths[0].parent,
        )
    return nested_model_paths[0].parent


def _load_checkpoint_model(model_dir: Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    model_path = model_dir / "model.pth"
    if not model_path.exists():
        raise FileNotFoundError(f"model.pth not found in {model_dir}")
    payload = torch.load(model_path, map_location=device)
    model_kwargs = payload["model_kwargs"]
    model = GNN_Temporal_Model(edge_index=None, **model_kwargs).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload


class ReservoirPairCollector:
    def __init__(self, max_pairs: int, seed: int = 42) -> None:
        self.max_pairs = max_pairs
        self.rng = random.Random(seed)
        self.total_seen = 0
        self.pred_buf = np.empty(max_pairs, dtype=np.float32)
        self.target_buf = np.empty(max_pairs, dtype=np.float32)
        self.step_buf = np.empty(max_pairs, dtype=np.int16)
        self.size = 0

    def add(self, preds: torch.Tensor, targets: torch.Tensor, forecast_steps: torch.Tensor) -> None:
        pred_np = preds.detach().cpu().reshape(-1).numpy().astype(np.float32, copy=False)
        target_np = targets.detach().cpu().reshape(-1).numpy().astype(np.float32, copy=False)
        step_np = forecast_steps.detach().cpu().reshape(-1).numpy().astype(np.int16, copy=False)
        valid = np.isfinite(pred_np) & np.isfinite(target_np) & np.isfinite(step_np)
        pred_np = pred_np[valid]
        target_np = target_np[valid]
        step_np = step_np[valid]
        for pred, target, step in zip(pred_np, target_np, step_np):
            self.total_seen += 1
            if self.size < self.max_pairs:
                self.pred_buf[self.size] = pred
                self.target_buf[self.size] = target
                self.step_buf[self.size] = step
                self.size += 1
                continue
            j = self.rng.randint(0, self.total_seen - 1)
            if j < self.max_pairs:
                self.pred_buf[j] = pred
                self.target_buf[j] = target
                self.step_buf[j] = step

    def arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            self.pred_buf[: self.size].copy(),
            self.target_buf[: self.size].copy(),
            self.step_buf[: self.size].copy(),
        )


def _apply_qmap(preds: np.ndarray, x_quantiles: np.ndarray, y_quantiles: np.ndarray) -> np.ndarray:
    return np.interp(preds, x_quantiles, y_quantiles, left=y_quantiles[0], right=y_quantiles[-1])


def _fit_quantile_map(preds: np.ndarray, targets: np.ndarray, quantiles: int) -> tuple[np.ndarray, np.ndarray]:
    q = np.linspace(0.0, 1.0, quantiles)
    xq = np.quantile(preds, q)
    yq = np.quantile(targets, q)
    x_unique, keep_idx = np.unique(xq, return_index=True)
    y_unique = yq[keep_idx]
    y_mono = np.maximum.accumulate(y_unique)
    return x_unique, y_mono


def _fit_zero_gate(preds: np.ndarray, targets: np.ndarray, candidates: int = 201) -> float:
    positive_mask = targets > 0
    thresholds = np.unique(np.quantile(preds, np.linspace(0.0, 1.0, candidates)))
    best_threshold = float(thresholds[0])
    best_score = -np.inf
    for threshold in thresholds:
        pred_positive = preds >= threshold
        tpr = float(np.mean(pred_positive[positive_mask])) if np.any(positive_mask) else 0.0
        tnr = float(np.mean(~pred_positive[~positive_mask])) if np.any(~positive_mask) else 0.0
        score = 0.5 * (tpr + tnr)
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold


def _apply_hurdle_qmap(
    preds: np.ndarray,
    zero_threshold: float,
    positive_x_quantiles: np.ndarray,
    positive_y_quantiles: np.ndarray,
    zero_blend_width: float = 0.2,
) -> np.ndarray:
    positive_preds = _apply_qmap(preds, positive_x_quantiles, positive_y_quantiles)
    if zero_blend_width <= 0:
        weight = (preds >= zero_threshold).astype(float)
    else:
        weight = np.clip((preds - zero_threshold) / zero_blend_width, 0.0, 1.0)
    return weight * positive_preds


def _apply_hybrid(preds: np.ndarray, qmap_preds: np.ndarray, threshold: float, blend_width: float = 0.2) -> np.ndarray:
    if blend_width <= 0:
        return np.where(preds < threshold, preds, qmap_preds)
    weight = np.clip((preds - threshold) / blend_width, 0.0, 1.0)
    return (1.0 - weight) * preds + weight * qmap_preds


def _compute_basic_metrics(preds: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    pred_levels = np.clip(np.rint(preds).astype(int), 0, 5)
    target_levels = np.clip(np.rint(targets).astype(int), 0, 5)
    distances = np.abs(pred_levels - target_levels)
    return {
        "count": int(len(preds)),
        "mae": float(np.mean(np.abs(preds - targets))) if len(preds) else float("nan"),
        "accuracy_exact": float(np.mean(distances == 0)) if len(preds) else float("nan"),
        "accuracy_pm1": float(np.mean(distances <= 1)) if len(preds) else float("nan"),
    }


def _summarize_overall(eval_df: pd.DataFrame, model_preds: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    targets = eval_df["level_mean"].to_numpy(dtype=float)
    for model_name, pred in model_preds.items():
        rows.append({"model": model_name, **_compute_basic_metrics(pred, targets)})
    return pd.DataFrame(rows)


def _summarize_by_level(eval_df: pd.DataFrame, model_preds: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    actual_levels = np.clip(np.rint(eval_df["level_mean"].to_numpy()).astype(int), 0, 5)
    targets = eval_df["level_mean"].to_numpy(dtype=float)
    for level in range(6):
        mask = actual_levels == level
        for model_name, pred in model_preds.items():
            rows.append({"actual_level": level, "model": model_name, **_compute_basic_metrics(pred[mask], targets[mask])})
    return pd.DataFrame(rows)


def _summarize_by_timestep_and_level(eval_df: pd.DataFrame, model_preds: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    actual_levels = np.clip(np.rint(eval_df["level_mean"].to_numpy()).astype(int), 0, 5)
    targets = eval_df["level_mean"].to_numpy(dtype=float)
    steps = eval_df["forecast_step"].to_numpy(dtype=int)
    for step in sorted(np.unique(steps)):
        step_mask = steps == step
        for level in range(6):
            mask = step_mask & (actual_levels == level)
            if not np.any(mask):
                continue
            row: dict[str, float | int] = {
                "forecast_step": int(step),
                "actual_level": int(level),
                "count": int(mask.sum()),
            }
            for model_name, pred in model_preds.items():
                metrics = _compute_basic_metrics(pred[mask], targets[mask])
                row[f"{model_name}_mae"] = metrics["mae"]
                row[f"{model_name}_acc_exact"] = metrics["accuracy_exact"]
                row[f"{model_name}_acc_pm1"] = metrics["accuracy_pm1"]
            rows.append(row)
    return pd.DataFrame(rows)


def _log_metric_group(prefix: str, metrics: dict[str, float]) -> None:
    parts = []
    for key, value in metrics.items():
        if value is None or not np.isfinite(value):
            continue
        parts.append(f"{key}={float(value):.6f}")
    if parts:
        logger.info("[METRIC] %s phase=%s", " ".join(parts), prefix)


def _make_seeded_loader(
    dataset: TrafficDataset,
    time_indices: list[int],
    num_time_windows: int,
    batch_size: int,
    drop_last: bool,
    loader_kwargs: dict,
    seed_offset: int,
) -> DataLoader:
    all_seed_ids = list(dataset.batch_mapping.keys())
    if not all_seed_ids:
        raise ValueError("Dataset has no batch_mapping seeds")
    num_batches = int(np.ceil(len(time_indices) / batch_size))
    seed_list = (all_seed_ids * int(np.ceil(num_batches / len(all_seed_ids))))[:num_batches]
    dataset.set_seed_list(seed_list)
    return DataLoader(
        dataset,
        batch_sampler=SeededBatchSampler(
            seed_list=seed_list,
            time_indices=time_indices,
            num_time_windows=num_time_windows,
            batch_size=batch_size,
            drop_last=drop_last,
            shuffle_time=False,
            rng_seed=42 + seed_offset,
        ),
        **loader_kwargs,
    )


def _run_collection_pass(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_pairs: int,
    seed: int,
    desc: str,
    level_scale: float,
) -> ReservoirPairCollector:
    collector = ReservoirPairCollector(max_pairs=max_pairs, seed=seed)
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc, leave=True):
            x_static = batch["x_static"].to(device)
            x_dyn_past = batch["x_dynamic_past"].to(device)
            x_dyn_future = batch["x_dynamic_future"].to(device)
            y_target = batch["y"].to(device)
            edge_index = batch["edge_index"].to(device)
            preds = model(
                edge_index,
                x_static[0],
                x_dyn_past.permute(0, 2, 1, 3),
                x_dyn_future.permute(0, 2, 1, 3),
            )
            level_pred, level_target = _extract_level_tensors(preds, y_target)
            level_pred, level_target = _to_real_level_space(level_pred, level_target, level_scale)
            forecast_steps = (
                torch.arange(1, level_pred.shape[-1] + 1, device=level_pred.device, dtype=torch.int16)
                .view(1, 1, -1)
                .expand_as(level_pred)
            )
            collector.add(level_pred, level_target, forecast_steps)
    return collector


def main() -> None:
    args = parse_args()
    thresholds = tuple(float(v.strip()) for v in args.calibration_thresholds.split(",") if v.strip())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_path = check_data_channels()
    if args.cache_local:
        cache_dir = Path(args.cache_dir)
        train_path = cache_training_data(
            train_path=train_path,
            cache_dir=cache_dir,
            include_weather=args.include_weather,
            use_minimal_datasets=args.use_minimal_datasets,
        )

    paths = setup_sagemaker_paths(
        train_path=train_path,
        include_weather=args.include_weather,
        use_minimal_datasets=args.use_minimal_datasets,
    )

    model_artifact_dir = _copy_or_extract_model_artifact(
        args.model_artifact_path,
        Path(args.cache_dir) / "calibration_model_artifact",
    )
    model, payload = _load_checkpoint_model(model_artifact_dir, device)
    model_kwargs = payload["model_kwargs"]

    dataset = TrafficDataset(
        graph_path=str(paths["graph"]),
        static_data_path=str(paths["static"]),
        dynamic_data_paths=[str(p) for p in paths["dynamic"]],
        metadata_paths=[str(p) for p in paths["metadata"]],
        target_data_path=str(paths["target"]),
        in_timesteps=args.input_seq_len,
        out_timesteps=args.output_seq_len,
        future_features=["coi_puntos", "coi_tramo", "AGLOMERACION", "EMERGENCIA", "MANIFESTACION"],
        node_batch_size=args.node_batch_size,
        weather_data_dir=str(paths["weather_dir"]),
        include_weather=args.include_weather,
        neighbor_features_path=str(paths["neighbor_features"]) if paths["neighbor_features"] else None,
        include_velocity_target=args.include_velocity_target,
        normalization_constants_path=str(paths["normalization"]) if paths["normalization"] else None,
    )
    level_scale = _get_level_scale(getattr(dataset, "target_norm_constants", None))

    num_time_windows = dataset.effective_total_timesteps - dataset.total_timesteps_window + 1
    train_end = int(num_time_windows * args.train_ratio)
    val_time_indices_all = list(range(train_end, num_time_windows))
    if not val_time_indices_all:
        raise ValueError("No validation time indices available")
    split_idx = max(1, len(val_time_indices_all) // 2)
    calib_time_indices = val_time_indices_all[:split_idx]
    eval_time_indices = val_time_indices_all[split_idx:] or val_time_indices_all[:split_idx]

    loader_kwargs = {"num_workers": args.num_workers, "collate_fn": pad_collate, "pin_memory": args.pin_memory}
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
        loader_kwargs["persistent_workers"] = args.persistent_workers

    fit_collector = _run_collection_pass(
        model=model,
        loader=_make_seeded_loader(
            dataset=dataset,
            time_indices=calib_time_indices,
            num_time_windows=num_time_windows,
            batch_size=args.batch_size,
            drop_last=args.drop_last,
            loader_kwargs=loader_kwargs,
            seed_offset=0,
        ),
        device=device,
        max_pairs=args.calibration_max_pairs,
        seed=42,
        desc="Calibration/Fit",
        level_scale=level_scale,
    )
    eval_collector = _run_collection_pass(
        model=model,
        loader=_make_seeded_loader(
            dataset=dataset,
            time_indices=eval_time_indices,
            num_time_windows=num_time_windows,
            batch_size=args.batch_size,
            drop_last=args.drop_last,
            loader_kwargs=loader_kwargs,
            seed_offset=1,
        ),
        device=device,
        max_pairs=args.calibration_max_pairs,
        seed=43,
        desc="Calibration/Eval",
        level_scale=level_scale,
    )

    pred_fit, target_fit, _ = fit_collector.arrays()
    pred_eval, target_eval, forecast_step_eval = eval_collector.arrays()
    x_quantiles, y_quantiles = _fit_quantile_map(pred_fit, target_fit, args.qmap_quantiles)
    qmap_pred = _apply_qmap(pred_eval, x_quantiles, y_quantiles)
    zero_threshold = _fit_zero_gate(pred_fit, target_fit)
    positive_mask = target_fit > 0
    if np.any(positive_mask):
        pos_x_quantiles, pos_y_quantiles = _fit_quantile_map(
            pred_fit[positive_mask],
            target_fit[positive_mask],
            args.qmap_quantiles,
        )
    else:
        pos_x_quantiles = np.asarray([zero_threshold, zero_threshold + 1e-6], dtype=float)
        pos_y_quantiles = np.asarray([0.0, 0.0], dtype=float)
    hurdle_qmap_pred = _apply_hurdle_qmap(
        pred_eval,
        zero_threshold=zero_threshold,
        positive_x_quantiles=pos_x_quantiles,
        positive_y_quantiles=pos_y_quantiles,
    )

    model_preds: dict[str, np.ndarray] = {
        "raw": pred_eval,
        "qmap_global": qmap_pred,
        "hurdle_qmap": hurdle_qmap_pred,
    }
    for thr in thresholds:
        model_preds[f"hybrid_{str(thr).replace('.', '_')}"] = _apply_hybrid(pred_eval, qmap_pred, threshold=thr)

    eval_df = pd.DataFrame(
        {
            "predicted_level": pred_eval,
            "level_mean": target_eval,
            "forecast_step": forecast_step_eval,
        }
    )
    overall_metrics = _summarize_overall(eval_df, model_preds).sort_values(
        [args.selection_metric, "mae", "model"],
        ascending=[False, True, True],
    )
    metrics_by_level = _summarize_by_level(eval_df, model_preds)
    metrics_by_timestep_and_level = _summarize_by_timestep_and_level(eval_df, model_preds)

    best_model = str(overall_metrics.iloc[0]["model"])
    best_threshold = None
    if best_model.startswith("hybrid_"):
        best_threshold = float(best_model.replace("hybrid_", "").replace("_", "."))

    for row in overall_metrics.to_dict(orient="records"):
        model_name = str(row.pop("model"))
        _log_metric_group(
            prefix=f"calibration_overall_{model_name}",
            metrics={f"{model_name}_{k}": float(v) for k, v in row.items() if isinstance(v, (float, int))},
        )
    for row in metrics_by_level.to_dict(orient="records"):
        level = int(row.pop("actual_level"))
        model_name = str(row.pop("model"))
        _log_metric_group(
            prefix=f"calibration_level_{model_name}_{level}",
            metrics={
                f"{model_name}_level_{level}_{k}": float(v)
                for k, v in row.items()
                if isinstance(v, (float, int))
            },
        )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    (MODEL_DIR / "qmap_calibrator.json").write_text(
        json.dumps(
            {
                "level_scale": float(level_scale),
                "calibration_space": "real_level_0_to_4",
                "calibrator_type": "composite",
                "qmap_global": {
                    "x_quantiles": x_quantiles.tolist(),
                    "y_quantiles": y_quantiles.tolist(),
                },
                "hurdle_qmap": {
                    "zero_threshold": float(zero_threshold),
                    "zero_blend_width": 0.2,
                    "positive_x_quantiles": pos_x_quantiles.tolist(),
                    "positive_y_quantiles": pos_y_quantiles.tolist(),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (MODEL_DIR / "transform_config.json").write_text(
        json.dumps(
            {
                "selected_model": best_model,
                "selection_metric": args.selection_metric,
                "available_models": list(model_preds.keys()),
                "raw_column": "predicted_level",
                "level_scale": float(level_scale),
                "calibration_space": "real_level_0_to_4",
                "qmap_artifact": "qmap_calibrator.json",
                "thresholds": list(thresholds),
                "best_hybrid_threshold": best_threshold,
                "hybrid_blend_width": 0.2,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (MODEL_DIR / "summary.json").write_text(
        json.dumps(
            {
                "selected_model": best_model,
                "selection_metric": args.selection_metric,
                "thresholds": list(thresholds),
                "best_hybrid_threshold": best_threshold,
                "overall_metrics": overall_metrics.to_dict(orient="records"),
                "fit_sampled_pairs": int(len(pred_fit)),
                "eval_sampled_pairs": int(len(pred_eval)),
                "fit_total_pairs_seen": int(fit_collector.total_seen),
                "eval_total_pairs_seen": int(eval_collector.total_seen),
                "model_kwargs": model_kwargs,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    overall_metrics.to_csv(MODEL_DIR / "overall_metrics.csv", index=False)
    metrics_by_level.to_csv(MODEL_DIR / "metrics_by_level.csv", index=False)
    metrics_by_timestep_and_level.to_csv(MODEL_DIR / "metrics_by_timestep_and_level.csv", index=False)
    logger.info("Calibration artifacts written to %s", MODEL_DIR)


if __name__ == "__main__":
    main()
