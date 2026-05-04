#!/usr/bin/env python3
"""
Reusable utilities for real-time congestion evaluation.

This module packages the calibration and metric logic used in the analysis:
- exact and +-1 class accuracy
- monotonic quantile-map calibration
- hybrid raw/qmap calibration
- metrics by model, level, and forecast step
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


@dataclass
class QuantileMapCalibrator:
    x_quantiles: np.ndarray
    y_quantiles: np.ndarray

    def transform(self, values: pd.Series | np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=float)
        return np.interp(
            arr,
            self.x_quantiles,
            self.y_quantiles,
            left=self.y_quantiles[0],
            right=self.y_quantiles[-1],
        )

    def to_dict(self) -> dict[str, list[float]]:
        return {
            "x_quantiles": self.x_quantiles.astype(float).tolist(),
            "y_quantiles": self.y_quantiles.astype(float).tolist(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, list[float]]) -> "QuantileMapCalibrator":
        return cls(
            x_quantiles=np.asarray(payload["x_quantiles"], dtype=float),
            y_quantiles=np.asarray(payload["y_quantiles"], dtype=float),
        )


@dataclass
class HurdleQMapCalibrator:
    zero_threshold: float
    positive_x_quantiles: np.ndarray
    positive_y_quantiles: np.ndarray
    zero_blend_width: float = 0.2

    def transform(self, values: pd.Series | np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=float)
        positive = np.interp(
            arr,
            self.positive_x_quantiles,
            self.positive_y_quantiles,
            left=self.positive_y_quantiles[0],
            right=self.positive_y_quantiles[-1],
        )
        if self.zero_blend_width <= 0:
            weight = (arr >= self.zero_threshold).astype(float)
        else:
            weight = np.clip((arr - self.zero_threshold) / self.zero_blend_width, 0.0, 1.0)
        return weight * positive

    def to_dict(self) -> dict[str, float | list[float]]:
        return {
            "zero_threshold": float(self.zero_threshold),
            "zero_blend_width": float(self.zero_blend_width),
            "positive_x_quantiles": self.positive_x_quantiles.astype(float).tolist(),
            "positive_y_quantiles": self.positive_y_quantiles.astype(float).tolist(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, float | list[float]]) -> "HurdleQMapCalibrator":
        return cls(
            zero_threshold=float(payload["zero_threshold"]),
            zero_blend_width=float(payload.get("zero_blend_width", 0.2)),
            positive_x_quantiles=np.asarray(payload["positive_x_quantiles"], dtype=float),
            positive_y_quantiles=np.asarray(payload["positive_y_quantiles"], dtype=float),
        )


def discretize_level(values: pd.Series | np.ndarray, min_level: int = 0, max_level: int = 4) -> np.ndarray:
    return np.clip(np.rint(np.asarray(values, dtype=float)), min_level, max_level).astype(int)


def fit_quantile_map(
    pred: pd.Series | np.ndarray,
    actual: pd.Series | np.ndarray,
    quantiles: int = 201,
) -> QuantileMapCalibrator:
    pred_arr = np.asarray(pred, dtype=float)
    actual_arr = np.asarray(actual, dtype=float)
    q = np.linspace(0.0, 1.0, quantiles)
    xq = np.quantile(pred_arr, q)
    yq = np.quantile(actual_arr, q)
    x_unique, keep_idx = np.unique(xq, return_index=True)
    y_unique = yq[keep_idx]
    y_mono = np.maximum.accumulate(y_unique)
    return QuantileMapCalibrator(x_quantiles=x_unique, y_quantiles=y_mono)


def fit_zero_gate(
    pred: pd.Series | np.ndarray,
    actual: pd.Series | np.ndarray,
    candidates: int = 201,
) -> float:
    pred_arr = np.asarray(pred, dtype=float)
    actual_arr = np.asarray(actual, dtype=float)
    positive_mask = actual_arr > 0
    thresholds = np.unique(np.quantile(pred_arr, np.linspace(0.0, 1.0, candidates)))
    best_threshold = float(thresholds[0])
    best_score = -np.inf
    for threshold in thresholds:
        pred_positive = pred_arr >= threshold
        tpr = float(np.mean(pred_positive[positive_mask])) if np.any(positive_mask) else 0.0
        tnr = float(np.mean(~pred_positive[~positive_mask])) if np.any(~positive_mask) else 0.0
        score = 0.5 * (tpr + tnr)
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold


def fit_hurdle_qmap(
    pred: pd.Series | np.ndarray,
    actual: pd.Series | np.ndarray,
    quantiles: int = 201,
    zero_blend_width: float = 0.2,
) -> HurdleQMapCalibrator:
    pred_arr = np.asarray(pred, dtype=float)
    actual_arr = np.asarray(actual, dtype=float)
    zero_threshold = fit_zero_gate(pred_arr, actual_arr)
    positive_mask = actual_arr > 0
    if not np.any(positive_mask):
        positive_x = np.asarray([zero_threshold, zero_threshold + 1e-6], dtype=float)
        positive_y = np.asarray([0.0, 0.0], dtype=float)
    else:
        positive_cal = fit_quantile_map(pred_arr[positive_mask], actual_arr[positive_mask], quantiles=quantiles)
        positive_x = positive_cal.x_quantiles
        positive_y = positive_cal.y_quantiles
    return HurdleQMapCalibrator(
        zero_threshold=zero_threshold,
        positive_x_quantiles=positive_x,
        positive_y_quantiles=positive_y,
        zero_blend_width=zero_blend_width,
    )


def apply_hybrid_calibration(
    pred: pd.Series | np.ndarray,
    calibrator: QuantileMapCalibrator,
    threshold: float,
    blend_width: float = 0.2,
) -> np.ndarray:
    pred_arr = np.asarray(pred, dtype=float)
    qmap = calibrator.transform(pred_arr)
    if blend_width <= 0:
        return np.where(pred_arr < threshold, pred_arr, qmap)
    weight = np.clip((pred_arr - threshold) / blend_width, 0.0, 1.0)
    return (1.0 - weight) * pred_arr + weight * qmap


def metric_row(actual: pd.Series | np.ndarray, pred: pd.Series | np.ndarray) -> dict[str, float]:
    actual_arr = np.asarray(actual, dtype=float)
    pred_arr = np.asarray(pred, dtype=float)
    pred_cls = discretize_level(pred_arr)
    actual_cls = discretize_level(actual_arr)
    return {
        "count": int(len(actual_arr)),
        "mae": float(np.mean(np.abs(pred_arr - actual_arr))),
        "accuracy_exact": float(np.mean(pred_cls == actual_cls)),
        "accuracy_pm1": float(np.mean(np.abs(pred_cls - actual_cls) <= 1)),
    }


def build_model_predictions(
    df: pd.DataFrame,
    calibrator: QuantileMapCalibrator,
    hurdle_calibrator: HurdleQMapCalibrator | None = None,
    thresholds: tuple[float, ...] = (0.6, 0.8),
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {
        "raw": df["predicted_level"].to_numpy(dtype=float),
        "qmap_global": calibrator.transform(df["predicted_level"]),
    }
    if hurdle_calibrator is not None:
        out["hurdle_qmap"] = hurdle_calibrator.transform(df["predicted_level"])
    for thr in thresholds:
        name = f"hybrid_{str(thr).replace('.', '_')}"
        out[name] = apply_hybrid_calibration(df["predicted_level"], calibrator, threshold=thr)
    return out


def summarize_overall(df: pd.DataFrame, model_preds: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for model, pred in model_preds.items():
        row = {"model": model}
        row.update(metric_row(df["level_mean"], pred))
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_by_level(df: pd.DataFrame, model_preds: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    actual_level = discretize_level(df["level_mean"])
    for level in [0, 1, 2, 3, 4]:
        mask = actual_level == level
        for model, pred in model_preds.items():
            row = {"actual_level": level, "model": model}
            row.update(metric_row(df.loc[mask, "level_mean"], pred[mask]))
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_by_timestep_and_level(
    df: pd.DataFrame,
    model_preds: dict[str, np.ndarray],
) -> pd.DataFrame:
    rows = []
    actual_level = discretize_level(df["level_mean"])
    for step in sorted(df["forecast_step"].unique()):
        step_mask = df["forecast_step"] == step
        for level in [0, 1, 2, 3, 4]:
            mask = step_mask & (actual_level == level)
            if not np.any(mask):
                continue
            row = {"forecast_step": int(step), "actual_level": level, "count": int(np.sum(mask))}
            for model, pred in model_preds.items():
                metrics = metric_row(df.loc[mask, "level_mean"], pred[mask])
                row[f"{model}_mae"] = metrics["mae"]
                row[f"{model}_acc_exact"] = metrics["accuracy_exact"]
                row[f"{model}_acc_pm1"] = metrics["accuracy_pm1"]
            rows.append(row)
    return pd.DataFrame(rows)


def confusion_matrix_table(
    actual: pd.Series | np.ndarray,
    pred: pd.Series | np.ndarray,
) -> pd.DataFrame:
    actual_cls = discretize_level(actual)
    pred_cls = discretize_level(pred)
    matrix = pd.crosstab(
        pd.Series(actual_cls, name="actual"),
        pd.Series(pred_cls, name="predicted"),
        normalize="index",
    )
    return matrix.reindex(index=[0, 1, 2, 3, 4], columns=[0, 1, 2, 3, 4], fill_value=0.0)


def fit_from_train_df(train_df: pd.DataFrame) -> QuantileMapCalibrator:
    return fit_quantile_map(train_df["predicted_level"], train_df["level_mean"])


def save_calibrator_artifact(
    calibrator: QuantileMapCalibrator,
    output_path: str | Path,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(calibrator.to_dict(), indent=2))
    return output_path


def load_calibrator_artifact(path: str | Path) -> QuantileMapCalibrator:
    payload = json.loads(Path(path).read_text())
    return QuantileMapCalibrator.from_dict(payload)


def export_calibration_bundle(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    output_dir: str | Path,
    thresholds: tuple[float, ...] = (0.6, 0.8),
    selection_metric: str = "accuracy_exact",
) -> dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    calibrator = fit_from_train_df(train_df)
    hurdle_calibrator = fit_hurdle_qmap(train_df["predicted_level"], train_df["level_mean"])
    artifact = {
        "calibrator_type": "composite",
        "qmap_global": calibrator.to_dict(),
        "hurdle_qmap": hurdle_calibrator.to_dict(),
    }
    output_path = output_dir / "qmap_calibrator.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2))

    model_preds = build_model_predictions(
        eval_df,
        calibrator,
        hurdle_calibrator=hurdle_calibrator,
        thresholds=thresholds,
    )
    overall_df = summarize_overall(eval_df, model_preds).sort_values(
        [selection_metric, "mae", "model"],
        ascending=[False, True, True],
    )
    overall_df.to_csv(output_dir / "overall_metrics.csv", index=False)
    summarize_by_level(eval_df, model_preds).to_csv(
        output_dir / "metrics_by_level.csv", index=False
    )
    if "forecast_step" in eval_df.columns:
        summarize_by_timestep_and_level(eval_df, model_preds).to_csv(
            output_dir / "metrics_by_timestep_and_level.csv", index=False
        )

    best_model = str(overall_df.iloc[0]["model"])
    best_threshold = None
    if best_model.startswith("hybrid_"):
        best_threshold = float(best_model.replace("hybrid_", "").replace("_", "."))

    transform_config = {
        "selected_model": best_model,
        "selection_metric": selection_metric,
        "available_models": list(model_preds.keys()),
        "raw_column": "predicted_level",
        "qmap_artifact": "qmap_calibrator.json",
        "thresholds": list(thresholds),
        "best_hybrid_threshold": best_threshold,
        "hybrid_blend_width": 0.2,
    }
    (output_dir / "transform_config.json").write_text(
        json.dumps(transform_config, indent=2)
    )

    summary = {
        "selected_model": best_model,
        "selection_metric": selection_metric,
        "thresholds": list(thresholds),
        "best_hybrid_threshold": best_threshold,
        "overall_metrics": overall_df.to_dict(orient="records"),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
