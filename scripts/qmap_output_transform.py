#!/usr/bin/env python3
"""
Minimal helper to apply the selected output calibration artifact.

Current selected transform:
- qmap_global

Required artifacts:
- qmap_calibrator.json
- transform_config.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


class QuantileMapCalibrator:
    def __init__(self, x_quantiles: list[float], y_quantiles: list[float]) -> None:
        self.x_quantiles = np.asarray(x_quantiles, dtype=float)
        self.y_quantiles = np.asarray(y_quantiles, dtype=float)

    def transform(self, values: np.ndarray | list[float]) -> np.ndarray:
        arr = np.asarray(values, dtype=float)
        return np.interp(
            arr,
            self.x_quantiles,
            self.y_quantiles,
            left=self.y_quantiles[0],
            right=self.y_quantiles[-1],
        )


class HurdleQMapCalibrator:
    def __init__(
        self,
        zero_threshold: float,
        positive_x_quantiles: list[float],
        positive_y_quantiles: list[float],
        zero_blend_width: float = 0.2,
    ) -> None:
        self.zero_threshold = float(zero_threshold)
        self.zero_blend_width = float(zero_blend_width)
        self.positive_x_quantiles = np.asarray(positive_x_quantiles, dtype=float)
        self.positive_y_quantiles = np.asarray(positive_y_quantiles, dtype=float)

    def transform(self, values: np.ndarray | list[float]) -> np.ndarray:
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


def load_transform_config(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def load_qmap_calibrator(path: str | Path) -> QuantileMapCalibrator:
    payload = json.loads(Path(path).read_text())
    if "qmap_global" in payload:
        payload = payload["qmap_global"]
    return QuantileMapCalibrator(
        x_quantiles=payload["x_quantiles"],
        y_quantiles=payload["y_quantiles"],
    )


def load_hurdle_calibrator(path: str | Path) -> HurdleQMapCalibrator:
    payload = json.loads(Path(path).read_text())
    if "hurdle_qmap" in payload:
        payload = payload["hurdle_qmap"]
    return HurdleQMapCalibrator(
        zero_threshold=payload["zero_threshold"],
        zero_blend_width=payload.get("zero_blend_width", 0.2),
        positive_x_quantiles=payload["positive_x_quantiles"],
        positive_y_quantiles=payload["positive_y_quantiles"],
    )


def apply_qmap_global(
    predicted_level: np.ndarray | list[float],
    calibrator_path: str | Path,
) -> np.ndarray:
    calibrator = load_qmap_calibrator(calibrator_path)
    return calibrator.transform(predicted_level)


def apply_hybrid_transform(
    predicted_level: np.ndarray | list[float],
    calibrator_path: str | Path,
    threshold: float,
    blend_width: float = 0.2,
) -> np.ndarray:
    raw = np.asarray(predicted_level, dtype=float)
    qmap = apply_qmap_global(raw, calibrator_path)
    if blend_width <= 0:
        return np.where(raw < threshold, raw, qmap)
    weight = np.clip((raw - threshold) / blend_width, 0.0, 1.0)
    return (1.0 - weight) * raw + weight * qmap


def apply_hurdle_qmap(
    predicted_level: np.ndarray | list[float],
    calibrator_path: str | Path,
) -> np.ndarray:
    calibrator = load_hurdle_calibrator(calibrator_path)
    return calibrator.transform(predicted_level)


def apply_selected_transform(
    predicted_level: np.ndarray | list[float],
    calibrator_path: str | Path,
    transform_config_path: str | Path,
    selected_model: str | None = None,
) -> np.ndarray:
    config = load_transform_config(transform_config_path)
    selected_model = selected_model or config["selected_model"]

    if selected_model == "raw":
        return np.asarray(predicted_level, dtype=float)

    if selected_model == "qmap_global":
        return apply_qmap_global(predicted_level, calibrator_path)

    if selected_model == "hurdle_qmap":
        return apply_hurdle_qmap(predicted_level, calibrator_path)

    if selected_model.startswith("hybrid_"):
        threshold = float(selected_model.replace("hybrid_", "").replace("_", "."))
        blend_width = float(config.get("hybrid_blend_width", 0.2))
        return apply_hybrid_transform(
            predicted_level=predicted_level,
            calibrator_path=calibrator_path,
            threshold=threshold,
            blend_width=blend_width,
        )

    raise ValueError(
        f"Unsupported selected_model={selected_model!r}. "
        "Supported values: raw, qmap_global, hurdle_qmap, hybrid_0_6, hybrid_0_8."
    )


if __name__ == "__main__":
    # Example usage for quick manual verification.
    artifact_dir = Path("data/artifacts/calibration_realtime_repr")
    raw = np.array([0.1, 0.5, 1.0, 2.0, 3.0], dtype=float)
    calibrated = apply_selected_transform(
        predicted_level=raw,
        calibrator_path=artifact_dir / "qmap_calibrator.json",
        transform_config_path=artifact_dir / "transform_config.json",
        selected_model=None,
    )
    print("raw:", raw.tolist())
    print("calibrated:", calibrated.tolist())
