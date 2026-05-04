from __future__ import annotations

import torch


def _weighted_nodewise_pearson_loss(
    pred_real: torch.Tensor,
    target_real: torch.Tensor,
    *,
    eps: float = 1e-6,
    min_std: float = 1e-3,
    weight_std: float = 0.75,
    weight_range: float = 0.25,
    weight_max: float = 1.0,
) -> torch.Tensor:
    """Weighted node-wise Pearson shape loss over the forecast horizon.

    The last dimension is treated as the temporal horizon. All previous
    dimensions are flattened into independent node-series.
    """
    if pred_real.shape != target_real.shape:
        raise ValueError(
            "pred_real and target_real must share the same shape for shape loss. "
            f"Got {tuple(pred_real.shape)} vs {tuple(target_real.shape)}."
        )
    if pred_real.dim() < 2:
        raise ValueError(
            "Pearson shape loss expects a temporal horizon in the last dimension."
        )

    horizon = pred_real.shape[-1]
    pred_series = pred_real.reshape(-1, horizon)
    target_series = target_real.reshape(-1, horizon)
    finite_mask = torch.isfinite(pred_series) & torch.isfinite(target_series)

    valid_counts = finite_mask.sum(dim=1)
    if not torch.any(valid_counts >= 2):
        return pred_real.new_zeros(())

    pred_masked = torch.where(finite_mask, pred_series, torch.zeros_like(pred_series))
    target_masked = torch.where(
        finite_mask, target_series, torch.zeros_like(target_series)
    )

    counts = valid_counts.to(dtype=pred_real.dtype).clamp_min(1.0)
    pred_mean = pred_masked.sum(dim=1) / counts
    target_mean = target_masked.sum(dim=1) / counts

    pred_centered = torch.where(
        finite_mask, pred_series - pred_mean.unsqueeze(1), torch.zeros_like(pred_series)
    )
    target_centered = torch.where(
        finite_mask,
        target_series - target_mean.unsqueeze(1),
        torch.zeros_like(target_series),
    )

    pred_var = (pred_centered.pow(2).sum(dim=1) / counts).clamp_min(0.0)
    target_var = (target_centered.pow(2).sum(dim=1) / counts).clamp_min(0.0)
    target_std = torch.sqrt(target_var + eps)

    valid_series = (valid_counts >= 2) & (target_std > float(min_std))
    if not torch.any(valid_series):
        return pred_real.new_zeros(())

    covariance = (pred_centered * target_centered).sum(dim=1) / counts
    pred_std = torch.sqrt(pred_var + eps)
    corr = covariance / (pred_std * target_std + eps)
    corr = corr.clamp(-1.0, 1.0)

    series_range = (
        target_masked.max(dim=1).values - target_masked.min(dim=1).values
    ).clamp_min(0.0)
    series_max = target_masked.max(dim=1).values.clamp_min(0.0)
    weights = (
        1.0
        + float(weight_std) * target_std
        + float(weight_range) * series_range
        + float(weight_max) * series_max
    )
    weights = torch.where(valid_series, weights, torch.zeros_like(weights))

    weighted_loss = weights * (1.0 - corr)
    return weighted_loss.sum() / weights.sum().clamp_min(eps)


def masked_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    level_weights: torch.Tensor | None = None,
    level_scale: float = 1.0,
    continuous_loss: str = "huber",
    huber_delta: float = 1.0,
    class_ce_lambda: float = 0.3,
    class_distance_lambda: float = 0.5,
    large_error_threshold: int = 2,
    large_error_lambda: float = 0.75,
    high_level_threshold: int = 2,
    high_level_lambda: float = 0.5,
    piecewise_threshold: int = 4,
    piecewise_mae_lambda: float = 0.5,
    directional_lambda: float = 0.3,
    shape_lambda: float = 0.0,
    shape_min_std: float = 1e-3,
    shape_eps: float = 1e-6,
    shape_weight_std: float = 0.75,
    shape_weight_range: float = 0.25,
    shape_weight_max: float = 1.0,
) -> torch.Tensor:
    """Composite loss for congestion level prediction.

    Balances:
    - continuous regression quality
    - exact class alignment
    - stronger penalties for large class-distance errors
    """
    mask = torch.isfinite(target)
    pred_real = pred * float(level_scale)
    target_real = target * float(level_scale)
    target_levels = torch.clamp(target_real.round(), 0, 5).long()

    if level_weights is None:
        base_weights = torch.tensor(
            [0.25, 1.0, 2.0, 3.0, 4.0, 5.0],
            device=pred.device,
            dtype=pred.dtype,
        )
    else:
        base_weights = level_weights.to(device=pred.device, dtype=pred.dtype)

    level_weights_expanded = base_weights[target_levels]
    level_weights_expanded = torch.where(
        mask,
        level_weights_expanded,
        torch.zeros((), device=pred.device, dtype=pred.dtype),
    )

    diff = pred - target
    abs_diff = diff.abs()
    denom = level_weights_expanded.sum().clamp_min(1)

    if continuous_loss == "mae":
        base_loss = (abs_diff * level_weights_expanded).sum() / denom
    elif continuous_loss == "mse":
        base_loss = ((diff**2) * level_weights_expanded).sum() / denom
    else:
        huber = torch.where(
            abs_diff <= huber_delta,
            0.5 * diff**2,
            huber_delta * (abs_diff - 0.5 * huber_delta),
        )
        base_loss = (huber * level_weights_expanded).sum() / denom

    total_loss = base_loss

    if class_ce_lambda > 0:
        class_centers = torch.arange(
            0, 6, device=pred.device, dtype=pred.dtype
        ).view(*([1] * pred.dim()), 6)
        logits = -((pred_real.unsqueeze(-1) - class_centers) ** 2)
        ce = torch.nn.functional.cross_entropy(
            logits[mask],
            target_levels[mask],
            reduction="none",
        )
        ce_weights = level_weights_expanded[mask]
        ce_loss = (ce * ce_weights).sum() / ce_weights.sum().clamp_min(1)
        total_loss = total_loss + class_ce_lambda * ce_loss

    pred_levels = torch.clamp(pred_real.round(), 0, 5).long()
    class_distance = (pred_levels - target_levels).abs().to(pred.dtype)

    if class_distance_lambda > 0:
        distance_penalty = (class_distance * level_weights_expanded).sum() / denom
        total_loss = total_loss + class_distance_lambda * distance_penalty

    if large_error_lambda > 0:
        large_mask = mask & (class_distance >= large_error_threshold)
        if large_mask.any():
            large_distance = (class_distance[large_mask] ** 2).to(pred.dtype)
            large_weights = level_weights_expanded[large_mask]
            large_loss = (large_distance * large_weights).sum() / large_weights.sum().clamp_min(1)
            total_loss = total_loss + large_error_lambda * large_loss

    if high_level_lambda > 0:
        high_mask = mask & (target_levels >= high_level_threshold)
        if high_mask.any():
            weighted_errors = abs_diff * level_weights_expanded
            high_weight_sum = level_weights_expanded.where(
                high_mask, torch.zeros_like(level_weights_expanded)
            ).sum().clamp_min(1)
            mae_loss = weighted_errors.where(
                high_mask, torch.zeros_like(weighted_errors)
            ).sum() / high_weight_sum
            total_loss = total_loss + high_level_lambda * mae_loss

    if piecewise_mae_lambda > 0:
        severe_mask = mask & (target_real >= piecewise_threshold)
        if severe_mask.any():
            severe_errors = abs_diff
            severe_weights = level_weights_expanded.where(
                severe_mask, torch.zeros_like(level_weights_expanded)
            )
            severe_weight_sum = severe_weights.sum().clamp_min(1)
            severe_loss = (severe_errors * severe_weights).sum() / severe_weight_sum
            total_loss = total_loss + piecewise_mae_lambda * severe_loss

    if directional_lambda > 0:
        under_mask = mask & (pred < target)
        if under_mask.any():
            directional_penalty = (target - pred) * level_weights_expanded
            under_weight_sum = level_weights_expanded.where(
                under_mask, torch.zeros_like(level_weights_expanded)
            ).sum().clamp_min(1)
            penalty_sum = directional_penalty.where(
                under_mask, torch.zeros_like(directional_penalty)
            ).sum()
            total_loss = total_loss + directional_lambda * penalty_sum / under_weight_sum

    if shape_lambda > 0:
        shape_loss = _weighted_nodewise_pearson_loss(
            pred_real,
            target_real,
            eps=shape_eps,
            min_std=shape_min_std,
            weight_std=shape_weight_std,
            weight_range=shape_weight_range,
            weight_max=shape_weight_max,
        )
        total_loss = total_loss + shape_lambda * shape_loss

    return total_loss
