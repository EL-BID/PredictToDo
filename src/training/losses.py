from __future__ import annotations

import torch


def _horizon_weights_like(
    pred: torch.Tensor,
    *,
    short_steps: int = 0,
    mid_steps: int = 0,
    short_weight: float = 1.0,
    mid_weight: float = 1.0,
    long_weight: float = 1.0,
) -> torch.Tensor:
    """Build per-forecast-step weights broadcastable to ``pred``.

    The last tensor dimension is treated as the forecast horizon. Defaults are
    all ones, so this is backward compatible with the previous loss.
    """
    horizon = int(pred.shape[-1])
    weights = torch.full((horizon,), float(long_weight), device=pred.device, dtype=pred.dtype)

    short_steps = max(0, min(int(short_steps), horizon))
    mid_steps = max(short_steps, min(int(mid_steps), horizon))
    if mid_steps > short_steps:
        weights[short_steps:mid_steps] = float(mid_weight)
    if short_steps > 0:
        weights[:short_steps] = float(short_weight)

    return weights.view(*([1] * (pred.dim() - 1)), horizon)


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
    valid_mask: torch.Tensor | None = None,
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
    horizon_short_steps: int = 0,
    horizon_mid_steps: int = 0,
    horizon_short_weight: float = 1.0,
    horizon_mid_weight: float = 1.0,
    horizon_long_weight: float = 1.0,
    zero_overprediction_lambda: float = 0.0,
    zero_overprediction_threshold: float = 0.5,
    high_underprediction_lambda: float = 0.0,
    high_underprediction_threshold: int = 3,
) -> torch.Tensor:
    """Composite loss for congestion level prediction.

    Balances:
    - continuous regression quality
    - exact class alignment
    - stronger penalties for large class-distance errors
    """
    mask = torch.isfinite(target) & torch.isfinite(pred)
    if valid_mask is not None:
        valid_mask = torch.broadcast_to(
            valid_mask.to(device=target.device, dtype=torch.bool),
            target.shape,
        )
        mask = mask & valid_mask
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
    horizon_weights = _horizon_weights_like(
        pred,
        short_steps=horizon_short_steps,
        mid_steps=horizon_mid_steps,
        short_weight=horizon_short_weight,
        mid_weight=horizon_mid_weight,
        long_weight=horizon_long_weight,
    )
    element_weights = level_weights_expanded * horizon_weights

    diff = pred - target
    abs_diff = diff.abs()
    denom = element_weights.sum().clamp_min(1)

    if continuous_loss == "mae":
        base_loss = (abs_diff * element_weights).sum() / denom
    elif continuous_loss == "mse":
        base_loss = ((diff**2) * element_weights).sum() / denom
    else:
        huber = torch.where(
            abs_diff <= huber_delta,
            0.5 * diff**2,
            huber_delta * (abs_diff - 0.5 * huber_delta),
        )
        base_loss = (huber * element_weights).sum() / denom

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
        ce_weights = element_weights[mask]
        ce_loss = (ce * ce_weights).sum() / ce_weights.sum().clamp_min(1)
        total_loss = total_loss + class_ce_lambda * ce_loss

    pred_levels = torch.clamp(pred_real.round(), 0, 5).long()
    class_distance = (pred_levels - target_levels).abs().to(pred.dtype)

    if class_distance_lambda > 0:
        distance_penalty = (class_distance * element_weights).sum() / denom
        total_loss = total_loss + class_distance_lambda * distance_penalty

    if large_error_lambda > 0:
        large_mask = mask & (class_distance >= large_error_threshold)
        if large_mask.any():
            large_distance = (class_distance[large_mask] ** 2).to(pred.dtype)
            large_weights = element_weights[large_mask]
            large_loss = (large_distance * large_weights).sum() / large_weights.sum().clamp_min(1)
            total_loss = total_loss + large_error_lambda * large_loss

    if high_level_lambda > 0:
        high_mask = mask & (target_levels >= high_level_threshold)
        if high_mask.any():
            weighted_errors = abs_diff * element_weights
            high_weight_sum = element_weights.where(
                high_mask, torch.zeros_like(element_weights)
            ).sum().clamp_min(1)
            mae_loss = weighted_errors.where(
                high_mask, torch.zeros_like(weighted_errors)
            ).sum() / high_weight_sum
            total_loss = total_loss + high_level_lambda * mae_loss

    if piecewise_mae_lambda > 0:
        severe_mask = mask & (target_real >= piecewise_threshold)
        if severe_mask.any():
            severe_errors = abs_diff
            severe_weights = element_weights.where(
                severe_mask, torch.zeros_like(element_weights)
            )
            severe_weight_sum = severe_weights.sum().clamp_min(1)
            severe_loss = (severe_errors * severe_weights).sum() / severe_weight_sum
            total_loss = total_loss + piecewise_mae_lambda * severe_loss

    if directional_lambda > 0:
        under_mask = mask & (pred < target)
        if under_mask.any():
            directional_penalty = (target - pred) * element_weights
            under_weight_sum = element_weights.where(
                under_mask, torch.zeros_like(element_weights)
            ).sum().clamp_min(1)
            penalty_sum = directional_penalty.where(
                under_mask, torch.zeros_like(directional_penalty)
            ).sum()
            total_loss = total_loss + directional_lambda * penalty_sum / under_weight_sum

    if zero_overprediction_lambda > 0:
        zero_mask = mask & (target_levels == 0)
        if zero_mask.any():
            zero_excess = (pred_real - float(zero_overprediction_threshold)).clamp_min(0.0)
            zero_horizon_weights = horizon_weights.expand_as(pred)
            zero_weights = zero_horizon_weights.where(
                zero_mask, torch.zeros_like(zero_horizon_weights)
            )
            zero_weight_sum = zero_weights.sum().clamp_min(1)
            zero_loss = (zero_excess.pow(2) * zero_weights).sum() / zero_weight_sum
            total_loss = total_loss + zero_overprediction_lambda * zero_loss

    if high_underprediction_lambda > 0:
        high_under_mask = (
            mask
            & (target_levels >= int(high_underprediction_threshold))
            & (pred_real < target_real)
        )
        if high_under_mask.any():
            high_under_error = (target_real - pred_real).clamp_min(0.0)
            high_under_weights = element_weights.where(
                high_under_mask, torch.zeros_like(element_weights)
            )
            high_under_weight_sum = high_under_weights.sum().clamp_min(1)
            high_under_loss = (
                high_under_error * high_under_weights
            ).sum() / high_under_weight_sum
            total_loss = total_loss + high_underprediction_lambda * high_under_loss

    if shape_lambda > 0:
        shape_pred_real = torch.where(mask, pred_real, torch.full_like(pred_real, float("nan")))
        shape_target_real = torch.where(mask, target_real, torch.full_like(target_real, float("nan")))
        shape_loss = _weighted_nodewise_pearson_loss(
            shape_pred_real,
            shape_target_real,
            eps=shape_eps,
            min_std=shape_min_std,
            weight_std=shape_weight_std,
            weight_range=shape_weight_range,
            weight_max=shape_weight_max,
        )
        total_loss = total_loss + shape_lambda * shape_loss

    return total_loss
