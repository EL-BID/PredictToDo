from __future__ import annotations

import torch


def masked_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    level_weights: torch.Tensor | None = None,
    high_level_threshold: int = 2,
    high_level_lambda: float = 0.5,
    piecewise_threshold: int = 4,
    piecewise_mae_lambda: float = 0.5,
    directional_lambda: float = 0.3,
) -> torch.Tensor:
    """Weighted MSE with extra penalties for high congestion levels."""
    mask = torch.isfinite(target)
    target_levels = torch.clamp(target.round(), 0, 5).long()

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

    diff2 = (pred - target) ** 2 * level_weights_expanded
    denom = level_weights_expanded.sum().clamp_min(1)
    mse_loss = diff2.sum() / denom

    total_loss = mse_loss

    if high_level_lambda > 0:
        high_mask = mask & (target_levels >= high_level_threshold)
        if high_mask.any():
            high_errors = (pred - target).abs()
            weighted_errors = high_errors * level_weights_expanded
            high_weight_sum = level_weights_expanded.where(
                high_mask, torch.zeros_like(level_weights_expanded)
            ).sum().clamp_min(1)
            mae_loss = weighted_errors.where(
                high_mask, torch.zeros_like(weighted_errors)
            ).sum() / high_weight_sum
            total_loss = total_loss + high_level_lambda * mae_loss

    if piecewise_mae_lambda > 0:
        severe_mask = mask & (target >= piecewise_threshold)
        if severe_mask.any():
            severe_errors = (pred - target).abs()
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

    return total_loss
