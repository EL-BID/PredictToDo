import torch

from src.training.losses import masked_mse


def test_zero_overprediction_penalty_increases_loss_for_false_positive_zero():
    target = torch.zeros((1, 1, 4))
    pred_low = torch.full((1, 1, 4), 0.25)
    pred_high = torch.full((1, 1, 4), 1.5)

    low_loss = masked_mse(
        pred_low,
        target,
        class_ce_lambda=0,
        class_distance_lambda=0,
        large_error_lambda=0,
        high_level_lambda=0,
        piecewise_mae_lambda=0,
        directional_lambda=0,
        zero_overprediction_lambda=1.0,
        zero_overprediction_threshold=0.5,
    )
    high_loss = masked_mse(
        pred_high,
        target,
        class_ce_lambda=0,
        class_distance_lambda=0,
        large_error_lambda=0,
        high_level_lambda=0,
        piecewise_mae_lambda=0,
        directional_lambda=0,
        zero_overprediction_lambda=1.0,
        zero_overprediction_threshold=0.5,
    )

    assert high_loss > low_loss


def test_high_underprediction_penalty_increases_loss_for_missed_high_levels():
    target = torch.full((1, 1, 4), 4.0)
    pred_low = torch.full((1, 1, 4), 1.0)
    pred_near = torch.full((1, 1, 4), 3.5)

    low_loss = masked_mse(
        pred_low,
        target,
        class_ce_lambda=0,
        class_distance_lambda=0,
        large_error_lambda=0,
        high_level_lambda=0,
        piecewise_mae_lambda=0,
        directional_lambda=0,
        high_underprediction_lambda=1.0,
        high_underprediction_threshold=3,
    )
    near_loss = masked_mse(
        pred_near,
        target,
        class_ce_lambda=0,
        class_distance_lambda=0,
        large_error_lambda=0,
        high_level_lambda=0,
        piecewise_mae_lambda=0,
        directional_lambda=0,
        high_underprediction_lambda=1.0,
        high_underprediction_threshold=3,
    )

    assert low_loss > near_loss


def test_horizon_weights_emphasize_short_horizon_errors():
    target = torch.zeros((1, 1, 4))
    pred = torch.tensor([[[2.0, 2.0, 0.0, 0.0]]])

    flat_loss = masked_mse(
        pred,
        target,
        class_ce_lambda=0,
        class_distance_lambda=0,
        large_error_lambda=0,
        high_level_lambda=0,
        piecewise_mae_lambda=0,
        directional_lambda=0,
    )
    short_weighted_loss = masked_mse(
        pred,
        target,
        class_ce_lambda=0,
        class_distance_lambda=0,
        large_error_lambda=0,
        high_level_lambda=0,
        piecewise_mae_lambda=0,
        directional_lambda=0,
        horizon_short_steps=2,
        horizon_mid_steps=4,
        horizon_short_weight=2.0,
        horizon_mid_weight=1.0,
        horizon_long_weight=1.0,
    )

    assert short_weighted_loss > flat_loss
