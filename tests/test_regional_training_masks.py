import pandas as pd
import torch

from src.training.losses import masked_mse
from src.training_entrypoint import (
    _apply_node_mask_to_target,
    _build_region_node_mask,
    _masked_tensor_mse,
    _node_valid_mask_for_target,
)


def test_build_region_node_mask_selects_central_nodes(tmp_path):
    region_map_path = tmp_path / "node_region_map.csv"
    pd.DataFrame(
        {
            "road_id": ["node_a", "node_b", "node_c"],
            "is_central": [True, False, True],
        }
    ).to_csv(region_map_path, index=False)

    mask, metadata = _build_region_node_mask(
        str(region_map_path),
        "central",
        {0: "node_a", 1: "node_b", 2: "node_c", 3: "node_d"},
    )

    assert mask.tolist() == [True, False, True, False]
    assert metadata["training_region"] == "central"
    assert metadata["selected_nodes"] == 2


def test_apply_node_mask_to_target_sets_non_region_nodes_to_nan():
    target = torch.arange(12, dtype=torch.float32).view(1, 3, 4)
    masked = _apply_node_mask_to_target(target, torch.tensor([True, False, True]))

    assert torch.equal(masked[:, 0, :], target[:, 0, :])
    assert torch.isnan(masked[:, 1, :]).all()
    assert torch.equal(masked[:, 2, :], target[:, 2, :])


def test_regional_masked_mse_ignores_non_region_nodes():
    pred = torch.tensor([[[1.0, 5.0], [100.0, 100.0]]])
    target = torch.tensor([[[2.0, 3.0], [0.0, 0.0]]])
    node_mask = torch.tensor([True, False])

    loss = _masked_tensor_mse(pred, target, node_mask)

    assert torch.isclose(loss, torch.tensor(2.5))


def test_node_valid_mask_broadcasts_to_full_target_shape():
    target = torch.zeros((16, 500, 50), dtype=torch.float32)
    node_mask = torch.zeros(500, dtype=torch.bool)
    node_mask[:10] = True

    valid_mask = _node_valid_mask_for_target(target, node_mask)

    assert valid_mask.shape == target.shape
    assert valid_mask.reshape(-1).numel() == target.numel()


def test_composite_masked_mse_uses_explicit_region_mask():
    pred = torch.tensor([[[1.0, 5.0], [100.0, 100.0]]])
    target = torch.tensor([[[2.0, 3.0], [0.0, 0.0]]])
    valid_mask = torch.tensor([[[True], [False]]])

    loss = masked_mse(pred, target, valid_mask=valid_mask)
    selected_loss = masked_mse(pred[:, :1, :], target[:, :1, :])

    assert torch.isfinite(loss)
    assert torch.isclose(loss, selected_loss)


def test_masked_mse_handles_empty_finite_target():
    pred = torch.ones((1, 2, 3), requires_grad=True)
    target = torch.full((1, 2, 3), float("nan"))

    loss = masked_mse(pred, target)

    assert loss.item() == 0.0
    loss.backward()
    assert pred.grad is not None
