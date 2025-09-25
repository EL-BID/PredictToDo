import time
from typing import Any, Callable, Dict

import torch

from src.sparse_dataloader import TrafficDataset
from src.training_utils import EpochUniquePicker

from ..utils.logging import log
from ..utils.tensor_stats import tensor_stats


@torch.no_grad()
def _sync_if_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def train_one_epoch(
    model: torch.nn.Module,
    dataloader,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: torch.device,
    picker: EpochUniquePicker,  # must support start_epoch() and pick()
    traffic_dataset: TrafficDataset,  # must support reasign_seed(str)
    epoch_idx: int,
) -> Dict[str, Any]:
    model.train()
    picker.start_epoch()

    totals = {
        k: 0.0
        for k in [
            "data_wait",
            "to_device",
            "reshape",
            "forward",
            "loss",
            "backward+step",
            "seed_pick",
            "batch_total",
        ]
    }
    total_loss = 0.0
    prev_end = time.perf_counter()

    for step, batch in enumerate(dataloader):
        iter_start = time.perf_counter()
        data_wait_s = iter_start - prev_end
        totals["data_wait"] += data_wait_s
        log(f"dataloader wait = {data_wait_s * 1000:.1f} ms", step)

        # Move to device
        t0 = time.perf_counter()
        x_static = batch["x_static"].to(device)
        x_dyn_past = batch["x_dynamic_past"].to(device)
        x_dyn_future = batch["x_dynamic_future"].to(device)
        y_target = batch["y"].to(device)
        edge_index = batch["edge_index"].to(device)
        _sync_if_cuda()
        move_s = time.perf_counter() - t0
        totals["to_device"] += move_s
        log(f"moved to device in {move_s * 1000:.1f} ms", step)

        # Print shapes & stats
        B, N = x_static.shape[0], x_static.shape[1]
        E = edge_index.shape[1]
        log(f"graph size -> B={B}, N={N}, E={E}", step)
        log(f"x_static: {tensor_stats(x_static)}", step)
        log(f"x_dyn_past: {tensor_stats(x_dyn_past)}", step)
        log(f"x_dyn_future: {tensor_stats(x_dyn_future)}", step)
        log(f"y_target: {tensor_stats(y_target)}", step)

        # Reshape
        t0 = time.perf_counter()
        static_inputs = x_static[0]
        dynamic_seq_inputs = x_dyn_past.permute(0, 2, 1, 3).contiguous()
        future_known_inputs = x_dyn_future.permute(0, 2, 1, 3).contiguous()
        _sync_if_cuda()
        reshape_s = time.perf_counter() - t0
        totals["reshape"] += reshape_s
        log(f"reshape/permutation {reshape_s * 1000:.1f} ms", step)

        # Forward
        t0 = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        preds = model(
            edge_index, static_inputs, dynamic_seq_inputs, future_known_inputs
        )
        _sync_if_cuda()
        fwd_s = time.perf_counter() - t0
        totals["forward"] += fwd_s
        log(f"forward {fwd_s * 1000:.1f} ms", step)

        # Loss
        t0 = time.perf_counter()
        loss = loss_fn(preds, y_target)
        _sync_if_cuda()
        loss_s = time.perf_counter() - t0
        totals["loss"] += loss_s
        log(f"loss {loss_s * 1000:.1f} ms", step)

        # Backward + step
        t0 = time.perf_counter()
        loss.backward()
        optimizer.step()
        _sync_if_cuda()
        bwd_s = time.perf_counter() - t0
        totals["backward+step"] += bwd_s
        log(f"backward+step {bwd_s * 1000:.1f} ms", step)

        total_loss += float(loss)

        # Next seed
        t0 = time.perf_counter()
        next_seed = picker.pick()
        traffic_dataset.reasign_seed(str(next_seed))
        seed_s = time.perf_counter() - t0
        totals["seed_pick"] += seed_s
        log(f"seed pick {seed_s * 1000:.1f} ms -> {next_seed}", step)

        batch_total = time.perf_counter() - iter_start
        totals["batch_total"] += batch_total
        log(f"TOTAL {batch_total * 1000:.1f} ms", step)
        prev_end = time.perf_counter()

    avg_loss = total_loss / max(1, len(dataloader))
    return {"avg_loss": avg_loss, **totals}
