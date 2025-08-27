import torch
import pandas as pd
import numpy as np
from dataclasses import dataclass

def pad_collate(batch):
    assert len(batch) > 0, "Empty batch."

    # --- Sanity: all samples must share same node set and edges ---
    N0 = batch[0].x_static.size(0)
    E0 = batch[0].edge_index.size(1)
    edge0 = batch[0].edge_index

    for i, d in enumerate(batch[1:], start=1):
        if d.x_static.size(0) != N0:
            raise ValueError(f"Item {i} has N={d.x_static.size(0)} but first has N={N0}.")
        if d.edge_index.size(1) != E0:
            raise ValueError(f"Item {i} has E={d.edge_index.size(1)} but first has E={E0}.")
        # Require identical edges (same ordering). If you only need same *set* of edges,
        # customize this check.
        if not torch.equal(d.edge_index, edge0):
            raise ValueError(f"Item {i} edge_index differs from first item.")

    # --- Always present ---
    x_static = torch.stack([d.x_static for d in batch], dim=0)  # (B, N, F_static)
    y        = torch.stack([d.y        for d in batch], dim=0)  # (B, N, Tout)

    # --- Dynamic features: support either split past/future or a single x_dynamic ---
    out = {
        "x_static":   x_static,
        "y":          y,
        "edge_index": edge0,     # (2, E) shared across batch
        "num_nodes":  N0,
        "num_edges":  E0,
    }

    has_past   = hasattr(batch[0], "x_dynamic_past")
    has_future = hasattr(batch[0], "x_dynamic_future")
    has_single = hasattr(batch[0], "x_dynamic")

    if has_past and has_future:
        out["x_dynamic_past"]   = torch.stack([d.x_dynamic_past   for d in batch], dim=0)  # (B, N, Tin, F_dyn)
        out["x_dynamic_future"] = torch.stack([d.x_dynamic_future for d in batch], dim=0)  # (B, N, Tout, F_future)
    elif has_single:
        out["x_dynamic"] = torch.stack([d.x_dynamic for d in batch], dim=0)  # (B, N, Tin, F_dyn)
    else:
        # If your dataset always has dynamics, you can raise instead:
        # raise AttributeError("Expected x_dynamic_past/x_dynamic_future or x_dynamic in Data.")
        pass

    # Optional: if your Data carries a node id tensor you want to pass through
    if hasattr(batch[0], "node_indices"):
        out["node_indices"] = batch[0].node_indices  # (N,) shared across items

    return out

@dataclass
class Node50Picker:
    top_nodes: np.ndarray
    other_nodes: np.ndarray
    rng: np.random.Generator

    def pick(self):
        # If one group is empty, fall back to the other
        if len(self.top_nodes) == 0:
            return self.rng.choice(self.other_nodes)
        if len(self.other_nodes) == 0:
            return self.rng.choice(self.top_nodes)
        # 50% from top-20%, 50% from the rest (uniform within each group)
        return (self.rng.choice(self.top_nodes)
                if self.rng.random() < 0.5
                else self.rng.choice(self.other_nodes))

def build_node_50_picker(csv_path: str, top_frac: float = 0.2, seed: int = 0) -> Node50Picker:
    df = pd.read_csv(csv_path)  # assumes: col0=node, col1=hours
    df = df.iloc[:, :2].copy()
    df.columns = ["node", "hours"]
    df["hours"] = pd.to_numeric(df["hours"], errors="coerce").fillna(0)

    d = df.sort_values("hours", ascending=False).reset_index(drop=True)
    cutoff = max(1, int(np.ceil(len(d) * top_frac)))
    top_nodes   = d["node"].iloc[:cutoff].to_numpy()
    other_nodes = d["node"].iloc[cutoff:].to_numpy()

    return Node50Picker(top_nodes=top_nodes,
                        other_nodes=other_nodes,
                        rng=np.random.default_rng(seed))