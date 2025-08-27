import torch
import pandas as pd
import numpy as np
from dataclasses import dataclass, field

import random
import re
from collections import deque
from typing import Optional, Callable, Literal, Any

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
        if len(self.top_nodes) == 0:
            return self.rng.choice(self.other_nodes)
        if len(self.other_nodes) == 0:
            return self.rng.choice(self.top_nodes)
        return (self.rng.choice(self.top_nodes)
                if self.rng.random() < 0.5
                else self.rng.choice(self.other_nodes))

def build_node_50_picker(csv_path: str, top_frac: float = 0.2, seed: int = 0) -> Node50Picker:
    df = pd.read_csv(csv_path)  # assumes: col0=node (literal), col1=hours
    df = df.iloc[:, :2].copy()
    df.columns = ["node", "hours"]
    df["hours"] = pd.to_numeric(df["hours"], errors="coerce").fillna(0)

    # keep nodes exactly as strings to preserve literals like "11013660_B.1"
    d = df.sort_values("hours", ascending=False, ignore_index=True)
    cutoff = max(1, int(np.ceil(len(d) * top_frac)))
    top_nodes   = d["node"].astype(str).to_numpy()
    other_nodes = d["node"].astype(str).to_numpy()[cutoff:]

    return Node50Picker(
        top_nodes=top_nodes,
        other_nodes=other_nodes,
        rng=np.random.default_rng(seed),
    )

def _default_node_key(seed_str: str) -> int:
    """
    Extracts a node id from a seed like '50008544_TF.1' -> 50008544.
    If no '_' present, tries to parse the whole string as an int.
    """
    m = re.match(r"^(\d+)", str(seed_str))
    if not m:
        raise ValueError(f"Cannot parse node id from seed '{seed_str}'")
    return int(m.group(1))

# --- epoch-unique wrapper ----------------------------------------------------
@dataclass
class EpochUniquePicker:
    base: Node50Picker
    p_top: float = 0.5
    on_exhaust: Literal["reset", "allow_repeat"] = "allow_repeat"
    key_fn: Optional[Callable[[Any], Any]] = None  # map literal -> uniqueness key

    _rng: np.random.Generator = field(init=False, repr=False)
    _seen: set = field(default_factory=set, init=False, repr=False)
    _top_perm: np.ndarray = field(default=None, init=False, repr=False)
    _rest_perm: np.ndarray = field(default=None, init=False, repr=False)
    _i_top: int = field(default=0, init=False, repr=False)
    _i_rest: int = field(default=0, init=False, repr=False)

    def __post_init__(self):
        self._rng = self.base.rng
        if self.key_fn is None:
            # default: uniqueness by the full literal
            self.key_fn = lambda x: x

    def start_epoch(self):
        self._seen.clear()
        self._top_perm  = self._rng.permutation(self.base.top_nodes)
        self._rest_perm = self._rng.permutation(self.base.other_nodes)
        self._i_top = 0
        self._i_rest = 0

    def _draw_from(self, which: str):
        if which == "top":
            if self._i_top < len(self._top_perm):
                v = self._top_perm[self._i_top]; self._i_top += 1; return v
        else:
            if self._i_rest < len(self._rest_perm):
                v = self._rest_perm[self._i_rest]; self._i_rest += 1; return v
        return None

    def pick(self):
        prefer_top = (self._rng.random() < self.p_top)
        node = self._draw_from("top" if prefer_top else "rest")
        if node is None:
            node = self._draw_from("rest" if prefer_top else "top")

        if node is None:
            if self.on_exhaust == "reset":
                self.start_epoch()
                return self.pick()
            # fallback: allow repeats using base picker (with replacement)
            return self.base.pick()

        key = self.key_fn(node)
        if key in self._seen:
            # try the other pool once
            alt = self._draw_from("rest" if prefer_top else "top")
            if alt is None:
                if self.on_exhaust == "reset":
                    self.start_epoch()
                    return self.pick()
                return self.base.pick()
            node = alt
            key = self.key_fn(node)

        self._seen.add(key)
        return node  # <-- return the literal as-is

    def remaining(self):
        return (len(self._top_perm) - self._i_top, len(self._rest_perm) - self._i_rest)