from dataclasses import dataclass
from typing import Callable, Iterable, Optional

import numpy as np
import pandas as pd


@dataclass
class Node50Picker:
    top_nodes: np.ndarray
    other_nodes: np.ndarray
    rng: np.random.Generator

    def pick(self) -> str | int:
        # Return a seed (literal from the arrays). Can be int or str.
        if len(self.top_nodes) == 0 and len(self.other_nodes) == 0:
            raise RuntimeError("Both top_nodes and other_nodes are empty.")
        if len(self.top_nodes) == 0:
            return self.rng.choice(self.other_nodes)
        if len(self.other_nodes) == 0:
            return self.rng.choice(self.top_nodes)
        return (
            self.rng.choice(self.top_nodes)
            if self.rng.random() < 0.5
            else self.rng.choice(self.other_nodes)
        )


def build_node_50_picker(
    csv_path: str, top_frac: float = 0.2, seed: int = 0
) -> Node50Picker:
    df = pd.read_csv(csv_path)  # assumes: col0=node, col1=hours
    df = df.iloc[:, :2].copy()
    df.columns = ["node", "hours"]
    df["hours"] = pd.to_numeric(df["hours"], errors="coerce").fillna(0)
    d = df.sort_values("hours", ascending=False).reset_index(drop=True)
    cutoff = max(1, int(np.ceil(len(d) * top_frac)))
    top_nodes = d["node"].iloc[:cutoff].to_numpy()
    other_nodes = d["node"].iloc[cutoff:].to_numpy()
    return Node50Picker(
        top_nodes=top_nodes, other_nodes=other_nodes, rng=np.random.default_rng(seed)
    )


class EpochUniquePicker:
    """
    Wraps a base picker and prevents selecting the same *node id* twice within an epoch.
    The 'key_fn' extracts the node id from the seed (e.g., for '50008544_TF.1' -> '50008544').
    Call start_epoch() before scheduling, and pick() to sample a new, unseen key.
    """

    def __init__(
        self,
        base: Node50Picker,
        key_fn: Optional[Callable[[object], str]] = None,
        max_tries: int = 1000,
    ):
        self.base = base
        self.key_fn = key_fn or (lambda s: str(s).split("_", 1)[0])
        self._seen: set[str] = set()
        self.max_tries = max_tries

    def start_epoch(self):
        self._seen.clear()

    def pick(self):
        for _ in range(self.max_tries):
            s = self.base.pick()
            k = self.key_fn(s)
            if k not in self._seen:
                self._seen.add(k)
                return s
        # Fallback: allow repeats if exhausted
        return self.base.pick()
