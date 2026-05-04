from torch_geometric.data import Data, Dataset

from src.sparse_dataloader import TrafficDataset


class TemporalSubset(Dataset):
    def __init__(
        self,
        dataset: TrafficDataset,
        start_time_idx: int,
        end_time_idx: int,
    ):
        """
        A temporal subset of a TrafficDataset, defined by a range of time indices.

        Args:
            dataset (TrafficDataset): The underlying full dataset.
            start_time_idx (int): Starting time index (inclusive) for this subset.
            end_time_idx (int): Ending time index (exclusive) for this subset.
        """
        self.dataset = dataset
        self.start_time_idx = start_time_idx
        self.end_time_idx = end_time_idx

        # Validate time indices
        max_time = dataset.total_timesteps_in_data - dataset.total_timesteps_window
        if start_time_idx < 0:
            raise ValueError(f"start_time_idx must be >= 0, got {start_time_idx}")
        if end_time_idx > max_time + 1:
            raise ValueError(
                f"end_time_idx must be <= {max_time + 1}, got {end_time_idx}"
            )
        if start_time_idx >= end_time_idx:
            raise ValueError(
                f"start_time_idx ({start_time_idx}) must be < end_time_idx ({end_time_idx})"
            )

    def __len__(self) -> int:
        """Returns the number of time windows in this temporal subset."""
        return self.end_time_idx - self.start_time_idx

    @property
    def total_training_samples(self) -> int:
        """Total samples across all node batches for this temporal subset."""
        return len(self) * self.dataset.total_possible_node_batches

    @property
    def node_seed(self) -> str:
        """Returns the current seed node from the underlying dataset."""
        return self.dataset.node_seed

    @property
    def total_possible_node_batches(self) -> int:
        """Returns the number of node batches from the underlying dataset."""
        return self.dataset.total_possible_node_batches

    @property
    def num_nodes(self) -> int:
        """Returns the number of nodes from the underlying dataset."""
        return self.dataset.num_nodes

    @property
    def node_batch_size(self) -> int:
        """Returns the node batch size from the underlying dataset."""
        return self.dataset.node_batch_size

    def __getitem__(self, idx) -> Data:
        """
        Fetches a sample from the subset.

        The idx is relative to this subset's time range, so it's automatically
        offset to the correct absolute time index.
        """
        assert isinstance(idx, int), f"Index must be int, got {type(idx)}"
        if idx < 0 or idx >= len(self):
            raise IndexError(
                f"Index {idx} out of range for subset with {len(self)} samples"
            )

        # Map the relative index to the absolute time index in the full dataset
        absolute_idx = self.start_time_idx + idx

        # Use the underlying dataset's __getitem__ with the absolute index
        # This will use the current seed from the dataset
        return self.dataset[absolute_idx]

    def reasign_seed(self, node_seed: str) -> None:
        """
        Reassigns the seed node in the underlying dataset.

        This affects all temporal subsets that share the same underlying dataset.
        """
        self.dataset.reasign_seed(node_seed)


def temporal_train_test_split(
    dataset: TrafficDataset,
    train_ratio: float = 0.8,
) -> tuple[TemporalSubset, TemporalSubset]:
    """
    Splits a TrafficDataset into train and test sets based on time.

    The split is done by time index, so the train set contains the earliest
    time windows and the test set contains the latest time windows. This
    preserves temporal ordering and prevents data leakage.

    **IMPORTANT**: Train and test share the same seed - reassigning one affects both
    - This is actually desired behavior for consistent node coverage
    - Use the same `EpochUniquePicker` instance for both to ensure all nodes are visited

    Args:
        dataset (TrafficDataset): The dataset to split.
        train_ratio (float): Fraction of time windows to use for training (default: 0.8).

    Returns:
        tuple[TemporalSubset, TemporalSubset]: (train_dataset, test_dataset)

    Example:
        >>> dataset = TrafficDataset(...)
        >>> train_ds, test_ds = temporal_train_test_split(dataset, train_ratio=0.8)
        >>>
        >>> # Both subsets share the same underlying dataset and seed
        >>> train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
        >>> test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)
        >>>
        >>> # Seed reassignment works transparently
        >>> for batch in train_loader:
        ...     # ... training ...
        ...     train_ds.reasign_seed(next_seed)  # Affects both train and test
    """
    if not 0 < train_ratio < 1:
        raise ValueError(f"train_ratio must be between 0 and 1, got {train_ratio}")

    # Calculate the split point
    total_time_windows = len(dataset)
    split_idx = int(total_time_windows * train_ratio)

    # Create temporal subsets
    train_dataset = TemporalSubset(
        dataset=dataset,
        start_time_idx=0,
        end_time_idx=split_idx,
    )

    test_dataset = TemporalSubset(
        dataset=dataset,
        start_time_idx=split_idx,
        end_time_idx=total_time_windows,
    )

    return train_dataset, test_dataset


def temporal_train_val_test_split(
    dataset: TrafficDataset,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> tuple[TemporalSubset, TemporalSubset, TemporalSubset]:
    """
    Splits a TrafficDataset into train, validation, and test sets based on time.

    The split is done by time index in chronological order:
    - Train: earliest time windows
    - Val: middle time windows
    - Test: latest time windows

    **IMPORTANT**: Train, val and test share the same seed - reassigning one affects all of them
    - This is actually desired behavior for consistent node coverage
    - Use the same `EpochUniquePicker` instance for all of them to ensure all nodes are visited

    Args:
        dataset (TrafficDataset): The dataset to split.
        train_ratio (float): Fraction of time windows for training (default: 0.7).
        val_ratio (float): Fraction of time windows for validation (default: 0.15).
            The remaining fraction (1 - train_ratio - val_ratio) is used for testing.

    Returns:
        tuple[TemporalSubset, TemporalSubset, TemporalSubset]:
            (train_dataset, val_dataset, test_dataset)

    Example:
        >>> dataset = TrafficDataset(...)
        >>> train_ds, val_ds, test_ds = temporal_train_val_test_split(
        ...     dataset, train_ratio=0.7, val_ratio=0.15
        ... )
        >>> print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    """
    test_ratio = 1.0 - train_ratio - val_ratio
    if test_ratio <= 0 or not (0 < train_ratio < 1) or not (0 < val_ratio < 1):
        raise ValueError(
            f"Invalid ratios: train={train_ratio}, val={val_ratio}, test={test_ratio}. "
            "All must be positive and sum to <= 1."
        )

    # Calculate split points
    total_time_windows = len(dataset)
    train_end = int(total_time_windows * train_ratio)
    val_end = int(total_time_windows * (train_ratio + val_ratio))

    # Create temporal subsets
    train_dataset = TemporalSubset(
        dataset=dataset,
        start_time_idx=0,
        end_time_idx=train_end,
    )

    val_dataset = TemporalSubset(
        dataset=dataset,
        start_time_idx=train_end,
        end_time_idx=val_end,
    )

    test_dataset = TemporalSubset(
        dataset=dataset,
        start_time_idx=val_end,
        end_time_idx=total_time_windows,
    )

    return train_dataset, val_dataset, test_dataset


if __name__ == "__main__":
    from torch_geometric.loader import DataLoader

    from src.sparse_dataloader import TrafficDataset
    from src.temporal_split import temporal_train_test_split
    from src.training_utils import pad_collate

    processed_data_dir = "data/datos_procesados"
    sparse_data_dir = f"{processed_data_dir}/sparse_datasets"

    print("\n1. Initializing full dataset...")

    dataset = TrafficDataset(
        graph_path=f"{processed_data_dir}/conectivity_graph/road_graph.pickle",
        static_data_path=f"{sparse_data_dir}/static_dataset.h5",
        dynamic_data_paths=[
            f"{sparse_data_dir}/coi_points_dataset.h5",
            f"{sparse_data_dir}/coi_tracts_dataset.h5",
            f"{sparse_data_dir}/irregularities_dataset.h5",
            f"{sparse_data_dir}/events_dataset.h5",
            f"{sparse_data_dir}/incidents_dataset.h5",
            f"{sparse_data_dir}/jam_velocity_dataset.h5",
            f"{sparse_data_dir}/jam_level_count_dataset.h5",
            f"{sparse_data_dir}/jam_delay_dataset.h5",
        ],
        metadata_paths=[
            f"{sparse_data_dir}/coi_points_dataset_metadata.pkl",
            f"{sparse_data_dir}/coi_tracts_dataset_metadata.pkl",
            f"{sparse_data_dir}/irregularities_dataset_metadata.pkl",
            f"{sparse_data_dir}/events_dataset_metadata.pkl",
            f"{sparse_data_dir}/incidents_dataset_metadata.pkl",
            f"{sparse_data_dir}/jam_velocity_dataset_metadata.pkl",
            f"{sparse_data_dir}/jam_level_count_dataset_metadata.pkl",
            f"{sparse_data_dir}/jam_delay_dataset_metadata.pkl",
        ],
        target_data_path=f"{sparse_data_dir}/jams_dataset.h5",
        in_timesteps=150,
        out_timesteps=50,
        future_features=[
            "coi_puntos",
            "coi_tramo",
            "AGLOMERACION",
            "EMERGENCIA",
            "MANIFESTACION",
        ],
        node_batch_size=500,
        node_seed="0_FT.5",  # Initial seed
        weather_data_dir=f"{processed_data_dir}/sparse_datasets",
        include_weather=True,
        neighbor_features_path=f"{processed_data_dir}/neighbor_features.parquet",
    )

    print(f"✓ Dataset loaded: {len(dataset):,} time windows")
    print(f"  - Total nodes: {dataset.num_nodes:,}")
    print(f"  - Node batch size: {dataset.node_batch_size}")
    print(f"  - Node batches needed: {dataset.total_possible_node_batches}")

    print("\n2. Splitting dataset temporally (80% train, 20% test)...")

    train_dataset, test_dataset = temporal_train_test_split(dataset, train_ratio=0.8)

    print(f"✓ Split complete:")
    print(f"  - Train: {len(train_dataset):,} time windows")
    print(f"  - Test: {len(test_dataset):,} time windows")
    print(f"  - Train total samples: {train_dataset.total_training_samples:,}")
    print(f"  - Test total samples: {test_dataset.total_training_samples:,}")

    print("\n3. Creating DataLoaders...")

    batch_size = 16

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle time windows
        num_workers=0,
        collate_fn=pad_collate,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle test set
        num_workers=0,
        collate_fn=pad_collate,
    )

    print(f"✓ DataLoaders created:")
    print(f"  - Train batches per seed: {len(train_loader):,}")
    print(f"  - Test batches per seed: {len(test_loader):,}")
    print(
        f"  - Actual train iterations per epoch: {len(train_loader) * dataset.total_possible_node_batches:,}"
    )
    print(
        f"  - Actual test iterations: {len(test_loader) * dataset.total_possible_node_batches:,}"
    )

    # =============================================================================
    # Training loop with seed reassignment (pseudo-code)
    # =============================================================================

    # # Initialize seed picker
    # picker = EpochUniquePicker(base=Node50Picker(...))

    # for epoch in range(num_epochs):
    #     picker.start_epoch()  # Reset for new epoch

    #     # Training phase
    #     model.train()
    #     for batch in train_loader:
    #         # ... training step ...

    #         # Reassign seed after each batch
    #         next_seed = picker.pick()
    #         train_dataset.reasign_seed(str(next_seed))

    #     # Evaluation phase on test set
    #     model.eval()
    #     picker.start_epoch()  # Reset picker for test evaluation

    #     with torch.no_grad():
    #         for batch in test_loader:
    #             # ... evaluation step ...

    #             # Reassign seed for test set too
    #             next_seed = picker.pick()
    #             test_dataset.reasign_seed(str(next_seed))

    #     # Both train_dataset and test_dataset share the same underlying
    #     # dataset, so seed reassignment affects both automatically!

    print("\n6. Sanity check - Time index boundaries:")
    print(f"  - Train time indices: [0, {train_dataset.end_time_idx})")
    print(f"  - Test time indices: [{test_dataset.start_time_idx}, {len(dataset)})")
    print(
        f"  - No overlap: {train_dataset.end_time_idx == test_dataset.start_time_idx} ✓"
    )

    print("\n7. Alternative: Three-way split (70-15-15):")

    from src.temporal_split import temporal_train_val_test_split

    train_ds, val_ds, test_ds = temporal_train_val_test_split(
        dataset, train_ratio=0.7, val_ratio=0.15
    )

    print(f"  - Train: {len(train_ds):,} time windows")
    print(f"  - Val: {len(val_ds):,} time windows")
    print(f"  - Test: {len(test_ds):,} time windows")
    print(f"  - Total: {len(train_ds) + len(val_ds) + len(test_ds):,} time windows")
