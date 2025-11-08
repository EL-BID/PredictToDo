import pickle
from typing import Optional

import pandas as pd
import torch

from sparse_dataloader import TrafficDataset


def reconstruct_predictions(
    predictions: torch.Tensor,
    dataset: TrafficDataset,
    batch_indices: list[int],
    metadata_path: str,
    value_column_name: str = "predicted_level",
) -> pd.DataFrame:
    """
    Reconstruct predictions from a batch into a DataFrame with MultiIndex (NodeID, Timestamp).

    This function maps predictions from the model output tensor back to the original
    node IDs and timestamps, creating a structured DataFrame suitable for analysis.

    Args:
        predictions (torch.Tensor): Predictions tensor of shape (batch_size, num_nodes, out_timesteps)
            or (num_nodes, out_timesteps) for a single sample.
        dataset: TrafficDataset instance used for generating the predictions.
        batch_indices (list[int]): List of sample indices from the dataset that were used
            to generate these predictions. Length should match batch_size.
        metadata_path (str): Path to a metadata pickle file that contains 'time_idx_map'
            (a dictionary mapping timestamps to indices). This is typically one of the
            metadata files used during dataset initialization.
        value_column_name (str): Name for the prediction value column in the output DataFrame.
            Defaults to "predicted_level".

    Returns:
        pd.DataFrame: DataFrame with MultiIndex (NodeID, Timestamp) and a column for predictions.

    Example:
        >>> # During evaluation loop
        >>> for i, batch in enumerate(dataloader):
        >>>     preds = model(batch)  # Shape: (batch_size, num_nodes, out_timesteps)
        >>>     batch_indices = list(range(i * batch_size, (i + 1) * batch_size))
        >>>     df = reconstruct_predictions(
        >>>         preds,
        >>>         traffic_dataset,
        >>>         batch_indices,
        >>>         "data/datos_procesados/sparse_datasets/jams_dataset_metadata.pkl"
        >>>     )
    """
    # Handle both batched and single sample predictions
    if predictions.dim() == 2:
        predictions = predictions.unsqueeze(0)
        batch_indices = (
            [batch_indices] if isinstance(batch_indices, int) else batch_indices
        )

    # Load metadata to get timestamp mappings
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)

    # Create reverse mapping from time index to timestamp
    idx_to_timestamp = {idx: ts for ts, idx in metadata["time_idx_map"].items()}

    # Prepare containers for results
    results = []

    # Process each sample in the batch
    for batch_idx, sample_idx in enumerate(batch_indices):
        # Get time and node indices for this sample
        time_idx, node_indices = dataset._get_time_and_node_indices(sample_idx)

        # Calculate output time range (predictions are for future timesteps)
        # time_idx is the start of the input window
        # Predictions start after (in_timesteps) and span (out_timesteps)
        output_start_time = time_idx + dataset.in_timesteps

        # Get predictions for this sample
        sample_preds = predictions[batch_idx]

        # Convert to CPU and numpy for processing
        if isinstance(sample_preds, torch.Tensor):
            sample_preds = sample_preds.cpu().numpy()

        # For each node in the batch
        for node_batch_idx, node_global_idx in enumerate(node_indices):
            # For each timestep in the output
            for t in range(dataset.out_timesteps):
                time_global_idx = output_start_time + t
                if time_global_idx in idx_to_timestamp:
                    results.append(
                        {
                            "NodeID": dataset.id_to_node[node_global_idx],
                            "Timestamp": idx_to_timestamp[time_global_idx],
                            value_column_name: sample_preds[node_batch_idx, t],
                        }
                    )

    # Create DataFrame
    df = pd.DataFrame(results).set_index(["NodeID", "Timestamp"])

    return df
