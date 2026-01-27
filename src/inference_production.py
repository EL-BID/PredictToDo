"""
Production-ready inference script with error handling, logging, and monitoring.

This script demonstrates best practices for running the inference pipeline in production:
- Robust error handling
- Structured logging
- Performance monitoring
- Batch processing
- Result caching
"""

import argparse
import logging
import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd
import torch

from inference_dataloader import InferenceDataLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("inference.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class InferencePipeline:
    """Production inference pipeline with monitoring and error handling."""

    def __init__(
        self,
        graph_path: str,
        static_data_path: str,
        normalization_constants_path: str,
        model_path: Optional[str] = None,
        use_ray: bool = False,
    ):
        """Initialize the inference pipeline."""
        logger.info("Initializing inference pipeline...")

        self.loader = InferenceDataLoader(
            graph_path=graph_path,
            static_data_path=static_data_path,
            normalization_constants_path=normalization_constants_path,
            use_ray=use_ray,
        )

        # Load target normalization constants for de-normalization
        self.target_norm_constants = self._load_target_norm_constants(
            normalization_constants_path
        )

        self.model = None
        if model_path:
            self.model = self._load_model(model_path)

        logger.info(f"Pipeline initialized with {self.loader.num_nodes} nodes")

    def _load_target_norm_constants(self, norm_path: str) -> dict:
        """Load target normalization constants for de-normalization."""
        logger.info(f"Loading normalization constants from {norm_path}")
        try:
            with open(norm_path, "rb") as f:
                norm_constants = pickle.load(f)
            target_norm = norm_constants.get("target", {})
            logger.info(
                f"Loaded target normalization constants: {list(target_norm.keys())}"
            )
            return target_norm
        except Exception as e:
            logger.warning(f"Could not load normalization constants: {e}")
            return {}

    def _denormalize_predictions(
        self,
        predictions: torch.Tensor,
        include_velocity: bool = True,
    ) -> torch.Tensor:
        """
        De-normalize predictions to actual values.

        The model outputs normalized predictions (divided by normalization constants).
        This method reverses that operation to get actual velocity/jam level values.

        Args:
            predictions: Normalized predictions tensor.
                If include_velocity=True: shape (batch, timesteps, 2) where
                    [:, :, 0] = jam_level, [:, :, 1] = velocity
                If include_velocity=False: shape (batch, timesteps) for jam_level only
            include_velocity: Whether predictions include velocity (index 1).

        Returns:
            De-normalized predictions in actual units.
        """
        if not self.target_norm_constants:
            logger.warning(
                "No target normalization constants available. Returning raw predictions."
            )
            return predictions

        denorm_predictions = predictions.clone()

        if include_velocity and denorm_predictions.dim() == 3:
            # Predictions have shape (batch, timesteps, 2)
            # De-normalize jam level (index 0)
            if "mean_jam_level" in self.target_norm_constants:
                denorm_predictions[:, :, 0] = (
                    denorm_predictions[:, :, 0]
                    * self.target_norm_constants["mean_jam_level"]
                )
                logger.debug(
                    f"De-normalized jam_level with constant: {self.target_norm_constants['mean_jam_level']}"
                )

            # De-normalize velocity (index 1)
            if "mean_jam_velocity" in self.target_norm_constants:
                denorm_predictions[:, :, 1] = (
                    denorm_predictions[:, :, 1]
                    * self.target_norm_constants["mean_jam_velocity"]
                )
                logger.debug(
                    f"De-normalized velocity with constant: {self.target_norm_constants['mean_jam_velocity']}"
                )
        else:
            # Predictions have shape (batch, timesteps) - only jam level
            if "mean_jam_level" in self.target_norm_constants:
                denorm_predictions = (
                    denorm_predictions * self.target_norm_constants["mean_jam_level"]
                )
                logger.debug(
                    f"De-normalized jam_level with constant: {self.target_norm_constants['mean_jam_level']}"
                )

        return denorm_predictions

    def _load_model(self, model_path: str):
        """Load trained model."""
        logger.info(f"Loading model from {model_path}")
        try:
            # Modify this based on your model loading logic
            model = torch.load(model_path)
            model.eval()
            logger.info("Model loaded successfully")
            return model
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def reconstruct_predictions(
        self,
        predictions: list[torch.Tensor],
        batches: list,
        include_velocity: bool = True,
    ) -> pd.DataFrame:
        """
        Reconstruct predictions from model output into a meaningful DataFrame.

        Maps the raw prediction tensors back to their corresponding node IDs and
        timestamps, creating a structured DataFrame suitable for analysis and
        downstream applications.

        Args:
            predictions: List of prediction tensors, one per batch.
                Each tensor has shape (1, num_nodes, out_timesteps) or
                (1, num_nodes, out_timesteps, 2) if include_velocity=True.
            batches: List of PyTorch Geometric Data objects from the inference loader.
                Each batch contains metadata: node_ids, future_timestamps.
            include_velocity: Whether predictions include velocity (affects output columns).
                If True, expects predictions with 2 features (jam_level, velocity).
                If False, expects predictions with only jam_level.

        Returns:
            pd.DataFrame: DataFrame with columns:
                - NodeID: Road segment identifier
                - Timestamp: Prediction timestamp
                - predicted_jam_level: Predicted jam level (0-5 scale)
                - predicted_velocity: Predicted velocity in km/h (if include_velocity=True)

        Example:
            >>> results = pipeline.predict(...)
            >>> df = pipeline.reconstruct_predictions(
            ...     results["predictions"],
            ...     results["batches"],
            ...     include_velocity=True
            ... )
            >>> print(df.head())
                                              predicted_jam_level  predicted_velocity
            NodeID     Timestamp
            road_12345 2025-01-27 10:15:00                   2.3               45.2
            road_12345 2025-01-27 10:30:00                   2.1               48.1
        """
        logger.info("Reconstructing predictions to DataFrame...")

        all_records = []

        for batch_idx, (pred_tensor, batch) in enumerate(zip(predictions, batches)):
            logger.debug(f"Processing batch {batch_idx + 1}/{len(batches)}")

            # Get batch metadata
            node_ids = batch.node_ids  # List of node ID strings
            future_timestamps = batch.future_timestamps  # DatetimeIndex

            if future_timestamps is None:
                logger.warning(
                    f"Batch {batch_idx} missing timestamp metadata, skipping"
                )
                continue

            # Convert tensor to numpy for processing
            # pred_tensor shape: (1, num_nodes, out_timesteps) or
            #                    (1, num_nodes, out_timesteps, 2)
            pred_np = pred_tensor.squeeze(0).cpu().numpy()

            num_nodes = pred_np.shape[0]
            num_timesteps = pred_np.shape[1]

            # Process each node and timestep
            for node_idx in range(num_nodes):
                node_id = node_ids[node_idx]

                for t_idx in range(num_timesteps):
                    if t_idx >= len(future_timestamps):
                        break

                    timestamp = future_timestamps[t_idx]

                    record = {
                        "NodeID": node_id,
                        "Timestamp": timestamp,
                    }

                    if include_velocity and pred_np.ndim == 3 and pred_np.shape[2] == 2:
                        # Predictions have shape (num_nodes, out_timesteps, 2)
                        record["predicted_jam_level"] = float(
                            pred_np[node_idx, t_idx, 0]
                        )
                        record["predicted_velocity"] = float(
                            pred_np[node_idx, t_idx, 1]
                        )
                    else:
                        # Predictions have shape (num_nodes, out_timesteps)
                        record["predicted_jam_level"] = float(pred_np[node_idx, t_idx])

                    all_records.append(record)

        # Create DataFrame
        df = pd.DataFrame(all_records)

        if not df.empty:
            df = df.set_index(["NodeID", "Timestamp"]).sort_index()
            logger.info(
                f"Reconstructed {len(df)} predictions for "
                f"{df.index.get_level_values('NodeID').nunique()} nodes and "
                f"{df.index.get_level_values('Timestamp').nunique()} timestamps"
            )
        else:
            logger.warning("No predictions were reconstructed")

        return df

    def predict(
        self,
        jams_path: str,
        irregularities_path: str,
        weather_path: str,
        pmt_puntos_path: str,
        pmt_tramos_path: str,
        events_path: str,
        incidents_path: str,
        output_path: Optional[str] = None,
    ) -> dict:
        """
        Run inference on provided data.

        Returns:
            Dictionary with predictions and metadata
        """
        start_time = time.time()
        logger.info("Starting inference...")

        try:
            # Load and prepare data
            logger.info("Loading data...")
            load_start = time.time()

            batches = self.loader.load_and_prepare_batch(
                jams_path=jams_path,
                irregularities_path=irregularities_path,
                weather_path=weather_path,
                pmt_puntos_path=pmt_puntos_path,
                pmt_tramos_path=pmt_tramos_path,
                events_path=events_path,
                incidents_path=incidents_path,
            )

            load_time = time.time() - load_start
            logger.info(f"Data loaded in {load_time:.2f}s - {len(batches)} batch(es)")

            # Run model inference if model is available
            predictions = None
            predictions_denorm = None
            inference_time = None

            if self.model is not None:
                logger.info("Running model inference...")
                inference_start = time.time()

                predictions = []
                predictions_denorm = []
                with torch.no_grad():
                    for i, batch in enumerate(batches):
                        logger.debug(f"Processing batch {i + 1}/{len(batches)}")

                        pred = self.model(
                            x_static=batch.x_static.unsqueeze(0),
                            x_dynamic_past=batch.x_dynamic_past.unsqueeze(0),
                            x_dynamic_future=batch.x_dynamic_future.unsqueeze(0),
                            x_weather=batch.x_weather.unsqueeze(0),
                            edge_index=batch.edge_index,
                        )

                        # De-normalize predictions to actual values
                        pred_denorm = self._denormalize_predictions(pred)

                        predictions.append(pred)
                        predictions_denorm.append(pred_denorm)

                inference_time = time.time() - inference_start
                logger.info(f"Inference completed in {inference_time:.2f}s")

            # Prepare results
            total_time = time.time() - start_time

            # Reconstruct predictions to DataFrame if available
            predictions_df = None
            if predictions_denorm:
                predictions_df = self.reconstruct_predictions(
                    predictions_denorm, batches, include_velocity=True
                )

            results = {
                "timestamp": datetime.now().isoformat(),
                "num_batches": len(batches),
                "num_nodes": batches[0].num_nodes if batches else 0,
                "load_time": load_time,
                "inference_time": inference_time,
                "total_time": total_time,
                "predictions_normalized": predictions,
                "predictions": predictions_denorm
                if predictions_denorm
                else predictions,
                "predictions_df": predictions_df,  # Reconstructed DataFrame
                "batches": batches,  # Include batches for custom reconstruction
                "normalization_constants": self.target_norm_constants,
            }

            # Save results if output path provided (saves de-normalized predictions)
            if output_path and results["predictions"]:
                self._save_predictions(
                    results["predictions"], output_path, predictions_df
                )

            logger.info(f"Inference completed successfully in {total_time:.2f}s")
            return results

        except Exception as e:
            logger.error(f"Inference failed: {e}", exc_info=True)
            raise

    def _save_predictions(
        self,
        predictions: list[torch.Tensor],
        output_path: str,
        predictions_df: Optional[pd.DataFrame] = None,
    ):
        """
        Save predictions to file.

        Saves both raw numpy arrays and the reconstructed DataFrame (if available).

        Args:
            predictions: List of prediction tensors.
            output_path: Base output path (will add .npy for arrays, .parquet for DataFrame).
            predictions_df: Optional reconstructed DataFrame with NodeID/Timestamp index.
        """
        logger.info(f"Saving predictions to {output_path}")
        try:
            # Convert to numpy and save
            pred_array = torch.cat(predictions, dim=0).cpu().numpy()

            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save as numpy array
            import numpy as np

            np.save(output_path, pred_array)
            logger.info(f"Raw predictions saved: {pred_array.shape}")

            # Save DataFrame as parquet if available
            if predictions_df is not None and not predictions_df.empty:
                # Replace .npy extension with .parquet if present
                df_output_path = (
                    str(output_path).replace(".npy", "") + "_reconstructed.parquet"
                )
                predictions_df.to_parquet(df_output_path)
                logger.info(f"Reconstructed predictions saved to {df_output_path}")

        except Exception as e:
            logger.error(f"Failed to save predictions: {e}")

    def batch_predict(
        self,
        data_dir: str,
        output_dir: str,
        file_pattern: str = "*.parquet",
    ):
        """
        Run inference on multiple data files in a directory.

        Args:
            data_dir: Directory containing data files
            output_dir: Directory to save predictions
            file_pattern: Pattern to match data files
        """
        logger.info(f"Starting batch prediction on {data_dir}")

        data_path = Path(data_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Find all data file sets
        # This is simplified - adapt to your file organization
        jams_files = sorted(data_path.glob(f"jams_{file_pattern}"))

        results_summary = []

        for i, jams_file in enumerate(jams_files):
            logger.info(f"Processing file set {i + 1}/{len(jams_files)}")

            # Derive other file names from jams file
            # Adapt this logic to your naming convention
            base_name = jams_file.stem.replace("jams_", "")

            try:
                results = self.predict(
                    jams_path=str(jams_file),
                    irregularities_path=str(
                        data_path / f"irregularidades_{base_name}.parquet"
                    ),
                    weather_path=str(data_path / f"clima_{base_name}.parquet"),
                    pmt_puntos_path=str(data_path / f"pmt_puntos_{base_name}.parquet"),
                    pmt_tramos_path=str(data_path / f"pmt_tramos_{base_name}.parquet"),
                    events_path=str(data_path / f"eventos_{base_name}.parquet"),
                    incidents_path=str(data_path / f"incidentes_{base_name}.parquet"),
                    output_path=str(output_path / f"predictions_{base_name}.npy"),
                )

                results_summary.append(
                    {
                        "file": base_name,
                        "status": "success",
                        "total_time": results["total_time"],
                    }
                )

            except Exception as e:
                logger.error(f"Failed to process {base_name}: {e}")
                results_summary.append(
                    {
                        "file": base_name,
                        "status": "failed",
                        "error": str(e),
                    }
                )

        # Save summary
        summary_df = pd.DataFrame(results_summary)
        summary_path = output_path / "batch_summary.csv"
        summary_df.to_csv(summary_path, index=False)

        logger.info(f"Batch prediction completed. Summary saved to {summary_path}")

        # Print summary
        success_count = (summary_df["status"] == "success").sum()
        logger.info(
            f"Successfully processed {success_count}/{len(results_summary)} files"
        )

    def cleanup(self):
        """Cleanup resources."""
        logger.info("Cleaning up resources...")
        self.loader.cleanup()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run traffic forecasting inference")

    parser.add_argument(
        "--mode",
        choices=["single", "batch"],
        default="single",
        help="Inference mode: single file or batch processing",
    )

    # Required paths
    parser.add_argument("--graph", required=True, help="Path to graph pickle file")
    parser.add_argument("--static", required=True, help="Path to static features HDF5")
    parser.add_argument("--norm", required=True, help="Path to normalization constants")

    # Optional model
    parser.add_argument("--model", help="Path to trained model (optional)")

    # Single mode arguments
    parser.add_argument("--jams", help="Path to jams parquet")
    parser.add_argument("--irregularities", help="Path to irregularities parquet")
    parser.add_argument("--weather", help="Path to weather parquet")
    parser.add_argument("--pmt-puntos", help="Path to PMT points parquet")
    parser.add_argument("--pmt-tramos", help="Path to PMT tracts parquet")
    parser.add_argument("--events", help="Path to events parquet")
    parser.add_argument("--incidents", help="Path to incidents parquet")
    parser.add_argument("--output", help="Output path for predictions")

    # Batch mode arguments
    parser.add_argument("--data-dir", help="Directory with input data (batch mode)")
    parser.add_argument("--output-dir", help="Directory for outputs (batch mode)")

    # Options
    parser.add_argument(
        "--use-ray", action="store_true", help="Enable Ray parallelization"
    )

    args = parser.parse_args()

    # Initialize pipeline
    pipeline = InferencePipeline(
        graph_path=args.graph,
        static_data_path=args.static,
        normalization_constants_path=args.norm,
        model_path=args.model,
        use_ray=args.use_ray,
    )

    try:
        if args.mode == "single":
            # Single file inference
            if not all(
                [
                    args.jams,
                    args.irregularities,
                    args.weather,
                    args.pmt_puntos,
                    args.pmt_tramos,
                    args.events,
                    args.incidents,
                ]
            ):
                parser.error("Single mode requires all data file paths")

            results = pipeline.predict(
                jams_path=args.jams,
                irregularities_path=args.irregularities,
                weather_path=args.weather,
                pmt_puntos_path=args.pmt_puntos,
                pmt_tramos_path=args.pmt_tramos,
                events_path=args.events,
                incidents_path=args.incidents,
                output_path=args.output,
            )

            logger.info(f"Results: {results}")

        else:
            # Batch processing
            if not all([args.data_dir, args.output_dir]):
                parser.error("Batch mode requires --data-dir and --output-dir")

            pipeline.batch_predict(
                data_dir=args.data_dir,
                output_dir=args.output_dir,
            )

    finally:
        pipeline.cleanup()


if __name__ == "__main__":
    import sys

    # Set up command line arguments for testing
    sys.argv = [
        "inference_production.py",
        "--mode",
        "single",
        "--graph",
        "data/datos_procesados/conectivity_graph/road_graph.pickle",
        "--static",
        "data/datos_procesados/sparse_datasets/static_dataset.h5",
        "--norm",
        "data/datos_procesados/sparse_datasets/normalization_constants.pkl",
        "--jams",
        "data/inference_tests/df_jams_test.parquet",
        "--irregularities",
        "data/inference_tests/df_irregularidades_test.parquet",
        "--weather",
        "data/inference_tests/df_clima_test.parquet",
        "--pmt-puntos",
        "data/inference_tests/df_pmt_puntos_test.parquet",
        "--pmt-tramos",
        "data/inference_tests/df_pmt_tramos_test.parquet",
        "--events",
        "data/inference_tests/df_eventos_test.parquet",
        "--incidents",
        "data/inference_tests/df_incidentes_test.parquet",
    ]

    main()
