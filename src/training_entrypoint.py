import math
import os  # noqa: I001
import sys
import time
import logging
import argparse
import pickle
import shutil
from pathlib import Path
from collections import defaultdict
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler
from tqdm.auto import tqdm

# Add project root to path to allow package imports when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.sparse_dataloader import TrafficDataset
from src.temporal_split import temporal_train_test_split
from src.training_utils import (
    pad_collate,
    build_node_50_picker,
    EpochUniquePicker,
    SeededBatchSampler,
)
from src.models.gnn_temporal import GNN_Temporal_Model
from src.training.loop import train_one_epoch
from src.training.losses import masked_mse
from src.utils.logging import log

LOG_LEVEL = int(os.getenv("SM_LOG_LEVEL", logging.INFO))
CHANNEL_TRAIN = os.getenv("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train")
CHANNEL_VALIDATION = os.getenv("SM_CHANNEL_VALIDATION", "/opt/ml/input/data/validation")
MODEL_DIR = os.getenv("SM_MODEL_DIR", "/opt/ml/model")
CHECKPOINT_DIR = os.getenv("SM_CHECKPOINTS_PATH", "/opt/ml/checkpoints")

logging.basicConfig(level=LOG_LEVEL, format="[%(asctime)s] %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "t"}:
        return True
    if normalized in {"0", "false", "no", "n", "f"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _tensor_stats(t: torch.Tensor) -> dict[str, float]:
    finite = torch.isfinite(t)
    nan_count = int((~finite & torch.isnan(t)).sum().item())
    inf_count = int((~finite & torch.isinf(t)).sum().item())
    max_abs = float(torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0).abs().max().item())
    return {"nan": nan_count, "inf": inf_count, "max_abs": max_abs}


def _sanitize_(t: torch.Tensor, clamp: float = 1e6) -> torch.Tensor:
    torch.nan_to_num_(t, nan=0.0, posinf=clamp, neginf=-clamp)
    t.clamp_(-clamp, clamp)
    return t


def _assert_edge_index_ok(edge_index: torch.Tensor, num_nodes: int) -> None:
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError("edge_index must be (2, E)")
    if edge_index.numel() == 0:
        raise ValueError("edge_index is empty")
    min_val = int(edge_index.min())
    max_val = int(edge_index.max())
    if min_val < 0:
        raise ValueError("edge_index has negative node ids")
    if max_val >= int(num_nodes):
        raise ValueError(f"edge_index has node id >= N ({max_val} >= {num_nodes})")


def _assert_finite(name: str, t: torch.Tensor) -> None:
    if not torch.isfinite(t).all():
        stats = _tensor_stats(t)
        raise ValueError(f"{name} has NaN/Inf values: {stats}")


def _log_metrics(prefix: str, metrics: dict[str, float], epoch: int, step: int) -> None:
    metrics_with_idx = dict(metrics)
    metrics_with_idx["epoch_idx"] = float(epoch)
    metrics_with_idx["step_idx"] = float(step)
    parts = [f"{k}={v:.6f}" for k, v in metrics_with_idx.items()]
    parts.append(f"epoch={epoch}")
    parts.append(f"step={step}")
    parts.append(f"phase={prefix}")
    logger.info("[METRIC] %s", " ".join(parts))


def _maybe_split_losses(
    preds: torch.Tensor,
    y_target: torch.Tensor,
    loss_fn: nn.Module,
) -> dict[str, float]:
    losses: dict[str, float] = {}
    if y_target.dim() == 4 and y_target.size(-1) >= 1:
        level_target = y_target[..., 0]
        if preds.shape == level_target.shape:
            losses["loss_level"] = float(loss_fn(preds, level_target))
        elif preds.dim() == 4 and preds.size(-1) >= 1:
            level_pred = preds[..., 0]
            if level_pred.shape == level_target.shape:
                losses["loss_level"] = float(loss_fn(level_pred, level_target))
        if y_target.size(-1) >= 2 and preds.dim() == 4 and preds.size(-1) >= 2:
            vel_target = y_target[..., 1]
            vel_pred = preds[..., 1]
            if vel_pred.shape == vel_target.shape:
                losses["loss_velocity"] = float(loss_fn(vel_pred, vel_target))
    return losses


def _num_batches(num_items: int, batch_size: int, drop_last: bool) -> int:
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if num_items <= 0:
        return 0
    if drop_last:
        return num_items // batch_size
    return int(math.ceil(num_items / batch_size))


def _build_seed_list(
    picker: EpochUniquePicker | None,
    fallback_seed: str,
    num_batches: int,
) -> list[str]:
    if num_batches <= 0:
        return []
    if picker is None:
        return [fallback_seed] * num_batches
    return [str(picker.pick()) for _ in range(num_batches)]


def _ensure_seed_list_length(seed_list: list[str], num_batches: int) -> list[str]:
    if num_batches <= 0:
        return []
    if not seed_list:
        raise ValueError("seed_list is empty but num_batches > 0")
    if len(seed_list) >= num_batches:
        return seed_list[:num_batches]
    return seed_list + [seed_list[-1]] * (num_batches - len(seed_list))


def parse_args() -> dict:
    parser = argparse.ArgumentParser(description="SageMaker Training for Traffic Forecasting GNN")

    # Hyperparameters
    parser.add_argument(
        "--epochs",
        help="Number of training epochs",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--batch-size",
        help="Batch size for training",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--learning-rate",
        help="Learning rate for Adam optimizer",
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        "--grad-clip-norm",
        help="Gradient clipping norm (None to disable)",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--model-type",
        help="GNN model type: GConvGRU, TGCN, GConvLSTM, A3TGCN",
        type=str,
        default="GConvGRU",
    )
    parser.add_argument(
        "--input-seq-len",
        help="Number of past timesteps",
        type=int,
        default=150,
    )
    parser.add_argument(
        "--output-seq-len",
        help="Number of forecast timesteps",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--node-batch-size",
        help="Number of nodes per sample (full network if >= total nodes)",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--train-ratio",
        help="Ratio of training samples (rest for validation)",
        type=float,
        default=0.8,
    )
    parser.add_argument(
        "--include-weather",
        help="Include weather features in dynamic inputs",
        type=_str2bool,
        default=True,
    )
    parser.add_argument(
        "--include-velocity-target",
        help="Include velocity as second target dimension",
        type=_str2bool,
        default=True,
    )
    parser.add_argument(
        "--sanity-checks",
        help="Enable tensor sanity checks and stats logging",
        type=_str2bool,
        default=False,
    )
    parser.add_argument(
        "--debug-batches",
        help="(DEBUG MODE) Limit training to N batches per epoch (for quick testing)",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--loss-high-level-threshold",
        help="Minimum jam level to apply extra MAE penalty",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--loss-high-level-lambda",
        help="Weight for high-level MAE penalty",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--loss-piecewise-threshold",
        help="Minimum jam level to apply severe MAE penalty",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--loss-piecewise-mae-lambda",
        help="Weight for severe MAE penalty",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--loss-directional-lambda",
        help="Weight for under-prediction directional penalty",
        type=float,
        default=0.3,
    )
    parser.add_argument(
        "--seeded-batch-sampler",
        help="Use a stateless seed list with a batch sampler (allows num_workers > 0)",
        type=_str2bool,
        default=True,
    )
    parser.add_argument(
        "--num-workers",
        help="Number of DataLoader workers (use >0 only with seeded-batch-sampler)",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--pin-memory",
        help="Enable DataLoader pin_memory for faster host->device transfer",
        type=_str2bool,
        default=True,
    )
    parser.add_argument(
        "--prefetch-factor",
        help="Number of prefetched batches per worker (only used when num_workers > 0)",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--persistent-workers",
        help="Keep DataLoader workers alive between epochs (only used when num_workers > 0)",
        type=_str2bool,
        default=True,
    )
    parser.add_argument(
        "--cache-local",
        help="Copy required training files to local cache for faster access",
        type=_str2bool,
        default=False,
    )
    parser.add_argument(
        "--cache-dir",
        help="Local cache directory when cache-local is enabled",
        type=str,
        default="/opt/ml/input/data/cache",
    )
    parser.add_argument(
        "--use-minimal-datasets",
        help="Only load the minimal set of dynamic datasets required for training",
        type=_str2bool,
        default=True,
    )
    parser.add_argument(
        "--drop-last",
        help="Drop last incomplete batch in each epoch",
        type=_str2bool,
        default=False,
    )
    parser.add_argument(
        "--shuffle-time",
        help="Shuffle time indices inside each epoch",
        type=_str2bool,
        default=True,
    )
    parser.add_argument(
        "--seeded-batch-seed",
        help="Seed for seeded batch sampler shuffling",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--timing-steps",
        help="Log batch timing every N steps (0 disables)",
        type=int,
        default=0,
    )

    payload = vars(parser.parse_args())
    logger.info(f"Parsed arguments:\n{payload}")
    return payload


def check_data_channels() -> tuple[Path, Path]:
    """
    Check if the data channels for training and validation exist.
    In SageMaker, the training data is in SM_CHANNEL_TRAIN containing:
    - Graph files
    - Static features HDF5
    - Dynamic features HDF5
    - Normalization constants
    """
    train_path = Path(CHANNEL_TRAIN)
    if not train_path.exists():
        raise ValueError(f"Training data channel '{CHANNEL_TRAIN}' does not exist.")
    
    logger.info(f"Training data files in {CHANNEL_TRAIN}:")
    for item in sorted(train_path.rglob("*")):
        if item.is_file():
            logger.info(f"  - {item.relative_to(train_path)}")

    return train_path


def _select_dynamic_files(sparse_dir: Path, minimal: bool) -> list[Path]:
    if minimal:
        minimal_dynamic = [
            "coi_points_dataset.h5",
            "coi_tracts_dataset.h5",
            "irregularities_dataset.h5",
            "events_dataset.h5",
            "incidents_dataset.h5",
            "jams_variables_dataset.h5",
        ]
        missing = [name for name in minimal_dynamic if not (sparse_dir / name).exists()]
        if missing:
            raise ValueError(f"Missing required dynamic files: {missing}")
        return [sparse_dir / name for name in minimal_dynamic]

    dynamic_files = sorted(
        [
            f
            for f in sparse_dir.glob("*_dataset.h5")
            if f.name not in ["static_dataset.h5", "jams_dataset.h5"]
        ]
    )
    if not dynamic_files:
        raise ValueError(f"No dynamic feature files found in {sparse_dir}")
    return dynamic_files


def _required_weather_files(sparse_dir: Path) -> list[Path]:
    names = [
        "weather_humidity.parquet",
        "weather_precipitation.parquet",
        "weather_wind_direction.parquet",
        "weather_wind_speed.parquet",
        "weather_stations.pkl",
    ]
    missing = [name for name in names if not (sparse_dir / name).exists()]
    if missing:
        raise ValueError(f"Missing required weather files: {missing}")
    return [sparse_dir / name for name in names]


def _copy_if_exists(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def cache_training_data(
    train_path: Path,
    cache_dir: Path,
    include_weather: bool,
    use_minimal_datasets: bool,
) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir = train_path / "sparse_datasets"
    if not sparse_dir.exists():
        raise ValueError(f"Sparse datasets directory not found: {sparse_dir}")

    dynamic_files = _select_dynamic_files(sparse_dir, use_minimal_datasets)
    dynamic_metadata = [sparse_dir / f"{f.with_suffix('').name}_metadata.pkl" for f in dynamic_files]
    missing_meta = [str(f) for f in dynamic_metadata if not f.exists()]
    if missing_meta:
        raise ValueError(f"Missing metadata files: {missing_meta}")
    all_metadata = sorted(sparse_dir.glob("*_metadata.pkl"))
    if not all_metadata:
        raise ValueError(f"No metadata files found in {sparse_dir}")

    required = [
        train_path / "conectivity_graph" / "road_graph.pickle",
        sparse_dir / "static_dataset.h5",
        sparse_dir / "jams_dataset.h5",
        sparse_dir / "normalization_constants.pkl",
        train_path / "neighbor_features.parquet",
        train_path / "nodes_traffic_info.csv",
    ]
    required.extend(dynamic_files)
    required.extend(all_metadata)
    if include_weather:
        required.extend(_required_weather_files(sparse_dir))

    for src in required:
        rel_path = src.relative_to(train_path)
        dst = cache_dir / rel_path
        _copy_if_exists(src, dst)

    return cache_dir


def setup_sagemaker_paths(
    train_path: Path,
    include_weather: bool,
    use_minimal_datasets: bool,
) -> dict:
    """
    Set up all required file paths from the SageMaker training data channel.
    SageMaker structure:
    /opt/ml/input/data/train/
    ├── conectivity_graph/
    │   └── road_graph.pickle
    ├── sparse_datasets/
    │   ├── static_dataset.h5
    │   ├── *.h5 (dynamic features)
    │   ├── *_metadata.pkl
    │   ├── jams_dataset.h5 (target)
    │   └── normalization_constants.pkl
    └── neighbor_features.parquet (optional)
    """
    paths = {}
    
    # Graph
    paths["graph"] = train_path / "conectivity_graph" / "road_graph.pickle"
    if not paths["graph"].exists():
        raise ValueError(f"Graph not found: {paths['graph']}")
    
    # Sparse data directory
    sparse_dir = train_path / "sparse_datasets"
    if not sparse_dir.exists():
        raise ValueError(f"Sparse datasets directory not found: {sparse_dir}")
    
    # Static data
    paths["static"] = sparse_dir / "static_dataset.h5"
    if not paths["static"].exists():
        raise ValueError(f"Static data not found: {paths['static']}")
    
    # Dynamic data files
    dynamic_files = _select_dynamic_files(sparse_dir, use_minimal_datasets)
    paths["dynamic"] = dynamic_files

    # Metadata files (matching dynamic files)
    metadata_files = [sparse_dir / f"{f.with_suffix('').name}_metadata.pkl" for f in dynamic_files]
    missing_meta = [str(f) for f in metadata_files if not f.exists()]
    if missing_meta:
        raise ValueError(f"Missing metadata files: {missing_meta}")
    paths["metadata"] = metadata_files
    
    # Target data
    paths["target"] = sparse_dir / "jams_dataset.h5"
    if not paths["target"].exists():
        raise ValueError(f"Target data not found: {paths['target']}")
    
    # Normalization constants
    paths["normalization"] = sparse_dir / "normalization_constants.pkl"
    if not paths["normalization"].exists():
        logger.warning(f"Normalization constants not found: {paths['normalization']}")
        logger.warning("Training will proceed WITHOUT normalization")
        paths["normalization"] = None
    
    # Optional neighbor features
    paths["neighbor_features"] = train_path / "neighbor_features.parquet"
    if not paths["neighbor_features"].exists():
        logger.warning(f"Neighbor features not found: {paths['neighbor_features']}")
        paths["neighbor_features"] = None

    # Optional node traffic info for seed picker
    paths["nodes_traffic_info"] = train_path / "nodes_traffic_info.csv"
    if not paths["nodes_traffic_info"].exists():
        logger.warning(f"Nodes traffic info not found: {paths['nodes_traffic_info']}")
        paths["nodes_traffic_info"] = None
    
    # Weather data directory (optional)
    if include_weather:
        _required_weather_files(sparse_dir)
    paths["weather_dir"] = sparse_dir
    
    logger.info("SageMaker paths validated successfully")
    return paths


def train_model(args: dict, paths: dict) -> None:
    """
    Full training pipeline:
    1. Load dataset with TrafficDataset
    2. Split into train/validation
    3. Create model
    4. Run training loop with seed rotation
    5. Save model and metrics
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Debug mode flag
    debug_batches = args.get("debug_batches", None)
    if debug_batches:
        logger.warning(f"🔴 DEBUG MODE ENABLED: Training/validation limited to {debug_batches} batches")
    sanity_checks = args.get("sanity_checks", False)
    
    # ============ 1. Load Dataset ============
    logger.info("Loading TrafficDataset...")
    traffic_dataset = TrafficDataset(
        graph_path=str(paths["graph"]),
        static_data_path=str(paths["static"]),
        dynamic_data_paths=[str(p) for p in paths["dynamic"]],
        metadata_paths=[str(p) for p in paths["metadata"]],
        target_data_path=str(paths["target"]),
        in_timesteps=args["input_seq_len"],
        out_timesteps=args["output_seq_len"],
        future_features=[
            "coi_puntos", "coi_tramo", "AGLOMERACION", "EMERGENCIA", "MANIFESTACION"
        ],
        node_batch_size=args["node_batch_size"],
        weather_data_dir=str(paths["weather_dir"]),
        include_weather=args["include_weather"],
        neighbor_features_path=str(paths["neighbor_features"]) if paths["neighbor_features"] else None,
        include_velocity_target=args["include_velocity_target"],
        normalization_constants_path=str(paths["normalization"]) if paths["normalization"] else None,
    )
    logger.info(f"Dataset loaded: {len(traffic_dataset)} samples")
    logger.info(f"  Num nodes: {traffic_dataset.num_nodes}")
    logger.info(f"  Static features: {traffic_dataset.num_static_features}")
    logger.info(f"  Dynamic features: {traffic_dataset.num_dynamic_features}")
    logger.info(f"  Target features: {traffic_dataset.num_target_features if hasattr(traffic_dataset, 'num_target_features') else 1 + (1 if args['include_velocity_target'] else 0)}")
    logger.info(
        "  include_weather=%s num_weather_features=%s",
        args["include_weather"],
        getattr(traffic_dataset, "num_weather_features", 0),
    )
    
    # ============ 2. Split Dataset ============
    use_seeded_batches = args.get("seeded_batch_sampler", True)
    if traffic_dataset.total_timesteps_in_data < traffic_dataset.total_timesteps_window:
        num_time_windows = 0
    else:
        num_time_windows = (
            traffic_dataset.total_timesteps_in_data
            - traffic_dataset.total_timesteps_window
            + 1
        )
    if use_seeded_batches:
        train_end = int(num_time_windows * args["train_ratio"])
        train_time_indices = list(range(train_end))
        val_time_indices = list(range(train_end, num_time_windows))
        train_dataset = traffic_dataset
        val_dataset = traffic_dataset
        logger.info(
            "Splitting time indices (train: %.1f%%, val: %.1f%%) -> %s/%s windows",
            args["train_ratio"] * 100,
            (1 - args["train_ratio"]) * 100,
            len(train_time_indices),
            len(val_time_indices),
        )
    else:
        logger.info(
            "Splitting dataset (train: %.1f%%, val: %.1f%%)...",
            args["train_ratio"] * 100,
            (1 - args["train_ratio"]) * 100,
        )
        train_dataset, val_dataset = temporal_train_test_split(
            traffic_dataset,
            train_ratio=args["train_ratio"],
        )
        logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # ============ 3. Create DataLoaders ============
    num_workers = int(args.get("num_workers", 0))
    if not use_seeded_batches and num_workers != 0:
        logger.warning("num_workers=%s is not supported with per-batch seed rotation. Forcing num_workers=0.", num_workers)
        num_workers = 0
    logger.info("Using num_workers=%s", num_workers)
    pin_memory = bool(args.get("pin_memory", False))
    persistent_workers = bool(args.get("persistent_workers", False))
    prefetch_factor = int(args.get("prefetch_factor", 2))
    if num_workers <= 0:
        persistent_workers = False
        prefetch_factor = None

    loader_kwargs = {
        "num_workers": num_workers,
        "collate_fn": pad_collate,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor
        loader_kwargs["persistent_workers"] = persistent_workers

    future_dim = getattr(traffic_dataset, "num_dynamic_future_features", None)
    if future_dim is None:
        raise ValueError("TrafficDataset does not expose num_dynamic_future_features.")

    base_dynamic_dim = traffic_dataset.num_dynamic_features
    weather_dim = getattr(traffic_dataset, "num_weather_features", 0) or 0
    dynamic_dim = base_dynamic_dim
    if args["include_weather"]:
        if weather_dim > 0:
            logger.info(
                "Including weather features in dynamic inputs (+%s features).",
                weather_dim,
            )
            dynamic_dim += weather_dim
        else:
            logger.warning(
                "include_weather=True but num_weather_features is missing or 0."
            )
    logger.info(
        "Input dims for GNN: static=%s dynamic_base=%s weather=%s dynamic_total=%s future=%s",
        traffic_dataset.num_static_features,
        base_dynamic_dim,
        weather_dim if args["include_weather"] else 0,
        dynamic_dim,
        future_dim,
    )
    logger.info(
        "Expected GNN input shapes: x_static=(B,N,%s) x_dynamic_past=(B,N,%s,%s) x_dynamic_future=(B,N,%s,%s)",
        traffic_dataset.num_static_features,
        args["input_seq_len"],
        dynamic_dim,
        args["output_seq_len"],
        future_dim,
    )
    
    # ============ 4. Create Model ============
    logger.info(f"Creating {args['model_type']} model...")
    model = GNN_Temporal_Model(
        edge_index=None,
        static_dim=traffic_dataset.num_static_features,
        dynamic_dim=dynamic_dim,
        future_dim=future_dim,
        input_seq_len=args["input_seq_len"],
        forecast_horizon=args["output_seq_len"],
        model_type=args["model_type"],
        include_velocity_target=args["include_velocity_target"],
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # ============ 5. Setup Training ============
    optimizer = torch.optim.Adam(model.parameters(), lr=args["learning_rate"])
    loss_fn = nn.MSELoss()
    level_weights = torch.tensor([0.25, 1.0, 2.0, 3.0, 4.0, 5.0])
    grad_clip_norm = args["grad_clip_norm"]
    
    # Seed picker for node rotation
    base_picker = None
    if paths.get("nodes_traffic_info"):
        base_picker = build_node_50_picker(
            str(paths["nodes_traffic_info"]),
            top_frac=0.2,
            seed=42,
        )
    
    if base_picker is None:
        logger.warning("Could not create seed picker, training will use single seed")
        picker = None
    else:
        picker = EpochUniquePicker(base_picker, p_top=0.5, on_exhaust="reset")
    
    # Create checkpoint directory
    Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
    
    metrics_history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "train_time": [],
    }
    
    # ============ 6. Training Loop ============
    logger.info(f"Starting training for {args['epochs']} epochs...")
    best_val_loss = float('inf')
    
    for epoch in range(args["epochs"]):
        epoch_start = time.time()
        epoch_num = epoch + 1
        
        # Train
        logger.info(f"\n{'='*80}")
        logger.info(f"Epoch {epoch_num}/{args['epochs']}")
        logger.info(f"{'='*80}")

        model.train()
        train_loss = 0.0
        train_samples = 0

        if use_seeded_batches:
            if picker is not None:
                picker.start_epoch()
            train_batches = _num_batches(
                len(train_time_indices),
                args["batch_size"],
                args.get("drop_last", False),
            )
            if train_batches == 0:
                raise ValueError("No training batches available for the selected time window split.")
            train_seed_list = _build_seed_list(
                picker,
                traffic_dataset.node_seed,
                train_batches,
            )
            traffic_dataset.set_seed_list(train_seed_list)
            train_loader = DataLoader(
                train_dataset,
                batch_sampler=SeededBatchSampler(
                    seed_list=train_seed_list,
                    time_indices=train_time_indices,
                    num_time_windows=num_time_windows,
                    batch_size=args["batch_size"],
                    drop_last=args.get("drop_last", False),
                    shuffle_time=args.get("shuffle_time", True),
                    rng_seed=args.get("seeded_batch_seed", 42) + epoch_num,
                ),
                **loader_kwargs,
            )
        else:
            if picker is not None:
                picker.start_epoch()
            train_loader = DataLoader(
                train_dataset,
                batch_size=args["batch_size"],
                shuffle=True,
                **loader_kwargs,
            )
        
        epoch_train_steps = len(train_loader)
        if debug_batches is not None:
            epoch_train_steps = min(epoch_train_steps, debug_batches)

        pbar = tqdm(train_loader, desc="Training", leave=True)
        prev_batch_end = time.perf_counter()
        timing_window_sums: dict[str, float] = defaultdict(float)
        timing_window_count = 0
        timing_epoch_sums: dict[str, float] = defaultdict(float)
        timing_epoch_count = 0
        metric_window_sums: dict[str, float] = defaultdict(float)
        metric_window_counts: dict[str, int] = defaultdict(int)
        metric_epoch_sums: dict[str, float] = defaultdict(float)
        metric_epoch_counts: dict[str, int] = defaultdict(int)
        for batch_idx, batch in enumerate(pbar):
            # Break if debug mode limit reached
            if debug_batches and batch_idx >= debug_batches:
                logger.info(f"Debug mode: Breaking training after {batch_idx} batches")
                break
            
            # Move to device
            batch_start = time.perf_counter()
            data_wait = batch_start - prev_batch_end
            x_static = batch["x_static"].to(device)
            x_dyn_past = batch["x_dynamic_past"].to(device)
            x_dyn_future = batch["x_dynamic_future"].to(device)
            y_target = batch["y"].to(device)
            edge_index = batch["edge_index"].to(device)
            after_transfer = time.perf_counter()

            if batch_idx == 0:
                dyn_feature_dim = x_dyn_past.shape[-1]
                if args["include_weather"]:
                    base_dim = traffic_dataset.num_dynamic_features
                    weather_dim = getattr(traffic_dataset, "num_weather_features", 0)
                    logger.info(
                        "First batch shapes: x_static=%s x_dynamic_past=%s x_dynamic_future=%s y_target=%s edge_index=%s",
                        tuple(x_static.shape),
                        tuple(x_dyn_past.shape),
                        tuple(x_dyn_future.shape),
                        tuple(y_target.shape),
                        tuple(edge_index.shape),
                    )
                    logger.info(
                        "Weather check: dynamic_features=%s (expected base=%s + weather=%s)",
                        dyn_feature_dim,
                        base_dim,
                        weather_dim,
                    )
                else:
                    logger.info(
                        "First batch shapes: x_static=%s x_dynamic_past=%s x_dynamic_future=%s y_target=%s edge_index=%s",
                        tuple(x_static.shape),
                        tuple(x_dyn_past.shape),
                        tuple(x_dyn_future.shape),
                        tuple(y_target.shape),
                        tuple(edge_index.shape),
                    )
                    logger.info(
                        "Weather check: include_weather=False dynamic_features=%s",
                        dyn_feature_dim,
                    )

            if sanity_checks and batch_idx == 0:
                for name, tensor in [
                    ("x_static", x_static),
                    ("x_dynamic_past", x_dyn_past),
                    ("x_dynamic_future", x_dyn_future),
                    ("y_target", y_target),
                ]:
                    stats = _tensor_stats(tensor)
                    logger.info("Sanity stats %s: %s", name, stats)
            if sanity_checks:
                _assert_edge_index_ok(edge_index, int(x_static.shape[1]))
                _sanitize_(x_static)
                _sanitize_(x_dyn_past)
                _sanitize_(x_dyn_future)
                _assert_finite("x_static", x_static)
                _assert_finite("x_dynamic_past", x_dyn_past)
                _assert_finite("x_dynamic_future", x_dyn_future)
                _assert_finite("y_target", y_target)
            
            B = x_static.shape[0]
            
            # Forward pass
            optimizer.zero_grad(set_to_none=True)
            preds = model(edge_index, x_static[0], x_dyn_past.permute(0, 2, 1, 3), x_dyn_future.permute(0, 2, 1, 3))
            after_forward = time.perf_counter()
            if sanity_checks:
                _assert_finite("preds", preds)
            
            # Loss
            if y_target.dim() == 4 and y_target.size(-1) >= 1:
                level_pred = preds[..., 0] if preds.dim() == 4 else preds
                level_target = y_target[..., 0]
                loss = masked_mse(
                    level_pred,
                    level_target,
                    level_weights=level_weights,
                    high_level_threshold=args["loss_high_level_threshold"],
                    high_level_lambda=args["loss_high_level_lambda"],
                    piecewise_threshold=args["loss_piecewise_threshold"],
                    piecewise_mae_lambda=args["loss_piecewise_mae_lambda"],
                    directional_lambda=args["loss_directional_lambda"],
                )
                if y_target.size(-1) >= 2 and preds.dim() == 4 and preds.size(-1) >= 2:
                    loss = loss + loss_fn(preds[..., 1], y_target[..., 1])
            else:
                loss = masked_mse(
                    preds,
                    y_target,
                    level_weights=level_weights,
                    high_level_threshold=args["loss_high_level_threshold"],
                    high_level_lambda=args["loss_high_level_lambda"],
                    piecewise_threshold=args["loss_piecewise_threshold"],
                    piecewise_mae_lambda=args["loss_piecewise_mae_lambda"],
                    directional_lambda=args["loss_directional_lambda"],
                )
            
            after_loss = time.perf_counter()
            # Backward
            loss.backward()
            after_backward = time.perf_counter()
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()
            after_step = time.perf_counter()
            
            train_loss += loss.item() * B
            train_samples += B
            
            pbar.set_postfix({"loss": f"{train_loss / max(train_samples, 1):.6f}"})

            step_num = epoch * epoch_train_steps + batch_idx + 1
            if args.get("timing_steps", 0) and step_num % args["timing_steps"] == 0:
                logger.info(
                    "[TIMING] epoch=%s step=%s data_wait=%.3f transfer=%.3f forward=%.3f loss=%.3f backward=%.3f step=%.3f total=%.3f",
                    epoch_num,
                    step_num,
                    data_wait,
                    after_transfer - batch_start,
                    after_forward - after_transfer,
                    after_loss - after_forward,
                    after_backward - after_loss,
                    after_step - after_backward,
                    after_step - batch_start,
                )
                _log_metrics(
                    "timing",
                    {
                        "time_data_wait": data_wait,
                        "time_transfer": after_transfer - batch_start,
                        "time_forward": after_forward - after_transfer,
                        "time_loss": after_loss - after_forward,
                        "time_backward": after_backward - after_loss,
                        "time_step": after_step - after_backward,
                        "time_total": after_step - batch_start,
                    },
                    epoch_num,
                    step_num,
                )
            split_losses = _maybe_split_losses(preds, y_target, loss_fn)
            metrics = {"train_loss": float(loss)}
            metrics.update(split_losses)
            _log_metrics("train", metrics, epoch_num, step_num)
            for key, value in metrics.items():
                metric_window_sums[key] += value
                metric_window_counts[key] += 1
                metric_epoch_sums[key] += value
                metric_epoch_counts[key] += 1
            prev_batch_end = time.perf_counter()

            if args.get("timing_steps", 0):
                timing_window_sums["time_data_wait"] += data_wait
                timing_window_sums["time_transfer"] += after_transfer - batch_start
                timing_window_sums["time_forward"] += after_forward - after_transfer
                timing_window_sums["time_loss"] += after_loss - after_forward
                timing_window_sums["time_backward"] += after_backward - after_loss
                timing_window_sums["time_step"] += after_step - after_backward
                timing_window_sums["time_total"] += after_step - batch_start
                timing_window_count += 1

                timing_epoch_sums["time_data_wait"] += data_wait
                timing_epoch_sums["time_transfer"] += after_transfer - batch_start
                timing_epoch_sums["time_forward"] += after_forward - after_transfer
                timing_epoch_sums["time_loss"] += after_loss - after_forward
                timing_epoch_sums["time_backward"] += after_backward - after_loss
                timing_epoch_sums["time_step"] += after_step - after_backward
                timing_epoch_sums["time_total"] += after_step - batch_start
                timing_epoch_count += 1

            if args.get("timing_steps", 0) and timing_window_count:
                if step_num % args["timing_steps"] == 0:
                    timing_window_avgs = {
                        key: timing_window_sums[key] / timing_window_count
                        for key in timing_window_sums
                    }
                    _log_metrics("timing_window", timing_window_avgs, epoch_num, step_num)
                    timing_window_sums = defaultdict(float)
                    timing_window_count = 0

                    metric_window_avgs = {
                        key: metric_window_sums[key] / metric_window_counts[key]
                        for key in metric_window_sums
                        if metric_window_counts[key]
                    }
                    if metric_window_avgs:
                        _log_metrics("train_window", metric_window_avgs, epoch_num, step_num)
                    metric_window_sums = defaultdict(float)
                    metric_window_counts = defaultdict(int)
            
            # Rotate seed if picker available
            if picker is not None and not use_seeded_batches:
                next_seed = picker.pick()
                traffic_dataset.reasign_seed(str(next_seed))
        
        avg_train_loss = train_loss / max(train_samples, 1)
        logger.info(f"Train Loss: {avg_train_loss:.6f}")
        _log_metrics("train_epoch", {"train_loss": avg_train_loss}, epoch_num, epoch_train_steps)
        metric_epoch_avgs = {
            key: metric_epoch_sums[key] / metric_epoch_counts[key]
            for key in metric_epoch_sums
            if metric_epoch_counts[key]
        }
        if metric_epoch_avgs:
            _log_metrics("train_epoch_avg", metric_epoch_avgs, epoch_num, epoch_train_steps)
        if timing_epoch_count:
            timing_epoch_avgs = {
                key: timing_epoch_sums[key] / timing_epoch_count
                for key in timing_epoch_sums
            }
            _log_metrics("timing_epoch", timing_epoch_avgs, epoch_num, epoch_train_steps)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_samples = 0
        
        with torch.no_grad():
            if use_seeded_batches:
                val_batches = _num_batches(
                    len(val_time_indices),
                    args["batch_size"],
                    args.get("drop_last", False),
                )
                val_seed_list = _ensure_seed_list_length(train_seed_list, val_batches)
                traffic_dataset.set_seed_list(
                    val_seed_list or [traffic_dataset.node_seed]
                )
                val_loader = DataLoader(
                    val_dataset,
                    batch_sampler=SeededBatchSampler(
                        seed_list=val_seed_list or [traffic_dataset.node_seed],
                        time_indices=val_time_indices,
                        num_time_windows=num_time_windows,
                        batch_size=args["batch_size"],
                        drop_last=args.get("drop_last", False),
                        shuffle_time=False,
                        rng_seed=args.get("seeded_batch_seed", 42) + epoch_num,
                    ),
                    **loader_kwargs,
                )
            else:
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=args["batch_size"],
                    shuffle=False,
                    **loader_kwargs,
                )

            epoch_val_steps = len(val_loader)
            if debug_batches is not None:
                epoch_val_steps = min(epoch_val_steps, debug_batches)

            pbar = tqdm(val_loader, desc="Validation", leave=True)
            for val_idx, batch in enumerate(pbar):
                # Break if debug mode limit reached
                if debug_batches and val_idx >= debug_batches:
                    logger.info(f"Debug mode: Breaking validation after {val_idx} batches")
                    break
                
                x_static = batch["x_static"].to(device)
                x_dyn_past = batch["x_dynamic_past"].to(device)
                x_dyn_future = batch["x_dynamic_future"].to(device)
                y_target = batch["y"].to(device)
                edge_index = batch["edge_index"].to(device)

                if sanity_checks and val_idx == 0:
                    for name, tensor in [
                        ("val_x_static", x_static),
                        ("val_x_dynamic_past", x_dyn_past),
                        ("val_x_dynamic_future", x_dyn_future),
                        ("val_y_target", y_target),
                    ]:
                        stats = _tensor_stats(tensor)
                        logger.info("Sanity stats %s: %s", name, stats)
                if sanity_checks:
                    _assert_edge_index_ok(edge_index, int(x_static.shape[1]))
                    _sanitize_(x_static)
                    _sanitize_(x_dyn_past)
                    _sanitize_(x_dyn_future)
                    _assert_finite("val_x_static", x_static)
                    _assert_finite("val_x_dynamic_past", x_dyn_past)
                    _assert_finite("val_x_dynamic_future", x_dyn_future)
                    _assert_finite("val_y_target", y_target)
                
                B = x_static.shape[0]
                
                preds = model(edge_index, x_static[0], x_dyn_past.permute(0, 2, 1, 3), x_dyn_future.permute(0, 2, 1, 3))
                if sanity_checks:
                    _assert_finite("val_preds", preds)
                if y_target.dim() == 4 and y_target.size(-1) >= 1:
                    level_pred = preds[..., 0] if preds.dim() == 4 else preds
                    level_target = y_target[..., 0]
                    loss = masked_mse(
                        level_pred,
                        level_target,
                        level_weights=level_weights,
                        high_level_threshold=args["loss_high_level_threshold"],
                        high_level_lambda=args["loss_high_level_lambda"],
                        piecewise_threshold=args["loss_piecewise_threshold"],
                        piecewise_mae_lambda=args["loss_piecewise_mae_lambda"],
                        directional_lambda=args["loss_directional_lambda"],
                    )
                    if y_target.size(-1) >= 2 and preds.dim() == 4 and preds.size(-1) >= 2:
                        loss = loss + loss_fn(preds[..., 1], y_target[..., 1])
                else:
                    loss = masked_mse(
                        preds,
                        y_target,
                        level_weights=level_weights,
                        high_level_threshold=args["loss_high_level_threshold"],
                        high_level_lambda=args["loss_high_level_lambda"],
                        piecewise_threshold=args["loss_piecewise_threshold"],
                        piecewise_mae_lambda=args["loss_piecewise_mae_lambda"],
                        directional_lambda=args["loss_directional_lambda"],
                    )
                
                val_loss += loss.item() * B
                val_samples += B
                pbar.set_postfix({"val_loss": f"{val_loss / max(val_samples, 1):.6f}"})

                step_num = epoch * epoch_val_steps + val_idx + 1
                split_losses = _maybe_split_losses(preds, y_target, loss_fn)
                metrics = {"val_loss": float(loss)}
                metrics.update({f"val_{k}": v for k, v in split_losses.items()})
                _log_metrics("val", metrics, epoch_num, step_num)
        
        avg_val_loss = val_loss / max(val_samples, 1)
        logger.info(f"Val Loss: {avg_val_loss:.6f}")
        _log_metrics("val_epoch", {"val_loss": avg_val_loss}, epoch_num, epoch_val_steps)
        
        model_config = {
            "static_dim": traffic_dataset.num_static_features,
            "dynamic_dim": dynamic_dim,
            "future_dim": future_dim,
            "input_seq_len": args["input_seq_len"],
            "forecast_horizon": args["output_seq_len"],
            "model_type": args["model_type"],
            "include_velocity_target": args["include_velocity_target"],
        }
        checkpoint_payload = {
            "state_dict": model.state_dict(),
            "model_kwargs": model_config,
            "torch_version": torch.__version__,
        }

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_path = Path(CHECKPOINT_DIR) / f"best_model.pth"
            torch.save(checkpoint_payload, checkpoint_path)
            logger.info(f"✓ New best model saved to {checkpoint_path}")
        
        # Save periodic checkpoint
        if (epoch + 1) % 5 == 0:
            checkpoint_path = Path(CHECKPOINT_DIR) / f"checkpoint_epoch_{epoch+1}.pth"
            torch.save(checkpoint_payload, checkpoint_path)
            logger.info(f"✓ Checkpoint saved to {checkpoint_path}")
        
        epoch_time = time.time() - epoch_start
        metrics_history["epoch"].append(epoch + 1)
        metrics_history["train_loss"].append(avg_train_loss)
        metrics_history["val_loss"].append(avg_val_loss)
        metrics_history["train_time"].append(epoch_time)
        
        logger.info(f"Epoch time: {epoch_time:.1f}s")
    
    # ============ 7. Save Final Model ============
    logger.info("\nTraining completed!")
    
    # Save final model to SM_MODEL_DIR
    model_dir = Path(MODEL_DIR)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    final_model_path = model_dir / "model.pth"
    torch.save(checkpoint_payload, final_model_path)
    logger.info(f"✓ Final model saved to {final_model_path}")
    
    # Save model config (separate for convenience)
    config_path = model_dir / "model_config.pkl"
    with open(config_path, "wb") as f:
        pickle.dump(model_config, f)
    logger.info(f"✓ Model config saved to {config_path}")

    training_args_path = model_dir / "training_args.pkl"
    with open(training_args_path, "wb") as f:
        pickle.dump(args, f)
    logger.info(f"✓ Training args saved to {training_args_path}")

    data_meta = {
        "paths": {k: (str(v) if v is not None else None) for k, v in paths.items()},
        "num_nodes": getattr(traffic_dataset, "num_nodes", None),
        "num_static_features": getattr(traffic_dataset, "num_static_features", None),
        "num_dynamic_features": getattr(traffic_dataset, "num_dynamic_features", None),
        "num_dynamic_future_features": getattr(
            traffic_dataset, "num_dynamic_future_features", None
        ),
        "num_target_features": getattr(traffic_dataset, "num_target_features", None),
        "target_feature_idx_map": getattr(
            traffic_dataset, "target_feature_idx_map", None
        ),
        "target_level_idx": getattr(traffic_dataset, "target_level_idx", None),
        "target_velocity_idx": getattr(traffic_dataset, "target_velocity_idx", None),
        "dynamic_feature_dims": getattr(traffic_dataset, "dynamic_feature_dims", None),
        "future_features": getattr(traffic_dataset, "future_features", None),
        "include_weather": getattr(traffic_dataset, "include_weather", None),
        "weather_features": getattr(traffic_dataset, "weather_features", None),
        "num_weather_features": getattr(traffic_dataset, "num_weather_features", None),
    }
    data_meta_path = model_dir / "data_meta.pkl"
    with open(data_meta_path, "wb") as f:
        pickle.dump(data_meta, f)
    logger.info(f"✓ Data metadata saved to {data_meta_path}")
    
    # Save metrics
    metrics_path = model_dir / "metrics.pkl"
    with open(metrics_path, "wb") as f:
        pickle.dump(metrics_history, f)
    logger.info(f"✓ Metrics saved to {metrics_path}")
    
    logger.info(f"\n{'='*80}")
    logger.info(f"TRAINING SUMMARY")
    logger.info(f"{'='*80}")
    logger.info(f"Best validation loss: {best_val_loss:.6f}")
    logger.info(f"Final training loss: {metrics_history['train_loss'][-1]:.6f}")
    logger.info(f"Total training time: {sum(metrics_history['train_time']):.1f}s")
    logger.info(f"Model saved to: {final_model_path}")
    logger.info(f"{'='*80}\n")


if __name__ == "__main__":
    try:
        logger.info(f"\n{'='*80}")
        logger.info(f"SageMaker Training Job Started at {datetime.now()}")
        logger.info(f"{'='*80}\n")
        
        # Parse arguments
        args = parse_args()
        
        # Check data channels
        train_path = check_data_channels()

        # Optionally cache to local storage
        if args.get("cache_local", False):
            cache_dir = Path(args.get("cache_dir", "/opt/ml/input/data/cache"))
            train_path = cache_training_data(
                train_path=train_path,
                cache_dir=cache_dir,
                include_weather=args["include_weather"],
                use_minimal_datasets=args["use_minimal_datasets"],
            )

        # Setup paths
        paths = setup_sagemaker_paths(
            train_path,
            include_weather=args["include_weather"],
            use_minimal_datasets=args["use_minimal_datasets"],
        )
        
        # Train model
        train_model(args, paths)
        
        logger.info(f"\n{'='*80}")
        logger.info(f"SageMaker Training Job Completed Successfully")
        logger.info(f"{'='*80}\n")
        
    except Exception as e:
        logger.error(f"\nTraining failed with error: {str(e)}", exc_info=True)
        sys.exit(1)
