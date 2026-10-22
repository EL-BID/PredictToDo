#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import boto3



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch central/noncentral regional training jobs.")
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--s3-prefix", default="congestion-bogota/regional-training")
    parser.add_argument("--role-arn", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--efs-id", required=True)
    parser.add_argument("--efs-path", default="/data/datos_procesados")
    parser.add_argument("--subnets", required=True)
    parser.add_argument("--security-group-ids", required=True)
    parser.add_argument("--instance-count", type=int, default=1)
    parser.add_argument("--volume-size-gb", type=int, default=100)
    parser.add_argument("--max-runtime-seconds", type=int, default=36 * 60 * 60)
    parser.add_argument(
        "--model-configs",
        default="GConvLSTM:ml.g4dn.8xlarge",
        help="Comma-separated model:instance entries, e.g. GConvLSTM:ml.g4dn.8xlarge,GConvGRU:ml.g5.4xlarge",
    )
    parser.add_argument("--regions", default="central,noncentral")
    parser.add_argument("--epochs", default="2")
    parser.add_argument("--batch-size", default="16")
    parser.add_argument("--learning-rate", default="0.001")
    parser.add_argument(
        "--seed-time-bucket-target-ratios",
        default="0.30,0.30,0.40",
        help="Training sampler target ratios in high,medium,low order.",
    )
    parser.add_argument("--source-root", type=Path, default=Path("."))
    parser.add_argument(
        "--node-region-map",
        type=Path,
        default=Path("data/artifacts/corridor_regions/node_region_map.csv"),
    )
    parser.add_argument(
        "--central-seed-time",
        type=Path,
        default=Path("data/artifacts/region_seed_time_final_partial/seed_time_bucket_rows_central.parquet"),
    )
    parser.add_argument(
        "--noncentral-seed-time",
        type=Path,
        default=Path("data/artifacts/region_seed_time_final_partial/seed_time_bucket_rows_noncentral.parquet"),
    )
    parser.add_argument("--upload-local-artifacts", action="store_true")
    parser.add_argument("--launch", action="store_true")
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("data/artifacts/regional_training_jobs"),
    )
    return parser.parse_args()


def parse_model_configs(raw: str) -> list[dict[str, str]]:
    configs = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Invalid model config {item!r}; expected model:instance")
        model_type, instance_type = item.split(":", 1)
        configs.append({"model_type": model_type.strip(), "instance_type": instance_type.strip()})
    if not configs:
        raise ValueError("At least one model config is required")
    return configs


def upload_file(s3, local_path: Path, bucket: str, key: str) -> str:
    s3.upload_file(str(local_path), bucket, key)
    return f"s3://{bucket}/{key}"


def tar_and_upload_source(source_root: Path, s3, bucket: str, key_prefix: str) -> str:
    include_files = ["requirements-sagemaker.txt"]
    with tempfile.TemporaryDirectory() as tmp_dir:
        tar_path = Path(tmp_dir) / "sourcedir.tar.gz"
        with tarfile.open(tar_path, mode="w:gz", dereference=True) as tar:
            src_root = source_root / "src"
            for root, dirs, files in os.walk(src_root):
                dirs[:] = [d for d in dirs if d != "__pycache__"]
                for filename in files:
                    if not filename.endswith(".py"):
                        continue
                    abs_path = Path(root) / filename
                    tar.add(abs_path, arcname=str(abs_path.relative_to(source_root)))
            for rel_path in include_files:
                abs_path = source_root / rel_path
                if abs_path.exists():
                    tar.add(abs_path, arcname=rel_path)
        return upload_file(s3, tar_path, bucket, f"{key_prefix.rstrip('/')}/sourcedir.tar.gz")


def metric_definitions() -> list[dict[str, str]]:
    numeric = r"([-+0-9eE\.]+)"
    names = [
        "train_loss",
        "loss_level",
        "loss_velocity",
        "train_probe_mae_level",
        "train_probe_accuracy_exact",
        "train_probe_accuracy_pm1",
        "train_probe_class_distance_mean",
        "train_probe_class_distance_ge2_rate",
        "val_loss",
        "val_mae_level",
        "val_accuracy_exact",
        "val_accuracy_pm1",
        "val_class_distance_mean",
        "val_class_distance_ge2_rate",
    ]
    return [{"Name": name, "Regex": rf"{name}={numeric}"} for name in names]


def base_hyperparameters(args: argparse.Namespace, source_code_uri: str) -> dict[str, str]:
    return {
        "epochs": str(args.epochs),
        "batch-size": str(args.batch_size),
        "learning-rate": str(args.learning_rate),
        "input-seq-len": "150",
        "output-seq-len": "50",
        "node-batch-size": "500",
        "train-ratio": "0.8",
        "include-weather": "True",
        "include-velocity-target": "True",
        "seeded-batch-sampler": "True",
        "seed-time-bucket-target-ratios": str(args.seed_time_bucket_target_ratios),
        "seed-time-epoch-cycles": "1",
        "seed-time-shuffle-seeds": "True",
        "num-workers": "4",
        "drop-last": "False",
        "shuffle-time": "True",
        "seeded-batch-seed": "42",
        "pin-memory": "True",
        "prefetch-factor": "2",
        "persistent-workers": "True",
        "cache-local": "True",
        "cache-dir": "/opt/ml/input/data/cache",
        "use-minimal-datasets": "True",
        "sagemaker_submit_directory": source_code_uri,
        "sagemaker_program": "src/training_entrypoint.py",
        "loss-continuous-type": "huber",
        "loss-huber-delta": "1.0",
        "loss-class-ce-lambda": "0.3",
        "loss-class-distance-lambda": "0.5",
        "loss-large-error-threshold": "2",
        "loss-large-error-lambda": "0.75",
        "loss-high-level-threshold": "2",
        "loss-high-level-lambda": "0.5",
        "loss-piecewise-threshold": "4",
        "loss-piecewise-mae-lambda": "0.5",
        "loss-directional-lambda": "0.3",
        "loss-shape-lambda": "0.05",
        "loss-shape-min-std": "1e-3",
        "loss-shape-eps": "1e-6",
        "loss-shape-weight-std": "0.75",
        "loss-shape-weight-range": "0.25",
        "loss-shape-weight-max": "1.0",
        "timing-steps": "20",
    }


def build_request(
    args: argparse.Namespace,
    *,
    job_name: str,
    model_type: str,
    instance_type: str,
    training_region: str,
    source_code_uri: str,
    node_region_map_uri: str,
    seed_time_uri: str,
) -> dict:
    hyperparameters = base_hyperparameters(args, source_code_uri)
    hyperparameters.update(
        {
            "model-type": model_type,
            "training-region": training_region,
            "node-region-map-path": node_region_map_uri,
            "seed-time-artifact-path": seed_time_uri,
        }
    )
    s3_output_path = (
        f"s3://{args.bucket}/congestion-bogota/training-runs/"
        f"regional/{training_region}/{model_type.lower()}"
    )
    return {
        "TrainingJobName": job_name,
        "RoleArn": args.role_arn,
        "AlgorithmSpecification": {
            "TrainingImage": args.image,
            "TrainingInputMode": "File",
            "MetricDefinitions": metric_definitions(),
            "EnableSageMakerMetricsTimeSeries": True,
        },
        "HyperParameters": hyperparameters,
        "InputDataConfig": [
            {
                "ChannelName": "train",
                "DataSource": {
                    "FileSystemDataSource": {
                        "FileSystemId": args.efs_id,
                        "FileSystemType": "EFS",
                        "FileSystemAccessMode": "ro",
                        "DirectoryPath": args.efs_path,
                    }
                },
            },
            {
                "ChannelName": "validation",
                "DataSource": {
                    "FileSystemDataSource": {
                        "FileSystemId": args.efs_id,
                        "FileSystemType": "EFS",
                        "FileSystemAccessMode": "ro",
                        "DirectoryPath": args.efs_path,
                    }
                },
            },
        ],
        "OutputDataConfig": {"S3OutputPath": s3_output_path},
        "ResourceConfig": {
            "InstanceType": instance_type,
            "InstanceCount": args.instance_count,
            "VolumeSizeInGB": args.volume_size_gb,
        },
        "StoppingCondition": {"MaxRuntimeInSeconds": args.max_runtime_seconds},
        "VpcConfig": {
            "Subnets": [x.strip() for x in args.subnets.split(",") if x.strip()],
            "SecurityGroupIds": [
                x.strip() for x in args.security_group_ids.split(",") if x.strip()
            ],
        },
    }


def main() -> None:
    args = parse_args()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    args.work_dir.mkdir(parents=True, exist_ok=True)
    s3 = boto3.client("s3", region_name=args.region)
    sm = boto3.client("sagemaker", region_name=args.region)

    source_code_uri = tar_and_upload_source(
        args.source_root,
        s3,
        args.bucket,
        f"{args.s3_prefix.rstrip('/')}/code/{timestamp}",
    )

    artifact_prefix = f"{args.s3_prefix.rstrip('/')}/artifacts/{timestamp}"
    if args.upload_local_artifacts:
        node_region_map_uri = upload_file(
            s3,
            args.node_region_map,
            args.bucket,
            f"{artifact_prefix}/{args.node_region_map.name}",
        )
        seed_time_uris = {
            "central": upload_file(
                s3,
                args.central_seed_time,
                args.bucket,
                f"{artifact_prefix}/{args.central_seed_time.name}",
            ),
            "noncentral": upload_file(
                s3,
                args.noncentral_seed_time,
                args.bucket,
                f"{artifact_prefix}/{args.noncentral_seed_time.name}",
            ),
        }
    else:
        node_region_map_uri = str(args.node_region_map)
        seed_time_uris = {
            "central": str(args.central_seed_time),
            "noncentral": str(args.noncentral_seed_time),
        }

    requests = []
    launched = []
    regions = [x.strip() for x in args.regions.split(",") if x.strip()]
    for training_region in regions:
        if training_region not in seed_time_uris:
            raise ValueError(f"Unsupported region: {training_region}")
        for cfg in parse_model_configs(args.model_configs):
            model_type = cfg["model_type"]
            job_name = (
                f"congestion-bogota-{training_region[:3]}-"
                f"{model_type.lower()}-{timestamp.replace('-', '')}"
            )
            request = build_request(
                args,
                job_name=job_name,
                model_type=model_type,
                instance_type=cfg["instance_type"],
                training_region=training_region,
                source_code_uri=source_code_uri,
                node_region_map_uri=node_region_map_uri,
                seed_time_uri=seed_time_uris[training_region],
            )
            request_path = args.work_dir / f"{job_name}.json"
            request_path.write_text(json.dumps(request, indent=2))
            item = {
                "region": training_region,
                "model_type": model_type,
                "instance_type": cfg["instance_type"],
                "job_name": job_name,
                "request_json": str(request_path),
            }
            requests.append(item)
            if args.launch:
                response = sm.create_training_job(**request)
                item["training_job_arn"] = response["TrainingJobArn"]
                launched.append(item)

    manifest = {
        "timestamp": timestamp,
        "source_code_uri": source_code_uri,
        "node_region_map_uri": node_region_map_uri,
        "seed_time_uris": seed_time_uris,
        "launch": bool(args.launch),
        "requests": requests,
        "launched_jobs": launched,
    }
    manifest_path = args.work_dir / f"regional_training_manifest_{timestamp}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
