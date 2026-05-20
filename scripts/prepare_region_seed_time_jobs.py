#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import tarfile
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import boto3
import pandas as pd


NUMERIC_REGEX = r"([-+0-9eE\.]+)"
METRIC_NAMES = [
    "num_time_windows",
    "train_time_indices",
    "seed_candidates",
    "node_batch_size",
    "input_seq_len",
    "output_seq_len",
    "pairs_evaluated",
    "seeds_completed",
    "partial_score_mean",
    "partial_score_min",
    "partial_score_max",
    "partial_low_ratio",
    "partial_medium_ratio",
    "partial_high_ratio",
    "num_seeds",
    "num_time_indices",
    "score_min",
    "score_mean",
    "score_max",
    "bucket_low_count",
    "bucket_medium_count",
    "bucket_high_count",
    "cache_local",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare SageMaker shard jobs for central/non-central seed-time artifacts."
    )
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--bucket", default="sagemaker-congestion-bogota")
    parser.add_argument("--role-arn", default="arn:aws:iam::195419001736:role/SagemakerExecutionRole")
    parser.add_argument(
        "--image",
        default="195419001736.dkr.ecr.us-east-1.amazonaws.com/congestion-bogota-sagemaker:temporal-features-20260331-194332",
    )
    parser.add_argument("--efs-id", default="fs-099e29197c0a3e353")
    parser.add_argument("--efs-path", default="/data/datos_procesados")
    parser.add_argument(
        "--security-group-ids",
        default="sg-0962698c0c8352ddd,sg-03590ecb70a82d570",
    )
    parser.add_argument("--subnets", default="subnet-017c79447d5151efe")
    parser.add_argument(
        "--instance-type",
        default="ml.r7i.4xlarge",
        help="Single instance type used when --instance-types is not provided.",
    )
    parser.add_argument(
        "--instance-types",
        default="",
        help="Comma-separated instance types to round-robin across jobs.",
    )
    parser.add_argument("--volume-size-gb", type=int, default=100)
    parser.add_argument("--max-runtime-seconds", type=int, default=14400)
    parser.add_argument("--shards-per-region", type=int, default=20)
    parser.add_argument("--max-active-jobs", type=int, default=10)
    parser.add_argument("--max-active-jobs-per-instance-type", type=int, default=10)
    parser.add_argument(
        "--seed-strategy-dir",
        type=Path,
        default=Path("data/artifacts/region_seed_strategy"),
    )
    parser.add_argument(
        "--time-indices-csv",
        type=Path,
        default=Path("data/artifacts/time_index_strategy/time_idx_strategy_2000.csv"),
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("data/artifacts/region_seed_time_jobs"),
    )
    parser.add_argument(
        "--s3-prefix",
        default="congestion-bogota/region-seed-time-artifacts",
    )
    parser.add_argument(
        "--source-code-s3-uri",
        default="",
        help="Existing S3 URI for sourcedir tarball. Required for launch unless --upload-source is used.",
    )
    parser.add_argument("--upload-inputs", action="store_true")
    parser.add_argument("--upload-source", action="store_true")
    parser.add_argument("--launch", action="store_true")
    return parser.parse_args()


def split_strategy(strategy_csv: Path, output_dir: Path, num_shards: int) -> list[Path]:
    df = pd.read_csv(strategy_csv)
    if "seed_id" not in df.columns:
        raise ValueError(f"{strategy_csv} must contain seed_id")
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for shard_id in range(num_shards):
        shard_df = df.iloc[shard_id::num_shards].copy()
        shard_path = output_dir / f"{strategy_csv.stem}_shard_{shard_id:02d}.csv"
        shard_df.to_csv(shard_path, index=False)
        paths.append(shard_path)
    return paths


def upload_file(s3_client, local_path: Path, bucket: str, key: str) -> str:
    s3_client.upload_file(str(local_path), bucket, key)
    return f"s3://{bucket}/{key}"


def tar_and_upload_source(source_root: Path, s3_client, bucket: str, key_prefix: str) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        tar_path = Path(tmp) / "sourcedir.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tar:
            for rel in [
                "src",
                "scripts",
                "requirements-sagemaker.txt",
                "pyproject.toml",
            ]:
                path = source_root / rel
                if path.exists():
                    tar.add(path, arcname=rel)
        key = f"{key_prefix.rstrip('/')}/code/sourcedir.tar.gz"
        return upload_file(s3_client, tar_path, bucket, key)


def build_request(
    *,
    args: argparse.Namespace,
    job_name: str,
    seed_s3_uri: str,
    time_idx_s3_uri: str,
    source_code_s3_uri: str,
    region_name: str,
    shard_id: int,
    instance_type: str,
) -> dict:
    output_prefix = f"{args.s3_prefix.rstrip('/')}/outputs/{region_name}/shard_{shard_id:02d}"
    hyper_parameters = {
        "sagemaker_program": "src/rebalance_sampling.py",
        "sagemaker_submit_directory": source_code_s3_uri,
        "input-seq-len": "150",
        "output-seq-len": "50",
        "node-batch-size": "500",
        "train-ratio": "0.8",
        "include-weather": "True",
        "include-velocity-target": "True",
        "use-minimal-datasets": "True",
        "cache-local": "True",
        "cache-dir": f"/opt/ml/input/data/cache/{region_name}/shard_{shard_id:02d}",
        "high-level-threshold": "3.0",
        "bucket-quantiles": "0.5,0.75",
        "seed-candidates-path": seed_s3_uri,
        "time-indices-path": time_idx_s3_uri,
        "shard-id": str(shard_id),
        "num-shards": "1",
        "progress-log-every": "5000",
        "output-parquet": f"/opt/ml/model/seed_time_bucket_rows_{region_name}_shard_{shard_id:02d}.parquet",
        "output-metadata": f"/opt/ml/model/seed_time_bucket_metadata_{region_name}_shard_{shard_id:02d}.json",
        "output-parquet-s3-uri": f"s3://{args.bucket}/{output_prefix}/seed_time_bucket_rows_{region_name}_shard_{shard_id:02d}.parquet",
        "output-metadata-s3-uri": f"s3://{args.bucket}/{output_prefix}/seed_time_bucket_metadata_{region_name}_shard_{shard_id:02d}.json",
    }
    metric_definitions = [
        {"Name": name, "Regex": rf"{name}={NUMERIC_REGEX}"} for name in METRIC_NAMES
    ]
    return {
        "TrainingJobName": job_name,
        "RoleArn": args.role_arn,
        "AlgorithmSpecification": {
            "TrainingImage": args.image,
            "TrainingInputMode": "File",
            "MetricDefinitions": metric_definitions,
            "EnableSageMakerMetricsTimeSeries": True,
        },
        "HyperParameters": hyper_parameters,
        "InputDataConfig": [
            {
                "ChannelName": "train",
                "DataSource": {
                    "FileSystemDataSource": {
                        "FileSystemId": args.efs_id,
                        "FileSystemAccessMode": "ro",
                        "FileSystemType": "EFS",
                        "DirectoryPath": args.efs_path,
                    }
                },
            }
        ],
        "OutputDataConfig": {
            "S3OutputPath": f"s3://{args.bucket}/{args.s3_prefix.rstrip('/')}/sagemaker-output/{region_name}/shard_{shard_id:02d}",
        },
        "ResourceConfig": {
            "InstanceType": instance_type,
            "InstanceCount": 1,
            "VolumeSizeInGB": args.volume_size_gb,
        },
        "StoppingCondition": {"MaxRuntimeInSeconds": args.max_runtime_seconds},
        "VpcConfig": {
            "SecurityGroupIds": [x.strip() for x in args.security_group_ids.split(",") if x.strip()],
            "Subnets": [x.strip() for x in args.subnets.split(",") if x.strip()],
        },
    }


def count_active_seedtime_jobs(sm_client, name_contains: str) -> int:
    response = sm_client.list_training_jobs(
        NameContains=name_contains,
        StatusEquals="InProgress",
        MaxResults=100,
    )
    count = len(response.get("TrainingJobSummaries", []))
    response = sm_client.list_training_jobs(
        NameContains=name_contains,
        StatusEquals="Stopping",
        MaxResults=100,
    )
    count += len(response.get("TrainingJobSummaries", []))
    return count


def count_active_seedtime_jobs_by_instance_type(sm_client, name_contains: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for status in ("InProgress", "Stopping"):
        paginator = sm_client.get_paginator("list_training_jobs")
        for page in paginator.paginate(
            NameContains=name_contains,
            StatusEquals=status,
        ):
            for summary in page.get("TrainingJobSummaries", []):
                job_name = summary["TrainingJobName"]
                detail = sm_client.describe_training_job(TrainingJobName=job_name)
                instance_type = detail["ResourceConfig"]["InstanceType"]
                counts[instance_type] = counts.get(instance_type, 0) + 1
    return counts


def main() -> None:
    args = parse_args()
    instance_types = [
        value.strip()
        for value in (args.instance_types or args.instance_type).split(",")
        if value.strip()
    ]
    if not instance_types:
        raise ValueError("At least one instance type is required")
    args.work_dir.mkdir(parents=True, exist_ok=True)
    shard_root = args.work_dir / "shards"
    request_root = args.work_dir / "requests"
    request_root.mkdir(parents=True, exist_ok=True)

    s3 = boto3.client("s3", region_name=args.region) if (args.upload_inputs or args.upload_source or args.launch) else None
    source_code_s3_uri = args.source_code_s3_uri
    if args.upload_source:
        source_code_s3_uri = tar_and_upload_source(
            Path.cwd(),
            s3,
            args.bucket,
            args.s3_prefix,
        )
    if args.launch and not source_code_s3_uri:
        raise ValueError("--launch requires --source-code-s3-uri or --upload-source")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    time_idx_key = f"{args.s3_prefix.rstrip('/')}/inputs/time_idx/{args.time_indices_csv.name}"
    time_idx_s3_uri = f"s3://{args.bucket}/{time_idx_key}"
    if args.upload_inputs:
        upload_file(s3, args.time_indices_csv, args.bucket, time_idx_key)

    requests = []
    shard_manifest = {"regions": {}, "total_jobs": 0}
    job_index = 0
    for region_name in ["central", "noncentral"]:
        strategy_csv = args.seed_strategy_dir / f"seed_strategy_{region_name}.csv"
        shard_paths = split_strategy(
            strategy_csv,
            shard_root / region_name,
            args.shards_per_region,
        )
        shard_manifest["regions"][region_name] = []
        for shard_id, shard_path in enumerate(shard_paths):
            seed_key = f"{args.s3_prefix.rstrip('/')}/inputs/seeds/{region_name}/{shard_path.name}"
            seed_s3_uri = f"s3://{args.bucket}/{seed_key}"
            if args.upload_inputs:
                upload_file(s3, shard_path, args.bucket, seed_key)
            job_name = f"congestion-bogota-seedtime-{region_name[:3]}-{shard_id:02d}-{timestamp}"
            request = build_request(
                args=args,
                job_name=job_name,
                seed_s3_uri=seed_s3_uri,
                time_idx_s3_uri=time_idx_s3_uri,
                source_code_s3_uri=source_code_s3_uri,
                region_name=region_name,
                shard_id=shard_id,
                instance_type=instance_types[job_index % len(instance_types)],
            )
            job_index += 1
            request_path = request_root / f"{job_name}.json"
            request_path.write_text(json.dumps(request, indent=2))
            requests.append(request)
            shard_manifest["regions"][region_name].append(
                {
                    "shard_id": shard_id,
                    "local_csv": str(shard_path),
                    "seed_s3_uri": seed_s3_uri,
                    "request_json": str(request_path),
                    "job_name": job_name,
                }
            )
    shard_manifest["total_jobs"] = len(requests)
    shard_manifest["source_code_s3_uri"] = source_code_s3_uri
    shard_manifest["time_idx_s3_uri"] = time_idx_s3_uri
    shard_manifest["upload_inputs"] = bool(args.upload_inputs)
    shard_manifest["upload_source"] = bool(args.upload_source)
    shard_manifest["launch"] = bool(args.launch)
    shard_manifest["max_active_jobs"] = int(args.max_active_jobs)
    shard_manifest["max_active_jobs_per_instance_type"] = int(args.max_active_jobs_per_instance_type)
    shard_manifest["instance_types"] = instance_types

    manifest_path = args.work_dir / "region_seed_time_job_manifest.json"
    manifest_path.write_text(json.dumps(shard_manifest, indent=2))

    if args.launch:
        sm = boto3.client("sagemaker", region_name=args.region)
        launched = []
        failed = []
        for request in requests:
            request_instance_type = request["ResourceConfig"]["InstanceType"]
            while True:
                active_total = count_active_seedtime_jobs(sm, "congestion-bogota-seedtime")
                active_by_type = count_active_seedtime_jobs_by_instance_type(
                    sm,
                    "congestion-bogota-seedtime",
                )
                active_for_type = active_by_type.get(request_instance_type, 0)
                if (
                    active_total < args.max_active_jobs
                    and active_for_type < args.max_active_jobs_per_instance_type
                ):
                    break
                time.sleep(60)
            try:
                response = sm.create_training_job(**request)
                launched.append(
                    {
                        "job_name": request["TrainingJobName"],
                        "instance_type": request_instance_type,
                        "training_job_arn": response["TrainingJobArn"],
                    }
                )
            except Exception as exc:
                failed.append(
                    {
                        "job_name": request["TrainingJobName"],
                        "instance_type": request_instance_type,
                        "error": repr(exc),
                    }
                )
        shard_manifest["launched_jobs"] = launched
        shard_manifest["failed_jobs"] = failed
        manifest_path.write_text(json.dumps(shard_manifest, indent=2))

    print(json.dumps(shard_manifest, indent=2))


if __name__ == "__main__":
    main()
