#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import boto3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retry failed regional seed-time SageMaker jobs.")
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("data/artifacts/region_seed_time_jobs_missing_only"),
    )
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument(
        "--instance-types",
        default="ml.c7i.4xlarge,ml.m5.4xlarge,ml.c6i.4xlarge,ml.c5.4xlarge",
    )
    parser.add_argument("--max-active-jobs", type=int, default=30)
    parser.add_argument("--wait-for-slots", action="store_true")
    parser.add_argument("--sleep-seconds", type=int, default=300)
    return parser.parse_args()


def active_training_jobs(sm) -> int:
    total = 0
    for status in ("InProgress", "Stopping"):
        paginator = sm.get_paginator("list_training_jobs")
        for page in paginator.paginate(
            NameContains="seedtime",
            StatusEquals=status,
        ):
            total += len(page.get("TrainingJobSummaries", []))
    return total


def latest_failed_jobs(work_dir: Path) -> list[dict]:
    manifest = json.loads((work_dir / "region_seed_time_job_manifest.json").read_text())
    failed = manifest.get("failed_jobs", [])
    for retry_manifest_path in manifest.get("retry_manifests", []):
        retry_manifest = json.loads(Path(retry_manifest_path).read_text())
        failed = retry_manifest.get("failed_jobs", [])
    return failed


def main() -> None:
    args = parse_args()
    instance_types = [x.strip() for x in args.instance_types.split(",") if x.strip()]
    if not instance_types:
        raise ValueError("At least one instance type is required")

    sm = boto3.client("sagemaker", region_name=args.region)
    pending = latest_failed_jobs(args.work_dir)
    retry_dir = args.work_dir / "retry_requests"
    retry_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

    launched = []
    failed = []
    for index, item in enumerate(pending):
        if args.wait_for_slots:
            while active_training_jobs(sm) >= args.max_active_jobs:
                time.sleep(args.sleep_seconds)

        old_job_name = item.get("old_job_name", item.get("job_name", ""))
        old_request_path = Path(item["request_json"])
        request = json.loads(old_request_path.read_text())
        match = re.search(r"seedtime-([a-z]+)-(\d+)", old_job_name)
        if not match:
            failed.append({"old_job_name": old_job_name, "error": "could not parse region/shard"})
            continue
        region_short, shard = match.group(1), match.group(2)
        new_name = f"cb-seedtime-{region_short}-{shard}-r2-{timestamp[-6:]}"
        instance_type = instance_types[index % len(instance_types)]
        request["TrainingJobName"] = new_name
        request["ResourceConfig"]["InstanceType"] = instance_type
        request_path = retry_dir / f"{new_name}.json"
        request_path.write_text(json.dumps(request, indent=2))
        try:
            response = sm.create_training_job(**request)
            launched.append(
                {
                    "old_job_name": old_job_name,
                    "job_name": new_name,
                    "instance_type": instance_type,
                    "training_job_arn": response["TrainingJobArn"],
                    "request_json": str(request_path),
                }
            )
        except Exception as exc:
            failed.append(
                {
                    "old_job_name": old_job_name,
                    "job_name": new_name,
                    "instance_type": instance_type,
                    "error": repr(exc),
                    "request_json": str(request_path),
                }
            )

    retry_manifest = {
        "timestamp": timestamp,
        "instance_types": instance_types,
        "launched_jobs": launched,
        "failed_jobs": failed,
    }
    retry_manifest_path = args.work_dir / f"retry_failed_jobs_{timestamp}.json"
    retry_manifest_path.write_text(json.dumps(retry_manifest, indent=2))

    main_manifest_path = args.work_dir / "region_seed_time_job_manifest.json"
    main_manifest = json.loads(main_manifest_path.read_text())
    main_manifest.setdefault("retry_manifests", []).append(str(retry_manifest_path))
    main_manifest_path.write_text(json.dumps(main_manifest, indent=2))
    print(json.dumps(retry_manifest, indent=2))


if __name__ == "__main__":
    main()
