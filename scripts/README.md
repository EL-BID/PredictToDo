# Scripts

Operational scripts:

- `compute_seed_cover.py`: computes seed coverage over the graph
- `build_corridor_region_map.py`: builds central/noncentral road-region artifacts from prioritized corridors
- `build_region_seed_strategies.py`: splits seed strategies into central and noncentral training artifacts
- `compute_regional_seed_cover.py`: computes minimal central/noncentral inference seed lists
- `prepare_region_seed_time_jobs.py`: prepares regional seed-time shard jobs
- `retry_failed_seed_time_jobs.py`: retries failed regional seed-time shard jobs
- `reuse_existing_seed_time_by_region.py`: reuses existing seed-time rows for regional artifacts
- `finalize_region_seed_time_artifacts.py`: merges/rebuckets regional seed-time shards
- `launch_region_training_jobs.py`: launches central/noncentral SageMaker model jobs
- `export_calibration_artifacts.py`: exports calibration artifacts from train/eval inputs
- `qmap_output_transform.py`: transforms quantile-map calibration outputs
- `realtime_eval_sample.py`: builds a recent real-time evaluation sample
- `realtime_eval_metrics.py`: computes real-time evaluation metrics
- `realtime_eval_backfill.py`: backfills real-time evaluation windows
- `realtime_eval_last7d_dense.py`: dense last-seven-day evaluation runner

Use `--help` on each script for CLI options:

```bash
uv run python scripts/realtime_eval_metrics.py --help
```
