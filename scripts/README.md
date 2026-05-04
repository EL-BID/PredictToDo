# Scripts

Operational scripts:

- `compute_seed_cover.py`: computes seed coverage over the graph
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
