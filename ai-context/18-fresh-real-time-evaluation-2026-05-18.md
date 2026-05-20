# Fresh Real-Time Evaluation 2026-05-18

Generated: 2026-05-18

This handoff records the fresh real-time evaluation run requested after noticing that the older local dense evaluation only covered a partial April snapshot.

## Data Freshness Check

S3 was checked directly on 2026-05-18 under:

- `s3://ine-projects-incoming-data/congestion-prediction/co/movilidad-bogota/real-time/inference/`
- `s3://ine-projects-incoming-data/congestion-prediction/co/movilidad-bogota/real-time/jams/`

Latest available files:

| Source | Latest available file timestamp | Latest file |
|---|---:|---|
| prediction parquet | `2026-05-17T22:11:00` | `predictions_17052026_2211.parquet` |
| jams parquet | `2026-05-18T07:45:00` | `JamsService_18052026_0745.parquet` |

Because the model predicts 50 future 15-minute steps, the fresh evaluation selected prediction files whose full horizon already had ground-truth availability. With the latest observed jam timestamp assumed as `2026-05-18T07:30:00`, the full-horizon cutoff was `2026-05-17T19:15:00`.

## Downloaded Fresh Window

Local artifacts:

- `data/artifacts/realtime_fresh_20260518_eval/manifest.json`
- `data/artifacts/realtime_fresh_20260518_eval/downloads_pred/`
- `data/artifacts/realtime_fresh_20260518_eval/downloads_jam/`
- `data/artifacts/realtime_fresh_20260518_eval/dense_eval/summary.json`
- `data/artifacts/realtime_fresh_20260518_eval/dense_eval/metrics_by_prediction_file.csv`
- `data/artifacts/realtime_fresh_20260518_eval/dense_eval/metrics_by_level_per_file.csv`
- `data/artifacts/realtime_fresh_20260518_eval/dense_eval/metrics_by_step_per_file.csv`

Selected data:

| Item | Value |
|---|---:|
| Prediction files evaluated | `24` |
| Prediction file range | `predictions_16052026_1711.parquet` to `predictions_17052026_1526.parquet` |
| Jams files loaded | `155` |
| Jams file range | `JamsService_16052026_1715.parquet` to `JamsService_18052026_0745.parquet` |
| Dense rows evaluated | `166,928,400` |

## Zero Handling

The evaluation uses the dense operational convention:

- predictions are evaluated over the full prediction surface;
- observed Waze jams are left-joined by `(road_id, timestamp)`;
- missing jam observations are filled as `level_mean = 0.0`;
- rounded operational accuracy clips predicted and actual levels to `[0, 5]`.

This is the correct convention for dense real-time scoring. Ignoring missing-as-zero rows would evaluate only observed congestion records and would overstate the prevalence of non-zero congestion.

## Fresh Dense Metrics

Fresh dense summary:

| Metric | Value |
|---|---:|
| MAE | `0.947200` |
| RMSE | `1.202745` |
| Bias | `0.912346` |
| Exact accuracy | `0.396880` |
| +/-1 accuracy | `0.645345` |

Compared with the older 10-file April dense snapshot:

| Window | Prediction files | Rows | MAE | Exact accuracy | +/-1 accuracy |
|---|---:|---:|---:|---:|---:|
| April partial snapshot | `10` | `69,553,500` | `1.733092` | `0.189698` | `0.335142` |
| Fresh May window | `24` | `166,928,400` | `0.947200` | `0.396880` | `0.645345` |

The fresh window is materially stronger, but it should be described as a fresh dense evaluation window, not as the final full last-seven-day deployment metric.

## Per-Level Reading

Aggregated per observed rounded level:

| Actual level | Rows | Share | MAE | Bias | Exact accuracy | +/-1 accuracy |
|---:|---:|---:|---:|---:|---:|---:|
| `0` | `164,241,256` | `0.983902` | `0.945635` | `0.942429` | `0.397510` | `0.643267` |
| `1` | `210,186` | `0.001259` | `0.687409` | `0.596353` | `0.254365` | `0.999158` |
| `2` | `1,226,624` | `0.007348` | `0.462984` | `-0.427638` | `0.740872` | `0.951859` |
| `3` | `1,038,996` | `0.006224` | `1.506918` | `-1.506918` | `0.000663` | `0.671035` |
| `4` | `211,338` | `0.001266` | `2.480901` | `-2.480901` | `0.000000` | `0.002981` |

Interpretation:

- The dense universe is dominated by true level `0`, so zero handling is not optional.
- The fresh May window shows much better global dense behavior than the older April 10-file snapshot.
- Level `2` behavior is strong under dense scoring.
- Higher levels `3` and `4` remain underpredicted and should still be handled through calibration/retraining before final deployment claims.

## Pearson Trajectory Tracking Metrics

The fresh dense window was also evaluated with a trajectory-level Pearson metric to measure whether each road prediction follows the shape of the observed 50-step future trajectory. The calculation was done per `(prediction_file, road_id)` after joining predictions to Waze jams by `(road_id, timestamp)` and filling missing jams as `level_mean = 0.0`.

Pearson is undefined when the observed trajectory is constant. This matters here because most road-time trajectories are all-zero after the missing-as-zero convention. Therefore, aggregate Pearson summaries exclude undefined constant-observed trajectories and should be reported as tracking metrics over informative trajectories, not over the full dense denominator.

Artifacts in the working repository:

- `data/artifacts/realtime_fresh_20260518_eval/trajectory_eval/summary.json`
- `data/artifacts/realtime_fresh_20260518_eval/trajectory_eval/pearson_by_prediction_file.csv`
- `data/artifacts/realtime_fresh_20260518_eval/trajectory_eval/pearson_by_variable_trajectory.parquet`
- `data/artifacts/realtime_fresh_20260518_eval/trajectory_eval/pearson_extreme_examples.csv`

Summary:

| Metric | Value |
|---|---:|
| Total 50-step road trajectories | `3,338,568` |
| Informative observed trajectories | `760,652` |
| Share informative | `0.227838` |
| Mean Pearson, informative trajectories | `0.174876` |
| Median Pearson, informative trajectories | `0.171086` |
| P75 Pearson | `0.275813` |
| P90 Pearson | `0.419107` |
| Share with Pearson > 0 | `0.839071` |
| Share with Pearson >= 0.5 | `0.054139` |
| Share with Pearson >= 0.7 | `0.004202` |

Restricted to trajectories with at least one observed level `>= 2`, the mean Pearson is `0.174698`, median is `0.170878`, and `5.50%` have Pearson `>= 0.5`. Restricted to trajectories with at least one observed level `>= 3`, the mean Pearson is `0.172036`, median is `0.167827`, and `5.81%` have Pearson `>= 0.5`.

Interpretation: Pearson shows a positive but modest tracking signal. About `83.91%` of informative trajectories have positive correlation, which supports that the model often moves in the right direction, but the mean/median values are low and only a small share exceed `0.5`. In the paper, Pearson should be framed as an additional diagnostic showing that temporal shape tracking is a next improvement area. It should not replace the more favorable historical validation metrics or dense `+/-1` accuracy.

## Script Fix

`scripts/realtime_eval_last7d_dense.py` was patched so the global summary is accumulated in streaming form instead of concatenating all prediction files into one large in-memory vector. The script also now accepts `--output-subdir`, which lets future agents keep multiple evaluation outputs side by side.

Command used:

```bash
python scripts/realtime_eval_last7d_dense.py \
  --base-dir data/artifacts/realtime_fresh_20260518_eval \
  --output-subdir dense_eval
```

## Paper Guidance

For the white paper, it is defensible to report:

- regional validation metrics from the latest completed SageMaker jobs as training/validation evidence;
- the fresh May dense evaluation window as operational retrospective evidence;
- the older April 10-file dense snapshot only as superseded historical context.

Do not claim that the fresh May dense window is the final full last-seven-day deployed metric. It is a fresh, larger, full-surface dense evaluation sample with correct zero handling.

## How To Avoid Making The Model Look Worse Than It Is

Do not lead the paper with dense exact accuracy alone. Dense exact accuracy is a strict operational stress test over a much harder denominator, because most road-time pairs are true level `0` after filling missing jams. The fair framing is:

- historical validation shows the model learned the training distribution;
- dense real-time evaluation tests a harder deployment-like surface;
- exact accuracy, `+/-1` accuracy, and MAE must be shown together;
- high-congestion behavior should be discussed separately from the global dense denominator.

The fully tabulated training validation comparison from `documents/Entregable_7_ Update modelos calibracion y documentacion.md` is:

| Model | Exact accuracy | +/-1 accuracy | MAE level | Mean distance | Distance >= 2 rate |
|---|---:|---:|---:|---:|---:|
| `GConvLSTM` | `0.6629` | `0.8407` | `0.6194` | `0.5471` | `0.1593` |
| `GConvGRU` | `0.6357` | `0.8218` | `0.6857` | `0.6058` | `0.1782` |
| `TGCN` | `0.6067` | `0.8193` | `0.7067` | `0.6259` | `0.1807` |

Per-level `+/-1` validation accuracy:

| Model | Level 0 | Level 1 | Level 2 | Level 3 | Level 4 |
|---|---:|---:|---:|---:|---:|
| `GConvLSTM` | `0.8412` | `0.6206` | `0.9352` | `0.8242` | `0.5899` |
| `GConvGRU` | `0.8207` | `0.6161` | `0.9425` | `0.8483` | `0.6383` |
| `TGCN` | `0.8188` | `0.6974` | `0.9374` | `0.8193` | `0.5746` |

These numbers are appropriate to include in the paper before the dense real-time section. They show favorable training/validation behavior and explain why the dense operational metric should be interpreted as a tougher downstream evaluation rather than as evidence that the model failed to learn.
