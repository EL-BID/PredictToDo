# Results Paper Update Proposal 2026-05-20

This note consolidates the latest training and real-time evaluation evidence for updating the white paper results section.

## Latest Operational-Loss Regional Jobs

The corrected `oploss3` SageMaker jobs all completed on 2026-05-19. They were launched from the original regional source package plus only the operational-loss additions. Despite completing successfully, they should not replace the previous regional baseline in the paper because validation performance worsened for most metrics.

| Region | Model | Job | Val exact | Val +/-1 | Val MAE | Val loss |
|---|---|---|---:|---:|---:|---:|
| central | GConvGRU | `congestion-bogota-oploss3-central-gconvgru-20260518-194900` | 0.5135 | 0.7494 | 0.9345 | 7.3505 |
| central | GConvLSTM | `congestion-bogota-oploss3-central-gconvlstm-20260518-194900` | 0.5043 | 0.7354 | 0.9607 | 7.3894 |
| central | TGCN | `congestion-bogota-oploss3-central-tgcn-20260518-194900` | 0.4770 | 0.7076 | 1.0395 | 7.4919 |
| noncentral | GConvGRU | `congestion-bogota-oploss3-noncentral-gconvgru-20260518-194900` | 0.5042 | 0.7992 | 0.8789 | 7.5267 |
| noncentral | GConvLSTM | `congestion-bogota-oploss3-noncentral-gconvlstm-20260518-194900` | 0.4978 | 0.7924 | 0.8726 | 7.4908 |
| noncentral | TGCN | `congestion-bogota-oploss3-noncentral-tgcn-20260518-194900` | 0.4018 | 0.7639 | 1.0027 | 7.4218 |

Compared with the previous useful regional line, the operational-loss jobs increased validation loss by roughly 1.63 to 2.09 points. Central TGCN improved exact accuracy and +/-1 accuracy relative to its failed previous line, but its MAE barely changed and it is not the best model in the region. Noncentral models degraded substantially, especially exact accuracy and MAE.

## Recommended Regional Validation Evidence

For the paper, use the previous regional GConvGRU models as the best regional candidates:

| Region | Recommended model | Job | Val exact | Val +/-1 | Val MAE |
|---|---|---|---:|---:|---:|
| central / principal corridors | GConvGRU | `congestion-bogota-cen-gconvgru-20260514230035` | 0.5895 | 0.7595 | 0.8035 |
| noncentral / non-principal roads | GConvGRU | `congestion-bogota-non-gconvgru-g512xl-20260514230153` | 0.6945 | 0.8485 | 0.5792 |

The operational-loss line should be described as a diagnostic experiment: it targeted zero false positives and high-level underprediction, but the initial hyperparameter setting over-regularized the model and harmed validation performance. It is evidence for future calibration work, not a final model line.

## Global Validation Evidence

The strongest global validation table remains appropriate for the paper:

| Model | Exact accuracy | +/-1 accuracy | MAE level | Mean distance | Distance >= 2 rate |
|---|---:|---:|---:|---:|---:|
| GConvLSTM | 0.6629 | 0.8407 | 0.6194 | 0.5471 | 0.1593 |
| GConvGRU | 0.6357 | 0.8218 | 0.6857 | 0.6058 | 0.1782 |
| TGCN | 0.6067 | 0.8193 | 0.7067 | 0.6259 | 0.1807 |

Per-level +/-1 validation accuracy:

| Model | Level 0 | Level 1 | Level 2 | Level 3 | Level 4 |
|---|---:|---:|---:|---:|---:|
| GConvLSTM | 0.8412 | 0.6206 | 0.9352 | 0.8242 | 0.5899 |
| GConvGRU | 0.8207 | 0.6161 | 0.9425 | 0.8483 | 0.6383 |
| TGCN | 0.8188 | 0.6974 | 0.9374 | 0.8193 | 0.5746 |

The most favorable but honest interpretation is that GConvLSTM is strongest globally, while GConvGRU is better for high-congestion levels 3 and 4.

## Dense Real-Time Evaluation With Zeros

Use the fresh May dense evaluation as the main operational evidence. It includes missing Waze jams as true level 0. Do not show the older April snapshot in the paper; it is superseded and distracts from the current result.

Summary:

| Scope | Prediction files | Jam files | Dense rows | MAE | RMSE | Bias | Exact | +/-1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Fresh May window | 24 | 155 | 166,928,400 | 0.9472 | 1.2027 | +0.9123 | 0.3969 | 0.6453 |

The fresh May window is the result to report. It shows useful aggregate performance under a dense operational denominator, but it still shows systematic calibration problems.

## Dense Behavior By Horizon

Selected per-step metrics for the fresh May dense window:

| Step | MAE | RMSE | Bias | Exact | +/-1 |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.4186 | 1.4463 | +1.3453 | 0.0136 | 0.5285 |
| 12 | 1.7356 | 1.7570 | +1.6991 | 0.0116 | 0.1388 |
| 24 | 1.0643 | 1.1975 | +1.0279 | 0.2137 | 0.7269 |
| 32 | 0.5189 | 0.6913 | +0.4802 | 0.5129 | 0.9944 |
| 40 | 0.1631 | 0.2654 | +0.1401 | 0.9568 | 0.9969 |
| 50 | 0.1503 | 0.2342 | +0.1338 | 0.9905 | 0.9975 |

The apparent improvement at late horizons should be interpreted carefully: by step 50, 99.74% of actual road-time pairs are level 0, and the prediction surface mostly collapses toward zero. The model is not necessarily more skillful at long horizons; the target distribution becomes easier under the dense zero-filled metric.

## Pearson Trajectory Tracking

Pearson is computed only for trajectories with variable actual and prediction series, because correlation is undefined for constant all-zero targets. Full 50-step trajectory Pearson:

| Scope | Count | Mean | Median | P75 | P90 | Positive share | >= 0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| variable actual trajectories | 760,652 | 0.1749 | 0.1711 | 0.2758 | 0.4191 | 0.8391 | 0.0541 |
| actual >= 2 at least once | 737,798 | 0.1747 | 0.1709 | 0.2766 | 0.4207 | 0.8371 | 0.0550 |
| actual >= 3 at least once | 546,767 | 0.1720 | 0.1678 | 0.2784 | 0.4256 | 0.8248 | 0.0581 |

Interpretation: Pearson is positive for most informative trajectories, so the model often moves in the correct temporal direction, but the magnitude is weak. This supports a claim of partial temporal tracking, not high-fidelity trajectory following.

## Proposed Results Section Structure

1. Start with historical validation: global three-model comparison and regional GConvGRU candidate results.
2. Explain that validation and dense real-time evaluation measure different surfaces.
3. Present fresh May dense real-time evaluation with zero handling.
4. Add a horizon table and explain that late-horizon gains are driven by the rising zero share.
5. Add Pearson trajectory tracking as an auxiliary temporal-shape metric.
6. Discuss the operational-loss ablation as a negative/diagnostic result, not as a final model improvement.
7. End with a calibration roadmap: threshold/post-processing, shorter operational horizon, per-region dense evaluation, and retraining with gentler loss weights.

## Generated Paper Artifacts

Generated on 2026-05-20:

- figures in `white_paper/figures/`:
  - `may_dense_step_metrics.{png,pdf}`
  - `may_dense_horizon_frontier_metrics.{png,pdf}`
  - `may_dense_zero_and_severity_distribution.{png,pdf}`
  - `may_dense_level_step_mae_heatmap.{png,pdf}`
  - `may_dense_level_step_pm1_heatmap.{png,pdf}`
  - `may_dense_pearson_by_horizon.{png,pdf}`
- tables in `white_paper/tables/`:
  - `may_dense_step_metrics_selected.{csv,tex}`
  - `may_dense_horizon_frontier_cumulative.{csv,tex}`
  - `may_dense_pearson_by_horizon_selected.{csv,tex}`
  - `may_dense_step_level_selected.csv`
  - `may_dense_step_metrics_all.csv`
  - `may_dense_horizon_frontier_all.csv`
  - `may_dense_step_distribution_all.csv`
  - `may_dense_step_level_all.csv`
  - `may_dense_horizon_level_all.csv`
  - `may_dense_pearson_by_horizon_all.csv`
  - `full_results_section_snippet.tex`

Use the selected tables in the main body and the `*_all.csv` tables as appendix/supporting material. This keeps the paper readable while preserving the full metric surface produced during evaluation.
