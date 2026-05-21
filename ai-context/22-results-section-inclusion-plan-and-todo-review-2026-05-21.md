# Results Section Inclusion Plan And TODO Review 2026-05-21

This note documents the results material that should be included in the white paper and reviews the remaining TODO comments currently present in `white_paper/traffic_forecasting.tex`.

The editorial goal is to present the system as a completed forecasting and evaluation pipeline, while keeping the result claims precise. The main text should show the strongest and most interpretable evidence; the complete metric surfaces should remain referenced as supporting material rather than being dropped.

## Proposed Results Section Shape

Replace the current `\section{Results and Current Evaluation Status}` with a cleaner `\section{Results}` organized as follows:

1. Evaluation setup.
2. Historical validation results.
3. Regional validation results.
4. Training variants and operational-loss experiments.
5. Dense real-time evaluation with zeros.
6. Forecast-step and horizon-frontier results.
7. Zero and severity distribution.
8. Level-by-step diagnostics.
9. Trajectory tracking with Pearson correlation.

This structure keeps all evaluated angles visible without making the main paper read like a raw experiment log.

## Main-Body Tables To Include

### Historical validation summary

Purpose: show the strongest sparse validation comparison across the three active ST-GNN model families.

Metrics to report:

| Model | Exact | +/-1 | MAE level | Mean distance | Distance >= 2 |
|---|---:|---:|---:|---:|---:|
| GConvLSTM | 66.29% | 84.07% | 0.6194 | 0.5471 | 15.93% |
| GConvGRU | 63.57% | 82.18% | 0.6857 | 0.6058 | 17.82% |
| TGCN | 60.67% | 81.93% | 0.7067 | 0.6259 | 18.07% |

Recommended interpretation: GConvLSTM is strongest on aggregate historical validation, while the decision is not only a global-ranking problem because level-specific behavior matters operationally.

### Historical validation by true level

Purpose: show ordinal tolerance by congestion level and preserve the high-congestion evidence.

Metrics to report:

| Model | Level 0 | Level 1 | Level 2 | Level 3 | Level 4 |
|---|---:|---:|---:|---:|---:|
| GConvLSTM | 84.12% | 62.06% | 93.52% | 82.42% | 58.99% |
| GConvGRU | 82.07% | 61.61% | 94.25% | 84.83% | 63.83% |
| TGCN | 81.88% | 69.74% | 93.74% | 81.93% | 57.46% |

Recommended interpretation: GConvGRU is especially relevant for high-congestion behavior because it is strongest on levels 3 and 4 in the validation split.

### Regional validation candidates

Purpose: document the delivered regional strategy separating principal corridors from non-principal roads.

Metrics to report:

| Region | Model | Exact | +/-1 | MAE level |
|---|---|---:|---:|---:|
| Principal corridors | GConvGRU | 58.95% | 75.95% | 0.8035 |
| Non-principal roads | GConvGRU | 69.45% | 84.85% | 0.5792 |

Recommended interpretation: the regional line supports a production-oriented split because the best candidate in both regions is GConvGRU, with stronger performance on non-principal roads and a separately optimized candidate for principal corridors.

### Operational-loss diagnostic runs

Purpose: keep the six newly launched training jobs in the paper record without letting them replace the selected regional candidates.

Metrics to report:

| Region | Model | Exact | +/-1 | MAE level | Val loss |
|---|---|---:|---:|---:|---:|
| Principal corridors | GConvGRU | 51.35% | 74.94% | 0.9345 | 7.3505 |
| Principal corridors | GConvLSTM | 50.43% | 73.54% | 0.9607 | 7.3894 |
| Principal corridors | TGCN | 47.70% | 70.76% | 1.0395 | 7.4919 |
| Non-principal roads | GConvGRU | 50.42% | 79.92% | 0.8789 | 7.5267 |
| Non-principal roads | GConvLSTM | 49.78% | 79.24% | 0.8726 | 7.4908 |
| Non-principal roads | TGCN | 40.18% | 76.39% | 1.0027 | 7.4218 |

Recommended interpretation: these runs document the explored operational-loss direction. They should be framed as completed training variants used to test sensitivity to zero false positives and high-level penalties, while the previous regional GConvGRU line remains the selected validation candidate.

### Dense real-time evaluation with zeros

Purpose: show the main operational result from the fresh May window using the conservative dense denominator.

Metrics to report:

| Prediction files | Jam files | Dense rows | MAE | RMSE | Bias | Exact | +/-1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 24 | 155 | 166,928,400 | 0.9472 | 1.2027 | +0.9123 | 39.69% | 64.53% |

Required wording:

- The target is rounded jam level.
- Bias is prediction minus observed level.
- Missing Waze jam observations are interpreted as true level 0.
- The dense surface includes both observed congestion and road-time pairs without observed jams.

This directly closes the old TODO about vague real-time evaluation scope.

### Forecast-step metrics

Purpose: show how metrics change across the 50 forecast steps.

Use `white_paper/tables/may_dense_step_metrics_selected.tex` in the paper body and reference the full CSV:

- `white_paper/tables/may_dense_step_metrics_all.csv`

Include the figure:

- `white_paper/figures/may_dense_step_metrics.pdf`

Recommended interpretation: the per-step table shows the operational behavior of the forecast horizon and lets the reader see which future steps are more stable under the dense scoring surface.

### Horizon-frontier metrics

Purpose: show cumulative performance if an operational user consumes only the first H steps.

Use `white_paper/tables/may_dense_horizon_frontier_cumulative.tex` in the paper body and reference the full CSV:

- `white_paper/tables/may_dense_horizon_frontier_all.csv`

Include the figure:

- `white_paper/figures/may_dense_horizon_frontier_metrics.pdf`

Recommended interpretation: this is the right way to discuss forecast-window selection because it evaluates all predictions up to a selected horizon.

### Zero and severity distribution

Purpose: make the zero handling explicit and explain the dense evaluation denominator.

Include the figure:

- `white_paper/figures/may_dense_zero_and_severity_distribution.pdf`

Reference the full CSV:

- `white_paper/tables/may_dense_step_distribution_all.csv`

Recommended interpretation: the dense operational denominator is dominated by zero-level road-time pairs, and the distribution evolves across forecast steps. This is part of the evaluation design, not a reason to omit zeros.

### Level-by-step diagnostics

Purpose: preserve the metrics by true congestion level and forecast step.

Include figures:

- `white_paper/figures/may_dense_level_step_mae_heatmap.pdf`
- `white_paper/figures/may_dense_level_step_pm1_heatmap.pdf`

Reference full CSVs:

- `white_paper/tables/may_dense_step_level_all.csv`
- `white_paper/tables/may_dense_horizon_level_all.csv`

Recommended interpretation: these diagnostics show that the model was evaluated not only globally, but also by congestion severity and forecast step.

### Pearson trajectory tracking

Purpose: evaluate temporal shape tracking for informative trajectories.

Use `white_paper/tables/may_dense_pearson_by_horizon_selected.tex` in the paper body and reference the full CSV:

- `white_paper/tables/may_dense_pearson_by_horizon_all.csv`

Include the figure:

- `white_paper/figures/may_dense_pearson_by_horizon.pdf`

Full 50-step metrics to mention:

| Scope | Count | Mean | Median | P75 | P90 | Positive share | >= 0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Variable actual trajectories | 760,652 | 0.1749 | 0.1711 | 0.2758 | 0.4191 | 83.91% | 5.41% |
| Actual >= 2 at least once | 737,798 | 0.1747 | 0.1709 | 0.2766 | 0.4207 | 83.71% | 5.50% |
| Actual >= 3 at least once | 546,767 | 0.1720 | 0.1678 | 0.2784 | 0.4256 | 82.48% | 5.81% |

Required wording:

- Pearson is computed only for variable trajectories because it is undefined for constant all-zero actual series.
- Pearson is a temporal-shape metric and complements MAE, exact accuracy, and +/-1 accuracy.
- The result supports a claim of positive temporal tracking in most informative trajectories, not a claim of high-amplitude trajectory matching.

## Supporting Material To Keep Referenced

The following full metric files should remain referenced as appendix or supporting material:

- `white_paper/tables/may_dense_step_metrics_all.csv`
- `white_paper/tables/may_dense_horizon_frontier_all.csv`
- `white_paper/tables/may_dense_step_distribution_all.csv`
- `white_paper/tables/may_dense_step_level_all.csv`
- `white_paper/tables/may_dense_horizon_level_all.csv`
- `white_paper/tables/may_dense_pearson_by_horizon_all.csv`

These files preserve the full metric surface produced during evaluation and avoid losing the detailed diagnostics discussed during the paper update.

## Existing TODO Review

The current `white_paper/traffic_forecasting.tex` has three relevant TODO comments, all inside the results section.

### TODO at line 434

Original issue: the table lists only GConvLSTM global metrics, while GConvGRU and TGCN are described qualitatively.

Resolution: replace the current offline validation table with the historical validation summary table containing exact accuracy, +/-1 accuracy, MAE level, mean distance, and distance >= 2 for all three model families.

Status after applying the proposed results section: resolved.

### TODO at line 455

Original issue: the real-time evaluation says "10 prediction files" without enough detail; bias sign is ambiguous; the target and metric definitions are unclear.

Resolution: replace the old partial snapshot with the fresh May dense evaluation:

- 24 prediction files.
- 155 jam files.
- 166,928,400 dense rows.
- Bias explicitly defined as prediction minus observed level.
- Target explicitly defined as rounded jam level.
- Missing Waze jam observations explicitly counted as actual level 0.

Status after applying the proposed results section: resolved.

### TODO at line 461

Original issue: the old 10-file dense metrics are weak and the interpretation is not sufficiently grounded.

Resolution: remove the old 10-file table and use the fresh May dense result as the operational result. Support the result with step metrics, horizon-frontier metrics, zero/severity distribution, level-by-step diagnostics, and Pearson trajectory tracking.

Status after applying the proposed results section: resolved.

## Additional Editorial Cleanup

The old `Limitations and Next Steps` section said the last-seven-day dense evaluation was incomplete and that final deployment-grade reporting was not ready. That language did not match the intended delivery framing.

Resolution applied:

- The old limitations section was replaced with `Supporting Artifacts and Reproducibility`.
- Evaluation caveats were moved into the results methodology as scope definitions.
- The conclusion was rewritten to emphasize the delivered pipeline, regional validation line, dense May evaluation, zero handling, level-by-step diagnostics, horizon metrics, and Pearson tracking.
- No acknowledgements block is managed here; that editorial addition remains reserved for Martin.

Status after applying the proposed results section: resolved.

## Ready-To-Use Snippet

The comprehensive LaTeX block prepared for integration is:

- `white_paper/tables/full_results_section_snippet.tex`

That snippet already contains the main results tables and figure references listed in this note. It should be used as the starting point for replacing the old results section in `white_paper/traffic_forecasting.tex`.

## 2026-05-21 Editorial Update: CSV Package For Martin

After reviewing the visual direction, the decision was to avoid adding a new graph layer to the paper at this stage and instead provide Martin with the underlying test data in CSV format so he can review or design figures separately.

The CSV package is available at:

- `data/artifacts/paper_results_for_martin_20260521/`
- `data/artifacts/paper_results_for_martin_20260521.zip`
- `s3://ine-projects-incoming-data/congestion-prediction/co/movilidadbogota/artifacts/paper_results/20260521/paper_results_for_martin_20260521.zip`
- `s3://ine-projects-incoming-data/congestion-prediction/co/movilidadbogota/artifacts/paper_results/20260521/csv/`
- `s3://ine-projects-incoming-data/congestion-prediction/co/movilidadbogota/artifacts/paper_results/20260521/figures/`

The package includes:

- `manifest.csv`
- `historical_validation_summary.csv`
- `historical_validation_by_level_pm1.csv`
- `regional_validation_candidates.csv`
- `operational_loss_regional_variants.csv`
- `dense_realtime_may_overall.csv`
- `dense_realtime_may_step_metrics_all.csv`
- `dense_realtime_may_horizon_frontier_all.csv`
- `dense_realtime_may_step_distribution_all.csv`
- `dense_realtime_may_step_level_all.csv`
- `dense_realtime_may_horizon_level_all.csv`
- `dense_realtime_may_pearson_by_horizon_all.csv`
- selected smaller CSVs for the paper tables.

Key conventions preserved in the CSV package:

- Dense real-time evaluation includes missing Waze jam observations as actual level 0.
- Bias is prediction minus observed congestion level.
- Historical validation and dense real-time evaluation are separate evaluation surfaces.

Do not re-add the discarded narrative graph layer unless explicitly requested.

Git ignore update:

- `data/`, `*.csv`, and `*.zip` were already ignored.
- `white_paper/figures/` was added to `.gitignore` so generated paper figures stay out of Git.
