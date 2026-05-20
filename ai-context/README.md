# AI Context

This folder is the canonical AI handoff layer for `congestion-bogota`.

It summarizes:
- repository structure
- current training/data architecture
- the new seed-time sampling system
- the newer shape-loss training line
- important artifacts and S3 locations
- decisions already made
- active experiments and remaining work
- current real-time evaluation and inference-traceability findings
- the regional central/noncentral corridor retraining work from 2026-05-14
- the regional job/model status verified on 2026-05-18
- the fresh dense real-time evaluation window downloaded and scored on 2026-05-18
- the minimal central/noncentral inference seed lists recalculated on 2026-05-18
- the central-first regional inference execution update from 2026-05-20

Human-facing repository documentation now lives in `docs/`. Use this folder for current operational state, decisions, and AI handoffs; use `README.md` and `docs/` for external readers.

Recommended reading order:
1. `00-start-here.md`
2. `20-regional-inference-execution-update-2026-05-20.md`
3. `19-regional-inference-seeds-2026-05-18.md`
4. `19-operational-loss-regional-training-2026-05-18.md`
5. `18-fresh-real-time-evaluation-2026-05-18.md`
6. `17-regional-status-2026-05-18.md`
7. `16-regional-corridor-retraining-2026-05-14.md`
8. `01-repository-map.md`
9. `02-architecture-and-dataflow.md`
10. `03-seed-time-sampling-status.md`
11. `04-artifacts-and-locations.md`
12. `05-decisions-and-assumptions.md`
13. `06-current-workspace-state.md`
14. `07-open-work-and-next-steps.md`
15. `08-module-catalog.md`
16. `09-preprocessing-pipeline.md`
17. `10-training-and-inference-runtime.md`
18. `11-tests-and-quality-signals.md`
19. `12-notebooks-and-operations.md`
20. `13-known-gaps-and-cleanup-needs.md`
21. `14-real-time-inference-repo.md`
