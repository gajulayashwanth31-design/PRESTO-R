# Changelog

## Unreleased

- Added Phase 0 project scaffolding, configuration loading, deterministic seeding, and setup checks.
- Implemented the Phase 1 simulator, event schedules, telemetry, observations, reward, Gymnasium environment, configurations, and tests.
- Added Phase 2 episode-isolated predictor data, GRU and logistic-regression training, evaluation, learned risk providers, and a risk-trace command.
- Completed Phase 2 training and held-out evaluation; added validation-tuned subgroup thresholds and corrected onset lead-time row alignment.
- Added Phase 3 routing heuristics, deterministic rollout/event logging, per-episode metrics, reactive DQN configurations and training, and learning-curve plotting.
- Completed the reactive DQN five-seed × 300k-step training sweep with validation curves and per-seed duration/reward summaries.
- Started Phase 4 with risk-aware DQN/PPO configs, oracle-risk training support, validation-only lambda screening/selection, risk-permutation and event-trace diagnostics, and training smoke tests.
- Completed the Phase 4 validation-only lambda screen (two training seeds for each of five penalties) and selected λ=0.5; full predictive, PRESTO-R, and oracle training runs are underway.
- Generated the seed-0 validation event trace and risk-permutation diagnostic; temporal risk permutation lowered proactive rate from 0.250 to 0.167 across 50 mixed validation episodes.
- Completed all full Phase 4 validation training (5 × 300k steps each for predictive DQN and PRESTO-R; 2 × 300k for the oracle), regenerated final diagnostics, and produced the all-method validation summary. PRESTO-R averaged 4.30 switches/episode at PDR 0.9677 versus predictive DQN's 21.36 switches at PDR 0.9654.
- Started Phase 5 with a resumable held-out grid runner, raw-parquet metric aggregation, training-seed confidence-interval summaries, paired Wilcoxon tests with Holm correction, and PRD success-check tooling. RGG-specific P0 agent training is underway; no Phase 5 test grid has been run yet.
- Finished all three-seed RGG training for reactive, predictive, and PRESTO-R. NSFNET P0 test grid, oracle diagnostic cells, and all five λ-screen variants on mixed test episodes are complete; RGG test grid is running.
- Completed the 150-cell Phase 5 test grid (30,000 episodes), raw-to-episode metric aggregation, both topology summary tables, paired tests with Holm correction, λ sensitivity table, and a passing reproducibility hash check. The success report records mixed PRD results: the switch/PDR and steady-delay checks pass; reactive-baseline recovery significance and onset-F1 target do not.
- Started the H=3/H=10 predictor dataset sensitivity sweep.
- Built both horizon-specific datasets, trained/evaluated the H=3 GRU (held-out predictor onset F1=0.5992), and started H=10 GRU plus the six 3-seed H-specific PRESTO agent runs.
- Added evaluation-only robustness perturbations (sudden failures, doubled measurement noise, and 50% higher background load), evaluation method filtering, and robustness summary output.
- Completed the H=3/H=10 held-out policy grids and all reactive/PRESTO-R robustness evaluations. Rebuilt the consolidated Phase 5 episode metrics (38,400 episodes), sensitivity/robustness summaries, paired tests, and reproducibility/success checks.
- Implemented the Phase 6 reporting pipeline, refreshed setup/reproduction instructions, and added Make targets for tests, metrics, tables, figures, and reports. Generated all nine figure pairs and `results/report.md`.
- Completed Phase 6 portability validation: verified clean environment creation, dependency installation, unit tests (74 passed), and smoke episode generation on Windows with Python 3.12.4. Recorded that GNU Make was unavailable and verified equivalent Python commands.
- Curated and packaged the minimum supported model bundle `models/presto_r_models_v1.0.zip` (12.61 MB, 88 files: 4 predictor checkpoints, 42 best agent checkpoints, and matching configs) with machine-readable SHA-256 manifest `models/SHA256SUMS.txt`; round-trip extraction verified. External publishing documented as pending remote repository configuration.
- Exported the technical report to publication-ready PDF `results/report.pdf` (12 pages, 1.63 MB) containing all 8 inline figures and 7 tables without numerical drift.
- Configured repository ignore rules to exclude large raw parquet grids (`results/raw/`), intermediate HTML, and caches while versioning reports, figures, tables, and model checksums.
- Created the `v1.0` release commit and annotated Git tag.
