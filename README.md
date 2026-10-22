# PRESTO-R

Predictive, Stability-Aware Reinforcement Learning Routing.

## Setup

Use Python 3.10–3.12 (tested on Python 3.12.4 and Python 3.11 on Windows) and install the pinned dependencies:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Quick check

Run the unit tests and the Phase 1 simulator smoke episode:

```powershell
python -m pytest
python -m scripts.smoke_env
```

The smoke run writes `results/smoke/episode.png`. The Makefile provides equivalent
convenience targets if GNU Make is installed (`make quick`, `make figures`,
`make report`, `make pdf`, `make package-models`, and `make reproduce-results`).

## Training

Build the episode-separated telemetry data and train/evaluate the predictor:

```powershell
python -m presto.predictor.dataset --config configs/predictor/gru.yaml
python -m presto.predictor.train --config configs/predictor/gru.yaml
python -m presto.predictor.train --config configs/predictor/logreg.yaml
python -m presto.predictor.evaluate --config configs/predictor/gru.yaml
python -m presto.predictor.evaluate --config configs/predictor/logreg.yaml
python -m presto.predictor.trace --config configs/predictor/gru.yaml
```

Dataset generation and training accept repeated `--set key=value` overrides for small
smoke runs. Train, validation, and test episodes are split by scenario seed. Predictor
thresholds are selected using validation data only; held-out test data is used only
for final metrics.

Train learned routing agents with the shared trainer:

```powershell
python -m presto.agents.train --config configs/agent/reactive_dqn.yaml --seed 0
python -m presto.agents.train --config configs/agent/predictive_dqn.yaml --seed 0
python -m presto.agents.train --config configs/agent/presto_dqn.yaml --seed 0
```

Training runs are stored under `runs/<method>/seed_<seed>/`. The best checkpoint is
selected using validation reward.

## Evaluation and Phase 5

The resumable P0 test grid evaluates the methods across five scenarios and NSFNET/RGG.
It skips cells with existing step and event logs. Run it after training all configured
models:

```powershell
python -m presto.eval.run_grid --grid configs/grid/main.yaml --workers 4
python -m presto.eval.metrics --raw-dir results/raw --out results/episode_metrics.parquet
python -m presto.eval.stats --metrics results/episode_metrics.parquet --out-dir results/tables --horizon-predictor 3=results/predictor/horizon_H3/metrics.csv --horizon-predictor 10=results/predictor/horizon_H10/metrics.csv
python -m presto.eval.check_success
```

The grid includes the H=3/H=10 PRESTO-R models and evaluation-only robustness
conditions (sudden failures, doubled measurement noise, and 50% more background load).
The corresponding predictor datasets, models, and trained agents must exist before
these evaluation cells are run. Test-set results must not be used to retune agents,
predictor thresholds, or λ.

## Phase 6: figures, report, and model release

Once Phase 5 raw results and derived tables are available, regenerate all figures and
the report:

```powershell
make figures
make report
```

Without GNU Make, run:

```powershell
python -m presto.eval.plots
python -m presto.eval.report
python -m scripts.export_report_pdf
```

Figures are written in PNG and PDF format to `results/figures/`; the generated markdown report
is `results/report.md`, and the submission-ready PDF export is `results/report.pdf` (12 pages,
including all 8 inline figures and 7 tables).

Regenerate tables and all report artifacts together with:

```powershell
make reproduce-results
```

This target requires the trained models, predictor data, and raw Phase 5 logs in the
workspace. The figures and report scripts do not run training.

The completed Phase 5 metrics contain 38,400 episode rows. Full training and
evaluation runtime depends on the machine, worker count, and storage; measure those
steps on the target system rather than relying on a fixed time estimate.

| Figure | Generated content |
|---|---|
| F1 | System architecture |
| F2 | Held-out onset precision–recall and predictor calibration |
| F3 | Learned and oracle link-risk trace |
| F4 / F9 | Main results for NSFNET / RGG |
| F5 | Mixed-scenario switch/PDR trade-off across λ |
| F6 | Held-out event trace for reactive, predictive, and PRESTO-R |
| F7 | Horizon sensitivity |
| F8 | Training reward and switches |

### Model release bundle and verification

The minimum supported model bundle contains 88 curated files (4 predictor checkpoints, 42
best agent policies, and their resolved `config.yaml` files), excluding raw training logs
and evaluation arrays:

- **Local archive:** `models/presto_r_models_v1.0.zip` (12.61 MB, 13,227,437 bytes)
- **Archive SHA-256:** `a0c3fcdd914f88ff2d899f4618fe9c4399ef3d3388806a067a5ea5dac4e1c398`
- **File manifest:** `models/SHA256SUMS.txt` (contains relative paths and SHA-256 digests for all 88 files)

To package, hash, and verify the model bundle locally:

```powershell
python -m scripts.package_models
```

To verify checksums against the manifest after extracting the archive:

```powershell
Get-FileHash -Algorithm SHA256 (Get-ChildItem -Recurse runs, models -Filter "*.zip", "*.pt", "*.joblib", "*.yaml")
```

*Note on publishing:* No remote Git repository or external release destination is currently configured.
Artifact publishing to a remote host is pending project owner configuration and authorization.

## Project structure

- `configs/`: simulator, scenario, predictor, agent, and experiment-grid settings.
- `presto/sim/`: topology, traffic, failure schedules, and network simulator.
- `presto/predictor/`: dataset creation, predictor models, training, and evaluation.
- `presto/policies/`, `presto/agents/`: heuristics and learned routing agents.
- `presto/eval/`: deterministic rollouts, metrics, statistics, figures, and report.
- `tests/`: unit and integration tests.
- `presto-r-plan/`: phase-by-phase implementation plan.
- `results/`: generated metrics, figures, tables, and report deliverables (`report.md`, `report.pdf`).
- `models/`: SHA-256 release checksum manifest (`SHA256SUMS.txt`) and curated model bundle.

Phase 7 (NS-3 validation) is optional and is not part of the completed core workflow.
