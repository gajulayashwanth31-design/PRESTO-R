# Phase 5: Experiments, Metrics and Statistics

**Goal:** run every method on the held-out test seeds across all scenarios, compute the PRD metrics, run the sensitivity sweeps, and check the success criteria automatically.

**Estimated effort:** 2–3 days (mostly compute time) · **Depends on:** Phase 4 · **PRD:** FR-10, sections 7–8

---

## 5.1 Experiment grid

| Axis | Values |
| --- | --- |
| Methods (P0) | `sp_failover`, `reactive_dqn`, `predictive_dqn`, `presto_dqn` |
| Methods (P1) | `greedy_delay`, `risk_threshold`, `reactive_dqn_pen`, `presto_ppo` |
| Diagnostic | `presto_oracle` (reported separately as an upper bound, never as a competitor) |
| Scenarios | `steady`, `congestion`, `node_failure`, `fluctuation`, `mixed` |
| Topologies | `nsfnet` (primary), `rgg` (secondary) |
| Training seeds | 0–4 for learned methods; heuristics are deterministic, so run them once |
| Test episodes | 200 per (scenario, topology): seeds 10000–10199, identical for every method |

All methods see **exactly the same 200 scenarios** per cell, which allows paired statistical tests.

### Agents for the RGG topology

Agents from Phases 3–4 are trained on NSFNET. For RGG, train `reactive_dqn`, `predictive_dqn` and `presto_dqn` with 3 seeds each (same configs, `defaults` pointing at `topo/rgg.yaml`, same λ*). Optionally also evaluate the NSFNET agents on RGG as a **generalisation** result. The observation is topology-agnostic (per-path features), so this is possible.

## 5.2 Compute budget (plan before running)

Fill in the per-run time you measured in Phase 3.

| Item | Runs | Steps each | Notes |
| --- | --- | --- | --- |
| Already done in Phases 3–4 | ~22 | 150k–300k | reactive, λ screening, predictive, presto, oracle |
| RGG agents | 9 | 300k | 3 methods × 3 seeds |
| H sweep (5.6) | 2 predictors + 6 agents | 300k | H ∈ {3, 10}, 3 seeds each; H = 5 exists |
| P1 methods | ~10 | 300k | `reactive_dqn_pen`, `presto_ppo` × 5 seeds |
| Evaluation rollouts | methods × 5 scenarios × 2 topologies × seeds × 200 episodes | — | Fast; parallelise with `multiprocessing.Pool` |

Training runs are independent, so run them in parallel, one process per CPU core, or overnight. `scripts/train_all.sh` should take a list of (config, seed) pairs and run them with `xargs -P <cores>`.

## 5.3 Tasks

### Task 5.3.1 — `presto/eval/run_grid.py`

```bash
python -m presto.eval.run_grid --grid configs/grid/main.yaml --workers 8
```

`configs/grid/main.yaml`:

```yaml
methods:
  sp_failover:    {type: heuristic, cls: SPFailover}
  risk_threshold: {type: heuristic, cls: RiskThreshold, kwargs: {tau: 0.5}, risk: gru}
  reactive_dqn:   {type: sb3, run_glob: "runs/reactive_dqn/{topo}_*_s{seed}/best_model.zip", risk: none}
  predictive_dqn: {type: sb3, run_glob: "runs/predictive_dqn/{topo}_*_s{seed}/best_model.zip", risk: gru}
  presto_dqn:     {type: sb3, run_glob: "runs/presto_dqn/{topo}_*_s{seed}/best_model.zip", risk: gru}
  presto_oracle:  {type: sb3, run_glob: "runs/presto_oracle/{topo}_*_s{seed}/best_model.zip", risk: oracle}
scenarios: [steady, congestion, node_failure, fluctuation, mixed]
topologies: [nsfnet, rgg]
seeds: [0, 1, 2, 3, 4]
test_seeds: [10000, 10200]
out_dir: results/raw
```

For each cell it calls `rollout(...)` and writes `results/raw/{method}/{topo}/{scenario}/s{seed}_steps.parquet` plus `_events.parquet`. **Skip cells whose outputs already exist**, so an interrupted run can resume.

### Task 5.3.2 — `presto/eval/metrics.py` (final)

Add the aggregation step:

```python
def compute_all(raw_dir: str) -> pd.DataFrame:
    """Walk results/raw, compute episode_metrics for every episode.
    Returns one row per (method, topo, scenario, train_seed, test_seed) with all
    PRD section 7 metrics. Saves results/episode_metrics.parquet."""
```

Use the metric definitions from Phase 3 unchanged. If you find a bug in a metric now, fix it, re-run its unit tests, and recompute everything. Never patch numbers by hand.

### Task 5.3.3 — `presto/eval/stats.py`

Two levels of aggregation:

1. **Per training seed:** average each metric over the 200 test episodes.
2. **Across training seeds:** report mean ± standard deviation, and a 95% confidence interval from the t-distribution with n − 1 degrees of freedom.

```python
def summary_table(df) -> pd.DataFrame:
    """Rows: method; columns: metric × scenario (one table per topology).
    Cell text: 'mean ± std'. Saves results/tables/summary_{topo}.csv."""

def paired_tests(df, target="presto_dqn", metrics=("pdr", "switches", "recovery_time")) -> pd.DataFrame:
    """For each (topo, scenario, metric, baseline): average over training seeds per test
    episode, then Wilcoxon signed-rank test target vs baseline over the 200 paired episodes.
    Holm–Bonferroni correction across all comparisons in the table.
    Report median difference, p (corrected), and effect size r = Z / sqrt(N)."""
```

Use `scipy.stats.wilcoxon` and `scipy.stats.t`. Do Holm correction by hand; it takes a few lines.

### Task 5.3.4 — `presto/eval/check_success.py`

Turns PRD section 8 into code that prints PASS/FAIL with the numbers, using **NSFNET test results**:

| # | Check | Implementation |
| --- | --- | --- |
| 1 | PRESTO-R PDR > reactive DQN, and recovery time < reactive DQN, in `node_failure` and `fluctuation` | Compare means across seeds, and require corrected p < 0.05 |
| 2 | PRESTO-R switches < predictive DQN, with PDR within 2 points | `mixed` and `fluctuation` scenarios |
| 3 | `steady`: PRESTO-R delay ≤ 1.10 × reactive DQN delay | Means |
| 4 | Predictor onset F1 ≥ 0.7 on precursor failures | Read `results/predictor/metrics.csv` |
| 5 | Reproducibility | Re-run 1 cell from scratch and compare the parquet hash with the stored one |

Save the output to `results/success_criteria.md`. Failed criteria stay in the report with an explanation (PRD section 8).

### Task 5.3.5 — λ sensitivity (test)

λ* was chosen on validation. For the report, also evaluate the Phase 4 λ-screening agents on **test** seeds (`mixed` scenario only) to show the trade-off curve on unseen data. This is reporting, not tuning, and λ* does not change.

### Task 5.3.6 — Horizon H sweep

For H ∈ {3, 10}:

1. Rebuild the dataset with `predictor.horizon=H` (Phase 2 script).
2. Retrain the GRU.
3. Train `presto_dqn` with 3 seeds, λ*, `risk_model` pointing at the new predictor.
4. Evaluate on `mixed` and `node_failure` test seeds.

Output `results/tables/horizon_sweep.csv`: H × (predictor onset F1, PDR, switches, proactive rate, recovery time). Expected pattern: small H gives an accurate predictor but little warning; large H gives more warning but more false alarms.

### Task 5.3.7 — Robustness checks (P1)

Small extra runs that strengthen the discussion section:

| Check | How |
| --- | --- |
| Sudden failures only | Test with `no_precursor_prob=1.0`. Prediction cannot help; PRESTO-R should be about equal to reactive DQN, not worse |
| Higher measurement noise | Test with noise ×2 |
| Heavier background load | Test with `n_background` +50% |

Apply all of them only at evaluation time, with no retraining. They show how gracefully the method degrades.

### Implementation status

- [x] Resumable grid runner and P0 grid config added (`presto.eval.run_grid`, `configs/grid/main.yaml`); it records each resolved cell config and SHA-256 hashes of raw logs. NSFNET has 5 training seeds; RGG has the planned 3. The λ-screen models are included for the mixed-scenario test sensitivity curve.
- [x] Raw parquet aggregation added (`presto.eval.metrics.compute_all`).
- [x] Training-seed summaries with t-based 95% confidence intervals and paired Wilcoxon/Holm tests added (`presto.eval.stats`).
- [x] PRD success-check implementation added (`presto.eval.check_success`).
- [x] RGG training complete (3 seeds each for reactive, predictive, and PRESTO-R).
- [x] P0 test grid complete: 5 scenarios × 200 seeds for NSFNET and RGG; 5-seed NSFNET and 3-seed RGG agents. Oracle diagnostic and λ sensitivity test cells are also complete.
- [x] Episode metrics, both topology summaries, paired tests, λ test sensitivity table, and `results/success_criteria.md` generated.
- [x] Reproducibility check passed (raw step/event parquet hashes match on a fresh rerun).
- [x] Horizon H=3/H=10 datasets built with episode-isolated train/validation/test splits.
- [x] H=3 predictor trained and evaluated on its held-out predictor split (onset F1=0.5992).
- [x] H=10 predictor and H-specific PRESTO agent training/evaluation completed; both H=3 and H=10 policies were evaluated on held-out mixed and node-failure scenarios.
- [x] Evaluation-only sudden-failure, doubled-noise, and +50% background-load cells completed for reactive DQN and PRESTO-R.
- [x] Robustness override handling, summary generation, and focused tests added; full test suite passes (97 tests).

**Observed success checks:** PRESTO-R passed the `mixed`/`fluctuation` switching comparison and steady-delay limit. It did not pass the statistically significant PDR-and-recovery comparison against reactive DQN in node-failure or fluctuation; predictor onset F1 was 0.5296 (below 0.7). All failures are retained in `results/success_criteria.md`; no test-based retuning was performed.

---

## 5.4 Outputs of this phase

```
results/
├── raw/                          # parquet step + event logs
├── episode_metrics.parquet
├── tables/
│   ├── summary_nsfnet.csv
│   ├── summary_rgg.csv
│   ├── paired_tests.csv
│   ├── lambda_sweep_test.csv
│   ├── horizon_sweep.csv
│   └── robustness.csv
├── predictor/metrics.csv         # from Phase 2
└── success_criteria.md
```

## Exit criteria

- [x] All P0 cells of the grid complete for both topologies
- [x] `episode_metrics.parquet` and the summary tables generated by script (38,400 episode rows)
- [x] Paired tests with Holm correction computed
- [x] `check_success.py` run and its output generated
- [x] λ and H sweeps done
- [x] Every number in the tables traces back to a parquet file and a config

## Pitfalls

- **Different scenarios per method.** Always pass the scenario seed explicitly to `env.reset(options={"scenario_seed": s})`, and never rely on the environment's internal RNG for test scenarios.
- **Treating 200 episodes as 200 independent samples of training randomness.** Report seed-level variance (mean ± std over 5 training seeds) as the main uncertainty. Use episode pairing only for the significance tests.
- **Cherry-picking.** Report all scenarios, including ones where PRESTO-R does not win, such as sudden failures.
- **Disk space.** Step logs for the full grid can reach a few GB. Store `float32`, drop columns you do not need, and compress parquet with zstd.
