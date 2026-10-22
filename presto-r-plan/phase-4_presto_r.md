# Phase 4: PRESTO-R Agent, Tuning and Ablations

**Goal:** train the full PRESTO-R agent (predicted risk in the state, switch penalty in the reward), choose λ on validation data, and confirm with diagnostics that the agent actually uses the predictor.

**Estimated effort:** 2–3 days · **Depends on:** Phases 2 and 3 · **PRD:** FR-6, FR-7, FR-8

---

## 4.1 Methods trained in this phase

| Method id | Risk in state | Risk source | λ | Purpose |
| --- | --- | --- | --- | --- |
| `predictive_dqn` | Yes | Learned GRU | 0 | Isolates the effect of prediction (PRD ablation) |
| **`presto_dqn`** | Yes | Learned GRU | λ* (tuned) | **Proposed method** |
| `presto_oracle` | Yes | `OracleRisk` | λ* | Upper bound: how good PRESTO-R could be with a perfect predictor. Diagnostic only |
| `presto_ppo` | Yes | Learned GRU | λ* | P1: algorithm comparison (PRD FR-6.3) |

Together with Phase 3's `reactive_dqn` (and optionally `reactive_dqn_pen`) this gives the ablation grid:

|  | λ = 0 | λ = λ* |
| --- | --- | --- |
| **No prediction** | `reactive_dqn` | `reactive_dqn_pen` (P1) |
| **Prediction** | `predictive_dqn` | **`presto_dqn`** |

## 4.2 Configs

`configs/agent/presto_dqn.yaml`:

```yaml
defaults: [agent/reactive_dqn.yaml]
agent:
  name: presto_dqn
  use_risk: true
  risk_model: models/predictor/gru_W10_H5.pt
reward: {lam: 0.5}            # selected λ* from validation screening
```

`configs/agent/predictive_dqn.yaml` is the same with `name: predictive_dqn` and `reward: {lam: 0.0}`.

**Keep every other hyperparameter identical** to `reactive_dqn`: network size, learning rate, steps and exploration schedule. A fair ablation changes one thing at a time.

## 4.3 Tasks

### Task 4.3.1 — λ tuning on validation seeds

Train `presto_dqn` for each λ ∈ {0.0, 0.05, 0.1, 0.25, 0.5} with **2 seeds × 150k steps** (a cheaper screening budget), and evaluate each on validation seeds 5000–5049 across the four failure scenarios.

```bash
for lam in 0.0 0.05 0.1 0.25 0.5; do
  for s in 0 1; do
    python -m presto.agents.train --config configs/agent/presto_dqn.yaml --seed $s \
      reward.lam=$lam agent.total_steps=150000 agent.name=presto_lam$lam
  done
done
python -m presto.eval.select_lambda --runs runs/presto_lam* --out results/tuning/lambda.csv
```

`select_lambda` writes a table of mean PDR, switches and unnecessary-switch rate per λ, plus a trade-off plot (x = switches per episode, y = PDR, one point per λ).

**Selection rule (decide it now, before seeing the numbers):** choose the **largest λ whose validation PDR is within 1 percentage point of the best PDR across all λ**. This is the most stable policy that gives up essentially no delivery. Write the chosen λ* into `presto_dqn.yaml` before any test evaluation.

**Selection result:** λ* = 0.5. The two-seed screening grid selected 0.5 under this rule. Aggregate validation PDR ranged from 0.9670 to 0.9681 across tested λ values; mean switches fell from 18.4 at λ=0 to 4.7 at λ=0.5. Detailed results are in `results/tuning/lambda.csv` and `results/tuning/lambda_overall.csv`.

**Validation results:** All 5 predictive and PRESTO-R training seeds and both oracle seeds completed 300k steps, evaluated only on validation seeds. Across the four scenarios, PRESTO-R averaged PDR 0.9677 and 4.30 switches/episode; predictive DQN averaged PDR 0.9654 and 21.36 switches/episode. PRESTO-R had lower switching and slightly higher PDR in each scenario. The per-run validation metrics are in `results/tuning/validation_summary.csv`.

**Risk-use diagnostic:** On 50 mixed validation episodes using PRESTO-R training seed 0, temporal risk permutation reduced proactive rate from 0.250 to 0.167 and PDR from 0.9570 to 0.9554, while switches increased from 6.66 to 9.00. The comparison event trace for reactive, predictive, and PRESTO-R seed 0 is saved at `results/diagnostics/event_trace.png`.

### Task 4.3.2 — Full training

With λ* fixed, train with **5 seeds × 300k steps** each:

- `predictive_dqn` (λ = 0)
- `presto_dqn` (λ*)
- `presto_oracle` (λ*, `risk: oracle`), 2 seeds are enough since it is a diagnostic
- P1: `reactive_dqn_pen` (λ*), `presto_ppo` (λ*)

`configs/agent/presto_ppo.yaml` overrides:

```yaml
agent:
  algo: ppo
  ppo: {learning_rate: 3.0e-4, n_steps: 2048, batch_size: 256, n_epochs: 10,
        gamma: 0.97, gae_lambda: 0.95, clip_range: 0.2, ent_coef: 0.01}
```

`train.py` builds `PPO(...)` when `algo == "ppo"`. Optionally use `MaskablePPO` from `sb3-contrib` with an action mask from `info["valid"]`. If you do, report it as a separate variant, because the DQN agents get no mask.

### Task 4.3.3 — Diagnostics: does the agent use the risk signal?

1. **Risk permutation test** (`presto/eval/diagnostics.py`). Evaluate `presto_dqn` on validation seeds twice: normally, and with the `max_risk` / `mean_risk` features of each path replaced by values from a *different* random tick. If PDR and proactive rate do not drop, the agent is ignoring the predictor. Fix this before Phase 5.
2. **Event trace figure.** For one validation episode with a precursor failure on the active path, plot over time: max risk on the active path, the action (path index), and path loss, for `reactive_dqn`, `predictive_dqn` and `presto_dqn`. The story the report needs: PRESTO-R leaves *during* the precursor, reactive DQN leaves *after* the failure, and predictive DQN without a penalty leaves early but also jumps around at other times.
3. **Gap to oracle.** Compare `presto_dqn` and `presto_oracle` on proactive rate. A large gap means the predictor is the bottleneck, which is useful for the discussion section.

### Task 4.3.4 — Validation summary

`results/tuning/validation_summary.csv`: every method from 4.1 and Phase 3, evaluated on validation seeds, all scenarios, mean ± std over training seeds. Use it to check the PRD success criteria *informally* before spending the full Phase 5 budget.

## 4.4 Expected outcome (validation)

| Comparison | Expected |
| --- | --- |
| `predictive_dqn` vs `reactive_dqn` | Higher proactive rate, lower recovery time, higher PDR in node-failure and fluctuation |
| `presto_dqn` vs `predictive_dqn` | Clearly fewer switches and lower unnecessary-switch rate; PDR within ~2 points |
| `presto_dqn` vs `sp_failover` | Higher PDR in all failure scenarios |
| `presto_dqn` in `steady` | Delay within 10% of `reactive_dqn` (few or no switches) |

If a comparison goes the other way, do **not** retune on test data. Investigate with the diagnostics, fix the cause, re-run validation, and record what changed in `CHANGELOG.md`.

## 4.5 Tests

| File | Checks |
| --- | --- |
| `tests/test_train_smoke.py` | `train.py` runs 2,000 steps for `presto_dqn` and `presto_ppo` configs without error and saves a model |
| `tests/test_configs.py` | Every agent config loads; `reactive_dqn` has `use_risk: false`; all DQN configs share identical `dqn:` blocks except where intended |

---

## Exit criteria

- [x] λ* chosen by the pre-declared rule and recorded in `presto_dqn.yaml` before any test evaluation (λ*=0.5)
- [x] `predictive_dqn` and `presto_dqn` trained with 5 seeds × 300k steps; `presto_oracle` with 2 × 300k
- [x] Risk permutation lowers proactive rate on the 50-episode validation diagnostic (0.250 → 0.167)
- [x] Event trace figure saved (`results/diagnostics/event_trace.png`)
- [x] Validation summary shows `presto_dqn` switching less than `predictive_dqn` at comparable PDR (4.30 vs. 21.36 switches/episode; PDR 0.9677 vs. 0.9654)

## Pitfalls

- **Tuning on test seeds.** Everything in this phase uses validation seeds 5000–5049 only. Test seeds are touched once, in Phase 5.
- **λ too large: the agent never leaves a failing path.** It shows as a high recovery time. The selection rule protects against it, but check the trade-off plot.
- **Agent ignores risk.** Common causes: the risk features are near zero most of the time (check their distribution), or the predictor's `horizon` differs from what the environment assumes. Increasing the learning steps or scaling the risk features to a larger range can help.
- **Unequal budgets.** If PRESTO-R gets more steps or tuning than the baselines, the comparison is unfair. Give `reactive_dqn` the same λ-screening effort (for `reactive_dqn_pen`) if you report it.
