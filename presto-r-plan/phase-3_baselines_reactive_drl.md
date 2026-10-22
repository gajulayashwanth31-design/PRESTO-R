# Phase 3: Baselines, Rollout Harness and Reactive DRL

**Goal:** every comparison method except PRESTO-R itself, plus one shared way to run any policy on fixed test seeds and turn the logs into metrics.

**Estimated effort:** 2–3 days · **Depends on:** Phase 1 · **Runs in parallel with:** Phase 2 · **PRD:** FR-6, FR-8, part of FR-10

---

## 3.1 Methods built in this phase

| Method id | Type | Uses risk | λ | Priority | Role |
| --- | --- | --- | --- | --- | --- |
| `sp_failover` | Heuristic | No | — | P0 | Classical baseline: stay on shortest path, fail over only when the current path is down |
| `greedy_delay` | Heuristic | No | — | P1 | Each tick, pick the valid path with the lowest observed delay. Shows flapping without a penalty |
| `risk_threshold` | Heuristic | Yes | — | P1 | Smart-Routing-like: leave a path when its max risk > τ. Needs Phase 2's predictor |
| `reactive_dqn` | DQN | No | 0 | P0 | Base-paper-style reactive RL baseline |
| `reactive_dqn_pen` | DQN | No | > 0 | P1 | Completes the 2×2 ablation (prediction × penalty) |

`predictive_dqn` (risk, λ = 0) and `presto_dqn` (risk, λ > 0) use the same training code and are run in Phase 4.

**On "RQSMR-style":** RQSMR splits flows across several paths and derives path lifetimes from vehicle mobility. Our reactive baseline keeps its essential property, an RL agent choosing paths from current path-state metrics with no forecast, inside our single-path setting. Call it "RQSMR-inspired reactive DRL" in the report, not a re-implementation.

## 3.2 Policy interface — `presto/policies/base.py`

```python
class Policy:
    name: str
    def reset(self, info: dict) -> None:
        """Called after env.reset(); info has 'paths', 'current', 'scenario_seed'."""
    def act(self, obs: np.ndarray, info: dict) -> int:
        raise NotImplementedError
```

The environment must expose in `info` everything a heuristic needs: `paths` (list of link-id lists), `current`, `obs_link` (observed latest link stats, shape (E, F)), `risk` (shape (E,), zeros if no predictor) and `valid` (bool per candidate path, from observed `up`).

## 3.3 Tasks

### Task 3.3.1 — `presto/policies/heuristics.py`

```python
class SPFailover(Policy):
    name = "sp_failover"
    def act(self, obs, info):
        cur = info["current"]
        if info["valid"][cur]:
            return cur
        valid = [i for i, v in enumerate(info["valid"]) if v]
        return valid[0] if valid else cur        # paths are sorted by base delay

class GreedyDelay(Policy):
    name = "greedy_delay"
    def act(self, obs, info):
        ...  # argmin of observed path delay over valid paths

class RiskThreshold(Policy):
    name = "risk_threshold"
    def __init__(self, tau: float = 0.5): self.tau = tau
    def act(self, obs, info):
        cur = info["current"]; paths = info["paths"]; risk = info["risk"]
        max_r = [max(risk[l] for l in p) for p in paths]
        if info["valid"][cur] and max_r[cur] <= self.tau:
            return cur
        cands = [i for i in range(len(paths)) if info["valid"][i]]
        return min(cands, key=lambda i: (max_r[i], i)) if cands else cur
```

Tune τ for `risk_threshold` on validation seeds (try 0.3, 0.5, 0.7) and fix it before the test runs.

### Task 3.3.2 — `presto/policies/sb3_policy.py`

```python
class SB3Policy(Policy):
    def __init__(self, model_path: str, algo: str = "dqn", name: str = "dqn"):
        self.model = (DQN if algo == "dqn" else PPO).load(model_path, device="cpu"); self.name = name
    def act(self, obs, info):
        action, _ = self.model.predict(obs, deterministic=True)
        return int(action)
```

### Task 3.3.3 — `presto/eval/rollout.py`

```python
def rollout(policy: Policy, cfg: dict, seeds: list[int], risk: RiskProvider | None,
            out_path: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run one episode per scenario seed. Returns:
      steps  — one row per (seed, t): action, switched, valid_chosen, path_delay,
               path_loss, delivered, demand, hops, reward, reward terms, current path
               links (as a string), max true health on the path, max risk on the path
      events — one row per (seed, event): kind, links, start, precursor, duration
    Writes both as parquet if out_path is given."""
```

The ground-truth event table is what the metrics need, so record it from `sim.schedules` and the generated `Event` list. It is never shown to the policy.

### Task 3.3.4 — `presto/eval/metrics.py` (first version)

Per-episode metrics, following PRD section 7:

```python
def episode_metrics(steps: pd.DataFrame, events: pd.DataFrame, sim_up: np.ndarray, H: int,
                    qos_ok_loss: float = 0.02) -> dict:
    return {
      "pdr":             steps.delivered.sum() / steps.demand.sum(),
      "delay_mean":      weighted mean of path_delay by delivered,
      "throughput":      steps.delivered.mean(),
      "switches":        steps.switched.sum(),
      "unnecessary_switch_rate": ...,   # see below
      "recovery_time":   ...,           # mean over failure events that hit the active path
      "proactive_rate":  ...,
      "energy":          steps.hops.sum(),
      "reward":          steps.reward.sum(),
    }
```

Exact definitions (write them as docstrings, and test them on hand-built examples):

| Metric | Definition |
| --- | --- |
| Unnecessary switch | A switch at tick *t* away from path P_old is **unnecessary** if no link of P_old is bad (2.1 definition) anywhere in `(t, t+H]` **and** P_old was not bad at *t*. Rate = unnecessary ÷ all switches (NaN if no switches) |
| Failure event (for recovery/proactive) | A tick `t_f` where a link goes from up to down, on a link that was on the active path at `t_f − H` |
| Proactive | The active path at `t_f` no longer contains that link. Proactive rate = proactive events ÷ all such events |
| Recovery time | For non-proactive events: ticks from `t_f` until the first tick where the active path is valid and path loss ≤ `qos_ok_loss`. Proactive events count as 0. Cap at the end of the episode |

`sim_up` is the true `up` schedule (T, E), saved by the rollout.

### Task 3.3.5 — `presto/agents/train.py`

One script trains every learned method. Method differences live only in config.

```bash
python -m presto.agents.train --config configs/agent/reactive_dqn.yaml --seed 0
```

`configs/agent/reactive_dqn.yaml`:

```yaml
defaults: [scenario/mixed.yaml]
agent:
  name: reactive_dqn
  algo: dqn
  use_risk: false
  risk_model: null
  total_steps: 300000
  policy_kwargs: {net_arch: [128, 128]}
  dqn:
    learning_rate: 5.0e-4
    buffer_size: 100000
    learning_starts: 5000
    batch_size: 128
    gamma: 0.97
    train_freq: 4
    target_update_interval: 2000
    exploration_fraction: 0.2
    exploration_final_eps: 0.02
  train_scenarios: [congestion, node_failure, fluctuation, mixed]   # sampled per episode
  eval_every: 10000
  eval_episodes: 50          # validation seeds 5000–5049
reward: {alpha: 1.0, beta: 4.0, gamma: 0.0, lam: 0.0, mu: 0.05}
```

Training loop outline:

```python
def main(cfg, seed):
    seed_everything(seed)
    risk = build_risk(cfg)                       # None / LearnedRisk / LogRegRisk
    train_env = Monitor(RoutingEnv(cfg, risk, seed_pool=TrainSeedPool(cfg, seed)))
    val_env   = Monitor(RoutingEnv(cfg, risk, seed_pool=FixedSeeds(range(5000, 5050))))
    model = DQN("MlpPolicy", train_env, seed=seed, verbose=0,
                policy_kwargs=cfg["agent"]["policy_kwargs"], **cfg["agent"]["dqn"])
    cb = EvalCallback(val_env, n_eval_episodes=50, eval_freq=cfg["agent"]["eval_every"],
                      best_model_save_path=run_dir, deterministic=True)
    model.learn(total_timesteps=cfg["agent"]["total_steps"], callback=[cb, SwitchLogger()])
    save_config(cfg, run_dir / "config.yaml")
```

- `TrainSeedPool` draws a fresh scenario seed below 5000 for every episode, plus a random scenario type from `train_scenarios`. Training on a mix makes one policy work across all scenarios.
- `SwitchLogger` (a small `BaseCallback`) logs mean switches per episode and mean PDR to `log.csv`. Watching switches during training is how you spot "never switch" and "always switch" policies early.
- Keep `best_model.zip` (best validation reward), not the last checkpoint.

### Task 3.3.6 — Learning-curve plot

`scripts/plot_training.py runs/reactive_dqn/*` plots validation reward and switches per episode against timesteps, averaged over seeds with a shaded band.

## 3.4 Sanity runs (do these before Phase 4)

1. [x] **Random policy vs `sp_failover`** on 50 validation seeds of `node_failure`: random PDR 0.866 vs 0.883 and switches 138.24 vs 0.30 per episode.
2. [x] **`reactive_dqn`, 1 seed, 100k steps:** run completed in 194 seconds; validation evaluations were recorded every 10k steps. The subsequent 300k-step five-seed runs also produced validation curves every 10k steps.
3. [x] **`reactive_dqn` vs `sp_failover`** on 50 validation seeds: DQN PDR was 0.970 vs 0.883 for `node_failure`, and 0.967 vs 0.865 for `congestion`.
4. [x] **Risk threshold tuning** on all 50 mixed validation seeds selected τ = 0.5 by mean reward.

## 3.5 Tests

| File | Checks |
| --- | --- |
| `tests/test_heuristics.py` | [x] `SPFailover` stays put on a valid path and fails over otherwise; `RiskThreshold` leaves when risk > τ |
| `tests/test_metrics.py` | [x] Hand-built necessary/proactive, unnecessary, recovery-time, and no-switch cases |
| `tests/test_rollout.py` | [x] Same policy + seeds → identical DataFrames; event table matches the generator output |

---

## Exit criteria

- [x] All heuristics implemented and tested
- [x] `rollout` + `episode_metrics` produce a per-episode table for any policy
- [x] Metric unit tests pass on hand-built cases
- [x] `reactive_dqn` trained with 5 seeds × 300k steps; five-seed learning curves and per-seed validation/training summaries saved
- [x] `reactive_dqn` ≥ `sp_failover` on validation PDR in `node_failure`, and better in `congestion`
- [x] Training time per run recorded (100k-step seed 0: 194 seconds)

Validation outputs: `results/phase3_sanity_comparison.csv`,
`results/phase3_training_curves.png`,
`results/phase3_training_curves_5seeds.png`,
`results/reactive_dqn_5seed_summary.csv`, and
`results/predictor/risk_threshold_tuning.csv` (selected τ is also recorded in the
matching JSON file).

The five full runs each completed 300,000 steps. Their best mean validation rewards
ranged from -155.38 to -154.58 (mean -154.96, sample standard deviation 0.37).
Logged training durations ranged from 601 to 11,053 seconds; the high seed-1 duration
is retained in the per-seed summary rather than hidden by averaging.

## Pitfalls

- **DQN never switches.** Common when the reward is dominated by the delay term. Check reward-term magnitudes in `info`: a failure tick should cost ~4 (β·1), and a switch at most ~0.5. Raise β or reduce α.
- **DQN switches every tick.** Exploration may be too high for too long, or the observation may lack `current` (one-hot) and `steps_since_switch`. Check `build_observation`.
- **Flat learning curve.** Too few events reach the managed flow. Check the event-on-candidate statistic from Phase 1, and use `mixed` with higher event rates for training.
- **Evaluating on training seeds.** Validation is 5000–5049 and test is 10000–10199. Assert this inside `rollout`.
- **Slow training.** Make sure `risk_model: null` really skips the predictor. Use `n_envs=4` with `SubprocVecEnv` if your SB3 setup allows; DQN supports it with `train_freq` adjusted.
