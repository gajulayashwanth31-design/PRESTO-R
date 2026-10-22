# PRESTO-R Implementation Plan: Overview

This is the master index for building PRESTO-R (Predictive, Stability-Aware RL Routing), as specified in `prd.md`. The work is split into eight phases, each in its own file. Read this file first, then work through the phases in order.

## How to use these files

- Each phase file lists **goal → tasks (file by file) → config → tests → exit criteria → pitfalls**.
- Do not start a phase until the previous phase's **exit criteria** are all ticked. Most failures in RL projects come from a broken environment that only shows up during training.
- Requirement IDs (FR-x, NFR-x) refer to `prd.md`.
- Code blocks are **skeletons**. Signatures, data shapes and formulas are fixed decisions. Bodies marked `...` are for you to fill in.

## Phase map

| Phase | File | What you build | PRD coverage | Est. days |
| --- | --- | --- | --- | --- |
| 0 | `phase-0_setup.md` | Repo, environment, config system, CI-style checks | NFR-1, 3, 4, 6, 7 | 0.5–1 |
| 1 | `phase-1_simulator_env.md` | Network simulator, failure scenarios, telemetry, state, reward, Gymnasium env | FR-1, 2, 3, 5, 7 | 4–5 |
| 2 | `phase-2_predictor.md` | Link-degradation dataset, logistic regression + GRU, risk providers | FR-4 | 2–3 |
| 3 | `phase-3_baselines_reactive_drl.md` | Heuristic baselines, rollout harness, reactive DQN | FR-6, 8, part of 10 | 2–3 |
| 4 | `phase-4_presto_r.md` | PRESTO-R agent, λ tuning, ablations, diagnostics | FR-6, 7, 8 | 2–3 |
| 5 | `phase-5_experiments.md` | Full experiment grid, metrics, statistics, sweeps | FR-10, section 7–8 | 2–3 |
| 6 | `phase-6_report.md` | Figures, tables, report, README, reproduction | FR-10.4–10.5, NFR-6 | 1–2 |
| 7 | `phase-7_ns3_optional.md` | NS-3 validation via ns3-gym | FR-9 (P2) | 3+ (optional) |

**Core path (Phases 0–6): about 14–20 working days.** Phase 7 is optional and starts only after Phase 5 is done.

## Dependency graph

```
Phase 0 ──► Phase 1 ──┬──► Phase 2 ──┐
                      │              ├──► Phase 4 ──► Phase 5 ──► Phase 6
                      └──► Phase 3 ──┘                   │
                                                         └──► Phase 7 (optional)
```

Phases 2 and 3 can run **in parallel** once Phase 1 is done. For a team of 2–3:

| Person | Phase 1 | Then | Then |
| --- | --- | --- | --- |
| A | Simulator + failures | Phase 2 (predictor) | Phase 5 (experiments) |
| B | Telemetry, state, reward, env wrapper, tests | Phase 3 (baselines + reactive DQN) | Phase 4 (PRESTO-R) |
| C (if any) | Config + tests + plotting utilities | Phase 6 groundwork (plots, report template) | Phase 7 (optional) |

## Final repository layout

This refines PRD section 11. The simulator is kept separate from the Gymnasium wrapper so that baselines and NS-3 validation can reuse it.

```
presto-r/
├── README.md
├── requirements.txt
├── pyproject.toml                # black/ruff/pytest settings
├── Makefile                      # make test / train / experiments / figures
├── configs/
│   ├── base.yaml                 # shared defaults (time step, seeds, paths)
│   ├── topo/nsfnet.yaml
│   ├── topo/rgg.yaml
│   ├── scenario/{steady,congestion,node_failure,fluctuation,mixed}.yaml
│   ├── predictor/{logreg,gru}.yaml
│   └── agent/{reactive_dqn,predictive_dqn,presto_dqn,presto_ppo}.yaml
├── presto/
│   ├── config.py                 # YAML loading + overrides
│   ├── seeding.py                # seed management
│   ├── sim/
│   │   ├── topology.py
│   │   ├── traffic.py
│   │   ├── failures.py
│   │   └── simulator.py          # NetworkSimulator (no RL code)
│   ├── telemetry.py
│   ├── risk.py                   # ZeroRisk, OracleRisk, LearnedRisk
│   ├── state.py                  # candidate paths + observation builder
│   ├── reward.py
│   ├── env.py                    # RoutingEnv (Gymnasium)
│   ├── predictor/
│   │   ├── dataset.py
│   │   ├── features.py
│   │   ├── logreg.py
│   │   ├── gru.py
│   │   ├── train.py
│   │   └── evaluate.py
│   ├── policies/
│   │   ├── base.py               # Policy interface
│   │   ├── heuristics.py         # shortest-path failover, greedy, risk-threshold
│   │   └── sb3_policy.py         # wraps a trained SB3 model
│   ├── agents/
│   │   └── train.py              # DQN / PPO training entry point
│   └── eval/
│       ├── rollout.py            # run a policy over test seeds → step logs
│       ├── metrics.py            # PRD section 7
│       ├── stats.py              # aggregation, CIs, paired tests
│       ├── run_grid.py           # method × scenario × topology × seed
│       ├── check_success.py      # PRD section 8, pass/fail
│       ├── plots.py
│       └── report.py             # writes results/report.md
├── scripts/                      # thin shell wrappers
├── tests/
├── data/                         # generated predictor datasets (git-ignored)
├── models/                       # trained predictors + agents (git-ignored)
└── results/                      # CSVs, figures, report.md
```

## Key design decisions (fixed for all phases)

These are settled here so the phases stay consistent. Change them only deliberately and update every phase file.

| Decision | Choice | Reason |
| --- | --- | --- |
| Simulation model | Discrete-time **fluid** model: 1 step = 1 tick (100 ms); traffic in rate units, not packets | Fast enough for 300k-step RL training on a CPU |
| Links | Undirected, indexed `0..E-1`, canonical key `(min(u,v), max(u,v))` | Simple indexing; NSFNET has E = 21 |
| Controlled traffic | **One managed flow per episode** (random source–destination); all other traffic is background | Keeps the action space at *k* and makes DQN tractable. Multi-flow is a P1 extension |
| Background traffic | ~20 flows on shortest paths, re-routed whenever the set of live links changes | Creates realistic congestion without being controlled by the agent |
| Hidden vs observed | Link **health** h ∈ [0,1] is hidden. The agent and predictor see only delay, loss, queue, utilisation and the up flag, with measurement noise | Otherwise prediction is trivial and results are meaningless |
| Candidate paths | k = 4 loop-free paths by base delay (Yen), fixed per episode | Fixed action meaning within an episode |
| Invalid path | A path with a down link delivers nothing (loss = 1) that step; no separate mask in DQN | Natural penalty; MaskablePPO can use a mask (P1) |
| Episode | 200 steps after a warm-up of W steps on path 0 | Warm-up fills the telemetry window |
| Window / horizon | W = 10 steps of history; H = 5 steps ahead (default) | Matches PRD; H is swept in Phase 5 |
| Seeds | Train scenarios: any seed < 5000. Validation: 5000–5049. Test: 10000–10199 | Strict separation; λ is tuned on validation only |
| Agent training seeds | 5 per learned method (torch/SB3 seed) | PRD FR-10.2 |

## Global conventions

- **Configuration:** every number lives in YAML. Code reads `cfg["section"]["key"]`. Command-line overrides are `key.sub=value`.
- **Randomness:** never call `np.random.*` directly. Pass a `np.random.Generator` created from `presto.seeding.make_rng(seed, stream)`.
- **Logging:** each training or evaluation run writes to `runs/<method>/<topo>_<scenario>_s<seed>/` with `config.yaml` (resolved config), `log.csv`, and model files.
- **Git:** one branch per phase (`phase-1-sim`, etc.), merged when the exit criteria pass. Commit generated results only from Phase 5 onwards, under `results/`.
- **Definition of done** for any task: code, a unit test (if it has logic), and a one-line note in `CHANGELOG.md`.

## Milestone checklist

- [ ] Phase 0: repo installs cleanly; `make test` passes on an empty test suite
- [ ] Phase 1: env passes `gymnasium.utils.env_checker`; failure scenarios visibly degrade QoS in a smoke plot
- [ ] Phase 2: GRU predictor reaches F1 ≥ 0.7 on validation (onset subset) for scenarios with precursors
- [ ] Phase 3: reactive DQN beats shortest-path failover on PDR in at least the node-failure scenario
- [ ] Phase 4: PRESTO-R switches less than λ = 0 predictive DQN at similar PDR on validation seeds
- [ ] Phase 5: full grid done; `check_success.py` output saved
- [ ] Phase 6: `make reproduce` regenerates every figure and table
- [ ] Phase 7 (optional): NS-3 ranking matches the Python simulator for at least one scenario
