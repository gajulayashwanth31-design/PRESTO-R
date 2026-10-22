# Phase 7 (Optional): NS-3 Validation

**Goal:** check that the method ranking found in the Python simulator also holds in a packet-level simulator. This is validation of the conclusion, not a second full study.

**Estimated effort:** 3+ days · **Depends on:** Phase 5 (start only after it is complete) · **Priority:** P2 · **PRD:** FR-9

Skip this phase if the deadline is tight. It is the most likely place to lose a week to tooling.

---

## 7.1 Scope

| In scope | Out of scope |
| --- | --- |
| NSFNET topology only | RGG topology |
| `node_failure` and `mixed` scenarios | All five scenarios |
| `sp_failover`, `reactive_dqn`, `presto_dqn` | Every method |
| Policies trained in Python, used **as is** | Retraining inside NS-3 |
| 30 test episodes per cell | 200 episodes |

**Success test:** the PDR and switch-count ordering of the three methods is the same as in the Python simulator, in at least one scenario. Exact values are expected to differ.

## 7.2 Setup

1. Install the NS-3 release that the current **ns3-gym** (OpenGym module) supports; check the ns3-gym README for the exact version pair. Version mismatches are the most common blocker.
2. Build NS-3 with the `opengym` contrib module and install the matching `ns3gym` Python package.
3. Verify the bundled ns3-gym example runs end to end before writing any project code.

Work on Linux or WSL2. Native Windows builds of NS-3 are painful.

## 7.3 Tasks

### Task 7.3.1 — NS-3 scenario script (`ns3/presto_scenario.cc`)

| Element | NS-3 implementation |
| --- | --- |
| Topology | 14 nodes, 21 `PointToPoint` links; data rate scaled from `capacity`, delay from the same `base_delay` values as the Python run (export them to JSON from the topology seed) |
| Queues | `DropTailQueue` sized to match `qmax` |
| Background traffic | `OnOffApplication` UDP flows on the same src/dst pairs and mean rates as the Python scenario seed |
| Managed flow | One UDP CBR flow between the episode's src and dst |
| Routing | Background flows: global routing, recomputed on link changes. Managed flow: `Ipv4StaticRouting` host routes installed along the path the agent chooses |
| Failures | Use the same event list (export `Event`s as JSON): precursors as gradually reduced `DataRate` plus `RateErrorModel` loss; failures as `SetDown()` on both net devices |
| Telemetry | Every step interval (100 ms): per-link queue length (queue disc trace), per-link drops, and per-link delay from `FlowMonitor` or a probe flow |

### Task 7.3.2 — OpenGym interface

- **Observation:** the same raw per-link telemetry `(E, F)` as Python, sent each step.
- **Action:** an integer path index.
- **Python side (`presto/ns3/ns3_env.py`):** a Gymnasium wrapper that receives the raw telemetry, pushes it into the **existing** `Telemetry` class, calls `LearnedRisk`, builds the observation with the **existing** `build_observation`, and computes the reward with the **existing** `compute_reward`. Reuse is the point; only the network itself is replaced.

### Task 7.3.3 — Feature alignment check

Before running policies, compare the distributions of `delay_rel`, `loss`, `queue_rel` and `util` between NS-3 and Python for the same scenario seed and the same fixed path, using histograms and summary statistics. If they differ a lot, rescale (for example, normalise queue by the NS-3 queue size). Otherwise the policies receive inputs they never saw in training. Document every scaling choice.

### Task 7.3.4 — Run and compare

Run the three methods on 30 shared scenario seeds per scenario. Produce:

- `results/ns3/summary.csv`: PDR, switches, recovery time, mean ± std
- `results/figures/F10_ns3_vs_python.png`: side-by-side bars for Python vs NS-3 per method

Compare rankings with the Kendall τ between the Python and NS-3 method orderings per metric.

## 7.4 Report addition

One subsection: "Packet-level validation". It contains the setup in a few sentences, F10, the ranking comparison, and an honest note on differences (packet effects, TCP/UDP behaviour, timing).

---

## Exit criteria

- [ ] ns3-gym example runs
- [ ] NSFNET scenario in NS-3 reproduces the event timeline from a Python scenario seed
- [ ] Feature alignment checked and documented
- [ ] Three methods compared on at least one scenario; ranking agreement reported

## Pitfalls

- **Spending the time before the core work is done.** Do not start until Phase 5's exit criteria pass.
- **Clock mismatch.** The NS-3 step interval must equal the Python tick (100 ms), or every time-based feature (slopes, H) shifts meaning.
- **Static routes and failures.** When a link on the managed path goes down, packets are dropped until the agent picks a new path. That is the intended behaviour; do not let global routing silently fix it.
