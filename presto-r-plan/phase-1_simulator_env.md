# Phase 1: Network Simulator and Gymnasium Environment

**Goal:** a fast, deterministic network simulator with realistic failure precursors, wrapped as a Gymnasium environment whose observation and reward follow the PRD exactly.

**Estimated effort:** 4–5 days · **Depends on:** Phase 0 · **PRD:** FR-1, FR-2, FR-3, FR-5, FR-7

This is the most important phase. Every later result depends on the simulator being correct, and on failures being *predictable but not trivial*.

---

## 1.1 Simulation model

Discrete time, 1 step = 1 tick (100 ms). Traffic is a fluid rate (units per tick), not individual packets.

### Per-link state (true, hidden from the agent)

| Variable | Meaning |
| --- | --- |
| `C[l]` | nominal capacity (units/tick) |
| `d0[l]` | base propagation delay (ms) |
| `h[l]` ∈ [0, 1] | health; set by the failure process; **hidden** |
| `up[l]` ∈ {0, 1} | link is up |
| `q[l]` | queue backlog (units), 0 ≤ q ≤ Qmax |

### Per-tick update for link `l`

```
load      = Σ rates of background flows on l  +  managed rate if l on managed path
cap_eff   = C[l] · h[l]                         (0 if link down)
if not up[l]:
    q = 0;  loss = 1;  delay = d0[l] + timeout_ms
else:
    q_raw    = q + load − cap_eff
    overflow = max(0, q_raw − Qmax)
    q        = clip(q_raw, 0, Qmax)
    loss     = clip( overflow / max(load, ε)  +  base_loss  +  (1 − h)·degr_loss , 0, 1)
    delay    = d0[l] + (q / max(cap_eff, ε)) · tick_ms        # queueing delay
util = load / C[l]
```

### Path QoS for the managed flow on path `P`

```
path_delay = Σ_{l∈P} delay[l]
path_loss  = 1 − Π_{l∈P} (1 − loss[l])          # = 1 if any link down
delivered  = rate · (1 − path_loss)
hops       = |P|
```

### Observations (what the agent and predictor see)

Each link reports `delay`, `loss`, `queue`, `util` and `up`, with Gaussian measurement noise (σ from config) on the continuous values. **Health `h` is never observed.**

---

## 1.2 Configs

`configs/topo/nsfnet.yaml`:

```yaml
topology:
  name: nsfnet
  capacity: 10.0          # units per tick, all links
  base_delay_ms: [2, 12]  # uniform per link, drawn once with topology seed
  qmax: 20.0
```

`configs/topo/rgg.yaml`:

```yaml
topology:
  name: rgg
  n_nodes: 24
  radius: 0.32
  capacity: 10.0
  delay_per_unit_dist_ms: 30   # base delay = distance × this
  qmax: 20.0
```

`configs/scenario/mixed.yaml` (the other scenarios are this file with some event rates set to 0):

```yaml
defaults: [base.yaml, topo/nsfnet.yaml]
traffic:
  n_background: 20
  bg_rate_lognormal: [0.6, 0.5]   # mean, sigma of log-rate
  diurnal_amp: 0.3
  diurnal_period: 100             # ticks
  noise_std: 0.05
  managed_rate: 2.0
link:
  base_loss: 0.001
  degr_loss: 0.15                 # extra loss at h = 0
  timeout_ms: 200
measurement:
  delay_noise_rel: 0.05
  loss_noise_abs: 0.005
  queue_noise_rel: 0.05
scenario:
  name: mixed
  events_per_episode:             # Poisson means
    congestion: 1.0
    link_failure: 1.0
    node_failure: 0.3
    fluctuation: 0.5
  target_on_candidate_prob: 0.7   # bias events onto links of the k candidate paths
  precursor_len: [3, 10]          # ticks of gradual degradation before failure
  no_precursor_prob: 0.2          # share of failures that happen suddenly
  down_duration: [15, 60]
  congestion_multiplier: [2.5, 4.0]
  congestion_duration: [10, 40]
  fluct_period: [6, 20]
  fluct_depth: [0.3, 0.7]
  h_min_precursor: 0.35
paths:
  k: 4
```

| Scenario file | congestion | link_failure | node_failure | fluctuation |
| --- | --- | --- | --- | --- |
| `steady.yaml` | 0 | 0 | 0 | 0 |
| `congestion.yaml` | 2.0 | 0 | 0 | 0 |
| `node_failure.yaml` | 0 | 0.5 | 1.0 | 0 |
| `fluctuation.yaml` | 0 | 0 | 0 | 1.5 |
| `mixed.yaml` | 1.0 | 1.0 | 0.3 | 0.5 |

---

## 1.3 Tasks

### Task 1.3.1 — `presto/sim/topology.py`

```python
import networkx as nx, numpy as np

NSFNET_EDGES = [(0,1),(0,2),(0,7),(1,2),(1,3),(2,5),(3,4),(3,10),(4,5),(4,6),(5,9),
                (5,12),(6,7),(7,8),(8,9),(8,11),(8,13),(10,11),(10,13),(11,12),(12,13)]

def load_topology(cfg: dict, rng: np.random.Generator) -> nx.Graph:
    """Return a connected undirected graph. Every edge has attributes:
    capacity, base_delay, link_id (0..E-1, stable order)."""
    ...

def link_index(G: nx.Graph) -> dict[tuple[int, int], int]:
    """Canonical (min(u,v), max(u,v)) -> link_id."""
    ...
```

- NSFNET: the 14-node, 21-link variant above is the one most often used in routing papers. Check it against the reference you cite.
- RGG: `nx.random_geometric_graph(n, radius, seed=...)`. Retry with the next seed until it is connected and every node has degree ≥ 2, so that alternative paths exist.
- Assign `link_id` in sorted canonical-edge order so ids are stable across runs.

### Task 1.3.2 — `presto/sim/traffic.py`

```python
from dataclasses import dataclass

@dataclass
class BgFlow:
    src: int; dst: int; base_rate: float; phase: float
    path_links: list[int]          # current route (link ids)

class BackgroundTraffic:
    def __init__(self, G, cfg, rng): ...
    def rates(self, t: int, multipliers: np.ndarray) -> np.ndarray:
        """Rate of each bg flow at tick t.
        rate = base · (1 + amp·sin(2πt/period + phase)) · mult_on_path + noise, clipped ≥ 0.
        multipliers[l] is the congestion multiplier active on link l (1.0 normally);
        a flow takes the max multiplier over its path."""
    def reroute(self, up: np.ndarray) -> None:
        """Shortest path by base_delay on the subgraph of live links.
        A flow with no path gets path_links = [] and sends nothing."""
    def link_load(self, t, multipliers) -> np.ndarray:  # shape (E,)
        ...
```

Background flows are re-routed only when the set of live links changes. Keep the previous `up` vector and compare it each tick. Re-routing every tick is slow and unrealistic.

### Task 1.3.3 — `presto/sim/failures.py`

```python
from dataclasses import dataclass

@dataclass
class Event:
    kind: str              # "congestion" | "link_failure" | "node_failure" | "fluctuation"
    links: list[int]       # affected link ids (node failure = all incident links)
    start: int             # tick at which the link goes down / congestion starts
    precursor: int         # ticks of degradation before `start` (0 = sudden)
    duration: int
    params: dict           # multiplier, period, depth, ...

class ScenarioGenerator:
    def __init__(self, cfg: dict, rng): ...
    def generate(self, G, episode_len: int, candidate_links: set[int],
                 protected_nodes: set[int]) -> list[Event]:
        """Number of each event kind ~ Poisson(mean). Start times uniform in
        [warmup + 15, warmup + episode_len − 10]. With prob target_on_candidate_prob
        the target link is drawn from candidate_links, otherwise from all links.
        Node failures never pick protected_nodes (the managed flow's src/dst)."""

def build_schedules(events, E: int, T: int, cfg, rng) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Precompute, for ticks 0..T-1:
       health   h[T, E] ∈ [0, 1]
       up       u[T, E] ∈ {0, 1}
       cong_mult m[T, E] ≥ 1
    Ground-truth labels and metrics are later derived from these arrays."""
```

How each event shapes the schedules:

| Event | Effect on `h`, `up`, `mult` |
| --- | --- |
| link_failure (with precursor P) | for t in [start−P, start): h falls linearly from 1 to `h_min_precursor` (+ small noise); for t in [start, start+duration): up = 0; afterwards h = 1, up = 1 |
| link_failure (sudden) | same, with P = 0 |
| node_failure | apply link_failure to every incident link with the same timing |
| congestion | for t in [start, start+duration): mult = multiplier, ramped up over the first 3 ticks |
| fluctuation | for t in [start, start+duration): h = 1 − depth·(0.5 + 0.5·sin(2πt/period)) + noise; with probability 0.3 it ends in a link_failure |

Precomputing the full schedule at reset makes ground-truth labels exact and cheap: link *l* is "bad" at *t* if `up[t,l] == 0` or its true loss or delay crosses a threshold.

### Task 1.3.4 — `presto/sim/simulator.py`

```python
@dataclass
class LinkStats:          # true values for one tick, all shape (E,)
    delay: np.ndarray; loss: np.ndarray; queue: np.ndarray; util: np.ndarray; up: np.ndarray

class NetworkSimulator:
    def __init__(self, cfg: dict, seed: int): ...
    def reset(self, managed_src: int, managed_dst: int, candidate_links: set[int]) -> None:
        """Build traffic + failure schedules for this episode (uses self.seed streams)."""
    def step(self, managed_path_links: list[int]) -> LinkStats:
        """Advance one tick with the managed flow on the given path; returns TRUE stats."""
    def observe(self, stats: LinkStats) -> LinkStats:
        """Add measurement noise (noise stream). Clip to valid ranges. up is not noised."""
    def path_qos(self, stats: LinkStats, path_links: list[int]) -> dict:
        """{'delay', 'loss', 'delivered', 'hops', 'valid'} from TRUE stats."""
    @property
    def t(self) -> int: ...
    @property
    def schedules(self) -> dict: ...   # 'health', 'up', 'mult' arrays (for labels/metrics)
```

Rules:

- `NetworkSimulator` has **no** Gymnasium or RL code, so baselines, the predictor dataset builder and NS-3 validation can all reuse it.
- Reward and metrics use **true** QoS. The agent observes **noisy** values.
- Vectorise per-link updates with NumPy. Python loops over links are the main speed risk.

### Task 1.3.5 — `presto/telemetry.py`

```python
RAW_FEATURES = ["delay_rel", "loss", "queue_rel", "util", "up"]   # F = 5

class Telemetry:
    """Ring buffer of the last W observed ticks for every link."""
    def __init__(self, n_links: int, window: int, base_delay: np.ndarray, qmax: float): ...
    def reset(self) -> None: ...
    def push(self, obs: LinkStats) -> None:
        """delay_rel = delay / base_delay (≥ 1), queue_rel = queue / qmax."""
    def window(self) -> np.ndarray:
        """Shape (E, W, F), oldest first. Used by the GRU predictor."""
    def summary(self) -> np.ndarray:
        """Shape (E, F·4 + 1): per raw feature [last, mean, slope, max] + flip count of `up`.
        Used by logistic regression and by baselines."""
    def latest(self) -> np.ndarray:   # shape (E, F)
        ...
```

Compute the slope as a least-squares fit over the window. Precompute the regression weights once.

### Task 1.3.6 — `presto/risk.py` (interface only in this phase)

```python
class RiskProvider:
    def reset(self, sim: "NetworkSimulator") -> None: ...
    def __call__(self, telemetry: Telemetry, t: int) -> np.ndarray:   # (E,) in [0, 1]
        raise NotImplementedError

class ZeroRisk(RiskProvider):
    def __call__(self, telemetry, t): return np.zeros(telemetry.n_links)

class OracleRisk(RiskProvider):
    """Uses the true future schedule: 1 if the link is bad within (t, t+H], else 0.
    Debugging upper bound only — never used for reported PRESTO-R results."""
```

`LearnedRisk` comes in Phase 2.

### Task 1.3.7 — `presto/state.py`

```python
def candidate_paths(G, src, dst, k, lid) -> list[list[int]]:
    """Up to k loop-free paths in increasing base delay
    (nx.shortest_simple_paths(G, src, dst, weight='base_delay')), as lists of link ids."""

def choose_flow(G, rng, k, lid) -> tuple[int, int, list[list[int]]]:
    """Draw src/dst until at least k candidate paths exist and the first two
    share at most half of their links (so there is a real alternative)."""

PATH_FEATURES = ["delay_norm", "loss", "bneck_queue", "bneck_util", "hops_norm",
                 "max_risk", "mean_risk", "valid"]                     # 8 per path

def build_observation(latest: np.ndarray, risk: np.ndarray, paths: list[list[int]],
                      current: int, steps_since_switch: int, ref_delay: float,
                      max_hops: int, use_risk: bool, k: int) -> np.ndarray:
    """Per path (k rows, padded with zeros + valid=0 if fewer paths):
         delay_norm  = clip(Σ delay / ref_delay, 0, 3) / 3
         loss        = 1 − Π(1 − loss_l)
         bneck_queue = max queue_rel on path
         bneck_util  = clip(max util, 0, 2) / 2
         hops_norm   = hops / max_hops
         max_risk, mean_risk over path links   (omitted entirely if not use_risk)
         valid       = all links up (observed)
       then one-hot(current, k) and min(steps_since_switch, 20) / 20.
    Length: k·8 + k + 1 = 37 with risk, k·6 + k + 1 = 29 without (k = 4)."""
```

`ref_delay` is the base delay of candidate path 0. It is fixed per episode, so `delay_norm` ≈ 1/3 on an idle shortest path.

### Task 1.3.8 — `presto/reward.py`

```python
@dataclass
class RewardWeights:
    alpha: float = 1.0    # delay
    beta: float = 4.0     # loss
    gamma: float = 0.0    # throughput (see note)
    lam: float = 0.1      # route-switch penalty  (λ)
    mu: float = 0.05      # energy

def compute_reward(qos: dict, switched: bool, ref_delay: float, max_hops: int,
                   demand: float, w: RewardWeights) -> tuple[float, dict]:
    D = min(qos["delay"] / ref_delay, 3.0) / 3.0
    L = qos["loss"]
    T = qos["delivered"] / demand
    S = float(switched)
    E = qos["hops"] / max_hops
    r = -w.alpha*D - w.beta*L + w.gamma*T - w.lam*S - w.mu*E
    return r, {"D": D, "L": L, "T": T, "S": S, "E": E}
```

**Note on γ:** with one managed flow, `T = 1 − L` exactly, so a non-zero γ just double-counts loss. Keep γ = 0 by default and say so in the report. γ becomes meaningful in the multi-flow extension.

Optional realism (config `reward.switch_transient_loss`, default 0.0): on a switch tick, add this extra loss to the delivered traffic to model packet reordering and drops during the flow-table update.

### Task 1.3.9 — `presto/env.py`

```python
class RoutingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, cfg: dict, risk: RiskProvider | None = None, seed_pool=None):
        """seed_pool: callable rng -> (scenario_seed, scenario_name) for training, or an
        iterator over fixed (seed, scenario) pairs for evaluation. The scenario name selects
        the event rates from configs/scenario/<name>.yaml, so one env can mix scenarios."""
        self.use_risk = cfg["agent"].get("use_risk", True)
        self.k = cfg["paths"]["k"]
        obs_dim = self.k * (8 if self.use_risk else 6) + self.k + 1
        self.observation_space = gym.spaces.Box(0.0, 1.0, (obs_dim,), np.float32)
        self.action_space = gym.spaces.Discrete(self.k)
        ...

    def reset(self, *, seed=None, options=None):
        # 1. pick scenario seed + scenario (options["scenario_seed"] / options["scenario"]
        #    if given, else seed_pool). Return the same `info` keys as step().
        # 2. build sim, choose managed flow + k candidate paths
        # 3. sim.reset(...); telemetry.reset(); risk.reset(sim)
        # 4. warm-up: W ticks on path 0, pushing observations into telemetry
        # 5. current = 0, steps_since_switch = W
        return obs, info

    def step(self, action: int):
        switched = action != self.current
        # apply path, sim.step, observe → telemetry.push, risk(...)
        # qos from TRUE stats on the chosen path; reward
        # obs from OBSERVED stats
        info = {"t", "action", "switched", "qos", "reward_terms", "path_links",
                "scenario_seed", "scenario",
                "paths",       # list of k link-id lists (for heuristics)
                "current",     # active path index after this step
                "valid",       # bool per candidate path, from OBSERVED up flags
                "obs_link",    # (E, F) latest observed link stats
                "risk"}        # (E,) risk vector (zeros if no predictor)
        truncated = self.sim.t >= warmup + episode_len
        return obs, reward, False, truncated, info
```

Invalid actions are not masked. Choosing a path with a down link gives `loss = 1` for that tick, which the reward already punishes heavily.

---

## 1.4 Tests

| File | What it checks |
| --- | --- |
| `tests/test_topology.py` | NSFNET has 14 nodes and 21 links; RGG is connected and has minimum degree ≥ 2; link ids are stable for the same seed |
| `tests/test_simulator.py` | Idle link: queue stays 0, delay = base delay. Overloaded link: queue rises to Qmax, then loss > 0. Down link: loss = 1. Same seed gives identical stats arrays |
| `tests/test_failures.py` | With precursor P, `h` decreases monotonically over P ticks before `up` goes to 0. Sudden failure: no decrease. Node failure takes down all incident links. Protected nodes never fail |
| `tests/test_telemetry.py` | Window shape (E, W, F); oldest-first order; slope is positive on a rising series |
| `tests/test_state.py` | Observation length 37 (risk) / 29 (no risk); all values in [0, 1]; padding when fewer than k paths |
| `tests/test_reward.py` | λ term is applied only on a switch; λ = 0 removes it; reward terms sum to the reward |
| `tests/test_env.py` | `gymnasium.utils.env_checker.check_env(env)` passes; an episode truncates at exactly `episode_len` steps; same seed + same actions give identical rewards |

```python
# tests/test_env.py
from gymnasium.utils.env_checker import check_env
from presto.config import load_config
from presto.env import RoutingEnv

def test_check_env():
    cfg = load_config("configs/scenario/mixed.yaml", ["agent.use_risk=false"])
    check_env(RoutingEnv(cfg), skip_render_check=True)
```

## 1.5 Smoke script — `scripts/smoke_env.py`

Runs one mixed-scenario episode with a "always path 0" policy and saves `results/smoke/episode.png`. The figure has four stacked panels over time: path delay, path loss, max true health on the path, and event markers (shaded spans).

**Look at this plot before moving on.** You should see delay and loss rise *during* precursors and spike at failures. If QoS does not react to events, the simulator is broken, and nothing downstream will work.

Also time it: `python -m timeit` over 1,000 env steps. **Target ≥ 2,000 steps/second** on a laptop CPU without the learned predictor. At this speed, 300k training steps take a few minutes of simulation; the network update dominates.

---

## Exit criteria

- [ ] All Phase 1 tests pass
- [ ] `check_env` passes for both observation variants
- [ ] Smoke plot shows QoS degrading before and during failures, and recovering afterwards
- [ ] Environment runs at ≥ 2,000 steps/s (without learned risk)
- [ ] Two runs with the same seed and actions produce identical step logs
- [ ] For the `mixed` scenario on NSFNET, at least 60% of episodes have one or more events on a candidate path (log this; tune `target_on_candidate_prob` if not)

## Pitfalls

- **Leaking hidden state.** Never put `health` or the schedules into the observation. Only `OracleRisk` may read them.
- **Failures that never affect the managed flow.** Then every method looks the same. That is why events are biased onto candidate links.
- **Background traffic overloading everything.** If `steady` already shows loss > 2% on the shortest path, lower `n_background` or the rate mean.
- **Delay normalisation blowing up.** A down link's delay uses `timeout_ms`, and `delay_norm` is clipped at 3× the reference.
- **Determinism breaks** when you iterate over Python sets. Always sort link ids before use.
