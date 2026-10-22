# Phase 2: Link Degradation Predictor

**Goal:** a model that reads the last W ticks of a link's telemetry and outputs p(l), the probability that the link becomes bad within the next H ticks. It is plugged into the environment as `LearnedRisk`.

**Estimated effort:** 2–3 days · **Depends on:** Phase 1 · **Runs in parallel with:** Phase 3 · **PRD:** FR-4

---

## 2.1 Label definition

Link *l* is **bad** at tick *t* if any of these hold, using **true** (not noisy) values:

```
up[t, l] == 0
or loss_true[t, l] > 0.05
or delay_true[t, l] > 3 · base_delay[l]
```

**Label** for a sample at tick *t*: `y = 1` if the link is bad at any tick in `(t, t+H]`, else 0.

Two evaluation subsets matter:

| Subset | Definition | Why |
| --- | --- | --- |
| All samples | every (link, t) | Overall quality |
| **Onset** | link is **not** bad at *t* | The useful case: predicting a failure that has not happened yet. Report this as the headline number |

A link that is already down is easy to "predict". Only the onset subset shows whether the model foresees failures.

## 2.2 Config — `configs/predictor/gru.yaml`

```yaml
predictor:
  type: gru
  window: 10             # W
  horizon: 5             # H
  bad_loss: 0.05
  bad_delay_factor: 3.0
  data:
    scenario_mix: [congestion, node_failure, fluctuation, mixed]
    topologies: [nsfnet, rgg]
    episodes_per_combo: 80      # → 4 × 2 × 80 = 640 episodes
    neg_subsample: 0.3          # keep 30% of negatives to balance classes
    split_by_episode: [0.7, 0.15, 0.15]
  model:
    hidden: 32
    layers: 1
    dropout: 0.1
  train:
    lr: 1.0e-3
    batch: 512
    epochs: 30
    patience: 5                 # early stopping on val PR-AUC (onset subset)
    pos_weight: auto            # = n_neg / n_pos after subsampling
```

`configs/predictor/logreg.yaml` is the same file with `type: logreg` and `model: {C: 1.0, class_weight: balanced}`.

## 2.3 Tasks

### Task 2.3.1 — `presto/predictor/dataset.py`

```python
def collect_episode(cfg: dict, scenario_seed: int) -> dict:
    """Run one episode with the shortest-path-failover policy (Phase 3 heuristic, or
    a simple 'path 0, else first valid path' stand-in until Phase 3 lands).
    Returns:
      windows   (N, W, F)  float32   telemetry windows (from Telemetry.window())
      summaries (N, 4F+1)  float32   Telemetry.summary() rows
      labels    (N,)       int8
      onset     (N,)       bool      link not bad at t
      meta      (N, 3)     int32     [scenario_seed, t, link_id]
    one row per (link, tick) after warm-up."""

def build_dataset(cfg: dict, out_path: str) -> None:
    """Loop over scenario_mix × topologies × episodes. Seeds are drawn from
    [0, seeds.train_max) only. Assign each EPISODE to train/val/test
    (never split rows of one episode). Subsample negatives in TRAIN only.
    Save data/predictor_W{W}_H{H}.npz plus a JSON with class balance per split."""
```

Labels come from `sim.schedules` and the true link stats, which you must record during the rollout. Store true loss and delay arrays of shape (T, E) per episode, then compute labels in one vectorised pass.

**Leakage guard:** the window for tick *t* covers ticks `t−W+1 … t`. The label covers `t+1 … t+H`. Add a unit test that rebuilds one sample by hand and checks these ranges.

### Task 2.3.2 — `presto/predictor/logreg.py`

```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

def make_logreg(cfg) -> Pipeline:
    return make_pipeline(StandardScaler(),
                         LogisticRegression(C=cfg["model"]["C"],
                                            class_weight="balanced", max_iter=2000))
```

Input: `summaries` (last, mean, slope and max of each raw feature, plus flip count). This is the interpretable baseline, and its coefficients are worth a sentence in the report.

### Task 2.3.3 — `presto/predictor/gru.py`

```python
class LinkGRU(nn.Module):
    """Shared across links: input one link's window (W, F), output a logit."""
    def __init__(self, n_features=5, hidden=32, layers=1, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(n_features, hidden, layers, batch_first=True,
                          dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 1))
    def forward(self, x):                 # x: (B, W, F)
        _, h = self.gru(x)
        return self.head(h[-1]).squeeze(-1)   # logits (B,)
```

Because the model is shared across links and only sees per-link features, it works on any topology. That is why the dataset mixes NSFNET and RGG.

### Task 2.3.4 — `presto/predictor/train.py`

```bash
python -m presto.predictor.train --config configs/predictor/gru.yaml
python -m presto.predictor.train --config configs/predictor/logreg.yaml
```

For the GRU:

- Normalise inputs: compute per-feature mean and std on the train split, and save them with the model.
- Loss: `BCEWithLogitsLoss(pos_weight=…)`.
- Adam optimiser; early stopping on **validation PR-AUC (onset subset)**.
- Save `models/predictor/gru_W10_H5.pt` containing `{state_dict, mean, std, model_kwargs, cfg}`.

For logistic regression, save with `joblib` to `models/predictor/logreg_W10_H5.joblib`.

### Task 2.3.5 — `presto/predictor/evaluate.py`

Report for the test split, for both "all" and "onset" subsets:

| Metric | Notes |
| --- | --- |
| PR-AUC | Main metric; robust to class imbalance |
| ROC-AUC | For completeness |
| Precision, recall, F1 | At the threshold that maximises F1 on **validation**, applied to test |
| Brier score | Calibration; the agent uses the raw probability, so calibration matters |
| Lead time | For true failures caught by the model, how many ticks before the bad tick did p first exceed the threshold? Report the median |
| Breakdown | F1 per scenario type, and for precursor vs sudden failures |

Write `results/predictor/metrics.csv` and a reliability diagram `results/predictor/calibration.png`.

**Expected pattern:** high recall on precursor failures, near zero on sudden ones. That is correct behaviour, not a bug: sudden failures are unpredictable by design (20% of failures). State it in the report.

### Task 2.3.6 — `LearnedRisk` in `presto/risk.py`

```python
class LearnedRisk(RiskProvider):
    def __init__(self, path: str, device="cpu"):
        ckpt = torch.load(path, map_location=device)
        self.model = LinkGRU(**ckpt["model_kwargs"]); self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval(); self.mean, self.std = ckpt["mean"], ckpt["std"]
    @torch.no_grad()
    def __call__(self, telemetry, t):
        x = (telemetry.window() - self.mean) / self.std          # (E, W, F)
        return torch.sigmoid(self.model(torch.as_tensor(x, dtype=torch.float32))).numpy()
```

Also add `LogRegRisk` with the same interface, using `telemetry.summary()`.

**Speed:** one GRU call for all E links per step. Measure the environment with `LearnedRisk`; it should stay ≥ 500 steps/s. If it is slower, call `torch.set_num_threads(1)`, which is usually faster for tiny batches.

### Task 2.3.7 — Calibration check with the oracle

Run 20 validation episodes with `OracleRisk` and with `LearnedRisk`, and plot both risk traces for one link that fails. This is the clearest picture for the report of what the predictor does.

## 2.4 Tests

| File | Checks |
| --- | --- |
| `tests/test_dataset.py` | Window and label tick ranges do not overlap (leakage guard); splits share no `scenario_seed`; negatives are subsampled only in train |
| `tests/test_predictor.py` | GRU forward pass has shape (B,); `LearnedRisk` output has shape (E,) and lies in [0, 1]; save/load round-trip gives identical outputs |

---

## Exit criteria

- [x] Dataset built; class balance logged (6.8% raw training positives before subsampling)
- [x] Logistic regression and GRU trained; held-out metrics written
- [x] **GRU precursor-onset F1 ≥ 0.7 on test** (0.748 at the threshold tuned on validation precursor-onset samples)
- [x] GRU beats logistic regression on onset PR-AUC (0.495 vs 0.266)
- [x] `LearnedRisk` plugged into `RoutingEnv`; environment speed 728 steps/s over 10,000 steps
- [x] Risk-trace figure for one failure saved

Validation artifacts: `results/predictor/metrics.csv`,
`results/predictor/risk_trace.png`, and `models/predictor/gru_W10_H5.pt`.
Sudden-failure ROC-AUC is 0.497 (near chance), consistent with sudden failures being
unpredictable; subgroup F1 uses a separately validation-tuned threshold.

## Pitfalls

- **Splitting rows randomly.** Neighbouring ticks of one episode are almost identical. A random row split leaks and inflates scores. Always split by episode.
- **Headline metric on all samples.** Already-failed links dominate the positives and make the model look great. Use the onset subset.
- **F1 below 0.7.** First check that precursors are visible in the observed telemetry: plot `delay_rel` and `loss` before failures. If they are drowned by noise, reduce `measurement.*_noise` or increase `degr_loss`. Then increase W to 15. Change one thing at a time and record it.
- **Distribution shift.** The dataset is collected under shortest-path failover, but the agent will route differently. The per-link model mostly sees link-local signals, so this is minor. Mention it as a limitation; optionally retrain on data from a trained agent (DAgger-style) if time allows.
