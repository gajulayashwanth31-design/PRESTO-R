# Phase 0: Project Setup

**Goal:** a clean, installable repository with the configuration and seeding utilities that every later phase depends on.

**Estimated effort:** 0.5–1 day · **Depends on:** nothing · **PRD:** NFR-1, NFR-3, NFR-4, NFR-6, NFR-7

---

## 0.1 Create the repository

```bash
mkdir presto-r && cd presto-r
git init
python3.11 -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
mkdir -p configs/{topo,scenario,predictor,agent} presto/{sim,predictor,policies,agents,eval} \
         scripts tests data models results runs
touch presto/__init__.py presto/{sim,predictor,policies,agents,eval}/__init__.py
```

Use **Python 3.11**. NumPy 1.26 does not support 3.13, and 3.10–3.12 are all fine.

## 0.2 Pin dependencies

`requirements.txt`:

```text
numpy==1.26.4
pandas==2.2.2
scipy==1.13.1
networkx==3.3
gymnasium==1.0.0
stable-baselines3==2.4.0
sb3-contrib==2.4.0
torch==2.4.1
scikit-learn==1.5.1
matplotlib==3.9.2
pyyaml==6.0.2
tqdm==4.66.5
pyarrow==17.0.0
pytest==8.3.2
black==24.8.0
ruff==0.6.4
```

```bash
pip install -r requirements.txt
python -c "import gymnasium, stable_baselines3, torch, networkx; print('ok')"
```

For CPU-only torch (smaller download): `pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu` before the rest.

## 0.3 Tooling files

`pyproject.toml`:

```toml
[project]
name = "presto-r"
version = "0.1.0"
requires-python = ">=3.10,<3.13"

[tool.black]
line-length = 100

[tool.ruff]
line-length = 100

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"
```

`.gitignore`:

```text
.venv/
__pycache__/
*.pyc
data/
models/
runs/
*.egg-info/
```

`Makefile` (targets get filled in by later phases):

```makefile
.PHONY: test lint fmt
test:
	pytest
lint:
	ruff check presto tests
fmt:
	black presto tests
```

## 0.4 Config system — `presto/config.py`

One function loads a YAML file, merges any `defaults:` files it lists, and applies command-line overrides.

```python
import copy, yaml
from pathlib import Path

def _deep_merge(base: dict, new: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out

def _parse_value(s: str):
    # "0.1" -> 0.1, "true" -> True, "[1,2]" -> [1, 2]
    return yaml.safe_load(s)

def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    """Load YAML with `defaults: [other.yaml, ...]` (relative to configs/) then apply
    overrides like ["reward.lam=0.25", "agent.total_steps=100000"]."""
    path = Path(path)
    raw = yaml.safe_load(path.read_text()) or {}
    cfg = {}
    for d in raw.pop("defaults", []):
        cfg = _deep_merge(cfg, load_config(Path("configs") / d))
    cfg = _deep_merge(cfg, raw)
    for ov in overrides or []:
        key, val = ov.split("=", 1)
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = _parse_value(val)
    return cfg

def save_config(cfg: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(yaml.safe_dump(cfg, sort_keys=False))
```

`configs/base.yaml`:

```yaml
sim:
  tick_ms: 100
  episode_len: 200
  warmup: 10              # = telemetry window W
seeds:
  train_max: 5000         # training scenarios use seeds in [0, 5000)
  val: [5000, 5050]       # half-open range
  test: [10000, 10200]
paths:
  data: data
  models: models
  runs: runs
  results: results
```

## 0.5 Seeding — `presto/seeding.py`

Separate random streams stop one component's randomness from shifting another's. For example, adding a noise draw to telemetry must not change the failure schedule.

```python
import random, numpy as np, torch

STREAMS = {"topology": 0, "traffic": 1, "failures": 2, "noise": 3, "flow": 4}

def make_rng(seed: int, stream: str) -> np.random.Generator:
    return np.random.default_rng([seed, STREAMS[stream]])

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
```

## 0.6 Tests for this phase — `tests/test_config.py`

```python
from presto.config import load_config
from presto.seeding import make_rng

def test_override(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("a:\n  b: 1\n")
    cfg = load_config(p, ["a.b=2", "a.c=[1,2]"])
    assert cfg["a"]["b"] == 2 and cfg["a"]["c"] == [1, 2]

def test_streams_independent():
    a = make_rng(7, "failures").random(3)
    b = make_rng(7, "failures").random(3)
    c = make_rng(7, "noise").random(3)
    assert (a == b).all() and not (a == c).all()
```

## 0.7 README stub

Create `README.md` with sections that later phases fill in: *Setup*, *Quick start*, *Training*, *Experiments*, *Reproducing figures*, *Project structure*.

---

## Exit criteria

- [ ] `pip install -r requirements.txt` works in a fresh virtual environment
- [ ] `make test` passes (config and seeding tests)
- [ ] `make lint` reports no errors
- [ ] `configs/base.yaml` loads through `load_config`
- [ ] First commit pushed to the shared remote

## Pitfalls

- **Mixing gymnasium and SB3 versions.** SB3 2.4 is the first release that supports Gymnasium 1.0. Keep the pinned pair together.
- **Global NumPy seeding.** Do not rely on `np.random.seed` for reproducibility in the simulator. Always pass explicit generators.
- **Windows users:** use `make` via Git Bash, or run the commands from the Makefile directly.
