# Phase 6: Figures, Report and Reproducibility

**Goal:** turn the Phase 5 results into a written report and figures that can be regenerated from the repository with one command, and package the code so a reviewer can run it.

**Estimated effort:** 1–2 days · **Depends on:** Phase 5 · **PRD:** FR-10.4, FR-10.5, NFR-6, success criterion 5

---

## 6.1 Figures — `presto/eval/plots.py`

Each figure has its own function and is saved as both PNG (300 dpi) and PDF under `results/figures/`. Use one fixed colour per method in every figure, defined once in a `METHOD_STYLE` dict, and a consistent method order.

| # | Figure | Content | Shows |
| --- | --- | --- | --- |
| F1 | Architecture | Redrawn from the Assessment-8 diagram | System overview |
| F2 | Predictor quality | Precision–recall curves (GRU vs logistic regression), onset subset; reliability diagram as a second panel | Predictor is useful and calibrated |
| F3 | Risk trace | Predicted risk vs oracle for one link around a precursor failure, with the failure tick marked | What the predictor "sees" |
| F4 | Main results | Grouped bars per scenario: PDR, recovery time, switches (3 panels), error bars = std over seeds, NSFNET | Headline comparison |
| F5 | Trade-off | x = route switches per episode, y = PDR; one point per λ for PRESTO-R (connected), single points for baselines; test seeds, `mixed` | Why the switch penalty matters |
| F6 | Event trace | For one episode: max risk on active path, active path index, path loss, for reactive / predictive / PRESTO-R (Phase 4 diagnostic, regenerated on a test seed) | Proactive vs reactive behaviour |
| F7 | Horizon sweep | H on x; predictor F1 and PRESTO-R proactive rate on two y-panels | Choosing H |
| F8 | Learning curves | Validation reward and switches vs timesteps for each learned method (mean ± band over seeds) | Training is stable and budgets are equal |
| F9 | RGG results | Same as F4 for the RGG topology | Results hold on a second topology |

Rules for every figure: axis labels with units, no titles that repeat the caption, readable at the width of one column of an A4 page, colour-blind-safe palette, and error bars defined in the caption.

## 6.2 Tables

| # | Table | Source |
| --- | --- | --- |
| T1 | Simulation and training parameters | resolved configs |
| T2 | Predictor metrics (all and onset; precursor vs sudden) | `results/predictor/metrics.csv` |
| T3 | Main results, NSFNET (mean ± std per method × scenario × metric) | `summary_nsfnet.csv` |
| T4 | Paired tests: PRESTO-R vs each baseline (median diff, corrected p, effect size) | `paired_tests.csv` |
| T5 | Ablation 2×2 (prediction × penalty) on `mixed` | `summary_nsfnet.csv` |
| T6 | Robustness checks | `robustness.csv` |
| T7 | Success criteria PASS/FAIL | `success_criteria.md` |

`presto/eval/report.py` renders these tables as markdown from the CSVs. **No number is typed by hand anywhere in the report.**

## 6.3 Report — `results/report.md`

`report.py` fills a template (`presto/eval/report_template.md`) with tables and figure links. You then write the prose sections around them. Structure:

1. **Introduction:** problem, gap, contributions (match Assessment-8 sections 2 and 4).
2. **Related work:** short version of the Assessment-8 literature table.
3. **System design:** architecture (F1), MDP, reward and predictor (match Assessment-8 section 5).
4. **Experimental setup:** simulator model, scenarios, topologies, baselines, metrics definitions, seeds and statistics (T1).
5. **Results:**
    - 5.1 Predictor (F2, F3, T2)
    - 5.2 Main comparison (F4, T3, T4)
    - 5.3 Ablation: prediction vs penalty (T5, F5)
    - 5.4 Behaviour during failures (F6)
    - 5.5 Sensitivity to H and λ (F7, F5)
    - 5.6 Second topology and robustness (F9, T6)
6. **Discussion and limitations:** fluid model rather than packets; single managed flow; synthetic failures with designed precursors; predictor trained on failover-policy data; RQSMR-inspired rather than re-implemented baseline; sudden failures not predictable.
7. **Conclusion and future work:** multi-flow agents, multipath splitting, NS-3/real traces, online predictor updates.
8. **Success criteria** (T7) and **reproducibility** instructions.

## 6.4 Reproducibility

### Reproduction commands

The repository Makefile exposes `quick`, `metrics`, `tables`, `success`, `figures`,
`report`, and `reproduce-results`. GNU Make is optional; run the equivalent Python
modules directly on systems without Make. The full training/grid reproduction still
requires the saved experiment models and data described in the README.

```makefile
quick: tests + simulator smoke episode
metrics: aggregate raw Phase 5 episode logs
tables: regenerate summary, paired, robustness, lambda, and horizon tables
success: recompute PRD checks and rerun the reproducibility cell
figures: regenerate PNG/PDF figures from saved data, models, and training logs
report: render results/report.md and results/report.pdf from generated tables and metrics
reproduce-results: success + figures + report
```

### Release checklist

- [x] Fresh virtual environment + `pip install -r requirements.txt` + quick check succeeds (tested on Windows 11 with Python 3.12.4; 74 tests pass, smoke episode generated; GNU Make unavailable, documented Python equivalent validated)
- [x] `make figures` / `python -m presto.eval.plots` regenerates all nine PNG/PDF figures from saved results and model artifacts
- [x] Trained models (88 curated predictor checkpoints and best agents with configs, 12.61 MB) packaged into `models/presto_r_models_v1.0.zip` with verified SHA-256 manifest `models/SHA256SUMS.txt`; external publishing pending remote repository configuration
- [x] README filled in: setup, quick start, full reproduction, project structure, model bundle verification, and report/figure commands
- [x] `results/report.md` exported to submission-ready PDF `results/report.pdf` (12 pages, 1.63 MB, 8 figures and 7 tables embedded)
- [x] Version tag `v1.0` on the release commit

## 6.5 Mapping to Assessment-8 deliverables

| Assessment-8 section | Updated by |
| --- | --- |
| Abstract | Add one sentence with the headline numbers from T3 |
| Literature review | Unchanged |
| Problem, gaps, objectives | Mark each objective as met or not, using T7 |
| Architecture and methodology | Replace the planned values with the final ones (λ*, H, W, k, steps) from T1 |
| New: Results and Discussion | Sections 5–6 of the report |

---

## Exit criteria

- [x] All core figures (F1–F9) and tables (T1–T7) generated; the missing `reactive_dqn_pen` ablation cell is called out rather than fabricated
- [x] Report written, including limitations and failed criteria
- [x] Focused and full test suites pass, and the simulator smoke episode runs locally
- [x] `python -m presto.eval.plots` regenerates all nine PNG/PDF figure pairs from saved results and required model artifacts
- [x] Tagged release `v1.0` with models and hashes

## Pitfalls

- **Numbers in prose drifting from tables.** Generate key sentences ("PRESTO-R reduced switches by X% vs …") with `report.py` too.
- **Over-claiming.** Simulator results are evidence for the method in this model, not proof for real networks. Say so plainly in the limitations section.
- **Figures only readable on screen.** Check every figure printed in black and white, at the size it will appear.
