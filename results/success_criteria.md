# Phase 5 Success Criteria

| Check | Result | Details |
|---|---|---|
| PRESTO-R vs reactive in node_failure | FAIL | PDR 0.9667 vs 0.9662; recovery 9.440 vs 8.915; Holm p PDR=1, recovery=0.2617 |
| PRESTO-R vs reactive in fluctuation | FAIL | PDR 0.9749 vs 0.9740; recovery 7.844 vs 7.386; Holm p PDR=0.01042, recovery=1 |
| PRESTO-R vs predictive in mixed | PASS | switches 5.197 vs 20.988; PDR 0.9625 vs 0.9608 |
| PRESTO-R vs predictive in fluctuation | PASS | switches 5.557 vs 23.419; PDR 0.9749 vs 0.9726 |
| Steady-state delay | PASS | PRESTO-R delay 46.2766; limit 50.9369 (reactive × 1.10) |
| Predictor onset F1 | FAIL | GRU validation onset F1=0.5296; threshold=0.7000 |
| Reproducibility | PASS | Step and event parquet hashes match on a fresh rerun |
