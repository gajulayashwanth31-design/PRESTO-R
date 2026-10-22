"""Interpretable per-link logistic-regression predictor."""

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler


def make_logreg(cfg: dict) -> Pipeline:
    """Build the standardized logistic-regression baseline."""
    model_cfg = cfg["predictor"]["model"]
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=float(model_cfg["C"]),
            class_weight=model_cfg.get("class_weight", "balanced"),
            max_iter=2000,
            random_state=int(cfg.get("seed", 0)),
        ),
    )
