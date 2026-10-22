"""Train the GRU or logistic-regression link-risk predictor."""

import argparse
import copy
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from presto.config import load_config
from presto.predictor.features import best_f1_threshold
from presto.predictor.gru import LinkGRU, predict_probabilities
from presto.predictor.logreg import make_logreg
from presto.seeding import seed_everything


def _load_split(archive: Any, split: str, feature: str) -> np.ndarray:
    key = f"{split}_{feature}"
    if key not in archive:
        raise ValueError(f"Dataset is missing required array {key!r}")
    return archive[key]


def _validation_threshold(
    labels: np.ndarray, probabilities: np.ndarray, onset: np.ndarray
) -> float:
    eligible = onset.astype(bool)
    if not np.any(eligible):
        raise ValueError("Validation data contains no onset samples")
    return best_f1_threshold(labels[eligible], probabilities[eligible])


def train_logreg(
    cfg: dict, dataset_path: str | Path, output_path: str | Path, seed: int = 0
) -> Path:
    """Fit logistic regression on training summaries and tune its threshold on validation."""
    with np.load(dataset_path, allow_pickle=False) as data:
        train_x = _load_split(data, "train", "summaries")
        train_y = _load_split(data, "train", "labels")
        val_x = _load_split(data, "val", "summaries")
        val_y = _load_split(data, "val", "labels")
        val_onset = _load_split(data, "val", "onset")
        model = make_logreg({**cfg, "seed": seed})
        model.fit(train_x, train_y)
        val_probs = model.predict_proba(val_x)[:, 1]
        threshold = _validation_threshold(val_y, val_probs, val_onset)
        checkpoint = {
            "model": model,
            "model_type": "logreg",
            "threshold": threshold,
            "cfg": cfg,
        }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(checkpoint, output)
    return output


def train_gru(
    cfg: dict,
    dataset_path: str | Path,
    output_path: str | Path,
    seed: int = 0,
    device: str = "cpu",
) -> Path:
    """Fit a GRU with train-only normalization and validation-onset PR-AUC stopping."""
    torch.set_num_threads(1)
    seed_everything(seed)
    target_device = torch.device(device)
    with np.load(dataset_path, allow_pickle=False) as data:
        train_x = _load_split(data, "train", "windows").astype(np.float32, copy=False)
        train_y = _load_split(data, "train", "labels").astype(np.float32, copy=False)
        val_x = _load_split(data, "val", "windows").astype(np.float32, copy=False)
        val_y = _load_split(data, "val", "labels").astype(np.int8, copy=False)
        val_onset = _load_split(data, "val", "onset").astype(bool, copy=False)

        if not np.any(train_y == 1):
            raise ValueError("Training data contains no positive labels")
        train_cfg = cfg["predictor"]["train"]
        model_cfg = cfg["predictor"]["model"]
        mean = train_x.mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
        std = train_x.std(axis=(0, 1), dtype=np.float64).astype(np.float32)
        std[std < 1e-6] = 1.0
        model_kwargs = {
            "n_features": int(train_x.shape[-1]),
            "hidden": int(model_cfg["hidden"]),
            "layers": int(model_cfg["layers"]),
            "dropout": float(model_cfg["dropout"]),
        }
        model = LinkGRU(**model_kwargs).to(target_device)
        labels_tensor = torch.from_numpy(train_y)
        training_data = TensorDataset(torch.from_numpy(train_x), labels_tensor)
        loader = DataLoader(
            training_data,
            batch_size=int(train_cfg["batch"]),
            shuffle=True,
            generator=torch.Generator().manual_seed(seed),
            num_workers=0,
        )
        pos_weight_config = train_cfg.get("pos_weight", "auto")
        if pos_weight_config == "auto":
            positives = float(np.count_nonzero(train_y))
            pos_weight = float((len(train_y) - positives) / positives)
        else:
            pos_weight = float(pos_weight_config)
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight], dtype=torch.float32, device=target_device)
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=float(train_cfg["lr"]))
        mean_tensor = torch.as_tensor(mean, dtype=torch.float32, device=target_device)
        std_tensor = torch.as_tensor(std, dtype=torch.float32, device=target_device)
        if not np.any(val_y[val_onset] == 1):
            raise ValueError("Validation onset data contains no positive labels")

        best_score = -np.inf
        best_state = None
        best_threshold = 0.5
        patience = int(train_cfg["patience"])
        wait = 0
        epochs = int(train_cfg["epochs"])
        for epoch in range(epochs):
            model.train()
            for batch_x, batch_y in loader:
                batch_x = (batch_x.to(target_device) - mean_tensor) / std_tensor
                batch_y = batch_y.to(target_device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

            val_probabilities = predict_probabilities(model, val_x, mean, std, target_device)
            val_onset_y = val_y[val_onset]
            val_onset_probs = val_probabilities[val_onset]
            score = float(average_precision_score(val_onset_y, val_onset_probs))
            threshold = best_f1_threshold(val_onset_y, val_onset_probs)
            if score > best_score:
                best_score = score
                best_threshold = threshold
                best_state = copy.deepcopy(model.state_dict())
                wait = 0
            else:
                wait += 1
            print(f"epoch {epoch + 1}/{epochs}: validation onset PR-AUC={score:.4f}")
            if wait >= patience:
                break

        if best_state is None:
            raise RuntimeError("GRU training did not produce a valid checkpoint")
        checkpoint = {
            "state_dict": best_state,
            "mean": mean,
            "std": std,
            "model_kwargs": model_kwargs,
            "cfg": cfg,
            "threshold": best_threshold,
            "validation_onset_pr_auc": best_score,
            "seed": seed,
        }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/predictor/gru.yaml")
    parser.add_argument("--data")
    parser.add_argument("--out")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    args = parser.parse_args()
    cfg = load_config(args.config, args.overrides)
    data_path = Path(
        args.data
        or Path(cfg["paths"]["data"])
        / f"predictor_W{cfg['predictor']['window']}_H{cfg['predictor']['horizon']}.npz"
    )
    extension = "joblib" if cfg["predictor"]["type"] == "logreg" else "pt"
    output = (
        Path(args.out)
        if args.out
        else Path(cfg["paths"]["models"])
        / "predictor"
        / (
            f"{cfg['predictor']['type']}_W{cfg['predictor']['window']}_"
            f"H{cfg['predictor']['horizon']}.{extension}"
        )
    )
    if cfg["predictor"]["type"] == "logreg":
        saved = train_logreg(cfg, data_path, output, seed=args.seed)
    elif cfg["predictor"]["type"] == "gru":
        saved = train_gru(cfg, data_path, output, seed=args.seed, device=args.device)
    else:
        raise ValueError(f"Unsupported predictor type: {cfg['predictor']['type']}")
    print(f"Saved predictor model to {saved}")


if __name__ == "__main__":
    main()
