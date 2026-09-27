"""Black-box model training and prediction utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, log_loss, roc_auc_score
from xgboost import XGBClassifier


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cfg: dict[str, Any],
    seed: int,
) -> XGBClassifier:
    params = dict(cfg["xgboost"])
    params["random_state"] = seed
    model = XGBClassifier(**params)
    model.fit(X_train, y_train)
    return model


def save_model(model: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: str | Path) -> Any:
    return joblib.load(path)


def predict_proba_df(model: Any, X: pd.DataFrame, prefix: str = "prob") -> pd.DataFrame:
    proba = model.predict_proba(X)
    columns = [f"{prefix}_{int(cls)}" for cls in model.classes_]
    return pd.DataFrame(proba, columns=columns, index=X.index)


def predict_labels(model: Any, X: pd.DataFrame) -> np.ndarray:
    return model.predict(X).astype(int)


def expected_calibration_error(y_true: np.ndarray, prob_pos: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(prob_pos, bins[1:-1], right=True)
    ece = 0.0
    for b in range(n_bins):
        mask = bin_ids == b
        if not np.any(mask):
            continue
        conf = float(np.mean(prob_pos[mask]))
        acc = float(np.mean(y_true[mask] == (prob_pos[mask] >= 0.5).astype(int)))
        ece += float(np.mean(mask)) * abs(acc - conf)
    return ece


def classification_metrics(y_true: np.ndarray, proba: np.ndarray, labels: list[int]) -> dict[str, float]:
    y_pred = np.asarray(labels)[np.argmax(proba, axis=1)]
    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "log_loss": float(log_loss(y_true, proba, labels=labels)),
    }
    if len(labels) == 2:
        pos_idx = labels.index(1) if 1 in labels else 1
        prob_pos = proba[:, pos_idx]
        metrics["auroc"] = float(roc_auc_score(y_true, prob_pos))
        metrics["brier"] = float(brier_score_loss(y_true, prob_pos))
        metrics["ece_10bin"] = expected_calibration_error(y_true, prob_pos)
    else:
        metrics["auroc_ovr_macro"] = float(roc_auc_score(y_true, proba, multi_class="ovr", average="macro"))
    return metrics


def prediction_frame(
    df: pd.DataFrame,
    split_col: str,
    y_true: pd.Series,
    model: Any,
    X: pd.DataFrame,
) -> pd.DataFrame:
    proba_df = predict_proba_df(model, X)
    pred = predict_labels(model, X)
    out = pd.DataFrame(
        {
            "row_id": df["row_id"].values,
            "split": df[split_col].values,
            "y_true": y_true.values,
            "m_label": pred,
        },
        index=df.index,
    )
    return pd.concat([out, proba_df], axis=1)
