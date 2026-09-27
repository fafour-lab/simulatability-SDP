#!/usr/bin/env python3
"""Stage 02: train XGBoost black-box M and save predictions for all splits."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cgrr.config import load_config, output_dir, project_seed
from cgrr.data import (
    PreprocessorState,
    attach_splits,
    labels_to_name,
    load_dataset,
    transform_features,
    transform_labels,
)
from cgrr.model import classification_metrics, prediction_frame, save_model, train_xgboost
from cgrr.utils.io import read_json, write_csv, write_json
from cgrr.utils.logging import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/heloc_cgrr.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(output_dir(cfg, "logs", "02_train_blackbox.log"))
    seed = project_seed(cfg)

    split_path = output_dir(cfg, "splits", f"split_assignments_seed{seed}.csv")
    split_df = pd.read_csv(split_path)
    state = PreprocessorState.from_dict(read_json(output_dir(cfg, "preprocess", "preprocessor_state.json")))
    raw = attach_splits(load_dataset(cfg), split_df)
    X = transform_features(raw, state)
    y = transform_labels(raw, state)

    train_mask = raw["split"] == "train_bb"
    X_train = X.loc[train_mask]
    y_train = y.loc[train_mask]
    sampling_rows = [{"stage": "train_bb_from_analysis_cohort", **_class_counts(y_train)}]
    model = train_xgboost(X_train, y_train, cfg, seed)
    save_model(model, output_dir(cfg, "models", "xgboost_blackbox.joblib"))
    write_csv(pd.DataFrame(sampling_rows), output_dir(cfg, "metrics", "blackbox_training_sampling.csv"))

    pred_df = prediction_frame(raw, "split", y, model, X)
    pred_df["y_true_name"] = labels_to_name(state, pred_df["y_true"])
    pred_df["m_label_name"] = labels_to_name(state, pred_df["m_label"])
    write_csv(pred_df, output_dir(cfg, "predictions", "blackbox_predictions.csv"))

    labels = [int(v) for v in model.classes_]
    prob_cols = [f"prob_{label}" for label in labels]
    rows = []
    for split, group in pred_df.groupby("split"):
        metrics = classification_metrics(
            group["y_true"].to_numpy(),
            group[prob_cols].to_numpy(),
            labels=labels,
        )
        metrics.update({"split": split, "n": int(len(group))})
        rows.append(metrics)
    metrics_df = pd.DataFrame(rows).sort_values("split")
    write_csv(metrics_df, output_dir(cfg, "metrics", "blackbox_metrics.csv"))
    write_json(cfg["xgboost"], output_dir(cfg, "models", "xgboost_params.json"))


def _class_counts(y: pd.Series) -> dict[str, int]:
    return {f"class_{int(label)}": int(count) for label, count in y.value_counts().sort_index().items()}


if __name__ == "__main__":
    main()
