#!/usr/bin/env python3
"""Stage 05: build flip-balanced real-instance counterfactual pairs for D_eval."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cgrr.config import load_config, output_dir, project_seed
from cgrr.counterfactuals import nearest_real_counterfactual_pairs
from cgrr.data import PreprocessorState, attach_splits, load_dataset, transform_features
from cgrr.utils.io import read_json, write_csv
from cgrr.utils.logging import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/heloc_cgrr.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(output_dir(cfg, "logs", "05_make_counterfactuals.log"))
    seed = project_seed(cfg)

    split_df = pd.read_csv(output_dir(cfg, "splits", f"split_assignments_seed{seed}.csv"))
    raw = attach_splits(load_dataset(cfg), split_df)
    state = PreprocessorState.from_dict(read_json(output_dir(cfg, "preprocess", "preprocessor_state.json")))
    X = transform_features(raw, state)
    bb = pd.read_csv(output_dir(cfg, "predictions", "blackbox_predictions.csv"))
    m = bb.set_index("row_id").loc[raw["row_id"], "m_label_name"].reset_index(drop=True)

    eval_mask = raw["split"] == "eval"
    pairs = nearest_real_counterfactual_pairs(
        X_eval=X.loc[eval_mask].reset_index(drop=True),
        raw_eval=raw.loc[eval_mask].reset_index(drop=True),
        m_labels=m.loc[eval_mask].reset_index(drop=True),
        seed=seed,
    )
    write_csv(pairs, output_dir(cfg, "evaluation_items", "counterfactual_pairs.csv"))


if __name__ == "__main__":
    main()
