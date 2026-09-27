#!/usr/bin/env python3
"""Stage 03: fit decision-tree surrogate T0 and export machine-readable rules R0."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cgrr.config import load_config, output_dir, project_seed
from cgrr.data import PreprocessorState, attach_splits, load_dataset, transform_features
from cgrr.rules.dsl import ruleset_to_text
from cgrr.rules.extract import extract_rules_from_tree
from cgrr.rules.verifier import score_ruleset
from cgrr.utils.io import read_json, write_csv, write_json
from cgrr.utils.logging import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/heloc_cgrr.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(output_dir(cfg, "logs", "03_build_surrogate.log"))
    seed = project_seed(cfg)

    split_df = pd.read_csv(output_dir(cfg, "splits", f"split_assignments_seed{seed}.csv"))
    raw = attach_splits(load_dataset(cfg), split_df)
    state = PreprocessorState.from_dict(read_json(output_dir(cfg, "preprocess", "preprocessor_state.json")))
    X = transform_features(raw, state)
    bb = pd.read_csv(output_dir(cfg, "predictions", "blackbox_predictions.csv"))
    m_numeric = bb.set_index("row_id").loc[raw["row_id"], "m_label"].reset_index(drop=True)
    m_name = bb.set_index("row_id").loc[raw["row_id"], "m_label_name"].reset_index(drop=True)

    def decode(label: int) -> str:
        return state.inverse_label_mapping[str(int(label))]

    default_label = m_name.loc[raw["split"] == "sur"].mode().iloc[0]
    rows = []
    best = None
    best_score = None
    best_tree = None
    for depth in cfg["surrogate"]["candidate_depths"]:
        tree = DecisionTreeClassifier(
            max_depth=int(depth),
            min_samples_leaf=int(cfg["surrogate"]["min_samples_leaf"]),
            random_state=seed + int(depth),
        )
        sur_mask = raw["split"] == "sur"
        refine_mask = raw["split"] == "refine"
        tree.fit(X.loc[sur_mask], m_numeric.loc[sur_mask])
        ruleset = extract_rules_from_tree(
            tree,
            feature_names=list(X.columns),
            label_decoder=decode,
            default_label=default_label,
            name=f"R0_depth{depth}",
        )
        score = score_ruleset(ruleset, X.loc[refine_mask], m_name.loc[refine_mask], cfg["refinement"]["objective"])
        score["depth"] = int(depth)
        rows.append(score)
        if best_score is None or score["objective"] > best_score["objective"]:
            best = ruleset
            best_score = score
            best_tree = tree

    if best is None or best_score is None or best_tree is None:
        raise RuntimeError("No surrogate tree was fitted")
    best.name = "R0"
    best.metadata["selection_score_on_refine"] = best_score

    write_csv(pd.DataFrame(rows).sort_values("objective", ascending=False), output_dir(cfg, "metrics", "surrogate_depth_search.csv"))
    write_json(best.to_dict(), output_dir(cfg, "rules", "R0.json"))
    text_path = output_dir(cfg, "rules", "R0.txt")
    text_path.parent.mkdir(parents=True, exist_ok=True)
    text_path.write_text(ruleset_to_text(best), encoding="utf-8")


if __name__ == "__main__":
    main()
