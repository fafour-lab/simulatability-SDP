#!/usr/bin/env python3
"""Stage 07: compute rule, symbolic-executor, proxy, and bootstrap metrics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cgrr.config import load_config, output_dir, project_seed
from cgrr.data import PreprocessorState, attach_splits, load_dataset, transform_features
from cgrr.metrics.reporting import bootstrap_proxy_effects, summarize_proxy_predictions
from cgrr.rules.dsl import RuleSet
from cgrr.rules.executor import per_rule_stats, predict_ruleset
from cgrr.rules.verifier import rule_counterexamples, score_ruleset
from cgrr.utils.io import read_json, write_csv
from cgrr.utils.logging import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/heloc_cgrr.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(output_dir(cfg, "logs", "07_evaluate.log"))
    seed = project_seed(cfg)

    split_df = pd.read_csv(output_dir(cfg, "splits", f"split_assignments_seed{seed}.csv"))
    raw = attach_splits(load_dataset(cfg), split_df)
    state = PreprocessorState.from_dict(read_json(output_dir(cfg, "preprocess", "preprocessor_state.json")))
    X = transform_features(raw, state)
    bb = pd.read_csv(output_dir(cfg, "predictions", "blackbox_predictions.csv"))
    m = bb.set_index("row_id").loc[raw["row_id"], "m_label_name"].reset_index(drop=True)

    rule_metric_rows = []
    symbolic_rows = []
    per_rule_frames = []
    for rules_path in sorted(output_dir(cfg, "rules").glob("*.json")):
        ruleset = RuleSet.from_dict(read_json(rules_path))
        for split in ["refine", "eval"]:
            mask = raw["split"] == split
            score = score_ruleset(
                ruleset,
                X.loc[mask].reset_index(drop=True),
                m.loc[mask].reset_index(drop=True),
                cfg["refinement"]["objective"],
            )
            score.update({"ruleset": ruleset.name, "split": split, "n": int(mask.sum())})
            rule_metric_rows.append(score)

            stats = per_rule_stats(
                ruleset,
                X.loc[mask].reset_index(drop=True),
                m.loc[mask].reset_index(drop=True),
            )
            stats["ruleset"] = ruleset.name
            stats["split"] = split
            per_rule_frames.append(stats)

        eval_mask = raw["split"] == "eval"
        pred, details = predict_ruleset(ruleset, X.loc[eval_mask].reset_index(drop=True), return_details=True)
        eval_raw = raw.loc[eval_mask].reset_index(drop=True)
        eval_m = m.loc[eval_mask].reset_index(drop=True)
        for row_id, pred_label, m_label, covered, conflict in zip(
            eval_raw["row_id"], pred, eval_m, details["covered"], details["conflict"]
        ):
            symbolic_rows.append(
                {
                    "row_id": int(row_id),
                    "condition": f"Symbolic-{ruleset.name}",
                    "mode": "forward",
                    "predicted_black_box_label": pred_label,
                    "m_label": m_label,
                    "covered": bool(covered),
                    "conflict": bool(conflict),
                }
            )

        refine_mask = raw["split"] == "refine"
        ce = rule_counterexamples(
            ruleset,
            X.loc[refine_mask].reset_index(drop=True),
            raw.loc[refine_mask].reset_index(drop=True),
            m.loc[refine_mask].reset_index(drop=True),
        )
        write_csv(ce, output_dir(cfg, "counterexamples", f"{ruleset.name}_refine_counterexamples.csv"))

    if rule_metric_rows:
        write_csv(pd.DataFrame(rule_metric_rows), output_dir(cfg, "metrics", "rule_metrics.csv"))
    if per_rule_frames:
        write_csv(pd.concat(per_rule_frames, ignore_index=True), output_dir(cfg, "metrics", "per_rule_metrics.csv"))
    if symbolic_rows:
        write_csv(pd.DataFrame(symbolic_rows), output_dir(cfg, "proxy", "symbolic_executor_forward.csv"))

    proxy_frames = []
    for proxy_csv in sorted(output_dir(cfg, "proxy").glob("*.csv")):
        proxy_frames.append(pd.read_csv(proxy_csv))
    if proxy_frames:
        proxy = pd.concat(proxy_frames, ignore_index=True)
        write_csv(proxy, output_dir(cfg, "metrics", "proxy_predictions_all.csv"))
        summary = summarize_proxy_predictions(proxy)
        write_csv(summary, output_dir(cfg, "metrics", "proxy_simulatability.csv"))
        comparisons = [
            ("Faithful-Refined", "R0"),
            ("Faithful-Refined", "x-only"),
            ("Proxy-Refined", "R0"),
            ("Hybrid-Refined", "R0"),
            ("R0", "x-only"),
            ("LIME", "x-only"),
            ("Anchors", "x-only"),
            ("Anchors", "LIME"),
            ("LIME", "R0"),
            ("Anchors", "R0"),
        ]
        effect_frames = []
        for mode in sorted(proxy["mode"].dropna().unique()):
            effects = bootstrap_proxy_effects(
                proxy,
                comparisons=comparisons,
                n_resamples=int(cfg["evaluation"]["bootstrap_resamples"]),
                seed=int(cfg["evaluation"]["bootstrap_seed"]),
                mode=str(mode),
            )
            if not effects.empty:
                effect_frames.append(effects)
        if effect_frames:
            write_csv(pd.concat(effect_frames, ignore_index=True), output_dir(cfg, "metrics", "proxy_bootstrap_effects.csv"))


if __name__ == "__main__":
    main()
