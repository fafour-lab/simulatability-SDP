#!/usr/bin/env python3
"""Stage 01: prepare the analysis cohort, splits, and preprocessing metadata."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cgrr.config import load_config, output_dir, project_seed
from cgrr.data import (
    build_feature_schema,
    fit_preprocessor,
    labels_to_int,
    make_split_assignments,
    prepare_dataset,
    prepared_dataset_path,
    split_counts,
)
from cgrr.utils.io import write_csv, write_json
from cgrr.utils.logging import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/heloc_cgrr.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(output_dir(cfg, "logs", "01_make_splits.log"))
    seed = project_seed(cfg)

    print(f"[01] Preparing dataset cohort: {cfg['dataset']['name']}", flush=True)
    df, sampling_summary = prepare_dataset(cfg)
    sampling_cfg = cfg["dataset"].get("sampling", {})
    if bool(sampling_cfg.get("enabled", False)):
        cohort_path = prepared_dataset_path(cfg)
        write_csv(df, cohort_path)
        sampling_summary["materialized_cohort"] = str(cohort_path)
        print(
            "[01] Balanced cohort ready: "
            f"{sampling_summary['selected_rows']:,}/{sampling_summary['source_rows']:,} rows "
            f"({sampling_summary['rows_per_class']:,} per class)",
            flush=True,
        )

    print("[01] Creating stratified experiment splits", flush=True)
    split_df = make_split_assignments(df, cfg, seed)
    state = fit_preprocessor(df, split_df, cfg)
    schema = build_feature_schema(state)
    split_class_counts = _split_class_counts(df, split_df, cfg)

    write_csv(split_df, output_dir(cfg, "splits", f"split_assignments_seed{seed}.csv"))
    write_csv(split_counts(split_df), output_dir(cfg, "splits", "split_counts.csv"))
    write_csv(split_class_counts, output_dir(cfg, "splits", "split_class_counts.csv"))
    write_csv(
        _sampling_summary_frame(sampling_summary),
        output_dir(cfg, "preprocess", "dataset_sampling_summary.csv"),
    )
    write_json(sampling_summary, output_dir(cfg, "preprocess", "dataset_sampling_summary.json"))
    write_json(state.to_dict(), output_dir(cfg, "preprocess", "preprocessor_state.json"))
    write_json(schema, output_dir(cfg, "preprocess", "feature_schema.json"))
    write_json(
        {
            "dataset_rows": int(len(df)),
            "source_dataset_rows": int(sampling_summary["source_rows"]),
            "target": cfg["dataset"]["target"],
            "seed": seed,
            "sampling": sampling_summary,
            "split_counts": split_counts(split_df).to_dict(orient="records"),
            "split_class_counts": split_class_counts.to_dict(orient="records"),
        },
        output_dir(cfg, "preprocess", "preprocess_summary.json"),
    )
    print("[01] Cohort, splits, and preprocessing metadata finished", flush=True)


def _split_class_counts(df: pd.DataFrame, split_df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    labeled = split_df.merge(
        df[["row_id", cfg["dataset"]["target"]]],
        on="row_id",
        how="left",
        validate="one_to_one",
    )
    labeled["class"] = labels_to_int(cfg, labeled[cfg["dataset"]["target"]])
    return (
        labeled.groupby(["split", "class"], as_index=False)
        .size()
        .rename(columns={"size": "n"})
        .sort_values(["split", "class"])
    )


def _sampling_summary_frame(summary: dict) -> pd.DataFrame:
    row = {key: value for key, value in summary.items() if not isinstance(value, dict)}
    for label, count in summary.get("source_class_counts", {}).items():
        row[f"source_class_{label}"] = count
    for label, count in summary.get("selected_class_counts", {}).items():
        row[f"selected_class_{label}"] = count
    return pd.DataFrame([row])


if __name__ == "__main__":
    main()
