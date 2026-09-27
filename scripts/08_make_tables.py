#!/usr/bin/env python3
"""Stage 08: write paper-ready CSV and LaTeX tables from metric artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cgrr.config import load_config, output_dir
from cgrr.tables import round_for_paper, write_csv_and_latex
from cgrr.utils.logging import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/heloc_cgrr.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(output_dir(cfg, "logs", "08_make_tables.log"))
    table_dir = output_dir(cfg, "tables")

    _table(
        source=output_dir(cfg, "metrics", "blackbox_metrics.csv"),
        dest=table_dir / "table_1_blackbox_metrics",
        columns=["split", "n", "accuracy", "macro_f1", "auroc", "brier", "ece_10bin"],
    )
    _table(
        source=output_dir(cfg, "metrics", "surrogate_depth_search.csv"),
        dest=table_dir / "table_2_surrogate_depth_search",
        columns=["depth", "objective", "full_fidelity", "covered_fidelity", "coverage", "conflict_rate", "n_rules", "avg_predicates"],
    )
    _table(
        source=output_dir(cfg, "metrics", "rule_metrics.csv"),
        dest=table_dir / "table_3_rule_metrics",
        columns=["ruleset", "split", "n", "objective", "full_fidelity", "covered_fidelity", "coverage", "conflict_rate", "n_rules", "avg_predicates"],
    )
    _table(
        source=output_dir(cfg, "metrics", "proxy_simulatability.csv"),
        dest=table_dir / "table_4_proxy_simulatability",
        columns=["condition", "mode", "n", "n_parseable", "parse_rate", "sim_accuracy"],
    )
    _table(
        source=output_dir(cfg, "metrics", "proxy_bootstrap_effects.csv"),
        dest=table_dir / "table_5_proxy_bootstrap_effects",
        columns=["mode", "treatment", "control", "mean_diff", "ci_low", "ci_high", "n", "n_resamples"],
    )


def _table(source: Path, dest: Path, columns: list[str]) -> None:
    if not source.exists():
        return
    df = pd.read_csv(source)
    keep = [col for col in columns if col in df.columns]
    df = round_for_paper(df[keep])
    write_csv_and_latex(df, dest)


if __name__ == "__main__":
    main()
