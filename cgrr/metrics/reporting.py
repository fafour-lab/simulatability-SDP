"""Metric summarizers for model, rule, and proxy-simulation outputs."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cgrr.metrics.bootstrap import paired_bootstrap_mean_diff


def summarize_proxy_predictions(proxy_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = ["condition", "mode"]
    if "flip" in proxy_df.columns:
        group_cols.append("flip")
    for keys, group in proxy_df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        record = dict(zip(group_cols, keys))
        valid = group["predicted_black_box_label"].notna()
        correct = group.loc[valid, "predicted_black_box_label"] == group.loc[valid, "m_label"]
        record.update(
            {
                "n": int(len(group)),
                "n_parseable": int(valid.sum()),
                "parse_rate": float(valid.mean()) if len(group) else 0.0,
                "sim_accuracy": float(correct.mean()) if len(correct) else np.nan,
            }
        )
        rows.append(record)
    return pd.DataFrame(rows)


def bootstrap_proxy_effects(
    proxy_df: pd.DataFrame,
    comparisons: list[tuple[str, str]],
    n_resamples: int,
    seed: int,
    mode: str = "forward",
) -> pd.DataFrame:
    df = proxy_df[proxy_df["mode"] == mode].copy()
    join_cols = ["row_id"]
    if mode == "counterfactual" and "cf_row_id" in df.columns:
        join_cols.append("cf_row_id")
    rows = []
    for treatment, control in comparisons:
        columns = join_cols + ["predicted_black_box_label", "m_label"]
        t = df[df["condition"] == treatment][columns]
        c = df[df["condition"] == control][columns]
        joined = t.merge(c, on=join_cols, suffixes=("_t", "_c"))
        if joined.empty:
            continue
        treatment_correct = (joined["predicted_black_box_label_t"] == joined["m_label_t"]).to_numpy(float)
        control_correct = (joined["predicted_black_box_label_c"] == joined["m_label_c"]).to_numpy(float)
        effect = paired_bootstrap_mean_diff(
            treatment_correct,
            control_correct,
            n_resamples=n_resamples,
            seed=seed + len(rows),
        )
        effect.update({"mode": mode, "treatment": treatment, "control": control})
        rows.append(effect)
    return pd.DataFrame(rows)
