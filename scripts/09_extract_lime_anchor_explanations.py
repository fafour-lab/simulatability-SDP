#!/usr/bin/env python3
"""Extract row-level LIME and Anchors explanations for proxy simulatability."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cgrr.config import load_config, output_dir, project_seed, resolve_path
from cgrr.data import PreprocessorState, attach_splits, load_dataset, transform_features
from cgrr.local_explanations import (
    LOCAL_EXPLANATION_FORMAT_VERSION,
    explain_anchor_instance,
    explain_lime_instance,
    fit_anchor_explainer,
    fit_lime_explainer,
)
from cgrr.model import load_model
from cgrr.utils.io import append_jsonl, read_json, read_jsonl, write_csv
from cgrr.utils.logging import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/heloc_cgrr.yaml")
    parser.add_argument("--explainer", choices=["lime", "anchors", "both"], default="both")
    parser.add_argument("--mode", choices=["forward", "counterfactual", "both"], default="both")
    parser.add_argument("--counterfactuals", default=None)
    parser.add_argument("--background-split", default=None, help="Split for explainer background data; default comes from config.")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(output_dir(cfg, "logs", "09_extract_lime_anchor_explanations.log"))
    seed = project_seed(cfg)
    root = cfg["project"]["root_dir"]
    local_cfg = cfg.get("local_explanations", {})

    split_df = pd.read_csv(output_dir(cfg, "splits", f"split_assignments_seed{seed}.csv"))
    raw = attach_splits(load_dataset(cfg), split_df)
    state = PreprocessorState.from_dict(read_json(output_dir(cfg, "preprocess", "preprocessor_state.json")))
    X = transform_features(raw, state)
    X_by_id = X.copy()
    X_by_id.index = raw["row_id"].to_numpy()
    bb = pd.read_csv(output_dir(cfg, "predictions", "blackbox_predictions.csv")).set_index("row_id")
    model = load_model(output_dir(cfg, "models", "xgboost_blackbox.joblib"))
    class_names = _class_names(model, state)

    background_split = args.background_split or local_cfg.get("background_split", "sur")
    background_mask = _background_mask(raw, background_split)
    if not bool(background_mask.any()):
        raise ValueError(f"No rows found for local explanation background split: {background_split}")
    X_background = X.loc[background_mask].reset_index(drop=True)

    explainers = _selected_explainers(args.explainer)
    modes = ["forward", "counterfactual"] if args.mode == "both" else [args.mode]

    lime_explainer = None
    lime_explanation_class = None
    if "LIME" in explainers:
        lime_cfg = local_cfg.get("lime", {})
        lime_explanation_class = str(
            lime_cfg.get("reference_class")
            or cfg["dataset"].get("positive_label")
            or class_names[-1]
        )
        lime_explainer = fit_lime_explainer(
            X_background=X_background,
            class_names=class_names,
            seed=seed,
            discretize_continuous=bool(lime_cfg.get("discretize_continuous", True)),
        )

    anchor_explainer = None
    if "Anchors" in explainers:
        anchor_cfg = local_cfg.get("anchors", {})
        anchor_explainer = fit_anchor_explainer(
            model=model,
            X_background=X_background,
            seed=seed,
            disc_perc=anchor_cfg.get("disc_perc", [25, 50, 75]),
        )

    for mode in modes:
        items = _items_for_mode(cfg, raw, mode, root, args.counterfactuals)
        if args.limit is not None:
            items = items.head(int(args.limit))
        for condition in explainers:
            out_jsonl = output_dir(cfg, "local_explanations", f"{condition}_{mode}.jsonl")
            out_csv = output_dir(cfg, "local_explanations", f"{condition}_{mode}.csv")
            done_keys = _completed_keys(out_jsonl)
            cache: dict[int, dict[str, Any]] = {}
            iterator = tqdm(items.iterrows(), total=len(items), desc=f"{condition}:{mode}")
            for _, item in iterator:
                row_id = int(item["row_id"])
                cf_row_id = _optional_int(item.get("cf_row_id"))
                explain_row_id = int(item["explain_row_id"])
                key = (row_id, cf_row_id, condition, mode)
                if key in done_keys:
                    continue
                if explain_row_id not in X_by_id.index:
                    raise KeyError(f"Missing transformed features for row_id={explain_row_id}")
                if explain_row_id not in cache:
                    x_row = X_by_id.loc[explain_row_id]
                    if condition == "LIME":
                        if lime_explainer is None:
                            raise RuntimeError("LIME explainer was not initialized")
                        if lime_explanation_class is None:
                            raise RuntimeError("LIME explanation class was not initialized")
                        cache[explain_row_id] = explain_lime_instance(
                            explainer=lime_explainer,
                            model=model,
                            x_row=x_row,
                            class_names=class_names,
                            explanation_class_name=lime_explanation_class,
                            **_lime_kwargs(local_cfg),
                        )
                    else:
                        if anchor_explainer is None:
                            raise RuntimeError("Anchor explainer was not initialized")
                        cache[explain_row_id] = explain_anchor_instance(
                            explainer=anchor_explainer,
                            model=model,
                            x_row=x_row,
                            class_names=class_names,
                            **_anchor_kwargs(local_cfg),
                        )
                target = bb.loc[explain_row_id]
                record = {
                    "row_id": row_id,
                    "cf_row_id": cf_row_id,
                    "explain_row_id": explain_row_id,
                    "target_part": str(item["target_part"]),
                    "condition": condition,
                    "mode": mode,
                    "flip": _json_scalar(item.get("flip")),
                    "original_m_label": str(bb.loc[row_id, "m_label_name"]),
                    "m_label": str(target["m_label_name"]),
                    "m_label_numeric": int(target["m_label"]),
                    **cache[explain_row_id],
                }
                append_jsonl([record], out_jsonl)

            records = _latest_records(read_jsonl(out_jsonl)) if out_jsonl.exists() else []
            cleaned = pd.DataFrame([{k: _csv_safe(v) for k, v in record.items()} for record in records])
            write_csv(cleaned, out_csv)


def _selected_explainers(value: str) -> list[str]:
    if value == "both":
        return ["LIME", "Anchors"]
    if value == "lime":
        return ["LIME"]
    return ["Anchors"]


def _class_names(model: Any, state: PreprocessorState) -> list[str]:
    return [state.inverse_label_mapping[str(int(label))] for label in model.classes_]


def _background_mask(raw: pd.DataFrame, split_spec: str) -> pd.Series:
    if split_spec == "all_non_eval":
        return raw["split"] != "eval"
    splits = [part.strip() for part in str(split_spec).split(",") if part.strip()]
    return raw["split"].isin(splits)


def _items_for_mode(
    cfg: dict[str, Any],
    raw: pd.DataFrame,
    mode: str,
    root: str,
    counterfactuals: str | None,
) -> pd.DataFrame:
    if mode == "forward":
        items = raw.loc[raw["split"] == "eval", ["row_id"]].copy()
        items["cf_row_id"] = None
        items["flip"] = None
        items["explain_row_id"] = items["row_id"]
        items["target_part"] = "x"
        return items

    cf_path = (
        resolve_path(counterfactuals, root)
        if counterfactuals
        else output_dir(cfg, "evaluation_items", "counterfactual_pairs.csv")
    )
    items = pd.read_csv(cf_path)
    items["explain_row_id"] = items["cf_row_id"]
    items["target_part"] = "x_counterfactual"
    return items


def _lime_kwargs(cfg: dict[str, Any]) -> dict[str, Any]:
    lime_cfg = cfg.get("lime", {})
    return {
        "num_features": int(lime_cfg.get("num_features", 8)),
        "num_samples": int(lime_cfg.get("num_samples", 5000)),
    }


def _anchor_kwargs(cfg: dict[str, Any]) -> dict[str, Any]:
    anchor_cfg = cfg.get("anchors", {})
    return {
        "threshold": float(anchor_cfg.get("threshold", 0.95)),
        "delta": float(anchor_cfg.get("delta", 0.1)),
        "tau": float(anchor_cfg.get("tau", 0.15)),
        "batch_size": int(anchor_cfg.get("batch_size", 100)),
        "coverage_samples": int(anchor_cfg.get("coverage_samples", 1000)),
        "beam_size": int(anchor_cfg.get("beam_size", 1)),
        "stop_on_first": bool(anchor_cfg.get("stop_on_first", True)),
        "max_anchor_size": anchor_cfg.get("max_anchor_size"),
    }


def _completed_keys(path: Path) -> set[tuple[int, int | None, str, str]]:
    if not path.exists():
        return set()
    keys = set()
    for record in read_jsonl(path):
        if int(record.get("explanation_format_version", 0)) != LOCAL_EXPLANATION_FORMAT_VERSION:
            continue
        keys.add(
            (
                int(record["row_id"]),
                _optional_int(record.get("cf_row_id")),
                str(record["condition"]),
                str(record["mode"]),
            )
        )
    return keys


def _latest_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the newest current-format record for each resumable item."""
    latest: dict[tuple[int, int | None, str, str], dict[str, Any]] = {}
    for record in records:
        if int(record.get("explanation_format_version", 0)) != LOCAL_EXPLANATION_FORMAT_VERSION:
            continue
        key = (
            int(record["row_id"]),
            _optional_int(record.get("cf_row_id")),
            str(record["condition"]),
            str(record["mode"]),
        )
        latest[key] = record
    return list(latest.values())


def _optional_int(value: Any) -> int | None:
    if value is None or pd.isna(value):
        return None
    return int(value)


def _json_scalar(value: Any) -> Any:
    if value is None or pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def _csv_safe(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value, sort_keys=True)
    return value


if __name__ == "__main__":
    main()
