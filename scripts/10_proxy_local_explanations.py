#!/usr/bin/env python3
"""Run proxy simulatability with row-level LIME or Anchors explanations."""

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
from cgrr.data import attach_splits, feature_columns, load_dataset
from cgrr.llm.prompts import build_proxy_prompt
from cgrr.llm.proxy import parse_proxy_response
from cgrr.llm.qwen_client import GenerationConfig, QwenClient
from cgrr.local_explanations import LOCAL_EXPLANATION_FORMAT_VERSION
from cgrr.utils.io import append_jsonl, read_json, read_jsonl, write_csv
from cgrr.utils.logging import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/heloc_cgrr.yaml")
    parser.add_argument("--condition", choices=["LIME", "Anchors"], required=True)
    parser.add_argument("--mode", choices=["forward", "counterfactual"], default="forward")
    parser.add_argument("--explanations", default=None, help="JSONL/CSV explanation artifact from script 09.")
    parser.add_argument("--counterfactuals", default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    safe_condition = args.condition.replace("/", "_").replace(" ", "_")
    setup_logging(output_dir(cfg, "logs", f"10_proxy_{safe_condition}_{args.mode}.log"))
    seed = project_seed(cfg)
    root = cfg["project"]["root_dir"]

    split_df = pd.read_csv(output_dir(cfg, "splits", f"split_assignments_seed{seed}.csv"))
    raw = attach_splits(load_dataset(cfg), split_df)
    raw_by_id = raw.set_index("row_id", drop=False)
    feature_cols = feature_columns(cfg, raw)
    feature_schema = read_json(output_dir(cfg, "preprocess", "feature_schema.json"))
    bb = pd.read_csv(output_dir(cfg, "predictions", "blackbox_predictions.csv"))
    bb_by_id = bb.set_index("row_id")

    explanation_path = (
        resolve_path(args.explanations, root)
        if args.explanations
        else output_dir(cfg, "local_explanations", f"{args.condition}_{args.mode}.jsonl")
    )
    explanations = _load_explanations(explanation_path)

    if args.mode == "forward":
        items = raw[raw["split"] == "eval"][["row_id"]].copy()
        items["cf_row_id"] = None
        items["flip"] = None
    else:
        cf_path = (
            resolve_path(args.counterfactuals, root)
            if args.counterfactuals
            else output_dir(cfg, "evaluation_items", "counterfactual_pairs.csv")
        )
        items = pd.read_csv(cf_path)

    max_items = args.limit
    if max_items is None:
        max_items = cfg.get("proxy", {}).get("max_items_per_condition")
    if max_items:
        items = items.head(int(max_items))

    out_jsonl = output_dir(cfg, "proxy", f"{safe_condition}_{args.mode}.jsonl")
    out_csv = output_dir(cfg, "proxy", f"{safe_condition}_{args.mode}.csv")
    done_keys = _completed_keys(out_jsonl)
    client_cfg = GenerationConfig.from_dict({**cfg["qwen"], **cfg.get("proxy", {})})
    client = QwenClient(client_cfg)

    for _, item in tqdm(items.iterrows(), total=len(items), desc=f"{args.condition}:{args.mode}"):
        row_id = int(item["row_id"])
        cf_row_id = _optional_int(item.get("cf_row_id"))
        key = (row_id, cf_row_id, args.condition, args.mode)
        if key in done_keys:
            continue
        explanation_record = explanations.get(key)
        if explanation_record is None:
            raise KeyError(
                "Missing local explanation for "
                f"row_id={row_id}, cf_row_id={cf_row_id}, condition={args.condition}, mode={args.mode}. "
                "Run scripts/09_extract_lime_anchor_explanations.py for this condition and mode first."
            )
        explanation_text = _proxy_explanation_text(explanation_record)

        x_row = _row_payload(raw_by_id.loc[row_id], feature_cols)
        if args.mode == "counterfactual":
            cf_row = _row_payload(raw_by_id.loc[cf_row_id], feature_cols)
            original_m_label = str(bb_by_id.loc[row_id, "m_label_name"])
        else:
            cf_row = None
            original_m_label = None
        system, user = build_proxy_prompt(
            feature_schema=feature_schema,
            x_row=x_row,
            condition_name=args.condition,
            explanation_text=explanation_text,
            mode=args.mode,
            cf_row=cf_row,
            original_m_label=original_m_label,
        )
        raw_response = client.generate(system, user)
        parsed = parse_proxy_response(raw_response)
        target_row_id = cf_row_id if args.mode == "counterfactual" else row_id
        record: dict[str, Any] = {
            "proxy_prompt_version": LOCAL_EXPLANATION_FORMAT_VERSION,
            "explanation_format_version": LOCAL_EXPLANATION_FORMAT_VERSION,
            "row_id": row_id,
            "cf_row_id": cf_row_id,
            "explain_row_id": int(explanation_record["explain_row_id"]),
            "condition": args.condition,
            "mode": args.mode,
            "flip": item.get("flip"),
            "m_label": str(bb_by_id.loc[target_row_id, "m_label_name"]),
            "explanation_target_label": str(explanation_record.get("target_label", explanation_record.get("m_label"))),
            "prompt": user,
            "raw_response": raw_response,
            **parsed,
        }
        append_jsonl([record], out_jsonl)

    records = _latest_records(read_jsonl(out_jsonl)) if out_jsonl.exists() else []
    cleaned = pd.DataFrame([{k: _csv_safe(v) for k, v in record.items() if k != "prompt"} for record in records])
    write_csv(cleaned, out_csv)


def _load_explanations(path: Path) -> dict[tuple[int, int | None, str, str], dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Explanation artifact does not exist: {path}")
    if path.suffix == ".csv":
        records = pd.read_csv(path).to_dict(orient="records")
    else:
        records = read_jsonl(path)
    out = {}
    for record in records:
        if int(record.get("explanation_format_version", 0)) != LOCAL_EXPLANATION_FORMAT_VERSION:
            continue
        key = (
            int(record["row_id"]),
            _optional_int(record.get("cf_row_id")),
            str(record["condition"]),
            str(record["mode"]),
        )
        out[key] = record
    if not out:
        raise ValueError(
            f"Explanation artifact {path} contains no current-format, label-safe records. "
            "Rerun scripts/09_extract_lime_anchor_explanations.py before proxy simulation."
        )
    return out


def _proxy_explanation_text(record: dict[str, Any]) -> str:
    """Return label-safe explanation text and reject legacy target-disclosing text."""
    if int(record.get("explanation_format_version", 0)) != LOCAL_EXPLANATION_FORMAT_VERSION:
        raise ValueError("Refusing to use a legacy local explanation that may disclose the target label.")
    text = str(record.get("explanation_text", "")).strip()
    if not text:
        raise ValueError("Local explanation text is empty.")
    lowered = text.lower()
    forbidden_phrases = ("the black-box predicts", "black-box prediction is")
    if any(phrase in lowered for phrase in forbidden_phrases):
        raise ValueError("Local explanation text discloses the target label and cannot be sent to the proxy.")
    return text


def _row_payload(row: pd.Series, feature_cols: list[str]) -> dict[str, Any]:
    payload = {"row_id": int(row["row_id"])}
    for col in feature_cols:
        value = row[col]
        if pd.isna(value):
            payload[col] = None
        elif hasattr(value, "item"):
            payload[col] = value.item()
        else:
            payload[col] = value
    return payload


def _completed_keys(path: Path) -> set[tuple[int, int | None, str, str]]:
    if not path.exists():
        return set()
    keys = set()
    for record in read_jsonl(path):
        if int(record.get("proxy_prompt_version", 0)) != LOCAL_EXPLANATION_FORMAT_VERSION:
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
    """Keep the newest current-prompt record for each resumable item."""
    latest: dict[tuple[int, int | None, str, str], dict[str, Any]] = {}
    for record in records:
        if int(record.get("proxy_prompt_version", 0)) != LOCAL_EXPLANATION_FORMAT_VERSION:
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


def _csv_safe(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value, sort_keys=True)
    return value


if __name__ == "__main__":
    main()
