#!/usr/bin/env python3
"""Stage 06: ask Qwen-8B proxy P to predict M(x) under explanation conditions."""

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
from cgrr.rules.dsl import RuleSet, ruleset_to_text
from cgrr.utils.io import append_jsonl, read_json, read_jsonl, write_csv
from cgrr.utils.logging import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/heloc_cgrr.yaml")
    parser.add_argument("--condition", required=True, help="Example: x-only, R0, Faithful-Refined")
    parser.add_argument("--rules", default=None, help="Rule JSON path for rule-based conditions.")
    parser.add_argument("--mode", choices=["forward", "counterfactual"], default="forward")
    parser.add_argument(
        "--split",
        choices=["train_bb", "val_bb", "sur", "refine", "eval"],
        default="eval",
        help=(
            "Dataset split used for forward simulation. Non-eval runs are written to "
            "refinement_inputs/ and are not included in final evaluation metrics."
        ),
    )
    parser.add_argument("--counterfactuals", default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    safe_condition = args.condition.replace("/", "_").replace(" ", "_")
    if args.mode == "counterfactual" and args.split != "eval":
        parser.error("--split only supports non-eval values in forward mode")
    log_suffix = f"_{args.split}" if args.mode == "forward" and args.split != "eval" else ""
    setup_logging(output_dir(cfg, "logs", f"06_proxy_{safe_condition}_{args.mode}{log_suffix}.log"))
    seed = project_seed(cfg)
    root = cfg["project"]["root_dir"]

    split_df = pd.read_csv(output_dir(cfg, "splits", f"split_assignments_seed{seed}.csv"))
    raw = attach_splits(load_dataset(cfg), split_df)
    feature_cols = feature_columns(cfg, raw)
    feature_schema = read_json(output_dir(cfg, "preprocess", "feature_schema.json"))
    bb = pd.read_csv(output_dir(cfg, "predictions", "blackbox_predictions.csv"))
    bb_by_id = bb.set_index("row_id")
    raw_by_id = raw.set_index("row_id", drop=False)

    ruleset = None
    if args.condition != "x-only":
        rules_path = resolve_path(args.rules, root) if args.rules else output_dir(cfg, "rules", f"{args.condition}.json")
        ruleset = RuleSet.from_dict(read_json(rules_path))
    explanation = "No rule explanation is provided." if ruleset is None else ruleset_to_text(ruleset)

    if args.mode == "forward":
        items = raw[raw["split"] == args.split][["row_id"]].copy()
        items["cf_row_id"] = None
        items["flip"] = None
    else:
        cf_path = (
            resolve_path(args.counterfactuals, root)
            if args.counterfactuals
            else output_dir(cfg, "evaluation_items", "counterfactual_pairs.csv")
        )
        items = pd.read_csv(cf_path)
    if items.empty:
        raise ValueError(f"No proxy-simulation items found for mode={args.mode}, split={args.split}")

    max_items = args.limit
    if max_items is None:
        max_items = cfg.get("proxy", {}).get("max_items_per_condition")
    if max_items:
        items = items.head(int(max_items))

    if args.mode == "forward" and args.split != "eval":
        artifact_dir = "refinement_inputs"
        artifact_stem = f"{safe_condition}_{args.mode}_{args.split}"
    else:
        artifact_dir = "proxy"
        artifact_stem = f"{safe_condition}_{args.mode}"
    out_jsonl = output_dir(cfg, artifact_dir, f"{artifact_stem}.jsonl")
    out_csv = output_dir(cfg, artifact_dir, f"{artifact_stem}.csv")
    done_keys = _completed_keys(out_jsonl)
    client_cfg = GenerationConfig.from_dict({**cfg["qwen"], **cfg.get("proxy", {})})
    client = QwenClient(client_cfg)

    for _, item in tqdm(items.iterrows(), total=len(items), desc=f"{args.condition}:{args.mode}"):
        row_id = int(item["row_id"])
        cf_row_id = None if pd.isna(item.get("cf_row_id")) else int(item["cf_row_id"])
        key = (row_id, cf_row_id, args.condition, args.mode)
        if key in done_keys:
            continue
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
            explanation_text=explanation,
            mode=args.mode,
            cf_row=cf_row,
            original_m_label=original_m_label,
        )
        raw_response = client.generate(system, user)
        parsed = parse_proxy_response(raw_response)
        target_row_id = cf_row_id if args.mode == "counterfactual" else row_id
        record: dict[str, Any] = {
            "row_id": row_id,
            "cf_row_id": cf_row_id,
            "condition": args.condition,
            "mode": args.mode,
            "data_split": args.split,
            "flip": item.get("flip"),
            "m_label": str(bb_by_id.loc[target_row_id, "m_label_name"]),
            "prompt": user,
            "raw_response": raw_response,
            **parsed,
        }
        append_jsonl([record], out_jsonl)

    records = read_jsonl(out_jsonl)
    cleaned = pd.DataFrame([{k: _csv_safe(v) for k, v in record.items() if k != "prompt"} for record in records])
    write_csv(cleaned, out_csv)


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
        cf = record.get("cf_row_id")
        keys.add(
            (
                int(record["row_id"]),
                None if cf is None else int(cf),
                str(record["condition"]),
                str(record["mode"]),
            )
        )
    return keys


def _csv_safe(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value, sort_keys=True)
    return value


if __name__ == "__main__":
    main()
