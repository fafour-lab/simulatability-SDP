"""Prompt builders for Qwen refiner and proxy evaluator."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from cgrr.rules.dsl import Rule, RuleSet, rule_to_text, ruleset_to_text


REFINER_SYSTEM = (
    "You are a careful rule-refinement assistant. You edit symbolic rules only "
    "through the allowed JSON DSL. Do not invent features or labels. Return JSON only. "
    "Do not include hidden reasoning, markdown, or <think> text."
)

PROXY_SYSTEM = (
    "You are simulating a fixed black-box classifier from the provided input and "
    "optional explanation. Predict the black-box label, not the ground-truth label. "
    "Return JSON only."
)


def compact_schema(feature_schema: dict[str, Any], max_features: int | None = None) -> str:
    rows = []
    features = feature_schema["features"]
    if max_features is not None:
        features = features[:max_features]
    for f in features:
        if f["kind"] == "numeric":
            rows.append(
                {
                    "name": f["name"],
                    "kind": "numeric",
                    "description": str(f.get("description", ""))[:180],
                    "min": round(float(f["min"]), 4),
                    "max": round(float(f["max"]), 4),
                }
            )
        else:
            rows.append(
                {
                    "name": f["name"],
                    "kind": "binary",
                    "description": str(f.get("description", ""))[:180],
                    "allowed_values": [0, 1],
                }
            )
    return json.dumps({"labels": list(feature_schema["labels"].keys()), "features": rows}, indent=2)


def examples_to_json(
    examples: pd.DataFrame,
    feature_columns: list[str],
    max_rows: int,
) -> str:
    rows = []
    for _, row in examples.head(max_rows).iterrows():
        item = {"row_id": int(row["row_id"])}
        for col in feature_columns:
            value = row[col]
            if pd.isna(value):
                item[col] = None
            elif hasattr(value, "item"):
                item[col] = value.item()
            else:
                item[col] = value
        for extra in [
            "m_label",
            "rule_label",
            "proxy_label",
            "proxy_confidence",
            "proxy_reason",
            "source",
            "first_rule_id",
        ]:
            if extra in row:
                value = row[extra]
                if pd.isna(value):
                    item[extra] = None
                else:
                    item[extra] = value.item() if hasattr(value, "item") else value
        rows.append(item)
    return json.dumps(rows, indent=2)


def build_refiner_prompt(
    ruleset: RuleSet,
    rule: Rule,
    rule_stats: dict[str, Any],
    feature_schema: dict[str, Any],
    examples: pd.DataFrame,
    feature_columns: list[str],
    max_examples: int,
    source: str,
) -> tuple[str, str]:
    user = f"""
Refine one weak rule from a surrogate rule set for a fixed black-box classifier.

Counterexample source: {source}

Feature schema:
{compact_schema(feature_schema)}

Current full rule set:
{ruleset_to_text(ruleset)}

Weak rule to edit:
{rule_to_text(rule)}

Weak rule statistics:
{json.dumps(rule_stats, indent=2, default=_json_default)}

Examples covered by, or relevant to, this rule:
{examples_to_json(examples, feature_columns, max_examples)}

Allowed operations:
- add_condition: {{"operation":"add_condition","parent_rule_id":"r_000","condition":{{"feature":"...","op":"<=","value":0}},"reason":"..."}}
- remove_condition: {{"operation":"remove_condition","parent_rule_id":"r_000","condition_index":0,"reason":"..."}}
- change_threshold: {{"operation":"change_threshold","parent_rule_id":"r_000","condition_index":0,"new_value":0,"reason":"..."}}
- split_rule: {{"operation":"split_rule","parent_rule_id":"r_000","new_rules":[{{"if":[{{"feature":"...","op":"<=","value":0}}],"then":"Good"}},{{"if":[{{"feature":"...","op":">","value":0}}],"then":"Bad"}}],"reason":"..."}}
- merge_rules: {{"operation":"merge_rules","parent_rule_ids":["r_000","r_001"],"new_rules":[{{"if":[{{"feature":"...","op":"<=","value":0}}],"then":"Good"}}],"reason":"..."}}
- change_default: {{"operation":"change_default","default_label":"Good","reason":"..."}}
- abstain: {{"operation":"abstain","reason":"..."}}

Return a JSON array with at most 3 candidate operations. Use only the feature names, operators, and labels in the schema.
Every condition must be an object with exactly these keys: "feature", "op", and "value".
Do not include prose outside JSON.
"""
    return REFINER_SYSTEM, user.strip()


def build_proxy_prompt(
    feature_schema: dict[str, Any],
    x_row: dict[str, Any],
    condition_name: str,
    explanation_text: str,
    mode: str = "forward",
    cf_row: dict[str, Any] | None = None,
    original_m_label: str | None = None,
) -> tuple[str, str]:
    payload: dict[str, Any] = {
        "valid_labels": list(feature_schema["labels"].keys()),
        "condition": condition_name,
        "mode": mode,
        "feature_schema": json.loads(compact_schema(feature_schema)),
        "x": x_row,
        "explanation": explanation_text,
        "required_output_format": {
            "predicted_black_box_label": "one valid label",
            "confidence": "number from 0.0 to 1.0",
            "rule_ids_used": ["rule ids if any"],
            "short_reason": "one sentence only",
        },
    }
    if mode == "counterfactual":
        payload["original_black_box_label"] = original_m_label
        payload["x_counterfactual"] = cf_row
    return PROXY_SYSTEM, json.dumps(payload, indent=2)


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return str(value)
