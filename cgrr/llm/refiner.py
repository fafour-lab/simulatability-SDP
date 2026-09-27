"""Candidate parsing and edit application for Qwen rule refinement."""

from __future__ import annotations

import copy
import json
from typing import Any

from cgrr.rules.dsl import Condition, Rule, RuleSet


def extract_json_payloads(text: str) -> list[Any]:
    """Extract JSON arrays/objects from a model response."""
    text = text.strip()
    if not text:
        return []
    candidates = [text]
    if "```" in text:
        blocks = text.split("```")
        candidates.extend(block for block in blocks if "{" in block or "[" in block)

    parsed: list[Any] = []
    decoder = json.JSONDecoder()
    for candidate in candidates:
        candidate = candidate.strip()
        if candidate.startswith("json"):
            candidate = candidate[4:].strip()
        try:
            parsed.append(json.loads(candidate))
            continue
        except json.JSONDecodeError:
            pass
        for start, char in enumerate(candidate):
            if char not in "[{":
                continue
            try:
                obj, _ = decoder.raw_decode(candidate[start:])
                parsed.append(obj)
                break
            except json.JSONDecodeError:
                continue
    return parsed


def flatten_candidates(payloads: list[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for payload in payloads:
        if isinstance(payload, list):
            out.extend([x for x in payload if isinstance(x, dict)])
        elif isinstance(payload, dict):
            if "candidates" in payload and isinstance(payload["candidates"], list):
                out.extend([x for x in payload["candidates"] if isinstance(x, dict)])
            else:
                out.append(payload)
    return out


def apply_candidate(ruleset: RuleSet, candidate: dict[str, Any]) -> RuleSet | None:
    op = str(candidate.get("operation", "")).strip().lower()
    if op == "abstain":
        return None
    edited = copy.deepcopy(ruleset)
    if op == "add_condition":
        rule = _find_rule(edited, str(candidate["parent_rule_id"]))
        rule.conditions.append(Condition.from_dict(candidate["condition"]))
    elif op == "remove_condition":
        rule = _find_rule(edited, str(candidate["parent_rule_id"]))
        idx = int(candidate["condition_index"])
        if idx < 0 or idx >= len(rule.conditions):
            raise ValueError("condition_index out of range")
        del rule.conditions[idx]
    elif op == "change_threshold":
        rule = _find_rule(edited, str(candidate["parent_rule_id"]))
        idx = int(candidate["condition_index"])
        if idx < 0 or idx >= len(rule.conditions):
            raise ValueError("condition_index out of range")
        rule.conditions[idx].value = candidate["new_value"]
    elif op == "split_rule":
        parent_id = str(candidate["parent_rule_id"])
        new_rules = _parse_new_rules(edited, candidate["new_rules"], parent_id)
        _replace_rules(edited, [parent_id], new_rules)
    elif op == "merge_rules":
        parent_ids = [str(v) for v in candidate["parent_rule_ids"]]
        new_rules = _parse_new_rules(edited, candidate["new_rules"], parent_ids[0])
        _replace_rules(edited, parent_ids, new_rules)
    elif op == "change_default":
        edited.default_label = str(candidate["default_label"])
    else:
        raise ValueError(f"Unsupported operation: {op}")
    edited.metadata = dict(edited.metadata)
    edited.metadata["last_operation"] = candidate
    return edited


def _find_rule(ruleset: RuleSet, rule_id: str) -> Rule:
    for rule in ruleset.rules:
        if rule.rule_id == rule_id:
            return rule
    raise KeyError(f"Unknown rule id: {rule_id}")


def _replace_rules(ruleset: RuleSet, parent_ids: list[str], new_rules: list[Rule]) -> None:
    parent_set = set(parent_ids)
    kept = [rule for rule in ruleset.rules if rule.rule_id not in parent_set]
    min_priority = min((_find_rule(ruleset, rid).priority for rid in parent_ids), default=len(kept))
    for i, rule in enumerate(new_rules):
        rule.priority = min_priority + i
    ruleset.rules = sorted(kept + new_rules, key=lambda r: (r.priority, r.rule_id))
    for priority, rule in enumerate(ruleset.rules):
        rule.priority = priority


def _parse_new_rules(ruleset: RuleSet, records: list[dict[str, Any]], parent_id: str) -> list[Rule]:
    new_rules: list[Rule] = []
    for idx, record in enumerate(records):
        conditions = record.get("conditions", record.get("if", []))
        rule_id = str(record.get("rule_id", f"{parent_id}_s{idx}"))
        if any(rule.rule_id == rule_id for rule in ruleset.rules) or any(r.rule_id == rule_id for r in new_rules):
            rule_id = ruleset.next_rule_id(prefix=f"{parent_id}_s")
        new_rules.append(
            Rule(
                rule_id=rule_id,
                conditions=[Condition.from_dict(c) for c in conditions],
                then=str(record["then"]),
                metadata={"created_from": parent_id},
            )
        )
    return new_rules
