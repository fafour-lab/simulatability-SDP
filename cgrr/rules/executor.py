"""Executable semantics for rule sets."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cgrr.rules.dsl import Condition, Rule, RuleSet


def condition_mask(X: pd.DataFrame, condition: Condition) -> pd.Series:
    if condition.feature not in X.columns:
        raise KeyError(f"Unknown feature in rule condition: {condition.feature}")
    left = X[condition.feature]
    value = condition.value
    if condition.op == "<=":
        return left <= float(value)
    if condition.op == "<":
        return left < float(value)
    if condition.op == ">=":
        return left >= float(value)
    if condition.op == ">":
        return left > float(value)
    if condition.op == "==":
        return left == value
    if condition.op == "!=":
        return left != value
    raise ValueError(f"Unsupported operator: {condition.op}")


def rule_mask(X: pd.DataFrame, rule: Rule) -> pd.Series:
    mask = pd.Series(True, index=X.index)
    for condition in rule.conditions:
        mask = mask & condition_mask(X, condition)
    return mask


def predict_ruleset(
    ruleset: RuleSet,
    X: pd.DataFrame,
    return_details: bool = False,
) -> tuple[np.ndarray, pd.DataFrame] | np.ndarray:
    ordered_rules = sorted(ruleset.rules, key=lambda r: (r.priority, r.rule_id))
    n = len(X)
    pred = np.array([ruleset.default_label] * n, dtype=object)
    first_match = np.array([None] * n, dtype=object)
    cover_count = np.zeros(n, dtype=int)
    label_sets: list[set[str]] = [set() for _ in range(n)]
    matched_rule_ids: list[list[str]] = [[] for _ in range(n)]

    for rule in ordered_rules:
        mask = rule_mask(X, rule).to_numpy()
        idx = np.flatnonzero(mask)
        if len(idx) == 0:
            continue
        cover_count[idx] += 1
        for i in idx:
            label_sets[i].add(rule.then)
            matched_rule_ids[i].append(rule.rule_id)
            if first_match[i] is None:
                first_match[i] = rule.rule_id
                pred[i] = rule.then

    if not return_details:
        return pred

    details = pd.DataFrame(
        {
            "covered": cover_count > 0,
            "cover_count": cover_count,
            "conflict": [len(s) > 1 for s in label_sets],
            "first_rule_id": first_match,
            "matched_rule_ids": [";".join(v) for v in matched_rule_ids],
        },
        index=X.index,
    )
    return pred, details


def per_rule_stats(ruleset: RuleSet, X: pd.DataFrame, m_labels: pd.Series | np.ndarray) -> pd.DataFrame:
    m = np.asarray(m_labels, dtype=object)
    records: list[dict[str, Any]] = []
    for rule in ruleset.rules:
        mask = rule_mask(X, rule).to_numpy()
        support = int(mask.sum())
        if support:
            precision = float(np.mean(m[mask] == rule.then))
        else:
            precision = np.nan
        records.append(
            {
                "rule_id": rule.rule_id,
                "then": rule.then,
                "support": support,
                "coverage": float(support / len(X)) if len(X) else 0.0,
                "precision": precision,
                "n_predicates": len(rule.conditions),
            }
        )
    return pd.DataFrame(records)
