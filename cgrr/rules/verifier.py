"""Verifier-constrained acceptance logic for candidate rule edits."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cgrr.rules.dsl import ALLOWED_OPS, Condition, RuleSet
from cgrr.rules.executor import per_rule_stats, predict_ruleset


def score_ruleset(
    ruleset: RuleSet,
    X: pd.DataFrame,
    m_labels: pd.Series | np.ndarray,
    objective_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    m = np.asarray(m_labels, dtype=object)
    pred, details = predict_ruleset(ruleset, X, return_details=True)
    covered = details["covered"].to_numpy()
    full_fidelity = float(np.mean(pred == m)) if len(m) else 0.0
    covered_fidelity = float(np.mean(pred[covered] == m[covered])) if np.any(covered) else 0.0
    coverage = float(np.mean(covered)) if len(m) else 0.0
    conflict_rate = float(details["conflict"].mean()) if len(details) else 0.0
    complexity = ruleset.complexity()
    objective_cfg = objective_cfg or {}
    objective = (
        full_fidelity
        + float(objective_cfg.get("alpha_coverage", 0.0)) * coverage
        - float(objective_cfg.get("lambda_complexity", 0.0)) * complexity["avg_predicates"]
        - float(objective_cfg.get("beta_conflict", 0.0)) * conflict_rate
    )
    return {
        "full_fidelity": full_fidelity,
        "covered_fidelity": covered_fidelity,
        "coverage": coverage,
        "conflict_rate": conflict_rate,
        "objective": float(objective),
        **complexity,
    }


def rule_counterexamples(
    ruleset: RuleSet,
    X: pd.DataFrame,
    raw_df: pd.DataFrame,
    m_labels: pd.Series | np.ndarray,
) -> pd.DataFrame:
    m = np.asarray(m_labels, dtype=object)
    pred, details = predict_ruleset(ruleset, X, return_details=True)
    out = raw_df[["row_id"]].copy()
    out["rule_label"] = pred
    out["m_label"] = m
    out["covered"] = details["covered"].values
    out["first_rule_id"] = details["first_rule_id"].values
    out["is_faithfulness_counterexample"] = pred != m
    return out


def _feature_map(feature_schema: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {f["name"]: f for f in feature_schema["features"]}


def validate_ruleset(
    ruleset: RuleSet,
    feature_schema: dict[str, Any],
    constraints: dict[str, Any],
) -> tuple[bool, list[str]]:
    """Reject malformed or unverifiable rules before scoring."""
    errors: list[str] = []
    labels = set(feature_schema["labels"].keys())
    features = _feature_map(feature_schema)
    complexity = ruleset.complexity()

    if ruleset.default_label not in labels:
        errors.append(f"default label {ruleset.default_label} is not in schema labels")
    if complexity["n_rules"] > float(constraints.get("max_rules", float("inf"))):
        errors.append("too many rules")
    if complexity["avg_predicates"] > float(constraints.get("max_avg_predicates", float("inf"))):
        errors.append("average predicate budget exceeded")
    if complexity["total_predicates"] > float(constraints.get("max_total_predicates", float("inf"))):
        errors.append("total predicate budget exceeded")

    seen_ids: set[str] = set()
    for rule in ruleset.rules:
        if rule.rule_id in seen_ids:
            errors.append(f"duplicate rule id {rule.rule_id}")
        seen_ids.add(rule.rule_id)
        if rule.then not in labels:
            errors.append(f"rule {rule.rule_id} has invalid consequent {rule.then}")
        bounds: dict[str, dict[str, float]] = {}
        for condition in rule.conditions:
            _validate_condition(condition, features, errors, rule.rule_id)
            _update_bounds(condition, bounds)
        for feature, bound in bounds.items():
            if bound.get("lower", -float("inf")) > bound.get("upper", float("inf")):
                errors.append(f"rule {rule.rule_id} has impossible bounds for {feature}")
    return len(errors) == 0, errors


def _validate_condition(
    condition: Condition,
    features: dict[str, dict[str, Any]],
    errors: list[str],
    rule_id: str,
) -> None:
    if condition.feature not in features:
        errors.append(f"rule {rule_id} uses unknown feature {condition.feature}")
        return
    if condition.op not in ALLOWED_OPS:
        errors.append(f"rule {rule_id} uses unsupported operator {condition.op}")
        return
    meta = features[condition.feature]
    if meta["kind"] == "binary":
        if condition.op not in {"==", "!="}:
            errors.append(f"binary feature {condition.feature} requires == or !=")
        if int(float(condition.value)) not in {0, 1}:
            errors.append(f"binary feature {condition.feature} threshold must be 0 or 1")
    elif meta["kind"] == "numeric":
        try:
            value = float(condition.value)
        except (TypeError, ValueError):
            errors.append(f"numeric feature {condition.feature} has nonnumeric value")
            return
        if "min" in meta and value < float(meta["min"]) - 1e-9:
            errors.append(f"{condition.feature} threshold below training range")
        if "max" in meta and value > float(meta["max"]) + 1e-9:
            errors.append(f"{condition.feature} threshold above training range")


def _update_bounds(condition: Condition, bounds: dict[str, dict[str, float]]) -> None:
    try:
        value = float(condition.value)
    except (TypeError, ValueError):
        return
    item = bounds.setdefault(condition.feature, {})
    if condition.op in {">", ">="}:
        item["lower"] = max(item.get("lower", -float("inf")), value)
    if condition.op in {"<", "<="}:
        item["upper"] = min(item.get("upper", float("inf")), value)


def accept_candidate(
    current: RuleSet,
    candidate: RuleSet,
    X_refine: pd.DataFrame,
    m_refine: pd.Series | np.ndarray,
    feature_schema: dict[str, Any],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    ref_cfg = cfg["refinement"]
    constraints = ref_cfg["constraints"]
    valid, errors = validate_ruleset(candidate, feature_schema, constraints)
    current_score = score_ruleset(current, X_refine, m_refine, ref_cfg["objective"])
    if not valid:
        return {
            "accepted": False,
            "reason": "invalid_ruleset",
            "errors": errors,
            "current_score": current_score,
            "candidate_score": None,
        }

    candidate_score = score_ruleset(candidate, X_refine, m_refine, ref_cfg["objective"])
    if candidate_score["coverage"] < float(constraints.get("min_coverage", 0.0)):
        return {
            "accepted": False,
            "reason": "coverage_constraint",
            "errors": [],
            "current_score": current_score,
            "candidate_score": candidate_score,
        }
    if candidate_score["conflict_rate"] > float(constraints.get("max_conflict_rate", 1.0)):
        return {
            "accepted": False,
            "reason": "conflict_constraint",
            "errors": [],
            "current_score": current_score,
            "candidate_score": candidate_score,
        }
    improvement = candidate_score["objective"] - current_score["objective"]
    if improvement <= float(ref_cfg.get("epsilon", 0.0)):
        return {
            "accepted": False,
            "reason": "objective_not_improved",
            "errors": [],
            "current_score": current_score,
            "candidate_score": candidate_score,
            "objective_improvement": float(improvement),
        }
    return {
        "accepted": True,
        "reason": "accepted",
        "errors": [],
        "current_score": current_score,
        "candidate_score": candidate_score,
        "objective_improvement": float(improvement),
    }


def weak_rules(
    ruleset: RuleSet,
    X_refine: pd.DataFrame,
    m_refine: pd.Series | np.ndarray,
    limit: int,
) -> pd.DataFrame:
    stats = per_rule_stats(ruleset, X_refine, m_refine)
    stats["precision_rank_value"] = stats["precision"].fillna(0.0)
    stats = stats.sort_values(
        by=["precision_rank_value", "support", "n_predicates"],
        ascending=[True, False, False],
    )
    return stats.head(limit).drop(columns=["precision_rank_value"])
