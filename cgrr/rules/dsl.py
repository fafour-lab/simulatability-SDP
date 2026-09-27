"""Machine-readable rule DSL used by the refiner and verifier."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


ALLOWED_OPS = {"<=", "<", ">=", ">", "==", "!="}


@dataclass
class Condition:
    feature: str
    op: str
    value: float | int | str

    def to_dict(self) -> dict[str, Any]:
        return {"feature": self.feature, "op": self.op, "value": self.value}

    @classmethod
    def from_dict(cls, obj: dict[str, Any] | list[Any] | tuple[Any, ...]) -> "Condition":
        """Parse a condition from the canonical object DSL or Qwen's compact list form."""
        if isinstance(obj, (list, tuple)):
            if len(obj) != 3:
                raise ValueError(f"Condition list must have exactly 3 items [feature, op, value], got: {obj}")
            feature, op, value = obj
            op = str(op).strip()
            if op == "=":
                op = "=="
            return cls(feature=str(feature), op=op, value=value)

        op = str(obj["op"]).strip()
        if op == "=":
            op = "=="
        return cls(feature=str(obj["feature"]), op=op, value=obj["value"])


@dataclass
class Rule:
    rule_id: str
    conditions: list[Condition]
    then: str
    priority: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "if": [c.to_dict() for c in self.conditions],
            "then": self.then,
            "priority": self.priority,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, obj: dict[str, Any]) -> "Rule":
        conditions = obj.get("conditions", obj.get("if", []))
        return cls(
            rule_id=str(obj["rule_id"]),
            conditions=[Condition.from_dict(c) for c in conditions],
            then=str(obj["then"]),
            priority=int(obj.get("priority", 0)),
            metadata=dict(obj.get("metadata", {})),
        )


@dataclass
class RuleSet:
    name: str
    default_label: str
    rules: list[Rule]
    labels: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "default_label": self.default_label,
            "labels": self.labels,
            "rules": [r.to_dict() for r in self.rules],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, obj: dict[str, Any]) -> "RuleSet":
        return cls(
            name=str(obj["name"]),
            default_label=str(obj["default_label"]),
            labels=[str(v) for v in obj["labels"]],
            rules=[Rule.from_dict(r) for r in obj["rules"]],
            metadata=dict(obj.get("metadata", {})),
        )

    def complexity(self) -> dict[str, float]:
        n_rules = len(self.rules)
        predicate_counts = [len(r.conditions) for r in self.rules]
        total = int(sum(predicate_counts))
        return {
            "n_rules": float(n_rules),
            "total_predicates": float(total),
            "avg_predicates": float(total / n_rules) if n_rules else 0.0,
            "max_predicates": float(max(predicate_counts)) if predicate_counts else 0.0,
        }

    def next_rule_id(self, prefix: str = "r") -> str:
        seen = {rule.rule_id for rule in self.rules}
        idx = len(seen)
        while f"{prefix}_{idx:03d}" in seen:
            idx += 1
        return f"{prefix}_{idx:03d}"


def condition_to_text(condition: Condition) -> str:
    value = condition.value
    if isinstance(value, float):
        value = f"{value:.4g}"
    return f"{condition.feature} {condition.op} {value}"


def rule_to_text(rule: Rule) -> str:
    if rule.conditions:
        premise = " AND ".join(condition_to_text(c) for c in rule.conditions)
    else:
        premise = "always"
    return f"{rule.rule_id}: IF {premise} THEN {rule.then}"


def ruleset_to_text(ruleset: RuleSet) -> str:
    lines = [f"Rule set: {ruleset.name}", f"Default label: {ruleset.default_label}", ""]
    for rule in sorted(ruleset.rules, key=lambda r: (r.priority, r.rule_id)):
        lines.append(rule_to_text(rule))
    return "\n".join(lines) + "\n"
