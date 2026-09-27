"""Extract executable rules from a fitted sklearn decision tree surrogate."""

from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.tree import DecisionTreeClassifier, _tree

from cgrr.rules.dsl import Condition, Rule, RuleSet, ruleset_to_text


def extract_rules_from_tree(
    tree: DecisionTreeClassifier,
    feature_names: list[str],
    label_decoder: Callable[[int], str],
    default_label: str,
    name: str = "R0",
) -> RuleSet:
    rules: list[Rule] = []
    tree_ = tree.tree_

    def walk(node_id: int, conditions: list[Condition]) -> None:
        left = tree_.children_left[node_id]
        right = tree_.children_right[node_id]
        if left == _tree.TREE_LEAF:
            counts = tree_.value[node_id][0]
            class_idx = int(np.argmax(counts))
            class_value = int(tree.classes_[class_idx])
            support = int(np.sum(counts))
            rule_id = f"r_{len(rules):03d}"
            rules.append(
                Rule(
                    rule_id=rule_id,
                    conditions=list(conditions),
                    then=label_decoder(class_value),
                    priority=len(rules),
                    metadata={"surrogate_leaf_support": support},
                )
            )
            return

        feature = feature_names[tree_.feature[node_id]]
        threshold = float(tree_.threshold[node_id])
        walk(left, conditions + [Condition(feature=feature, op="<=", value=threshold)])
        walk(right, conditions + [Condition(feature=feature, op=">", value=threshold)])

    walk(0, [])
    labels = [label_decoder(int(v)) for v in sorted(tree.classes_)]
    return RuleSet(name=name, default_label=default_label, labels=labels, rules=rules)


def write_rules_text(ruleset: RuleSet, path: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(ruleset_to_text(ruleset))
