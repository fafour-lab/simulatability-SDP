"""Simple rule baselines used by the evaluation scripts."""

from __future__ import annotations

import copy
import random

from cgrr.rules.dsl import RuleSet


def corrupted_ruleset(ruleset: RuleSet, seed: int) -> RuleSet:
    """Shuffle consequents as a negative-control explanation."""
    rng = random.Random(seed)
    edited = copy.deepcopy(ruleset)
    labels = [rule.then for rule in edited.rules]
    rng.shuffle(labels)
    for rule, label in zip(edited.rules, labels):
        rule.then = label
    edited.name = f"{ruleset.name}-Corrupted"
    edited.metadata = dict(edited.metadata)
    edited.metadata["baseline"] = "shuffled_consequents"
    return edited
