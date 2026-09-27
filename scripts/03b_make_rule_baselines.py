#!/usr/bin/env python3
"""Stage 03b: create rule-only baseline conditions such as corrupted R0."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cgrr.baselines import corrupted_ruleset
from cgrr.config import load_config, output_dir, project_seed, resolve_path
from cgrr.rules.dsl import RuleSet, ruleset_to_text
from cgrr.utils.io import read_json, write_json
from cgrr.utils.logging import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/heloc_cgrr.yaml")
    parser.add_argument("--input-rules", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(output_dir(cfg, "logs", "03b_make_rule_baselines.log"))
    seed = project_seed(cfg)
    root = cfg["project"]["root_dir"]
    rules_path = resolve_path(args.input_rules, root) if args.input_rules else output_dir(cfg, "rules", "R0.json")
    r0 = RuleSet.from_dict(read_json(rules_path))

    corrupted = corrupted_ruleset(r0, seed=seed)
    corrupted.name = "Corrupted-R0"
    write_json(corrupted.to_dict(), output_dir(cfg, "rules", "Corrupted-R0.json"))
    output_dir(cfg, "rules", "Corrupted-R0.txt").write_text(ruleset_to_text(corrupted), encoding="utf-8")


if __name__ == "__main__":
    main()
