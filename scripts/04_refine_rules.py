#!/usr/bin/env python3
"""Stage 04: run verifier-constrained Qwen-8B rule refinement."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cgrr.config import load_config, output_dir, project_seed, resolve_path
from cgrr.data import PreprocessorState, attach_splits, feature_columns, load_dataset, transform_features
from cgrr.llm.prompts import build_refiner_prompt
from cgrr.llm.qwen_client import GenerationConfig, QwenClient
from cgrr.llm.refiner import apply_candidate, extract_json_payloads, flatten_candidates
from cgrr.rules.dsl import Rule, RuleSet, ruleset_to_text
from cgrr.rules.executor import rule_mask
from cgrr.rules.verifier import accept_candidate, rule_counterexamples, score_ruleset, weak_rules
from cgrr.utils.io import append_jsonl, read_json, write_json
from cgrr.utils.logging import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/heloc_cgrr.yaml")
    parser.add_argument("--input-rules", default=None, help="Defaults to outputs/.../rules/R0.json")
    parser.add_argument("--output-name", default="Faithful-Refined")
    parser.add_argument(
        "--source",
        choices=["faithfulness", "proxy", "hybrid", "none"],
        default="faithfulness",
        help="Counterexample source used in prompts.",
    )
    parser.add_argument(
        "--proxy-log",
        default=None,
        help=(
            "Refine-split CSV from Stage 06. For proxy/hybrid sources, defaults to "
            "outputs/.../refinement_inputs/R0_forward_refine.csv."
        ),
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(output_dir(cfg, "logs", f"04_refine_{args.output_name}.log"))
    seed = project_seed(cfg)

    split_df = pd.read_csv(output_dir(cfg, "splits", f"split_assignments_seed{seed}.csv"))
    raw = attach_splits(load_dataset(cfg), split_df)
    original_feature_cols = feature_columns(cfg, raw)
    state = PreprocessorState.from_dict(read_json(output_dir(cfg, "preprocess", "preprocessor_state.json")))
    feature_schema = read_json(output_dir(cfg, "preprocess", "feature_schema.json"))
    X = transform_features(raw, state)
    bb = pd.read_csv(output_dir(cfg, "predictions", "blackbox_predictions.csv"))
    m_name = bb.set_index("row_id").loc[raw["row_id"], "m_label_name"].reset_index(drop=True)

    root = cfg["project"]["root_dir"]
    rules_path = resolve_path(args.input_rules, root) if args.input_rules else output_dir(cfg, "rules", "R0.json")
    current = RuleSet.from_dict(read_json(rules_path))
    refine_mask = raw["split"] == "refine"
    X_refine = X.loc[refine_mask]
    raw_refine = raw.loc[refine_mask].reset_index(drop=True)
    m_refine = m_name.loc[refine_mask].reset_index(drop=True)
    X_refine = X_refine.reset_index(drop=True)

    proxy_log = resolve_path(args.proxy_log, root) if args.proxy_log else None
    if args.source in {"proxy", "hybrid"}:
        if proxy_log is None:
            proxy_log = output_dir(cfg, "refinement_inputs", "R0_forward_refine.csv")
        expected_labels = dict(zip(raw_refine["row_id"].astype(int), m_refine.astype(str)))
        proxy_failures = _load_proxy_failures(proxy_log, expected_labels, current.name)
    else:
        proxy_failures = pd.DataFrame()
    client = QwenClient(GenerationConfig.from_dict(cfg["qwen"]))
    log_path = output_dir(cfg, "logs", f"refinement_{args.output_name}.jsonl")
    prompt_dir = output_dir(cfg, "prompts", args.output_name)
    prompt_dir.mkdir(parents=True, exist_ok=True)

    for iteration in range(1, int(cfg["refinement"]["max_iterations"]) + 1):
        weak = _select_weak_rules(
            source=args.source,
            ruleset=current,
            X_refine=X_refine,
            raw_refine=raw_refine,
            m_refine=m_refine,
            proxy_failures=proxy_failures,
            limit=int(cfg["refinement"]["weak_rule_count"]),
        )
        if weak.empty:
            if iteration == 1 and args.source == "proxy":
                raise ValueError(
                    "Proxy/black-box disagreements do not map to any rule in the input rule set"
                )
            break
        accepted_this_iteration = False
        for _, weak_row in weak.iterrows():
            rule = _get_rule(current, weak_row["rule_id"])
            examples = _examples_for_rule(
                source=args.source,
                rule=rule,
                ruleset=current,
                X_refine=X_refine,
                raw_refine=raw_refine,
                m_refine=m_refine,
                proxy_failures=proxy_failures,
                max_examples=int(cfg["refinement"]["examples_per_rule"]),
            )
            system, user = build_refiner_prompt(
                ruleset=current,
                rule=rule,
                rule_stats=weak_row.to_dict(),
                feature_schema=feature_schema,
                examples=examples,
                feature_columns=original_feature_cols,
                max_examples=int(cfg["refinement"]["examples_per_rule"]),
                source=args.source,
            )
            prompt_file = prompt_dir / f"iter{iteration:02d}_{rule.rule_id}.json"
            prompt_file.write_text(json.dumps({"system": system, "user": user}, indent=2), encoding="utf-8")
            raw_response = client.generate(system, user)
            candidates = flatten_candidates(extract_json_payloads(raw_response))[: int(cfg["refinement"]["beam_size"])]
            best_candidate: tuple[dict[str, Any], RuleSet, dict[str, Any]] | None = None
            for candidate_idx, candidate in enumerate(candidates):
                record: dict[str, Any] = {
                    "iteration": iteration,
                    "rule_id": rule.rule_id,
                    "candidate_idx": candidate_idx,
                    "candidate": candidate,
                    "raw_response": raw_response,
                }
                try:
                    proposed = apply_candidate(current, candidate)
                    if proposed is None:
                        record.update({"accepted": False, "reason": "abstain"})
                        append_jsonl([record], log_path)
                        continue
                    verdict = accept_candidate(
                        current=current,
                        candidate=proposed,
                        X_refine=X_refine,
                        m_refine=m_refine,
                        feature_schema=feature_schema,
                        cfg=cfg,
                    )
                    record.update(verdict)
                    append_jsonl([record], log_path)
                    if verdict["accepted"]:
                        if best_candidate is None:
                            best_candidate = (candidate, proposed, verdict)
                        elif verdict["candidate_score"]["objective"] > best_candidate[2]["candidate_score"]["objective"]:
                            best_candidate = (candidate, proposed, verdict)
                except Exception as exc:  # noqa: BLE001 - logged as rejection evidence.
                    record.update({"accepted": False, "reason": "exception", "errors": [repr(exc)]})
                    append_jsonl([record], log_path)
            if best_candidate is not None:
                current = best_candidate[1]
                current.name = args.output_name
                accepted_this_iteration = True
                break
        if not accepted_this_iteration:
            break

    current.name = args.output_name
    final_score = score_ruleset(current, X_refine, m_refine, cfg["refinement"]["objective"])
    current.metadata["final_refine_score"] = final_score
    current.metadata["refinement_source"] = args.source
    if proxy_log is not None:
        current.metadata["proxy_log"] = str(proxy_log)
        current.metadata["proxy_disagreement_count"] = int(len(proxy_failures))
    write_json(current.to_dict(), output_dir(cfg, "rules", f"{args.output_name}.json"))
    output_dir(cfg, "rules").mkdir(parents=True, exist_ok=True)
    output_dir(cfg, "rules", f"{args.output_name}.txt").write_text(ruleset_to_text(current), encoding="utf-8")


def _get_rule(ruleset: RuleSet, rule_id: str) -> Rule:
    for rule in ruleset.rules:
        if rule.rule_id == rule_id:
            return rule
    raise KeyError(rule_id)


def _load_proxy_failures(
    path: Path,
    expected_labels: dict[int, str],
    expected_condition: str,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Proxy-refinement log does not exist: {path}. "
            "Run scripts/06_proxy_simulation.py with --condition R0 --mode forward --split refine first."
        )
    proxy = pd.read_csv(path)
    required = {"row_id", "condition", "mode", "m_label", "predicted_black_box_label"}
    missing = sorted(required - set(proxy.columns))
    if missing:
        raise ValueError(f"Proxy-refinement log is missing required columns: {missing}")
    if proxy.empty:
        raise ValueError(f"Proxy-refinement log is empty: {path}")
    if not proxy["row_id"].is_unique:
        raise ValueError("Proxy-refinement log must contain at most one forward prediction per row_id")
    modes = set(proxy["mode"].dropna().astype(str))
    if modes != {"forward"}:
        raise ValueError(f"Proxy-refinement log must contain only forward rows; found modes={sorted(modes)}")
    conditions = set(proxy["condition"].dropna().astype(str))
    if conditions != {expected_condition}:
        raise ValueError(
            f"Proxy-refinement log must use condition={expected_condition!r}; "
            f"found conditions={sorted(conditions)}"
        )
    if "data_split" in proxy.columns:
        data_splits = set(proxy["data_split"].dropna().astype(str))
        if data_splits != {"refine"}:
            raise ValueError(
                "Proxy-refinement log must contain only data_split=refine rows; "
                f"found data_split={sorted(data_splits)}"
            )

    proxy["row_id"] = proxy["row_id"].astype(int)
    expected_ids = set(expected_labels)
    observed_ids = set(proxy["row_id"])
    unexpected = sorted(observed_ids - expected_ids)
    if unexpected:
        preview = unexpected[:10]
        raise ValueError(
            "Proxy-refinement log contains rows outside the refinement split "
            f"(first unexpected row_ids: {preview})"
        )
    for row in proxy[["row_id", "m_label"]].itertuples(index=False):
        expected = expected_labels[int(row.row_id)]
        if str(row.m_label) != expected:
            raise ValueError(
                f"Proxy log black-box label mismatch for row_id={int(row.row_id)}: "
                f"log={row.m_label!r}, expected={expected!r}"
            )

    parseable = proxy["predicted_black_box_label"].notna()
    failures = proxy.loc[
        parseable
        & (proxy["predicted_black_box_label"].astype(str) != proxy["m_label"].astype(str))
    ].copy()
    if failures.empty:
        raise ValueError("Proxy-refinement log contains no parseable proxy/black-box disagreements")
    return failures


def _select_weak_rules(
    source: str,
    ruleset: RuleSet,
    X_refine: pd.DataFrame,
    raw_refine: pd.DataFrame,
    m_refine: pd.Series,
    proxy_failures: pd.DataFrame,
    limit: int,
) -> pd.DataFrame:
    if source not in {"proxy", "hybrid"}:
        return weak_rules(ruleset, X_refine, m_refine, limit=limit)

    stats = weak_rules(ruleset, X_refine, m_refine, limit=len(ruleset.rules))
    counterexamples = rule_counterexamples(ruleset, X_refine, raw_refine, m_refine)
    failure_ids = set(proxy_failures["row_id"].astype(int))
    counts = (
        counterexamples.loc[counterexamples["row_id"].isin(failure_ids), "first_rule_id"]
        .dropna()
        .value_counts()
    )
    stats["proxy_disagreement_count"] = stats["rule_id"].map(counts).fillna(0).astype(int)
    if source == "proxy":
        stats = stats[stats["proxy_disagreement_count"] > 0]
    stats = stats.sort_values(
        by=["proxy_disagreement_count", "precision", "support", "n_predicates"],
        ascending=[False, True, False, False],
        na_position="first",
    )
    return stats.head(limit)


def _examples_for_rule(
    source: str,
    rule: Rule,
    ruleset: RuleSet,
    X_refine: pd.DataFrame,
    raw_refine: pd.DataFrame,
    m_refine: pd.Series,
    proxy_failures: pd.DataFrame,
    max_examples: int,
) -> pd.DataFrame:
    ce = rule_counterexamples(ruleset, X_refine, raw_refine, m_refine)
    ce = ce.merge(raw_refine, on="row_id", how="left")
    rule_specific = ce[ce["first_rule_id"] == rule.rule_id].copy()
    faith = rule_specific[rule_specific["is_faithfulness_counterexample"]].copy()
    faith["source"] = "faithfulness"

    mask = rule_mask(X_refine, rule).reset_index(drop=True)
    covered = ce.loc[mask].copy()
    covered["source"] = "covered"

    if source == "none":
        return covered.head(max_examples)
    if source == "faithfulness":
        return pd.concat([faith, covered], ignore_index=True).drop_duplicates("row_id").head(max_examples)

    proxy = pd.DataFrame()
    if not proxy_failures.empty:
        proxy_columns = ["row_id", "predicted_black_box_label"]
        proxy_columns.extend(
            col for col in ["confidence", "short_reason"] if col in proxy_failures.columns
        )
        proxy_details = proxy_failures[proxy_columns].rename(
            columns={
                "predicted_black_box_label": "proxy_label",
                "confidence": "proxy_confidence",
                "short_reason": "proxy_reason",
            }
        )
        proxy = rule_specific.merge(proxy_details, on="row_id", how="inner")
        proxy["source"] = "proxy"
    if source == "proxy":
        return pd.concat([proxy, covered], ignore_index=True).drop_duplicates("row_id").head(max_examples)
    if source == "hybrid":
        return pd.concat([faith, proxy, covered], ignore_index=True).drop_duplicates("row_id").head(max_examples)
    raise ValueError(source)


if __name__ == "__main__":
    main()
