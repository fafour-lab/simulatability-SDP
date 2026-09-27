from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "refine_rules_script", ROOT / "scripts" / "04_refine_rules.py"
)
assert SPEC is not None and SPEC.loader is not None
REFINE_RULES = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REFINE_RULES)


class ProxyRefinementLogTests(unittest.TestCase):
    def _write_csv(self, rows: list[dict[str, object]]) -> Path:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        path = Path(temp_dir.name) / "proxy.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        return path

    def test_loads_only_parseable_disagreements_from_refine_split(self) -> None:
        path = self._write_csv(
            [
                {
                    "row_id": 10,
                    "condition": "R0",
                    "mode": "forward",
                    "data_split": "refine",
                    "m_label": "Good",
                    "predicted_black_box_label": "Bad",
                },
                {
                    "row_id": 11,
                    "condition": "R0",
                    "mode": "forward",
                    "data_split": "refine",
                    "m_label": "Bad",
                    "predicted_black_box_label": "Bad",
                },
                {
                    "row_id": 12,
                    "condition": "R0",
                    "mode": "forward",
                    "data_split": "refine",
                    "m_label": "Good",
                    "predicted_black_box_label": None,
                },
            ]
        )

        failures = REFINE_RULES._load_proxy_failures(
            path, {10: "Good", 11: "Bad", 12: "Good"}, "R0"
        )

        self.assertEqual(failures["row_id"].tolist(), [10])

    def test_rejects_eval_proxy_log(self) -> None:
        path = self._write_csv(
            [
                {
                    "row_id": 20,
                    "condition": "R0",
                    "mode": "forward",
                    "data_split": "eval",
                    "m_label": "Good",
                    "predicted_black_box_label": "Bad",
                }
            ]
        )

        with self.assertRaisesRegex(ValueError, "data_split=refine"):
            REFINE_RULES._load_proxy_failures(path, {20: "Good"}, "R0")

    def test_rejects_rows_outside_refinement_split_without_split_column(self) -> None:
        path = self._write_csv(
            [
                {
                    "row_id": 99,
                    "condition": "R0",
                    "mode": "forward",
                    "m_label": "Good",
                    "predicted_black_box_label": "Bad",
                }
            ]
        )

        with self.assertRaisesRegex(ValueError, "outside the refinement split"):
            REFINE_RULES._load_proxy_failures(path, {20: "Good"}, "R0")

    def test_rejects_proxy_log_for_a_different_rule_condition(self) -> None:
        path = self._write_csv(
            [
                {
                    "row_id": 20,
                    "condition": "x-only",
                    "mode": "forward",
                    "data_split": "refine",
                    "m_label": "Good",
                    "predicted_black_box_label": "Bad",
                }
            ]
        )

        with self.assertRaisesRegex(ValueError, "condition='R0'"):
            REFINE_RULES._load_proxy_failures(path, {20: "Good"}, "R0")


if __name__ == "__main__":
    unittest.main()
