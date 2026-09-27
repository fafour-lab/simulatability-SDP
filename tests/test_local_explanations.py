from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from cgrr.local_explanations import (
    LOCAL_EXPLANATION_FORMAT_VERSION,
    anchor_explanation_text,
    explain_lime_instance,
    lime_explanation_text,
)


class _FakeModel:
    classes_ = np.array([0, 1])

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(frame), dtype=int)

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        return np.tile(np.array([[0.857, 0.143]]), (len(frame), 1))


class _FakeLimeExplanation:
    score = 0.91
    local_pred = np.array([0.14])

    def __init__(self) -> None:
        self.requested_label: int | None = None

    def as_list(self, label: int) -> list[tuple[str, float]]:
        self.requested_label = label
        return [("feature <= 1", 0.25)]


class _FakeLimeExplainer:
    def __init__(self) -> None:
        self.labels: tuple[int, ...] | None = None
        self.explanation = _FakeLimeExplanation()

    def explain_instance(self, **kwargs: object) -> _FakeLimeExplanation:
        self.labels = kwargs["labels"]  # type: ignore[assignment]
        return self.explanation


class LocalExplanationLeakageTests(unittest.TestCase):
    def test_lime_uses_fixed_class_instead_of_predicted_class(self) -> None:
        explainer = _FakeLimeExplainer()
        result = explain_lime_instance(
            explainer=explainer,
            model=_FakeModel(),
            x_row=pd.Series({"feature": 0.0}),
            class_names=["Bad", "Good"],
            explanation_class_name="Good",
        )

        self.assertEqual(explainer.labels, (1,))
        self.assertEqual(result["target_label"], "Bad")
        self.assertEqual(result["explanation_format_version"], LOCAL_EXPLANATION_FORMAT_VERSION)
        self.assertNotIn("predicts Bad", result["explanation_text"])
        self.assertNotIn("0.857", result["explanation_text"])

    def test_lime_proxy_text_has_no_row_specific_prediction(self) -> None:
        text = lime_explanation_text("Good", [{"description": "feature <= 1", "weight": 0.25}])

        self.assertIn("fixed reference class Good", text)
        self.assertNotIn("black-box predicts", text.lower())
        self.assertNotIn("probability", text.lower())

    def test_anchor_proxy_text_withholds_outcome(self) -> None:
        text = anchor_explanation_text(["feature <= 1"], precision=0.973, coverage=0.099)

        self.assertIn("feature <= 1", text)
        self.assertIn("outcome label is withheld", text)
        self.assertNotIn("black-box predicts", text.lower())
        self.assertNotIn("Bad", text)
        self.assertNotIn("Good", text)


if __name__ == "__main__":
    unittest.main()
