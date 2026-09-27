"""Local LIME and Anchors explanations for black-box simulatability tasks."""

from __future__ import annotations

import inspect
import math
from typing import Any, Callable

import numpy as np
import pandas as pd


LOCAL_EXPLANATION_FORMAT_VERSION = 2


def fit_lime_explainer(
    X_background: pd.DataFrame,
    class_names: list[str],
    seed: int,
    discretize_continuous: bool = True,
) -> Any:
    """Fit a LIME tabular explainer on non-evaluation background data."""
    try:
        from lime.lime_tabular import LimeTabularExplainer
    except ImportError as exc:
        raise ImportError("LIME explanations require the `lime` package. Install with `pip install lime`.") from exc

    return LimeTabularExplainer(
        training_data=X_background.to_numpy(dtype=float),
        feature_names=list(X_background.columns),
        class_names=class_names,
        mode="classification",
        discretize_continuous=bool(discretize_continuous),
        random_state=int(seed),
    )


def fit_anchor_explainer(
    model: Any,
    X_background: pd.DataFrame,
    seed: int,
    disc_perc: list[int] | tuple[int, ...] = (25, 50, 75),
) -> Any:
    """Fit an AnchorTabular explainer on non-evaluation background data."""
    try:
        from alibi.explainers import AnchorTabular
    except ImportError as exc:
        raise ImportError("Anchor explanations require the `alibi` package. Install with `pip install alibi`.") from exc

    feature_names = list(X_background.columns)
    predict_labels = make_predict_label_fn(model, feature_names)
    constructor_kwargs = {"seed": int(seed)}
    try:
        explainer = AnchorTabular(predict_labels, feature_names, **constructor_kwargs)
    except TypeError:
        explainer = AnchorTabular(predict_labels, feature_names)

    fit_kwargs = {"disc_perc": tuple(int(v) for v in disc_perc)}
    _call_with_supported_kwargs(explainer.fit, X_background.to_numpy(dtype=float), fit_kwargs)
    return explainer


def explain_lime_instance(
    explainer: Any,
    model: Any,
    x_row: pd.Series,
    class_names: list[str],
    explanation_class_name: str | None = None,
    num_features: int = 8,
    num_samples: int = 5000,
) -> dict[str, Any]:
    """Explain one transformed row with LIME for a fixed, non-target-derived class."""
    feature_names = list(x_row.index)
    context = prediction_context(model, x_row, class_names)
    predict_proba = make_predict_proba_fn(model, feature_names)
    if explanation_class_name is None:
        explanation_class_name = str(class_names[-1])
    if explanation_class_name not in class_names:
        raise ValueError(
            f"LIME explanation class {explanation_class_name!r} is not in class_names={class_names!r}"
        )
    explanation_class_index = class_names.index(explanation_class_name)
    explanation = explainer.explain_instance(
        data_row=x_row.to_numpy(dtype=float),
        predict_fn=predict_proba,
        labels=(explanation_class_index,),
        num_features=int(num_features),
        num_samples=int(num_samples),
    )
    terms = _lime_terms(explanation, explanation_class_index)
    return {
        "explanation_format_version": LOCAL_EXPLANATION_FORMAT_VERSION,
        "explainer": "LIME",
        # These target fields are retained for evaluation/auditing. They are not
        # included in the proxy-facing explanation text.
        "target_label": context["label_name"],
        "target_label_numeric": context["label_numeric"],
        "target_probability": context["probability"],
        "features": terms,
        "explanation_text": lime_explanation_text(explanation_class_name, terms),
        "metadata": {
            "num_features": int(num_features),
            "num_samples": int(num_samples),
            "explanation_class_name": explanation_class_name,
            "explanation_class_index": int(explanation_class_index),
            "local_fidelity_score": _maybe_float(getattr(explanation, "score", None)),
            "local_prediction": _json_safe(getattr(explanation, "local_pred", None)),
        },
    }


def explain_anchor_instance(
    explainer: Any,
    model: Any,
    x_row: pd.Series,
    class_names: list[str],
    threshold: float = 0.95,
    delta: float = 0.1,
    tau: float = 0.15,
    batch_size: int = 100,
    coverage_samples: int = 1000,
    beam_size: int = 1,
    stop_on_first: bool = True,
    max_anchor_size: int | None = None,
) -> dict[str, Any]:
    """Explain one transformed row with AnchorTabular."""
    context = prediction_context(model, x_row, class_names)
    kwargs: dict[str, Any] = {
        "threshold": float(threshold),
        "delta": float(delta),
        "tau": float(tau),
        "batch_size": int(batch_size),
        "coverage_samples": int(coverage_samples),
        "beam_size": int(beam_size),
        "stop_on_first": bool(stop_on_first),
    }
    if max_anchor_size is not None:
        kwargs["max_anchor_size"] = int(max_anchor_size)

    explanation = _call_with_supported_kwargs(
        explainer.explain,
        x_row.to_numpy(dtype=float),
        kwargs,
    )
    anchor = [str(item) for item in _as_list(_explanation_value(explanation, "anchor", []))]
    precision = _maybe_float(_explanation_value(explanation, "precision", None))
    coverage = _maybe_float(_explanation_value(explanation, "coverage", None))
    features = [{"description": item} for item in anchor]
    return {
        "explanation_format_version": LOCAL_EXPLANATION_FORMAT_VERSION,
        "explainer": "Anchors",
        # These target fields are retained for evaluation/auditing. They are not
        # included in the proxy-facing explanation text.
        "target_label": context["label_name"],
        "target_label_numeric": context["label_numeric"],
        "target_probability": context["probability"],
        "features": features,
        "explanation_text": anchor_explanation_text(anchor, precision, coverage),
        "metadata": {
            "threshold": float(threshold),
            "delta": float(delta),
            "tau": float(tau),
            "batch_size": int(batch_size),
            "coverage_samples": int(coverage_samples),
            "beam_size": int(beam_size),
            "stop_on_first": bool(stop_on_first),
            "max_anchor_size": max_anchor_size,
            "precision": precision,
            "coverage": coverage,
        },
    }


def make_predict_proba_fn(model: Any, feature_names: list[str]) -> Callable[[np.ndarray], np.ndarray]:
    def predict_proba(samples: np.ndarray) -> np.ndarray:
        return model.predict_proba(_as_feature_frame(samples, feature_names))

    return predict_proba


def make_predict_label_fn(model: Any, feature_names: list[str]) -> Callable[[np.ndarray], np.ndarray]:
    def predict_labels(samples: np.ndarray) -> np.ndarray:
        return model.predict(_as_feature_frame(samples, feature_names)).astype(int)

    return predict_labels


def prediction_context(model: Any, x_row: pd.Series, class_names: list[str]) -> dict[str, Any]:
    feature_names = list(x_row.index)
    frame = _as_feature_frame(x_row.to_numpy(dtype=float), feature_names)
    probabilities = model.predict_proba(frame)[0]
    prediction = model.predict(frame)[0]
    classes = list(model.classes_)
    class_index = classes.index(prediction)
    return {
        "class_index": int(class_index),
        "label_numeric": int(prediction),
        "label_name": str(class_names[class_index]),
        "probability": float(probabilities[class_index]),
        "probabilities": {
            str(class_names[i]): float(probabilities[i])
            for i in range(len(class_names))
        },
    }


def lime_explanation_text(explanation_class_name: str, terms: list[dict[str, Any]]) -> str:
    lines = [
        (
            "LIME local explanation for the target row with respect to the fixed "
            f"reference class {explanation_class_name}."
        ),
        (
            f"Feature contributions toward {explanation_class_name}; positive weights "
            f"support {explanation_class_name} and negative weights push away from it:"
        ),
    ]
    if terms:
        for term in terms:
            lines.append(f"- {term['description']}: {term['weight']:+.4f}")
    else:
        lines.append("- No LIME terms were returned.")
    return "\n".join(lines)


def anchor_explanation_text(
    anchor: list[str],
    precision: float | None,
    coverage: float | None,
) -> str:
    metrics = []
    if precision is not None:
        metrics.append(f"estimated precision {_format_float(precision)}")
    if coverage is not None:
        metrics.append(f"coverage {_format_float(coverage)}")
    metric_text = f" ({', '.join(metrics)})" if metrics else ""
    lines = [
        "Anchors local explanation for the target row; the outcome label is withheld.",
        f"Anchor conditions{metric_text}:",
    ]
    if anchor:
        for condition in anchor:
            lines.append(f"- {condition}")
        lines.append(
            "The anchor states that rows satisfying these conditions are expected to receive the same black-box prediction under Anchor perturbations."
        )
    else:
        lines.append("- No non-empty anchor was found under the configured search budget.")
    return "\n".join(lines)


def _as_feature_frame(samples: Any, feature_names: list[str]) -> pd.DataFrame:
    if isinstance(samples, pd.DataFrame):
        return samples.loc[:, feature_names]
    array = np.asarray(samples, dtype=float)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    return pd.DataFrame(array, columns=feature_names)


def _lime_terms(explanation: Any, class_index: int) -> list[dict[str, Any]]:
    try:
        weighted_terms = explanation.as_list(label=class_index)
    except Exception:
        weighted_terms = explanation.as_list()
    return [
        {"description": str(description), "weight": float(weight)}
        for description, weight in weighted_terms
    ]


def _explanation_value(explanation: Any, key: str, default: Any) -> Any:
    if hasattr(explanation, key):
        return getattr(explanation, key)
    data = getattr(explanation, "data", None)
    if isinstance(data, dict) and key in data:
        return data[key]
    return default


def _call_with_supported_kwargs(func: Callable[..., Any], arg: Any, kwargs: dict[str, Any]) -> Any:
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return func(arg, **kwargs)
    accepts_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values())
    if accepts_kwargs:
        return func(arg, **kwargs)
    supported = {key: value for key, value in kwargs.items() if key in signature.parameters}
    return func(arg, **supported)


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return [value]


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple, np.ndarray)):
        values = _as_list(value)
        if not values:
            return None
        value = values[-1]
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return value


def _format_float(value: float) -> str:
    return f"{float(value):.3f}"
