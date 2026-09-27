"""Bootstrap confidence intervals for paired effects."""

from __future__ import annotations

import numpy as np


def paired_bootstrap_mean_diff(
    treatment: np.ndarray,
    control: np.ndarray,
    n_resamples: int,
    seed: int,
    ci: float = 0.95,
) -> dict[str, float]:
    treatment = np.asarray(treatment, dtype=float)
    control = np.asarray(control, dtype=float)
    if len(treatment) != len(control):
        raise ValueError("paired bootstrap inputs must have the same length")
    rng = np.random.default_rng(seed)
    n = len(treatment)
    diffs = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        diffs[i] = float(np.mean(treatment[idx] - control[idx]))
    alpha = (1.0 - ci) / 2.0
    return {
        "mean_diff": float(np.mean(treatment - control)),
        "ci_low": float(np.quantile(diffs, alpha)),
        "ci_high": float(np.quantile(diffs, 1.0 - alpha)),
        "n": float(n),
        "n_resamples": float(n_resamples),
    }
