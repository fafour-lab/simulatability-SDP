"""Construct feature-valid counterfactual pairs from held-out real examples."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def nearest_real_counterfactual_pairs(
    X_eval: pd.DataFrame,
    raw_eval: pd.DataFrame,
    m_labels: pd.Series,
    seed: int,
) -> pd.DataFrame:
    """Pair each eval row with nearest same-label and opposite-label real rows.

    This creates valid HELOC-like counterfactual items without synthesizing
    impossible credit records. The output is balanced by flip/no-flip whenever
    both neighbors exist.
    """
    rng = np.random.default_rng(seed)
    X_scaled = StandardScaler().fit_transform(X_eval)
    nn = NearestNeighbors(n_neighbors=min(50, len(X_eval)), metric="euclidean")
    nn.fit(X_scaled)
    distances, indices = nn.kneighbors(X_scaled)

    labels = np.asarray(m_labels)
    rows = []
    for i, row_id in enumerate(raw_eval["row_id"].to_numpy()):
        same_candidates = []
        flip_candidates = []
        for j in indices[i][1:]:
            if labels[j] == labels[i]:
                same_candidates.append(j)
            else:
                flip_candidates.append(j)
        for flip, candidates in [(False, same_candidates), (True, flip_candidates)]:
            if not candidates:
                continue
            chosen = int(candidates[0])
            rows.append(
                {
                    "row_id": int(row_id),
                    "cf_row_id": int(raw_eval.iloc[chosen]["row_id"]),
                    "flip": bool(flip),
                    "m_label": labels[i],
                    "m_label_cf": labels[chosen],
                    "neighbor_rank": int(np.where(indices[i] == chosen)[0][0]),
                    "distance": float(distances[i][np.where(indices[i] == chosen)[0][0]]),
                    "order_key": float(rng.random()),
                }
            )
    return pd.DataFrame(rows).sort_values(["flip", "order_key"]).drop(columns=["order_key"])
