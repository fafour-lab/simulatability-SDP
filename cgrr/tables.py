"""Paper-ready CSV and LaTeX table writers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from cgrr.utils.io import ensure_dir


def write_csv_and_latex(
    df: pd.DataFrame,
    stem: str | Path,
    float_format: str = "%.3f",
    index: bool = False,
) -> None:
    stem = Path(stem)
    ensure_dir(stem.parent)
    df.to_csv(stem.with_suffix(".csv"), index=index)
    with stem.with_suffix(".tex").open("w", encoding="utf-8") as handle:
        handle.write(
            df.to_latex(
                index=index,
                escape=True,
                float_format=lambda x: float_format % x,
            )
        )


def round_for_paper(df: pd.DataFrame, digits: int = 3) -> pd.DataFrame:
    out = df.copy()
    for col in out.select_dtypes(include=["float", "float64", "float32"]).columns:
        out[col] = out[col].round(digits)
    return out
