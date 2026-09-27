"""Dataset loading, preprocessing, and immutable split creation."""

from __future__ import annotations

import importlib.util
import hashlib
import re
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


HELOC_SENTINEL_MEANINGS = {
    -9: "special missing value -9 in the HELOC data dictionary",
    -8: "special missing value -8 in the HELOC data dictionary",
    -7: "special missing value -7 in the HELOC data dictionary",
}


@dataclass
class PreprocessorState:
    target: str
    feature_columns: list[str]
    model_feature_columns: list[str]
    medians: dict[str, float]
    ranges: dict[str, dict[str, float]]
    sentinel_values: list[int]
    add_missing_indicators: bool
    label_mapping: dict[str, int]
    inverse_label_mapping: dict[str, str]
    label_names: dict[str, str] = field(default_factory=dict)
    feature_kinds: dict[str, str] = field(default_factory=dict)
    categorical_levels: dict[str, list[str]] = field(default_factory=dict)
    feature_descriptions: dict[str, str] = field(default_factory=dict)
    value_descriptions: dict[str, dict[str, str]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, obj: dict[str, Any]) -> "PreprocessorState":
        obj = dict(obj)
        obj.setdefault("label_names", {})
        obj.setdefault("feature_kinds", {col: "numeric" for col in obj.get("feature_columns", [])})
        obj.setdefault("categorical_levels", {})
        obj.setdefault("feature_descriptions", {})
        obj.setdefault("value_descriptions", {})
        return cls(**obj)


def load_dataset(cfg: dict[str, Any]) -> pd.DataFrame:
    """Load the immutable analysis cohort created by Stage 01.

    Datasets without cohort sampling keep the original behavior and are read
    directly from their configured CSV path.
    """
    sampling_cfg = cfg["dataset"].get("sampling", {})
    if bool(sampling_cfg.get("enabled", False)):
        path = prepared_dataset_path(cfg)
        if not path.exists():
            raise FileNotFoundError(
                f"Prepared analysis cohort does not exist: {path}. "
                "Run scripts/01_make_splits.py first."
            )
    else:
        path = Path(cfg["dataset"]["path"])

    df = pd.read_csv(path).reset_index(drop=True)
    if "row_id" not in df.columns:
        df.insert(0, "row_id", np.arange(len(df), dtype=int))
    return df


def prepare_dataset(cfg: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Create the deterministic dataset cohort consumed by every later stage."""
    sampling_cfg = cfg["dataset"].get("sampling", {})
    if bool(sampling_cfg.get("enabled", False)):
        if not bool(sampling_cfg.get("balance_classes", True)):
            raise ValueError("dataset.sampling currently requires balance_classes: true")
        return _stream_balanced_binary_sample(cfg)

    df = pd.read_csv(cfg["dataset"]["path"]).reset_index(drop=True)
    if "row_id" not in df.columns:
        df.insert(0, "row_id", np.arange(len(df), dtype=int))
    y = labels_to_int(cfg, df[cfg["dataset"]["target"]])
    counts = _label_counts(y)
    return df, {
        "sampling_enabled": False,
        "balance_classes": False,
        "source_rows": int(len(df)),
        "selected_rows": int(len(df)),
        "source_class_counts": counts,
        "selected_class_counts": counts,
    }


def prepared_dataset_path(cfg: dict[str, Any]) -> Path:
    filename = str(
        cfg["dataset"].get("sampling", {}).get(
            "materialized_filename", "balanced_sample.csv"
        )
    )
    return Path(cfg["project"]["output_dir"]).joinpath("preprocess", filename)


def _stream_balanced_binary_sample(
    cfg: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Uniformly undersample both binary classes with bounded memory use."""
    sampling_cfg = cfg["dataset"]["sampling"]
    max_records = int(sampling_cfg.get("max_records", 10_000))
    chunk_size = int(sampling_cfg.get("chunk_size", 50_000))
    progress_every_chunks = int(sampling_cfg.get("progress_every_chunks", 10))
    seed = int(sampling_cfg.get("seed", cfg.get("project", {}).get("seed", 2027)))
    if max_records < 2:
        raise ValueError("dataset.sampling.max_records must be at least 2")
    if chunk_size < 1:
        raise ValueError("dataset.sampling.chunk_size must be positive")

    target = cfg["dataset"]["target"]
    per_class_cap = max_records // 2
    rng = np.random.default_rng(seed)
    reservoirs: dict[int, pd.DataFrame] = {}
    source_counts = {0: 0, 1: 0}
    source_rows = 0

    for chunk_number, chunk in enumerate(
        pd.read_csv(cfg["dataset"]["path"], chunksize=chunk_size), start=1
    ):
        if target not in chunk.columns:
            raise ValueError(f"Target column {target!r} is missing from the dataset")
        chunk = chunk.copy()
        chunk_labels = labels_to_int(cfg, chunk[target])
        chunk["__sampling_key"] = rng.random(len(chunk))
        chunk["__source_row"] = np.arange(source_rows, source_rows + len(chunk), dtype=np.int64)
        source_rows += len(chunk)

        for label in (0, 1):
            candidates = chunk.loc[chunk_labels == label]
            source_counts[label] += int(len(candidates))
            if candidates.empty:
                continue
            if label in reservoirs:
                candidates = pd.concat([reservoirs[label], candidates], ignore_index=True)
            reservoirs[label] = candidates.nsmallest(per_class_cap, "__sampling_key").copy()

        if chunk_number == 1 or (
            progress_every_chunks > 0 and chunk_number % progress_every_chunks == 0
        ):
            print(
                f"[sampling] Scanned {source_rows:,} rows; "
                f"class 0={source_counts[0]:,}, class 1={source_counts[1]:,}",
                flush=True,
            )

    missing_classes = [label for label in (0, 1) if source_counts[label] == 0]
    if missing_classes:
        raise ValueError(
            "Cannot construct a balanced binary cohort; source dataset has no rows "
            f"for encoded class(es): {missing_classes}"
        )

    rows_per_class = min(per_class_cap, source_counts[0], source_counts[1])
    selected_parts = [
        reservoirs[label].nsmallest(rows_per_class, "__sampling_key").copy()
        for label in (0, 1)
    ]
    selected = pd.concat(selected_parts, ignore_index=True)
    selected = selected.sort_values("__sampling_key", kind="stable").reset_index(drop=True)
    source_positions = selected["__source_row"].astype(np.int64).sort_values().to_numpy()
    fingerprint = hashlib.sha256(source_positions.tobytes()).hexdigest()
    selected = selected.drop(columns=["__sampling_key", "__source_row"])
    if "row_id" in selected.columns:
        selected = selected.drop(columns=["row_id"])
    selected.insert(0, "row_id", np.arange(len(selected), dtype=int))

    selected_counts = _label_counts(labels_to_int(cfg, selected[target]))
    if len(selected) > max_records or selected_counts.get("0") != selected_counts.get("1"):
        raise RuntimeError("Balanced cohort invariant failed")
    summary = {
        "sampling_enabled": True,
        "balance_classes": True,
        "sampling_method": "streaming_random_undersampling",
        "sampling_seed": seed,
        "chunk_size": chunk_size,
        "progress_every_chunks": progress_every_chunks,
        "max_records": max_records,
        "source_rows": int(source_rows),
        "selected_rows": int(len(selected)),
        "rows_per_class": int(rows_per_class),
        "source_class_counts": {str(k): int(v) for k, v in source_counts.items()},
        "selected_class_counts": selected_counts,
        "source_row_fingerprint_sha256": fingerprint,
    }
    return selected, summary


def _label_counts(y: pd.Series) -> dict[str, int]:
    return {str(int(label)): int(count) for label, count in y.value_counts().sort_index().items()}


def feature_columns(cfg: dict[str, Any], df: pd.DataFrame) -> list[str]:
    target = cfg["dataset"]["target"]
    drop_columns = set(cfg["dataset"].get("drop_columns", []))
    return [col for col in df.columns if col not in {"row_id", target, "split"} and col not in drop_columns]


def labels_to_int(cfg: dict[str, Any], y: pd.Series) -> pd.Series:
    dataset_cfg = cfg["dataset"]
    if "positive_values" in dataset_cfg or "negative_values" in dataset_cfg:
        positives = {_canonical_value(v) for v in dataset_cfg.get("positive_values", [])}
        negatives = {_canonical_value(v) for v in dataset_cfg.get("negative_values", [])}
        encoded = []
        missing_values = set()
        for value in y:
            canonical = _canonical_value(value)
            if canonical in positives:
                encoded.append(1)
            elif canonical in negatives:
                encoded.append(0)
            else:
                missing_values.add(canonical)
                encoded.append(np.nan)
        if missing_values:
            raise ValueError(f"Labels missing from positive/negative value config: {sorted(missing_values)}")
        return pd.Series(encoded, index=y.index, dtype=int)

    mapping = {_canonical_value(k): int(v) for k, v in dataset_cfg["label_mapping"].items()}
    canonical_y = y.map(_canonical_value)
    missing = sorted(set(canonical_y.unique()) - set(mapping))
    if missing:
        raise ValueError(f"Labels missing from config mapping: {missing}")
    return canonical_y.map(mapping).astype(int)


def labels_to_name(state: PreprocessorState, y: pd.Series | np.ndarray) -> list[str]:
    inverse = {int(k): v for k, v in state.label_names.items()} if state.label_names else {
        int(v): k for k, v in state.label_mapping.items()
    }
    return [inverse[int(value)] for value in list(y)]


def make_split_assignments(
    df: pd.DataFrame,
    cfg: dict[str, Any],
    seed: int,
) -> pd.DataFrame:
    """Create leakage-controlled train/val/surrogate/refine/eval split IDs."""
    target = cfg["dataset"]["target"]
    y = labels_to_int(cfg, df[target])
    fractions = cfg["splits"]

    train_idx, temp_idx = train_test_split(
        df.index,
        train_size=float(fractions["train_bb"]),
        random_state=seed,
        stratify=stratify_or_none(y),
    )
    temp_y = y.loc[temp_idx]
    val_fraction_within_temp = float(fractions["val_bb"]) / (1.0 - float(fractions["train_bb"]))
    val_idx, rem_idx = train_test_split(
        temp_idx,
        train_size=val_fraction_within_temp,
        random_state=seed + 1,
        stratify=stratify_or_none(temp_y),
    )

    rem_y = y.loc[rem_idx]
    rem_total = float(fractions["sur"]) + float(fractions["refine"]) + float(fractions["eval"])
    sur_fraction_within_rem = float(fractions["sur"]) / rem_total
    sur_idx, rem2_idx = train_test_split(
        rem_idx,
        train_size=sur_fraction_within_rem,
        random_state=seed + 2,
        stratify=stratify_or_none(rem_y),
    )

    rem2_y = y.loc[rem2_idx]
    refine_fraction_within_rem2 = float(fractions["refine"]) / (
        float(fractions["refine"]) + float(fractions["eval"])
    )
    refine_idx, eval_idx = train_test_split(
        rem2_idx,
        train_size=refine_fraction_within_rem2,
        random_state=seed + 3,
        stratify=stratify_or_none(rem2_y),
    )

    split = pd.Series(index=df.index, dtype=object)
    split.loc[train_idx] = "train_bb"
    split.loc[val_idx] = "val_bb"
    split.loc[sur_idx] = "sur"
    split.loc[refine_idx] = "refine"
    split.loc[eval_idx] = "eval"
    return pd.DataFrame({"row_id": df["row_id"], "split": split.values})


def fit_preprocessor(
    df: pd.DataFrame,
    split_df: pd.DataFrame,
    cfg: dict[str, Any],
) -> PreprocessorState:
    target = cfg["dataset"]["target"]
    features = feature_columns(cfg, df)
    sentinels = [int(v) for v in cfg["dataset"].get("sentinel_values", [])]
    add_indicators = bool(cfg["dataset"].get("add_missing_indicators", True))
    max_categorical_levels = int(cfg["dataset"].get("max_categorical_levels", 30))
    configured_categorical = set(cfg["dataset"].get("categorical_features", []))
    metadata = load_feature_metadata(cfg)
    feature_descriptions = {col: str(metadata["feature_map"].get(col, col)) for col in features}
    value_descriptions = {
        col: {_canonical_value(k): str(v) for k, v in values.items()}
        for col, values in metadata["value_mapper"].items()
    }

    train_ids = split_df.loc[split_df["split"] == "train_bb", "row_id"]
    train = df[df["row_id"].isin(train_ids)]
    medians: dict[str, float] = {}
    ranges: dict[str, dict[str, float]] = {}
    model_features: list[str] = []
    feature_kinds: dict[str, str] = {}
    categorical_levels: dict[str, list[str]] = {}

    for col in features:
        kind = infer_feature_kind(train[col], col, configured_categorical, value_descriptions)
        feature_kinds[col] = kind
        if kind == "categorical":
            levels = infer_categorical_levels(train[col], max_categorical_levels)
            categorical_levels[col] = levels
            medians[col] = 0.0
            ranges[col] = {"min": 0.0, "max": 1.0, "median": 0.0}
            for level in levels:
                model_features.append(onehot_feature_name(col, level))
            continue

        values = pd.to_numeric(train[col], errors="coerce")
        clean = values.mask(values.isin(sentinels), np.nan)
        median = float(clean.median()) if clean.notna().any() else 0.0
        medians[col] = median
        ranges[col] = {
            "min": float(clean.min()) if clean.notna().any() else median,
            "max": float(clean.max()) if clean.notna().any() else median,
            "median": median,
        }
        model_features.append(col)
        if add_indicators:
            model_features.append(f"{col}__is_special")

    label_mapping = make_label_mapping(cfg, df[target])
    label_names = {str(k): str(v) for k, v in cfg["dataset"].get("label_names", {}).items()}
    if not label_names:
        label_names = {str(v): k for k, v in label_mapping.items()}
    inverse = {str(k): v for k, v in label_names.items()}
    return PreprocessorState(
        target=target,
        feature_columns=features,
        model_feature_columns=model_features,
        medians=medians,
        ranges=ranges,
        sentinel_values=sentinels,
        add_missing_indicators=add_indicators,
        label_mapping=label_mapping,
        inverse_label_mapping=inverse,
        label_names=label_names,
        feature_kinds=feature_kinds,
        categorical_levels=categorical_levels,
        feature_descriptions=feature_descriptions,
        value_descriptions=value_descriptions,
    )


def transform_features(df: pd.DataFrame, state: PreprocessorState) -> pd.DataFrame:
    """Transform raw columns into the numeric feature matrix used by M and T0."""
    pieces: dict[str, pd.Series] = {}
    sentinels = set(state.sentinel_values)
    for col in state.feature_columns:
        if state.feature_kinds.get(col, "numeric") == "categorical":
            raw = df[col].map(_canonical_category)
            levels = state.categorical_levels[col]
            known = set(levels)
            mapped = raw.where(raw.isin(known), "__OTHER__")
            if "__OTHER__" not in known:
                mapped = mapped.where(mapped != "__OTHER__", levels[-1])
            for level in levels:
                pieces[onehot_feature_name(col, level)] = (mapped == level).astype(int)
            continue

        raw = pd.to_numeric(df[col], errors="coerce")
        special = raw.isin(sentinels) | raw.isna()
        clean = raw.mask(special, np.nan).fillna(state.medians[col])
        pieces[col] = clean.astype(float)
        if state.add_missing_indicators:
            pieces[f"{col}__is_special"] = special.astype(int)
    return pd.DataFrame(pieces, index=df.index)[state.model_feature_columns]


def transform_labels(df: pd.DataFrame, state: PreprocessorState) -> pd.Series:
    canonical = df[state.target].map(_canonical_value)
    return canonical.map(state.label_mapping).astype(int)


def build_feature_schema(state: PreprocessorState) -> dict[str, Any]:
    features: list[dict[str, Any]] = []
    for col in state.feature_columns:
        description = state.feature_descriptions.get(col, col)
        if state.feature_kinds.get(col, "numeric") == "categorical":
            for level in state.categorical_levels[col]:
                value_label = state.value_descriptions.get(col, {}).get(level, level)
                features.append(
                    {
                        "name": onehot_feature_name(col, level),
                        "source_column": col,
                        "kind": "binary",
                        "description": f"{description} equals {value_label}",
                        "allowed_values": [0, 1],
                    }
                )
            continue

        sentinels = {
            str(value): HELOC_SENTINEL_MEANINGS.get(value, "special missing value")
            for value in state.sentinel_values
        }
        features.append(
            {
                "name": col,
                "source_column": col,
                "kind": "numeric",
                "description": description,
                "min": state.ranges[col]["min"],
                "max": state.ranges[col]["max"],
                "median_imputation": state.ranges[col]["median"],
                "sentinel_values_treated_as_missing": sentinels,
            }
        )
        if state.add_missing_indicators:
            features.append(
                {
                    "name": f"{col}__is_special",
                    "source_column": col,
                    "kind": "binary",
                    "description": f"1 when {description} had a special missing/sentinel value, else 0",
                    "allowed_values": [0, 1],
                }
            )
    return {
        "target": state.target,
        "labels": {name: int(code) for code, name in state.label_names.items()}
        if state.label_names
        else state.label_mapping,
        "features": features,
    }


def attach_splits(df: pd.DataFrame, split_df: pd.DataFrame) -> pd.DataFrame:
    return df.merge(split_df, on="row_id", how="left", validate="one_to_one")


def split_counts(split_df: pd.DataFrame) -> pd.DataFrame:
    return (
        split_df.groupby("split", as_index=False)
        .size()
        .rename(columns={"size": "n"})
        .sort_values("split")
    )


def make_label_mapping(cfg: dict[str, Any], y: pd.Series) -> dict[str, int]:
    dataset_cfg = cfg["dataset"]
    if "positive_values" in dataset_cfg or "negative_values" in dataset_cfg:
        mapping = {_canonical_value(v): 1 for v in dataset_cfg.get("positive_values", [])}
        mapping.update({_canonical_value(v): 0 for v in dataset_cfg.get("negative_values", [])})
        return mapping
    return {_canonical_value(k): int(v) for k, v in dataset_cfg["label_mapping"].items()}


def load_feature_metadata(cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    metadata_path = cfg["dataset"].get("metadata_path")
    if not metadata_path:
        return {"feature_map": {}, "value_mapper": {}}
    path = Path(metadata_path).expanduser()
    if not path.is_absolute():
        path = Path(cfg["project"]["root_dir"]).joinpath(path)
    spec = importlib.util.spec_from_file_location("dataset_feature_metadata", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not load feature metadata module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return {
        "feature_map": dict(getattr(module, "feature_map", {})),
        "value_mapper": dict(getattr(module, "value_mapper", {})),
    }


def infer_feature_kind(
    series: pd.Series,
    col: str,
    configured_categorical: set[str],
    value_descriptions: dict[str, dict[str, str]],
) -> str:
    if col in configured_categorical or col in value_descriptions:
        return "categorical"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    numeric = pd.to_numeric(series, errors="coerce")
    non_missing = series.notna().sum()
    if non_missing and numeric.notna().sum() / non_missing >= 0.95:
        return "numeric"
    return "categorical"


def infer_categorical_levels(series: pd.Series, max_levels: int) -> list[str]:
    values = series.map(_canonical_category)
    counts = values.value_counts(dropna=False)
    levels = [str(v) for v in counts.head(max_levels).index.tolist()]
    if "__OTHER__" not in levels:
        levels.append("__OTHER__")
    return levels


def onehot_feature_name(column: str, level: str) -> str:
    safe_level = re.sub(r"[^0-9A-Za-z]+", "_", str(level)).strip("_")
    if not safe_level:
        safe_level = "value"
    digest = hashlib.sha1(str(level).encode("utf-8")).hexdigest()[:8]
    return f"{column}__eq__{safe_level[:36]}_{digest}"


def _canonical_value(value: Any) -> str:
    if pd.isna(value):
        return "__MISSING__"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value)


def _canonical_category(value: Any) -> str:
    text = _canonical_value(value).strip()
    return text if text else "__MISSING__"


def stratify_or_none(y: pd.Series) -> pd.Series | None:
    counts = y.value_counts()
    if counts.empty or counts.min() < 2:
        return None
    return y
