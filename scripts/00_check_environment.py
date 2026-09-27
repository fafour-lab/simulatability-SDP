#!/usr/bin/env python3
"""Print and validate the Python environment used by the pipeline."""

from __future__ import annotations

import importlib
import sys


REQUIRED_IMPORTS = {
    "numpy": "numpy",
    "pandas": "pandas",
    "sklearn": "scikit-learn",
    "xgboost": "xgboost",
    "joblib": "joblib",
    "yaml": "pyyaml",
    "tqdm": "tqdm",
    "scipy": "scipy",
    "torch": "torch",
    "transformers": "transformers",
    "accelerate": "accelerate",
}

OPTIONAL_IMPORTS = {
    "lime": "lime",
    "alibi": "alibi",
}


def main() -> None:
    print(f"[env] sys.executable={sys.executable}", flush=True)
    print(f"[env] python_version={sys.version.split()[0]}", flush=True)
    missing = []
    for import_name, package_name in REQUIRED_IMPORTS.items():
        try:
            module = importlib.import_module(import_name)
            version = getattr(module, "__version__", "unknown")
            print(f"[env] OK {package_name} ({import_name}) version={version}", flush=True)
        except Exception as exc:  # noqa: BLE001 - report the real import failure.
            missing.append((package_name, import_name, repr(exc)))

    for import_name, package_name in OPTIONAL_IMPORTS.items():
        try:
            module = importlib.import_module(import_name)
            version = getattr(module, "__version__", "unknown")
            print(f"[env] OPTIONAL OK {package_name} ({import_name}) version={version}", flush=True)
        except Exception as exc:  # noqa: BLE001 - optional, but useful for local explanation stages.
            print(f"[env] OPTIONAL MISSING {package_name} ({import_name}): {exc!r}", flush=True)

    if missing:
        print("[env] Missing required imports:", flush=True)
        for package_name, import_name, error in missing:
            print(f"[env]   {package_name} ({import_name}): {error}", flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
