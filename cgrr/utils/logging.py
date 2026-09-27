"""Logging setup for command line scripts."""

from __future__ import annotations

import logging
from pathlib import Path

from cgrr.utils.io import ensure_dir


def setup_logging(log_path: str | Path | None = None, level: int = logging.INFO) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_path is not None:
        log_path = Path(log_path)
        ensure_dir(log_path.parent)
        handlers.append(logging.FileHandler(log_path, mode="a", encoding="utf-8"))
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        handlers=handlers,
        force=True,
    )
