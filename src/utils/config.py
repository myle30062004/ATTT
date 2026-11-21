"""Project configuration loading and path resolution."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "config.yaml"


@lru_cache(maxsize=4)
def load_config(path: str | None = None) -> dict[str, Any]:
    with open(path or DEFAULT_CONFIG, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def project_path(key: str, config: dict[str, Any] | None = None) -> Path:
    """Absolute path for an entry of the ``paths`` section, created if missing."""
    cfg = config or load_config()
    path = (PROJECT_ROOT / cfg["paths"][key]).resolve()
    if path.suffix == "":
        path.mkdir(parents=True, exist_ok=True)
    return path
