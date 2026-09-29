"""Configuration loading: YAML file, overridable by environment variables."""
from __future__ import annotations
import os
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | None = None) -> dict:
    path = path or os.environ.get("VERITY_CONFIG", str(ROOT / "configs" / "default.yaml"))
    cfg = yaml.safe_load(Path(path).read_text())
    if os.environ.get("QDRANT_URL"):
        cfg["qdrant"]["url"] = os.environ["QDRANT_URL"]
    if os.environ.get("VERITY_COLLECTION"):
        cfg["qdrant"]["collection"] = os.environ["VERITY_COLLECTION"]
    return cfg
