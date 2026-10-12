from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from ..utils import expand_env


def load_baseline_registry(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = expand_env(yaml.safe_load(f) or {})
    root = Path(cfg.get("third_party_root", ""))
    for item in cfg.get("baselines", {}).values():
        local = item.get("local_path")
        if local:
            item["resolved_path"] = str((root / local).resolve())
            item["exists"] = (root / local).exists()
        else:
            item["resolved_path"] = None
            item["exists"] = False
    return cfg


def baseline_rows(registry: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for name, item in registry.get("baselines", {}).items():
        rows.append(
            {
                "name": name,
                "category": item.get("category", ""),
                "status": item.get("status", ""),
                "exists": bool(item.get("exists")),
                "path": item.get("resolved_path"),
                "repo": item.get("repo"),
                "notes": item.get("notes", ""),
            }
        )
    return rows
