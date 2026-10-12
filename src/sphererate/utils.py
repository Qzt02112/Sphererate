from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


def expand_env(value: Any) -> Any:
    """Recursively expand ``${VAR}`` references in config values.

    Shipped configs keep machine-specific roots as ``${SCANDMM_ROOT}`` /
    ``${TEMPSAL_ROOT}`` placeholders. Point the matching environment variables
    at your local checkouts instead of editing every config file. Unknown
    variables are left untouched, so an unset root stays visibly unset.
    """
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, dict):
        return {key: expand_env(item) for key, item in value.items()}
    if isinstance(value, list):
        return [expand_env(item) for item in value]
    return value


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_jsonl(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(rows: Iterable[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_np(x: np.ndarray, eps: float = 1e-7, weights: np.ndarray | None = None) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = np.maximum(x, 0.0)
    if weights is None:
        denom = float(x.sum())
    else:
        denom = float((x * weights).sum())
    if denom <= eps:
        return np.ones_like(x, dtype=np.float32) / float(x.size)
    return x / denom


def spatial_softmax(logits: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    b, c, h, w = logits.shape
    flat = logits.reshape(b, c, h * w)
    flat = flat - flat.amax(dim=-1, keepdim=True)
    probs = torch.softmax(flat, dim=-1)
    return probs.reshape(b, c, h, w).clamp_min(eps)


def normalize_torch(x: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    dims = tuple(range(2, x.ndim))
    denom = x.sum(dim=dims, keepdim=True).clamp_min(eps)
    return x.clamp_min(0) / denom
