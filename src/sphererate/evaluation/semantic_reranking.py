from __future__ import annotations

import hashlib

import numpy as np


def _spherical_normalize(density: np.ndarray, area: np.ndarray, eps: float) -> np.ndarray:
    values = np.maximum(np.asarray(density, dtype=np.float64), 0.0)
    weights = np.asarray(area, dtype=np.float64)
    if values.shape != weights.shape:
        raise ValueError(f"density/area shape mismatch: {values.shape} != {weights.shape}")
    denominator = float(np.sum(values * weights))
    if not np.isfinite(denominator) or denominator <= eps:
        raise ValueError("density must have positive finite spherical mass")
    return values / denominator


def geometric_semantic_fusion(
    template: np.ndarray,
    semantic: np.ndarray,
    area: np.ndarray,
    *,
    alpha: float,
    eps: float = 1e-12,
) -> np.ndarray:
    """Fuse a start/time prior and image semantic evidence in log-density space."""

    coefficient = float(alpha)
    if not 0.0 <= coefficient <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    template_n = _spherical_normalize(template, area, eps)
    semantic_n = _spherical_normalize(semantic, area, eps)
    logits = (1.0 - coefficient) * np.log(template_n + eps)
    logits += coefficient * np.log(semantic_n + eps)
    logits -= float(np.max(logits))
    return _spherical_normalize(np.exp(logits), area, eps)


def deterministic_derangement(length: int, *, seed: int, key: str) -> np.ndarray:
    """Return a stable permutation with no sample paired to itself."""

    count = int(length)
    if count < 2:
        raise ValueError("a derangement requires at least two items")
    digest = hashlib.sha256(f"{int(seed)}:{key}".encode("utf-8")).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    identity = np.arange(count, dtype=np.int64)
    for _ in range(1000):
        candidate = rng.permutation(count)
        if np.all(candidate != identity):
            return candidate.astype(np.int64)
    return np.roll(identity, 1)
