from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np


EPS = 1e-7
LOWER_IS_BETTER = {"kld", "kld_sph"}


def _area_normalize(value: np.ndarray, area: np.ndarray, eps: float = EPS) -> np.ndarray:
    array = np.maximum(np.asarray(value, dtype=np.float64), 0.0)
    weights = np.asarray(area, dtype=np.float64)
    denominator = float((array * weights).sum())
    if not np.isfinite(denominator) or denominator <= eps:
        array = np.ones_like(array, dtype=np.float64)
        denominator = float((array * weights).sum())
    return (array / max(denominator, eps)).astype(np.float32)


def density_mix(
    start: np.ndarray,
    static: np.ndarray,
    alpha: float,
    area: np.ndarray,
) -> np.ndarray:
    coefficient = float(alpha)
    if not 0.0 <= coefficient <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    mixed = coefficient * np.asarray(start, dtype=np.float64)
    mixed += (1.0 - coefficient) * np.asarray(static, dtype=np.float64)
    return _area_normalize(mixed, area)


def log_density_mix(
    start: np.ndarray,
    static: np.ndarray,
    alpha: float,
    area: np.ndarray,
    eps: float = EPS,
) -> np.ndarray:
    coefficient = float(alpha)
    if not 0.0 <= coefficient <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    start_log = np.log(np.maximum(np.asarray(start, dtype=np.float64), eps))
    static_log = np.log(np.maximum(np.asarray(static, dtype=np.float64), eps))
    logits = coefficient * start_log + (1.0 - coefficient) * static_log
    mixed = np.exp(logits - float(np.max(logits)))
    return _area_normalize(mixed, area, eps=eps)


def grouped_bootstrap_improvement(
    candidate: dict[str, dict[str, float]],
    control: dict[str, dict[str, float]],
    groups_by_image: dict[str, str],
    metric: str,
    iterations: int = 10000,
    seed: int = 20260711,
) -> dict[str, float | int]:
    if set(candidate) != set(control):
        raise ValueError("Candidate and control must contain the same image IDs")
    oriented_by_group: dict[str, list[float]] = defaultdict(list)
    for image_id in sorted(candidate):
        if image_id not in groups_by_image:
            raise KeyError(f"Missing bootstrap group for {image_id}")
        if metric in LOWER_IS_BETTER:
            difference = float(control[image_id][metric]) - float(candidate[image_id][metric])
        else:
            difference = float(candidate[image_id][metric]) - float(control[image_id][metric])
        oriented_by_group[str(groups_by_image[image_id])].append(difference)
    values = np.asarray(
        [np.mean(oriented_by_group[group]) for group in sorted(oriented_by_group)],
        dtype=np.float64,
    )
    if not len(values):
        raise ValueError("No groups available for bootstrap")
    rng = np.random.default_rng(int(seed))
    samples = np.empty(int(iterations), dtype=np.float64)
    for index in range(int(iterations)):
        draw = rng.integers(0, len(values), size=len(values))
        samples[index] = float(values[draw].mean())
    return {
        "oriented_improvement_mean": float(values.mean()),
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
        "num_groups": int(len(values)),
    }


def deterministic_wrong_image_indices(
    rows: list[dict[str, Any]],
    seed: int,
    max_attempts: int = 10000,
) -> list[int]:
    """Return a seeded one-to-one donor permutation with no same-group assignments."""

    if len(rows) < 2:
        raise ValueError("At least two rows are required for a wrong-image mapping")
    groups = [str(row["group_id"]) for row in rows]
    largest_group = max(groups.count(group) for group in set(groups))
    if largest_group * 2 > len(rows):
        raise ValueError("No cross-group one-to-one donor mapping exists")
    rng = np.random.default_rng(int(seed))
    indices = np.arange(len(rows), dtype=np.int64)
    for _ in range(int(max_attempts)):
        donors = rng.permutation(indices)
        if all(target != donor and groups[target] != groups[int(donor)] for target, donor in enumerate(donors)):
            return [int(value) for value in donors]
    raise RuntimeError(f"Failed to find a cross-group donor permutation in {max_attempts} attempts")
