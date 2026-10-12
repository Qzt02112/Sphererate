from __future__ import annotations

import hashlib
from typing import Sequence

import numpy as np


EPS = 1e-7


def normalize_longitude_deg(value: float) -> float:
    return float((float(value) + 180.0) % 360.0 - 180.0)


def longitude_pixel_shift(longitude_deg: float, width: int) -> int:
    return int(round(int(width) * normalize_longitude_deg(longitude_deg) / 360.0))


def reference_alignment_pixel_shift(
    actual_longitude_deg: float,
    reference_longitude_deg: float,
    width: int,
) -> int:
    """Pixel roll that moves an actual ERP longitude onto a reference longitude."""
    return longitude_pixel_shift(float(reference_longitude_deg) - float(actual_longitude_deg), width)


def align_to_start(value: np.ndarray, longitude_deg: float) -> np.ndarray:
    array = np.asarray(value)
    return np.roll(array, shift=-longitude_pixel_shift(longitude_deg, array.shape[-1]), axis=-1)


def restore_from_start(value: np.ndarray, longitude_deg: float) -> np.ndarray:
    array = np.asarray(value)
    return np.roll(array, shift=longitude_pixel_shift(longitude_deg, array.shape[-1]), axis=-1)


def area_normalize(value: np.ndarray, area: np.ndarray, eps: float = EPS) -> np.ndarray:
    array = np.maximum(np.asarray(value, dtype=np.float64), 0.0)
    weights = np.asarray(area, dtype=np.float64)
    denominator = float((array * weights).sum())
    if not np.isfinite(denominator) or denominator <= eps:
        array = np.ones_like(array, dtype=np.float64)
        denominator = float((array * weights).sum())
    return (array / max(denominator, eps)).astype(np.float32)


def mean_temporal_template(aligned_maps: Sequence[np.ndarray], area: np.ndarray) -> np.ndarray:
    if not aligned_maps:
        raise ValueError("At least one aligned temporal map is required")
    mean = np.mean(np.stack(aligned_maps, axis=0).astype(np.float64), axis=0)
    return np.stack([area_normalize(current, area) for current in mean], axis=0)


def select_monotonic_alphas(costs: np.ndarray, alpha_grid: Sequence[float]) -> list[float]:
    matrix = np.asarray(costs, dtype=np.float64)
    alphas = [float(value) for value in alpha_grid]
    if matrix.ndim != 2 or matrix.shape[1] != len(alphas):
        raise ValueError("costs must have shape [num_bins, len(alpha_grid)]")
    if any(left >= right for left, right in zip(alphas, alphas[1:])):
        raise ValueError("alpha_grid must be strictly increasing")
    num_bins, num_alphas = matrix.shape
    dp = np.full((num_bins, num_alphas), np.inf, dtype=np.float64)
    parent = np.full((num_bins, num_alphas), -1, dtype=np.int64)
    dp[0] = matrix[0]
    for bin_index in range(1, num_bins):
        for current in range(num_alphas):
            allowed_previous = np.arange(current, num_alphas, dtype=np.int64)
            best_previous = int(allowed_previous[np.argmin(dp[bin_index - 1, allowed_previous])])
            dp[bin_index, current] = matrix[bin_index, current] + dp[bin_index - 1, best_previous]
            parent[bin_index, current] = best_previous
    selected_indices = [0] * num_bins
    selected_indices[-1] = int(np.argmin(dp[-1]))
    for bin_index in range(num_bins - 1, 0, -1):
        selected_indices[bin_index - 1] = int(parent[bin_index, selected_indices[bin_index]])
    return [alphas[index] for index in selected_indices]


def accumulate_temporal_maps(
    temporal_maps: np.ndarray,
    bin_weights: Sequence[float],
    area: np.ndarray,
) -> np.ndarray:
    maps = np.asarray(temporal_maps, dtype=np.float64)
    weights = np.asarray(list(bin_weights), dtype=np.float64)
    if maps.ndim != 3 or maps.shape[0] != len(weights):
        raise ValueError("temporal_maps must have shape [num_bins, height, width]")
    if np.any(weights < 0.0) or float(weights.sum()) <= 0.0:
        raise ValueError("bin_weights must be nonnegative with positive sum")
    weights = weights / weights.sum()
    mixed = (maps * weights[:, None, None]).sum(axis=0)
    return area_normalize(mixed, area)


def centered_log_residual(
    target: np.ndarray,
    template: np.ndarray,
    area: np.ndarray,
    eps: float = EPS,
) -> np.ndarray:
    residual = np.log(np.maximum(np.asarray(target, dtype=np.float64), eps))
    residual -= np.log(np.maximum(np.asarray(template, dtype=np.float64), eps))
    weights = np.asarray(area, dtype=np.float64)
    residual -= float((residual * weights).sum() / max(weights.sum(), eps))
    return residual.astype(np.float32)


def weighted_cosine(left: np.ndarray, right: np.ndarray, area: np.ndarray, eps: float = EPS) -> float:
    first = np.asarray(left, dtype=np.float64)
    second = np.asarray(right, dtype=np.float64)
    weights = np.asarray(area, dtype=np.float64)
    numerator = float((weights * first * second).sum())
    denominator = float(
        np.sqrt((weights * first * first).sum() * (weights * second * second).sum())
    )
    if not np.isfinite(denominator) or denominator <= eps:
        return 0.0
    return float(np.clip(numerator / denominator, -1.0, 1.0))


def deterministic_observer_split(
    observer_ids: Sequence[int],
    key: str,
    seed: int,
) -> tuple[list[int], list[int]]:
    unique = sorted({int(value) for value in observer_ids})
    if len(unique) < 2:
        raise ValueError("At least two observers are required")
    ranked = sorted(
        unique,
        key=lambda observer: hashlib.sha256(f"{int(seed)}:{key}:{observer}".encode("utf-8")).digest(),
    )
    left_size = len(ranked) // 2
    return sorted(ranked[:left_size]), sorted(ranked[left_size:])
