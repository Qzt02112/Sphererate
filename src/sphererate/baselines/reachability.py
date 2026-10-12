from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from sphererate.data.early_horizon import clipped_dwell_weights
from sphererate.data.maps import erp_pixel_center_unit_vectors, normalized_coords_to_unit_vectors


def angular_distance_grid_deg(
    height: int,
    width: int,
    *,
    start_x_norm: float,
    start_y_norm: float,
) -> np.ndarray:
    """Great-circle distance from one ERP start point to every pixel centre."""
    start = normalized_coords_to_unit_vectors(
        np.asarray([start_x_norm], dtype=np.float32),
        np.asarray([start_y_norm], dtype=np.float32),
    )[0]
    grid = erp_pixel_center_unit_vectors(int(height), int(width))
    dots = np.clip(np.einsum("hwc,c->hw", grid, start, optimize=True), -1.0, 1.0)
    return np.rad2deg(np.arccos(dots)).astype(np.float32)


def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if values.shape != weights.shape or values.size == 0:
        raise ValueError("values and weights must be non-empty arrays with the same shape")
    if not 0.0 < float(quantile) < 1.0:
        raise ValueError("quantile must be strictly between zero and one")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("values and weights must be finite and weights non-negative")
    if float(weights.sum()) <= 0.0:
        raise ValueError("weights must have positive mass")
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    cumulative = np.cumsum(weights[order]) / float(weights.sum())
    return float(sorted_values[min(int(np.searchsorted(cumulative, quantile, side="left")), len(values) - 1)])


def calibrate_reachability_radii(
    fixation_arrays: Iterable[np.ndarray],
    horizons_s: Iterable[float],
    *,
    start_x_norm: float,
    start_y_norm: float,
    quantile: float = 0.95,
) -> dict[float, float]:
    """Estimate a train-only dwell-weighted angular reachability envelope."""
    arrays = [np.asarray(value, dtype=np.float32) for value in fixation_arrays]
    start = normalized_coords_to_unit_vectors(
        np.asarray([start_x_norm], dtype=np.float32),
        np.asarray([start_y_norm], dtype=np.float32),
    )[0].astype(np.float64)
    radii: dict[float, float] = {}
    previous = 0.0
    for raw_horizon in horizons_s:
        horizon = float(raw_horizon)
        distances, weights = [], []
        for fixations in arrays:
            if fixations.ndim != 2 or fixations.shape[1] < 5:
                raise ValueError(f"fixations must be [N,>=5], got {fixations.shape}")
            vectors = normalized_coords_to_unit_vectors(fixations[:, 0], fixations[:, 1]).astype(np.float64)
            current_weights = clipped_dwell_weights(fixations, horizon).astype(np.float64)
            active = current_weights > 0.0
            distances.append(np.rad2deg(np.arccos(np.clip(vectors[active] @ start, -1.0, 1.0))))
            weights.append(current_weights[active])
        radius = weighted_quantile(np.concatenate(distances), np.concatenate(weights), quantile)
        previous = max(previous, radius)
        radii[horizon] = previous
    return radii


def gate_final_map(final_map: np.ndarray, distance_deg: np.ndarray, radius_deg: float) -> np.ndarray:
    """Remove final-map saliency outside the frozen hard reachable region."""
    final = np.maximum(np.asarray(final_map, dtype=np.float32), 0.0)
    distance = np.asarray(distance_deg, dtype=np.float32)
    if final.shape != distance.shape:
        raise ValueError(f"final map and distance grid must match, got {final.shape} and {distance.shape}")
    if not np.isfinite(radius_deg) or float(radius_deg) <= 0.0:
        raise ValueError("radius_deg must be finite and positive")
    gated = final * (distance <= float(radius_deg))
    if float(gated.sum()) <= 0.0:
        gated = (distance <= float(radius_deg)).astype(np.float32)
    return gated.astype(np.float32)
