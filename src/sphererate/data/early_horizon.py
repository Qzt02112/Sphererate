from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence
import hashlib
import math

import numpy as np

from sphererate.data.maps import MapGeometry, erp_gaussian_map, spherical_gaussian_map
from sphererate.data.salient360 import _split_sequences


DEFAULT_HORIZONS_S = (0.5, 1.0, 3.0, 5.0)
MILLISECONDS_TO_SECONDS = 1e-3


def horizon_key(value: float) -> str:
    return f"{float(value):g}"


def load_salient360_eye_fixations(
    path: str | Path,
    *,
    timestamp_scale: float = MILLISECONDS_TO_SECONDS,
) -> np.ndarray:
    """Load one eye file, keeping one sequence ID per physical observer.

    Salient360 stores left and right eye outputs in separate files.  Call this
    function on exactly one file for a primary observer analysis; never append
    both eyes and treat them as independent observers.
    """

    sequences = _split_sequences(Path(path), eye="single", start_observer_id=0)
    if not sequences:
        return np.zeros((0, 6), dtype=np.float32)
    rows: list[np.ndarray] = []
    for observer_id, sequence in enumerate(sequences):
        current = np.asarray(sequence, dtype=np.float32).copy()
        current[:, 2] *= float(timestamp_scale)
        current[:, 3] = float(observer_id)
        current[:, 4] *= float(timestamp_scale)
        rows.append(current)
    return np.concatenate(rows, axis=0).astype(np.float32)


def clipped_dwell_weights(fixations: np.ndarray, horizon_s: float) -> np.ndarray:
    array = np.asarray(fixations, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] < 5:
        raise ValueError(f"Expected fixation array [N, >=5], got {array.shape}")
    horizon = float(horizon_s)
    if not np.isfinite(horizon) or horizon <= 0.0:
        raise ValueError("horizon_s must be finite and positive")
    starts = array[:, 2]
    durations = np.maximum(array[:, 4], 0.0)
    ends = starts + durations
    overlap = np.minimum(ends, horizon) - np.maximum(starts, 0.0)
    return np.maximum(overlap, 0.0).astype(np.float32)


def raw_dwell_tile_distribution(
    fixations: np.ndarray,
    *,
    observer_ids: Iterable[int],
    horizon_s: float,
    tile_rows: int,
    tile_cols: int,
) -> np.ndarray:
    """Aggregate clipped raw fixation dwell into a normalized ERP tile target.

    This deliberately performs no Gaussian rendering.  It is the train-time
    counterpart of the fixed-budget coverage endpoint, whose ground-truth mass
    is raw clipped dwell rather than mass from a smoothed saliency map.
    """

    array = np.asarray(fixations, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] < 5:
        raise ValueError(f"Expected fixation array [N, >=5], got {array.shape}")
    rows, cols = int(tile_rows), int(tile_cols)
    if rows <= 0 or cols <= 0:
        raise ValueError("tile grid must be positive")
    mask = _observer_mask(array, observer_ids)
    current = array[mask]
    weights = clipped_dwell_weights(current, float(horizon_s)).astype(np.float64)
    positive = np.isfinite(weights) & (weights > 0.0)
    current = current[positive]
    weights = weights[positive]
    if not len(current) or float(weights.sum()) <= 0.0:
        raise ValueError("Selected observers have no positive dwell at this horizon")

    x_tile = np.floor(np.mod(current[:, 0], 1.0) * cols).astype(np.int64)
    y_tile = np.floor(
        np.clip(current[:, 1], 0.0, np.nextafter(1.0, 0.0)) * rows
    ).astype(np.int64)
    mass = np.zeros(rows * cols, dtype=np.float64)
    np.add.at(mass, y_tile * cols + x_tile, weights)
    mass /= mass.sum()
    return mass.reshape(rows, cols).astype(np.float32)


def raw_dwell_topk_consensus(
    fixations: np.ndarray,
    *,
    horizons_s: Sequence[float],
    budgets_percent: Sequence[float],
    tile_rows: int,
    tile_cols: int,
    observer_splits: int,
    seed: int,
    key: str,
) -> np.ndarray:
    """Estimate reliable top-k membership by deterministic observer-half resampling."""

    array = np.asarray(fixations, dtype=np.float32)
    observer_ids = np.unique(array[:, 3]).astype(np.int64)
    if len(observer_ids) < 4:
        raise ValueError("top-k consensus requires at least four observers")
    split_count = int(observer_splits)
    if split_count <= 0:
        raise ValueError("observer_splits must be positive")
    rows, cols = int(tile_rows), int(tile_cols)
    budgets = [float(value) for value in budgets_percent]
    counts = [
        min(rows * cols, max(1, int(math.ceil(rows * cols * budget / 100.0))))
        for budget in budgets
    ]
    consensus = np.zeros(
        (len(horizons_s), len(budgets), rows * cols), dtype=np.float64
    )
    samples = 0
    for split_index in range(split_count):
        digest = hashlib.sha256(
            f"{int(seed)}:{key}:{split_index}".encode("utf-8")
        ).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
        permutation = rng.permutation(observer_ids)
        midpoint = len(permutation) // 2
        halves = (permutation[:midpoint], permutation[midpoint:])
        for selected in halves:
            samples += 1
            for horizon_index, horizon in enumerate(horizons_s):
                mass = raw_dwell_tile_distribution(
                    array,
                    observer_ids=selected.tolist(),
                    horizon_s=float(horizon),
                    tile_rows=rows,
                    tile_cols=cols,
                ).ravel()
                tile_ids = np.arange(mass.size, dtype=np.int64)
                order = np.lexsort((tile_ids, -mass))
                for budget_index, count in enumerate(counts):
                    consensus[horizon_index, budget_index, order[:count]] += 1.0
    consensus /= float(samples)
    return consensus.reshape(len(horizons_s), len(budgets), rows, cols).astype(np.float32)


def _observer_mask(fixations: np.ndarray, observer_ids: Iterable[int]) -> np.ndarray:
    selected = np.asarray(sorted({int(value) for value in observer_ids}), dtype=np.int64)
    if selected.size == 0:
        raise ValueError("At least one observer ID is required")
    ids = np.asarray(fixations[:, 3], dtype=np.int64)
    return np.isin(ids, selected)


def build_cumulative_density(
    fixations: np.ndarray,
    *,
    observer_ids: Iterable[int],
    horizon_s: float | None,
    height: int,
    width: int,
    sigma_px: float | None = None,
    renderer: str = "legacy_erp",
    sigma_deg: float | None = None,
) -> np.ndarray:
    array = np.asarray(fixations, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] < 5:
        raise ValueError(f"Expected fixation array [N, >=5], got {array.shape}")
    mask = _observer_mask(array, observer_ids)
    selected = array[mask]
    if horizon_s is None:
        weights = np.maximum(selected[:, 4], 0.0)
    else:
        weights = clipped_dwell_weights(selected, float(horizon_s))
    positive = np.isfinite(weights) & (weights > 0.0)
    selected = selected[positive]
    weights = weights[positive]
    if selected.size == 0 or float(weights.sum()) <= 0.0:
        scope = "final" if horizon_s is None else f"horizon={float(horizon_s):g}s"
        raise ValueError(f"Selected observer half has no positive dwell for {scope}")
    geometry = MapGeometry(int(height), int(width))
    renderer = str(renderer).lower()
    if renderer in {"legacy", "legacy_erp", "erp"}:
        if sigma_px is None:
            raise ValueError("legacy_erp renderer requires sigma_px")
        return erp_gaussian_map(
            selected[:, 0],
            selected[:, 1],
            int(height),
            int(width),
            sigma_px=float(sigma_px),
            weights=weights,
            area_weights=geometry.area_weights,
            area_weighted=True,
        ).astype(np.float32)
    if renderer in {"sphere_chord", "spherical_chord"}:
        if sigma_deg is None:
            raise ValueError("sphere_chord renderer requires sigma_deg")
        return spherical_gaussian_map(
            selected[:, 0],
            selected[:, 1],
            int(height),
            int(width),
            sigma_deg=float(sigma_deg),
            weights=weights,
            area_weights=geometry.area_weights,
            distance="chord",
            truncate_sigma=4.0,
        ).astype(np.float32)
    raise ValueError(f"Unknown saliency renderer: {renderer}")


def audit_fixation_sequences(
    fixations: np.ndarray,
    *,
    horizons_s: Sequence[float] = DEFAULT_HORIZONS_S,
) -> dict[str, object]:
    array = np.asarray(fixations, dtype=np.float32)
    observer_ids = sorted(np.unique(array[:, 3]).astype(int).tolist()) if len(array) else []
    non_monotonic = 0
    first_times: list[float] = []
    last_ends: list[float] = []
    for observer_id in observer_ids:
        current = array[array[:, 3].astype(int) == observer_id]
        timestamps = current[:, 2].astype(np.float64)
        if np.any(np.diff(timestamps) < -1e-9):
            non_monotonic += 1
        first_times.append(float(timestamps.min()))
        last_ends.append(float(np.max(timestamps + np.maximum(current[:, 4], 0.0))))
    with_mass: dict[str, int] = {}
    covering: dict[str, int] = {}
    for horizon in horizons_s:
        key = horizon_key(horizon)
        with_mass[key] = sum(
            bool((clipped_dwell_weights(array[array[:, 3].astype(int) == observer], horizon) > 0).any())
            for observer in observer_ids
        )
        covering[key] = sum(end >= float(horizon) for end in last_ends)
    return {
        "observer_count": len(observer_ids),
        "fixation_count": int(len(array)),
        "non_monotonic_observer_count": int(non_monotonic),
        "first_timestamp_s": _percentiles(first_times),
        "last_end_s": _percentiles(last_ends),
        "observers_with_mass_by_horizon": with_mass,
        "observers_covering_horizon": covering,
    }


def _percentiles(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "median": None, "p95": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(array.min()),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "max": float(array.max()),
    }
