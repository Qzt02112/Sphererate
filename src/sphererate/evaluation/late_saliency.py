from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from sphererate.data.maps import MapGeometry, erp_gaussian_map
from sphererate.evaluation.content_interaction import log_density_mix
from sphererate.evaluation.metrics import normalize_density


def build_session_window_map(
    fixations: np.ndarray,
    start_s: float,
    end_s: float,
    geometry: MapGeometry,
    sigma_px: float,
) -> np.ndarray:
    if float(start_s) < 0.0 or float(end_s) <= float(start_s):
        raise ValueError(f"Invalid time window [{start_s}, {end_s}]")
    rows = np.asarray(fixations, dtype=np.float32)
    if rows.ndim != 2 or rows.shape[1] < 5:
        raise ValueError(f"Expected fixation rows [N,>=5], got {rows.shape}")
    mask = np.isfinite(rows[:, :5]).all(axis=1)
    mask &= rows[:, 2] >= float(start_s)
    mask &= rows[:, 2] <= float(end_s)
    selected = rows[mask]
    return erp_gaussian_map(
        selected[:, 0] if len(selected) else np.array([], dtype=np.float32),
        selected[:, 1] if len(selected) else np.array([], dtype=np.float32),
        geometry.height,
        geometry.width,
        sigma_px=float(sigma_px),
        weights=selected[:, 4] if len(selected) else None,
        area_weights=geometry.area_weights,
        area_weighted=True,
    ).astype(np.float32)


def _yaw_shift_pixels(width: int, start_longitude_deg: float) -> int:
    shift = float(start_longitude_deg) * float(width) / 360.0
    rounded = int(round(shift))
    if not np.isclose(shift, rounded, atol=1e-6):
        raise ValueError(
            f"Longitude {start_longitude_deg}° does not map to an integer roll at width={width}"
        )
    return rounded


def yaw_align_map(density: np.ndarray, start_longitude_deg: float) -> np.ndarray:
    array = np.asarray(density, dtype=np.float32)
    return np.roll(array, shift=-_yaw_shift_pixels(array.shape[-1], start_longitude_deg), axis=-1)


def restore_start_template(template: np.ndarray, start_longitude_deg: float) -> np.ndarray:
    array = np.asarray(template, dtype=np.float32)
    return np.roll(array, shift=_yaw_shift_pixels(array.shape[-1], start_longitude_deg), axis=-1)


def estimate_start_template(
    samples: Iterable[tuple[np.ndarray, float]],
    area_weights: np.ndarray,
) -> np.ndarray:
    aligned = [yaw_align_map(density, longitude) for density, longitude in samples]
    if not aligned:
        raise ValueError("At least one train sample is required for the start template")
    mean = np.mean(np.stack(aligned, axis=0).astype(np.float64), axis=0)
    return normalize_density(mean, np.asarray(area_weights, dtype=np.float32)).astype(np.float32)


def fuse_log_density(
    start_map: np.ndarray,
    content_map: np.ndarray,
    alpha: float,
    area_weights: np.ndarray,
) -> np.ndarray:
    return log_density_mix(start_map, content_map, alpha=float(alpha), area=area_weights)


def deterministic_wrong_content_map(content_ids: Iterable[str], seed: int) -> dict[str, str]:
    contents = sorted(set(str(value) for value in content_ids))
    if len(contents) < 2:
        raise ValueError("At least two contents are required for a wrong-image derangement")
    rng = np.random.default_rng(int(seed))
    indices = np.arange(len(contents), dtype=np.int64)
    for _ in range(10000):
        donors = rng.permutation(indices)
        if np.all(donors != indices):
            return {contents[index]: contents[int(donors[index])] for index in indices}
    raise RuntimeError("Failed to construct a deterministic wrong-content derangement")
