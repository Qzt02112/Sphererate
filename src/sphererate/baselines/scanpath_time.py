from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from sphererate.data.maps import MapGeometry, erp_gaussian_map, spherical_gaussian_map


def official_second_step_weights(
    horizon_s: float,
    length: int,
    *,
    half_second_policy: str = "first_second_approximation",
) -> np.ndarray:
    """Return cumulative weights for scanpaths sampled once per second.

    ScanDMM's official preprocessing samples one gaze point from every one-second
    bin. It cannot resolve the internal ordering of the first second, so 0.5 s
    uses the first predicted point as an explicitly labelled approximation.
    """
    if length <= 0:
        raise ValueError("length must be positive")
    horizon = float(horizon_s)
    if horizon <= 0.0:
        raise ValueError("horizon_s must be positive")
    weights = np.zeros(int(length), dtype=np.float32)
    if horizon < 1.0:
        if half_second_policy != "first_second_approximation":
            raise ValueError(f"Unsupported sub-second policy: {half_second_policy}")
        weights[0] = 1.0
        return weights
    complete_steps = min(int(length), max(1, int(np.floor(horizon + 1e-8))))
    weights[:complete_steps] = 1.0
    return weights


def second_sampled_scanpaths_to_cumulative_maps(
    scanpaths: np.ndarray,
    *,
    horizons_s: Sequence[float],
    height: int,
    width: int,
    sigma_px: float | None = None,
    renderer: str = "legacy_erp",
    sigma_deg: float | None = None,
    coordinate_order: str = "yx",
    half_second_policy: str = "first_second_approximation",
) -> np.ndarray:
    """Convert official one-point-per-second scanpaths to cumulative ERP maps."""
    paths = np.asarray(scanpaths, dtype=np.float32)
    if paths.ndim != 3 or paths.shape[-1] < 2:
        raise ValueError(f"Expected scanpaths shaped (N,T,2+), got {paths.shape}")
    if coordinate_order not in {"yx", "xy"}:
        raise ValueError("coordinate_order must be 'yx' or 'xy'")
    if paths.shape[0] == 0 or paths.shape[1] == 0:
        raise ValueError("scanpaths cannot be empty")
    valid = np.isfinite(paths[..., :2]).all(axis=-1)
    if coordinate_order == "yx":
        y_all, x_all = paths[..., 0], paths[..., 1]
    else:
        x_all, y_all = paths[..., 0], paths[..., 1]
    area = MapGeometry(int(height), int(width)).area_weights
    maps = []
    for horizon in horizons_s:
        step_weights = official_second_step_weights(
            float(horizon),
            paths.shape[1],
            half_second_policy=half_second_policy,
        )
        point_weights = np.broadcast_to(step_weights[None, :], valid.shape)
        keep = valid & (point_weights > 0.0)
        if str(renderer).lower() in {"legacy", "legacy_erp", "erp"}:
            if sigma_px is None:
                raise ValueError("legacy_erp renderer requires sigma_px")
            rendered = erp_gaussian_map(
                np.mod(x_all[keep], 1.0),
                np.clip(y_all[keep], 0.0, 1.0),
                int(height),
                int(width),
                sigma_px=float(sigma_px),
                weights=point_weights[keep],
                area_weights=area,
                area_weighted=True,
            )
        elif str(renderer).lower() in {"sphere_chord", "spherical_chord"}:
            if sigma_deg is None:
                raise ValueError("sphere_chord renderer requires sigma_deg")
            rendered = spherical_gaussian_map(
                np.mod(x_all[keep], 1.0),
                np.clip(y_all[keep], 0.0, 1.0),
                int(height),
                int(width),
                sigma_deg=float(sigma_deg),
                weights=point_weights[keep],
                area_weights=area,
                distance="chord",
                truncate_sigma=4.0,
            )
        else:
            raise ValueError(f"Unknown saliency renderer: {renderer}")
        maps.append(rendered)
    return np.stack(maps, axis=0).astype(np.float32)
