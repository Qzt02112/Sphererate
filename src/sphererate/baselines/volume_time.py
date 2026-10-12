from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from sphererate.data.maps import MapGeometry
from sphererate.evaluation.metrics import normalize_density


def uniform_slice_overlap_weights(
    horizon_s: float,
    *,
    num_slices: int,
    duration_s: float,
) -> np.ndarray:
    """Compute overlap duration between uniform temporal slices and [0, horizon]."""
    horizon = float(horizon_s)
    duration = float(duration_s)
    if horizon <= 0.0 or duration <= 0.0 or int(num_slices) <= 0:
        raise ValueError("horizon, duration and num_slices must be positive")
    width = duration / int(num_slices)
    starts = np.arange(int(num_slices), dtype=np.float64) * width
    ends = starts + width
    return np.maximum(0.0, np.minimum(ends, min(horizon, duration)) - starts).astype(np.float32)


def temporal_volume_to_cumulative_maps(
    volume: np.ndarray,
    *,
    horizons_s: Sequence[float],
    duration_s: float,
    height: int,
    width: int,
) -> np.ndarray:
    """Accumulate a source-time-quantized saliency volume at real-time horizons."""
    array = np.maximum(np.asarray(volume, dtype=np.float32), 0.0)
    if array.ndim != 3 or array.shape[0] <= 0:
        raise ValueError(f"Expected volume shaped (T,H,W), got {array.shape}")
    resized = np.stack(
        [cv2.resize(current, (int(width), int(height)), interpolation=cv2.INTER_LINEAR) for current in array],
        axis=0,
    ).astype(np.float32)
    area = MapGeometry(int(height), int(width)).area_weights
    maps = []
    for horizon in horizons_s:
        weights = uniform_slice_overlap_weights(
            float(horizon), num_slices=resized.shape[0], duration_s=float(duration_s)
        )
        accumulated = (resized * weights[:, None, None]).sum(axis=0)
        maps.append(normalize_density(accumulated, area))
    return np.stack(maps, axis=0).astype(np.float32)
