from __future__ import annotations

from typing import Any

import numpy as np

from sphererate.data.maps import fixation_binary_map
from sphererate.utils import normalize_np


def _validate_prefix_bins(num_bins: int, prefix_bins: int) -> int:
    prefix_bins = int(prefix_bins)
    if prefix_bins < 0:
        raise ValueError(f"condition_prefix_bins must be >= 0, got {prefix_bins}")
    if prefix_bins >= num_bins:
        raise ValueError(f"condition_prefix_bins must be < num_bins ({num_bins}), got {prefix_bins}")
    return prefix_bins


def cumulative_from_temporal_np(
    temporal: np.ndarray,
    area_weights: np.ndarray | None = None,
    eps: float = 1e-7,
) -> np.ndarray:
    temporal = np.asarray(temporal, dtype=np.float32)
    cumulative: list[np.ndarray] = []
    running = np.zeros_like(temporal[0], dtype=np.float32)
    for idx in range(temporal.shape[0]):
        running = running + temporal[idx]
        cumulative.append(normalize_np(running, eps=eps, weights=area_weights))
    return np.stack(cumulative, axis=0).astype(np.float32)


def prefix_condition_channels_np(
    target: dict[str, np.ndarray],
    prefix_bins: int,
    include_step_channel: bool = True,
) -> np.ndarray:
    temporal = np.asarray(target["temporal"], dtype=np.float32)
    prefix_bins = _validate_prefix_bins(temporal.shape[0], prefix_bins)
    if prefix_bins == 0:
        prefix_map = np.zeros_like(temporal[0], dtype=np.float32)
    else:
        prefix_map = np.asarray(target["cumulative"], dtype=np.float32)[prefix_bins - 1]
    channels = [prefix_map.astype(np.float32)]
    if include_step_channel:
        step = np.full_like(prefix_map, float(prefix_bins) / float(max(temporal.shape[0] - 1, 1)), dtype=np.float32)
        channels.append(step)
    return np.stack(channels, axis=0).astype(np.float32)


def future_target_from_prefix_np(
    target: dict[str, np.ndarray],
    prefix_bins: int,
    fixations: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    temporal = np.asarray(target["temporal"], dtype=np.float32)
    area = np.asarray(target["area_weights"], dtype=np.float32)
    prefix_bins = _validate_prefix_bins(temporal.shape[0], prefix_bins)
    future_temporal = temporal[prefix_bins:].astype(np.float32)
    future_cumulative = cumulative_from_temporal_np(future_temporal, area)
    future_final = future_cumulative[-1].astype(np.float32)
    future_fixation = _future_fixation_map(fixations, prefix_bins, future_final.shape)
    if future_fixation is None:
        future_fixation = np.asarray(target.get("fixation", future_final), dtype=np.float32)
    return {
        "temporal": future_temporal,
        "cumulative": future_cumulative,
        "final": future_final,
        "fixation": future_fixation.astype(np.float32),
        "area_weights": area,
    }


def apply_prefix_condition_np(
    target: dict[str, np.ndarray],
    prefix_bins: int,
    fixations: np.ndarray | None = None,
    include_step_channel: bool = True,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    conditioned_target = future_target_from_prefix_np(target, prefix_bins, fixations=fixations)
    channels = prefix_condition_channels_np(target, prefix_bins, include_step_channel=include_step_channel)
    return conditioned_target, channels


def target_npz_to_dict(data: Any) -> dict[str, np.ndarray]:
    return {key: data[key].astype(np.float32).squeeze() for key in data.files}


def _future_fixation_map(
    fixations: np.ndarray | None,
    prefix_bins: int,
    shape_hw: tuple[int, int],
) -> np.ndarray | None:
    if fixations is None:
        return None
    fix = np.asarray(fixations, dtype=np.float32)
    if fix.ndim != 2 or fix.shape[1] < 6:
        return None
    future = fix[fix[:, 5] >= float(prefix_bins)]
    h, w = shape_hw
    return fixation_binary_map(
        future[:, 0] if len(future) else np.array([], dtype=np.float32),
        future[:, 1] if len(future) else np.array([], dtype=np.float32),
        h,
        w,
    ).astype(np.float32)
