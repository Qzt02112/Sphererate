from __future__ import annotations

import numpy as np


def _pixel_unit_vectors(height: int, width: int, ys: np.ndarray, xs: np.ndarray) -> np.ndarray:
    lon = 2.0 * np.pi * ((xs.astype(np.float32) + 0.5) / float(width)) - np.pi
    lat = np.pi / 2.0 - np.pi * ((ys.astype(np.float32) + 0.5) / float(height))
    cos_lat = np.cos(lat)
    return np.stack([cos_lat * np.cos(lon), cos_lat * np.sin(lon), np.sin(lat)], axis=-1)


def angular_error_deg(pred: np.ndarray, fixation: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=np.float32).squeeze()
    fixation = np.asarray(fixation, dtype=np.float32).squeeze()
    if pred.ndim != 2 or fixation.ndim != 2:
        raise ValueError(f"Expected 2D pred/fixation maps, got {pred.shape}/{fixation.shape}")
    h, w = pred.shape
    pred_y, pred_x = np.unravel_index(int(np.nanargmax(pred)), pred.shape)
    gt_y, gt_x = np.nonzero(fixation > 0.5)
    if len(gt_y) == 0:
        return float("nan")
    pred_vec = _pixel_unit_vectors(h, w, np.asarray([pred_y]), np.asarray([pred_x]))[0]
    gt_vec = _pixel_unit_vectors(h, w, gt_y, gt_x)
    dots = np.clip(gt_vec @ pred_vec, -1.0, 1.0)
    return float(np.degrees(np.arccos(dots)).min())


def topk_hit(pred: np.ndarray, fixation: np.ndarray, frac: float) -> float:
    pred = np.asarray(pred, dtype=np.float32).squeeze()
    fixation = np.asarray(fixation, dtype=np.float32).squeeze()
    if pred.ndim != 2 or fixation.ndim != 2:
        raise ValueError(f"Expected 2D pred/fixation maps, got {pred.shape}/{fixation.shape}")
    gt = np.flatnonzero(fixation.reshape(-1) > 0.5)
    if len(gt) == 0:
        return float("nan")
    frac = min(max(float(frac), 1e-6), 1.0)
    k = max(1, int(round(frac * pred.size)))
    top = np.argpartition(pred.reshape(-1), -k)[-k:]
    return float(bool(np.intersect1d(top, gt, assume_unique=False).size))


def next_fixation_scores(pred: np.ndarray, fixation: np.ndarray) -> dict[str, float]:
    return {
        "peak_error_deg": angular_error_deg(pred, fixation),
        "hit_top1pct": topk_hit(pred, fixation, 0.01),
        "hit_top5pct": topk_hit(pred, fixation, 0.05),
    }
