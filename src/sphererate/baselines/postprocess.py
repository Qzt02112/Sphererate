from __future__ import annotations

from pathlib import Path

import numpy as np

from sphererate.data.maps import build_temporal_maps
from sphererate.utils import ensure_dir


def scanpaths_to_fixation_rows(scanpaths: np.ndarray, num_bins: int, order: str = "yx") -> np.ndarray:
    """Convert scanpath array (N, T, 2) into unified fixation rows.

    Most ScanDMM-style outputs use normalized (y, x). Set order="xy" for
    normalized (x, y).
    """
    if scanpaths.ndim != 3 or scanpaths.shape[-1] < 2:
        raise ValueError(f"Expected scanpaths shape (N,T,2+), got {scanpaths.shape}")
    rows = []
    for obs_id in range(scanpaths.shape[0]):
        seq = scanpaths[obs_id]
        valid = np.isfinite(seq[:, :2]).all(axis=1)
        seq = seq[valid]
        n = len(seq)
        if n == 0:
            continue
        bins = np.floor(np.arange(n, dtype=np.float32) * num_bins / max(n, 1)).astype(np.int64)
        bins = np.clip(bins, 0, num_bins - 1)
        arr = np.zeros((n, 6), dtype=np.float32)
        if order == "yx":
            arr[:, 0] = seq[:, 1] % 1.0
            arr[:, 1] = np.clip(seq[:, 0], 0.0, 1.0)
        else:
            arr[:, 0] = seq[:, 0] % 1.0
            arr[:, 1] = np.clip(seq[:, 1], 0.0, 1.0)
        arr[:, 2] = np.arange(n, dtype=np.float32)
        arr[:, 3] = float(obs_id)
        arr[:, 4] = 1.0
        arr[:, 5] = bins
        rows.append(arr)
    if not rows:
        return np.zeros((0, 6), dtype=np.float32)
    return np.concatenate(rows, axis=0)


def postprocess_scanpath_dir(
    scanpath_dir: str | Path,
    out_dir: str | Path,
    height: int,
    width: int,
    num_bins: int,
    sigma_px: float,
    order: str = "yx",
) -> Path:
    scanpath_dir = Path(scanpath_dir)
    out_dir = ensure_dir(out_dir)
    for path in sorted(scanpath_dir.glob("*.npy")):
        scanpaths = np.load(path)
        fix = scanpaths_to_fixation_rows(scanpaths, num_bins=num_bins, order=order)
        maps = build_temporal_maps(
            fix,
            num_bins=num_bins,
            height=height,
            width=width,
            sigma_px=sigma_px,
            area_weighted=True,
        )
        np.savez_compressed(out_dir / f"{path.stem}.npz", final=maps["final"], temporal=maps["temporal"], cumulative=maps["cumulative"])
    return out_dir
