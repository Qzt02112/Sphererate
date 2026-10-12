from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from sphererate.data.maps import MapGeometry, erp_gaussian_map
from sphererate.utils import ensure_dir, normalize_np, read_jsonl


def _prior_map(kind: str, height: int, width: int) -> np.ndarray:
    geom = MapGeometry(height, width)
    area = geom.area_weights
    kind = kind.lower()
    if kind == "uniform":
        return normalize_np(np.ones((height, width), dtype=np.float32), weights=area)
    if kind == "equator":
        y = np.linspace(0.0, 1.0, height, endpoint=False, dtype=np.float32) + 0.5 / height
        band = np.exp(-((y - 0.5) ** 2) / (2 * 0.16**2)).astype(np.float32)
        pred = band[:, None].repeat(width, axis=1)
        return normalize_np(pred, weights=area)
    if kind == "center":
        return erp_gaussian_map(
            np.array([0.5], dtype=np.float32),
            np.array([0.5], dtype=np.float32),
            height,
            width,
            sigma_px=max(height, width) / 10.0,
            area_weights=area,
            area_weighted=True,
        )
    raise ValueError(f"Unknown prior kind: {kind}")


def _load_train_average_prior(rows: list[dict[str, Any]]) -> tuple[np.ndarray, int, int]:
    """Build an empirical prior from train targets without reading target-split GT."""
    train_rows = [row for row in rows if row.get("split") == "train"]
    if not train_rows:
        raise ValueError("Cannot build train_average prior: manifest has no train rows")

    total: np.ndarray | None = None
    reference_area: np.ndarray | None = None
    reference_shape: tuple[int, int] | None = None
    num_bins: int | None = None
    for row in train_rows:
        with np.load(row["target"]) as target:
            final = np.asarray(target["final"], dtype=np.float64)
            area = np.asarray(target["area_weights"], dtype=np.float64)
            temporal = np.asarray(target["temporal"])
        if final.ndim != 2 or area.shape != final.shape or temporal.ndim != 3:
            raise ValueError(f"Invalid target tensors for train row {row.get('image_id')}")
        if not np.isfinite(final).all() or not np.isfinite(area).all():
            raise ValueError(f"Non-finite train target for {row.get('image_id')}")
        if (final < 0).any() or (area < 0).any():
            raise ValueError(f"Negative train target/area weight for {row.get('image_id')}")

        shape = tuple(final.shape)
        row_num_bins = int(temporal.shape[0])
        if total is None:
            total = np.zeros_like(final, dtype=np.float64)
            reference_area = area
            reference_shape = shape
            num_bins = row_num_bins
        elif shape != reference_shape or row_num_bins != num_bins or not np.allclose(area, reference_area):
            raise ValueError(f"Inconsistent train target geometry for {row.get('image_id')}")
        total += final

    assert total is not None and reference_area is not None and num_bins is not None
    average = total / float(len(train_rows))
    pred = normalize_np(average.astype(np.float32), weights=reference_area.astype(np.float32))
    return pred, num_bins, len(train_rows)


def make_prior_predictions(processed_root: str | Path, out_dir: str | Path, kind: str, split: str = "val") -> Path:
    processed_root = Path(processed_root)
    out_dir = ensure_dir(out_dir)
    all_rows = read_jsonl(processed_root / "manifest.jsonl")
    output_rows = [row for row in all_rows if row["split"] == split]
    if not output_rows:
        raise ValueError(f"No rows for split={split} in {processed_root}")

    kind = kind.lower()
    source_split: str | None = None
    source_count = 0
    if kind == "train_average":
        pred, num_bins, source_count = _load_train_average_prior(all_rows)
        source_split = "train"
    else:
        with np.load(output_rows[0]["target"]) as first:
            height, width = first["final"].shape
            num_bins = int(first["temporal"].shape[0])
        pred = _prior_map(kind, height, width)

    if pred.ndim != 2 or not np.isfinite(pred).all() or (pred < 0).any():
        raise ValueError(f"Prior {kind} is not a finite non-negative 2D map")
    temporal = np.repeat(pred[None, :, :], num_bins, axis=0)
    cumulative = temporal.copy()
    for row in output_rows:
        np.savez_compressed(out_dir / f"{row['image_id']}.npz", final=pred, temporal=temporal, cumulative=cumulative)

    metadata = {
        "kind": kind,
        "prediction_split": split,
        "prediction_count": len(output_rows),
        "source_split": source_split,
        "source_count": source_count,
        "processed_root": str(processed_root),
        "shape": list(pred.shape),
        "num_bins": num_bins,
    }
    with (out_dir / "prior_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    return out_dir
