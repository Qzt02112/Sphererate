from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from sphererate.evaluation.metrics import kld, normalize_density
from sphererate.utils import ensure_dir, read_jsonl


HARDNESS_KEYS = (
    "temporal_step_tv_gt",
    "cumulative_early_late_sym_kld_sph",
    "temporal_first_last_peak_displacement",
)


def _as_stack(value: np.ndarray, key: str, image_id: str) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32).squeeze()
    if value.ndim != 3:
        raise ValueError(f"{image_id}/{key} must be a KxHxW stack, got shape={value.shape}")
    return value


def _step_tv(stack: np.ndarray, area: np.ndarray) -> float:
    if stack.shape[0] < 2:
        return float("nan")
    values: list[float] = []
    for idx in range(stack.shape[0] - 1):
        cur = normalize_density(stack[idx], area)
        nxt = normalize_density(stack[idx + 1], area)
        values.append(0.5 * float((np.abs(nxt - cur) * area).sum()))
    return float(np.nanmean(values))


def _sym_kld(a: np.ndarray, b: np.ndarray, area: np.ndarray) -> float:
    return 0.5 * (kld(a, b, area) + kld(b, a, area))


def _peak_xy(value: np.ndarray) -> tuple[int, int]:
    y, x = np.unravel_index(int(np.asarray(value).argmax()), value.shape)
    return int(x), int(y)


def _erp_peak_distance(a: np.ndarray, b: np.ndarray) -> float:
    h, w = a.shape
    x1, y1 = _peak_xy(a)
    x2, y2 = _peak_xy(b)
    dx = abs(x2 - x1)
    dx = min(dx, w - dx) / max(float(w - 1), 1.0)
    dy = abs(y2 - y1) / max(float(h - 1), 1.0)
    return float(np.sqrt(dx * dx + dy * dy))


def _mean_consecutive_peak_distance(stack: np.ndarray) -> float:
    if stack.shape[0] < 2:
        return float("nan")
    return float(np.nanmean([_erp_peak_distance(stack[idx], stack[idx + 1]) for idx in range(stack.shape[0] - 1)]))


def _add_hardness_scores(rows: list[dict[str, Any]], keys: Iterable[str] = HARDNESS_KEYS) -> list[dict[str, Any]]:
    keys = tuple(keys)
    stats: dict[str, tuple[float, float]] = {}
    for key in keys:
        values = np.asarray([float(row[key]) for row in rows], dtype=np.float32)
        mean = float(np.nanmean(values))
        std = float(np.nanstd(values))
        stats[key] = (mean, std)

    for row in rows:
        z_values: list[float] = []
        for key in keys:
            mean, std = stats[key]
            z_values.append(0.0 if std <= 1e-7 else (float(row[key]) - mean) / std)
        row["temporal_hardness_score"] = float(np.nanmean(z_values))

    rows.sort(key=lambda row: (-float(row["temporal_hardness_score"]), str(row["image_id"])))
    for idx, row in enumerate(rows, start=1):
        row["temporal_hardness_rank"] = idx
    return rows


def temporal_hardness_scores(processed_root: str | Path, split: str = "test") -> list[dict[str, Any]]:
    processed_root = Path(processed_root)
    manifest = [row for row in read_jsonl(processed_root / "manifest.jsonl") if row["split"] == split]
    rows: list[dict[str, Any]] = []
    for manifest_row in manifest:
        image_id = str(manifest_row["image_id"])
        with np.load(manifest_row["target"]) as target_data:
            temporal = _as_stack(target_data["temporal"], "temporal", image_id)
            cumulative = _as_stack(target_data["cumulative"], "cumulative", image_id)
            area = np.asarray(target_data["area_weights"], dtype=np.float32)

        first_last_peak = _erp_peak_distance(temporal[0], temporal[-1])
        rows.append(
            {
                "image_id": image_id,
                "split": split,
                "temporal_step_tv_gt": _step_tv(temporal, area),
                "cumulative_step_tv_gt": _step_tv(cumulative, area),
                "cumulative_early_late_sym_kld_sph": _sym_kld(cumulative[0], cumulative[-1], area),
                "temporal_first_last_peak_displacement": first_last_peak,
                "temporal_mean_peak_displacement": _mean_consecutive_peak_distance(temporal),
            }
        )
    return _add_hardness_scores(rows)


def select_hard_rows(
    rows: list[dict[str, Any]],
    top_k: int | None = None,
    top_fraction: float | None = None,
) -> list[dict[str, Any]]:
    if top_k is None and top_fraction is None:
        return list(rows)
    if top_k is None:
        assert top_fraction is not None
        top_k = max(1, int(np.ceil(len(rows) * top_fraction)))
    top_k = max(0, min(int(top_k), len(rows)))
    return list(rows[:top_k])


def write_hard_subset(
    rows: list[dict[str, Any]],
    selected_rows: list[dict[str, Any]],
    csv_path: str | Path,
    id_path: str | Path,
) -> None:
    selected_ids = {row["image_id"] for row in selected_rows}
    csv_path = Path(csv_path)
    id_path = Path(id_path)
    ensure_dir(csv_path.parent)
    ensure_dir(id_path.parent)

    output_rows: list[dict[str, Any]] = []
    for row in rows:
        output_rows.append({**row, "selected": row["image_id"] in selected_ids})

    if output_rows:
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(output_rows[0].keys()))
            writer.writeheader()
            writer.writerows(output_rows)
    else:
        csv_path.write_text("", encoding="utf-8")

    with open(id_path, "w", encoding="utf-8") as f:
        for row in selected_rows:
            f.write(f"{row['image_id']}\n")
