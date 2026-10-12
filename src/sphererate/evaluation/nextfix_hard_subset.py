from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from sphererate.utils import ensure_dir, read_jsonl


def _load_fixations(path: str | Path | None) -> np.ndarray | None:
    if not path:
        return None
    with np.load(path) as data:
        if "fixations" not in data:
            return None
        return data["fixations"].astype(np.float32)


def _unit_vector(point: np.ndarray) -> np.ndarray:
    lon = 2.0 * np.pi * float(point[0]) - np.pi
    lat = np.pi / 2.0 - np.pi * float(point[1])
    cos_lat = np.cos(lat)
    return np.asarray([cos_lat * np.cos(lon), cos_lat * np.sin(lon), np.sin(lat)], dtype=np.float64)


def angular_distance_deg(a: np.ndarray, b: np.ndarray) -> float:
    va = _unit_vector(a)
    vb = _unit_vector(b)
    dot = float(np.clip(np.dot(va, vb), -1.0, 1.0))
    return float(np.degrees(np.arccos(dot)))


def _wrapped_delta(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    dx = (float(b[0]) - float(a[0]) + 0.5) % 1.0 - 0.5
    mid_y = 0.5 * (float(a[1]) + float(b[1]))
    lat = np.pi / 2.0 - np.pi * mid_y
    dy = float(b[1]) - float(a[1])
    return np.asarray([dx * np.cos(lat), dy], dtype=np.float64)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-8:
        return 0.0
    return float(np.clip(np.dot(a, b) / denom, -1.0, 1.0))


def _zscore_rows(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> None:
    for key in keys:
        values = np.asarray([float(row[key]) for row in rows], dtype=np.float32)
        mean = float(np.nanmean(values))
        std = float(np.nanstd(values))
        for row in rows:
            row[f"z_{key}"] = 0.0 if std <= 1e-7 else float((float(row[key]) - mean) / std)


def nextfix_hardness_scores(
    processed_root: str | Path,
    split: str = "test",
    prefix_length: int = 3,
) -> list[dict[str, Any]]:
    processed_root = Path(processed_root)
    image_rows = [row for row in read_jsonl(processed_root / "manifest.jsonl") if row["split"] == split]
    rows: list[dict[str, Any]] = []
    for image_row in image_rows:
        fix = _load_fixations(image_row.get("fixations"))
        if fix is None or fix.ndim != 2 or fix.shape[1] < 6:
            continue
        by_observer: dict[int, list[np.ndarray]] = defaultdict(list)
        for fixation in fix:
            by_observer[int(fixation[3])].append(fixation)
        for observer_id, points in by_observer.items():
            ordered = np.asarray(sorted(points, key=lambda item: (float(item[2]), float(item[5]))), dtype=np.float32)
            if len(ordered) <= prefix_length:
                continue
            for target_index in range(prefix_length, len(ordered)):
                prefix = ordered[target_index - prefix_length : target_index]
                target = ordered[target_index]
                current = prefix[-1]
                previous = prefix[-2] if len(prefix) >= 2 else current
                older = prefix[-3] if len(prefix) >= 3 else previous
                prev_vec = _wrapped_delta(previous, current)
                older_vec = _wrapped_delta(older, previous)
                next_vec = _wrapped_delta(current, target)
                next_deg = angular_distance_deg(current, target)
                prev_deg = angular_distance_deg(previous, current)
                older_deg = angular_distance_deg(older, previous)
                turn_cos = _cosine(prev_vec, next_vec)
                history_consistency = abs(_cosine(older_vec, prev_vec))
                motion_support = np.sqrt(max(next_deg, 0.0) * max(prev_deg, 0.0))
                rows.append(
                    {
                        "sample_id": f"{image_row['image_id']}__obs{observer_id}__t{target_index}",
                        "image_id": str(image_row["image_id"]),
                        "dataset": str(image_row.get("dataset") or "unknown"),
                        "split": split,
                        "observer_id": int(observer_id),
                        "target_index": int(target_index),
                        "prefix_length": int(prefix_length),
                        "next_deg": float(next_deg),
                        "prev_deg": float(prev_deg),
                        "older_deg": float(older_deg),
                        "turn_cos": float(turn_cos),
                        "abs_turn_cos": float(abs(turn_cos)),
                        "history_consistency": float(history_consistency),
                        "motion_support": float(motion_support),
                    }
                )
    if not rows:
        return rows
    _zscore_rows(rows, ("next_deg", "prev_deg", "motion_support", "abs_turn_cos", "history_consistency"))
    for row in rows:
        row["order_hardness_score"] = float(
            row["z_next_deg"]
            + 0.75 * row["z_prev_deg"]
            + 0.75 * row["z_motion_support"]
            + 0.35 * row["z_abs_turn_cos"]
            + 0.15 * row["z_history_consistency"]
        )
    rows.sort(key=lambda row: (-float(row["order_hardness_score"]), str(row["sample_id"])))
    for idx, row in enumerate(rows, start=1):
        row["order_hardness_rank"] = idx
    return rows


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
    sample_id_path: str | Path,
) -> None:
    selected_ids = {str(row["sample_id"]) for row in selected_rows}
    csv_path = Path(csv_path)
    sample_id_path = Path(sample_id_path)
    ensure_dir(csv_path.parent)
    ensure_dir(sample_id_path.parent)

    output_rows = [{**row, "selected": str(row["sample_id"]) in selected_ids} for row in rows]
    if output_rows:
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(output_rows[0].keys()))
            writer.writeheader()
            writer.writerows(output_rows)
    else:
        csv_path.write_text("", encoding="utf-8")

    with open(sample_id_path, "w", encoding="utf-8") as f:
        for row in selected_rows:
            f.write(f"{row['sample_id']}\n")
