from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from sphererate.data.maps import build_temporal_maps
from sphererate.evaluation.metrics import mean_scores, saliency_scores
from sphererate.utils import ensure_dir, read_jsonl


def _score_one_image(
    row: dict[str, Any],
    num_bins: int,
    sigma_px: float,
    min_fixations: int = 2,
) -> list[dict[str, Any]]:
    target = np.load(row["target"])
    height, width = target["final"].shape
    fix_data = np.load(row["fixations"])
    fixations = fix_data["fixations"].astype(np.float32)
    if fixations.size == 0:
        return []

    observer_ids = np.unique(fixations[:, 3].astype(np.int64))
    out: list[dict[str, Any]] = []
    for observer_id in observer_ids:
        held_mask = fixations[:, 3].astype(np.int64) == observer_id
        held = fixations[held_mask]
        others = fixations[~held_mask]
        if len(held) < min_fixations or len(others) < min_fixations:
            continue
        pred_maps = build_temporal_maps(
            others,
            num_bins=num_bins,
            height=height,
            width=width,
            sigma_px=sigma_px,
            area_weighted=True,
        )
        held_maps = build_temporal_maps(
            held,
            num_bins=num_bins,
            height=height,
            width=width,
            sigma_px=sigma_px,
            area_weighted=True,
        )
        scores = saliency_scores(
            pred_maps["final"],
            held_maps["final"],
            held_maps["fixation"],
            pred_maps["area_weights"],
        ).as_dict()
        out.append({"image_id": row["image_id"], "observer_id": int(observer_id), **scores})
    return out


def _score_one_image_split_half(
    row: dict[str, Any],
    num_bins: int,
    sigma_px: float,
    repeats: int,
    seed: int,
    min_fixations: int = 2,
) -> list[dict[str, Any]]:
    target = np.load(row["target"])
    height, width = target["final"].shape
    fix_data = np.load(row["fixations"])
    fixations = fix_data["fixations"].astype(np.float32)
    if fixations.size == 0:
        return []

    observer_ids = np.unique(fixations[:, 3].astype(np.int64))
    if len(observer_ids) < 2:
        return []
    out: list[dict[str, Any]] = []
    for repeat in range(repeats):
        rng = np.random.default_rng(seed + repeat * 1009 + int(row["image_id"].lstrip("P")))
        shuffled = observer_ids.copy()
        rng.shuffle(shuffled)
        split = len(shuffled) // 2
        pred_ids = set(int(x) for x in shuffled[:split])
        target_ids = set(int(x) for x in shuffled[split:])
        pred_mask = np.array([int(x) in pred_ids for x in fixations[:, 3].astype(np.int64)])
        target_mask = np.array([int(x) in target_ids for x in fixations[:, 3].astype(np.int64)])
        pred_fix = fixations[pred_mask]
        target_fix = fixations[target_mask]
        if len(pred_fix) < min_fixations or len(target_fix) < min_fixations:
            continue
        pred_maps = build_temporal_maps(
            pred_fix,
            num_bins=num_bins,
            height=height,
            width=width,
            sigma_px=sigma_px,
            area_weighted=True,
        )
        target_maps = build_temporal_maps(
            target_fix,
            num_bins=num_bins,
            height=height,
            width=width,
            sigma_px=sigma_px,
            area_weighted=True,
        )
        scores = saliency_scores(
            pred_maps["final"],
            target_maps["final"],
            target_maps["fixation"],
            pred_maps["area_weights"],
        ).as_dict()
        out.append({"image_id": row["image_id"], "repeat": repeat, **scores})
    return out


def evaluate_human_upper_bound(
    processed_root: str | Path,
    split: str,
    num_bins: int,
    sigma_px: float,
    output_dir: str | Path | None = None,
    max_images: int | None = None,
    protocol: str = "leave_one_observer",
    repeats: int = 5,
    seed: int = 20260708,
) -> dict[str, Any]:
    processed_root = Path(processed_root)
    rows = [row for row in read_jsonl(processed_root / "manifest.jsonl") if row["split"] == split]
    rows.sort(key=lambda row: row["image_id"])
    if max_images is not None:
        rows = rows[: int(max_images)]

    per_observer: list[dict[str, Any]] = []
    for row in rows:
        if protocol == "leave_one_observer":
            per_observer.extend(_score_one_image(row, num_bins=num_bins, sigma_px=sigma_px))
        elif protocol == "split_half":
            per_observer.extend(
                _score_one_image_split_half(
                    row,
                    num_bins=num_bins,
                    sigma_px=sigma_px,
                    repeats=repeats,
                    seed=seed,
                )
            )
        else:
            raise ValueError(f"Unknown human consistency protocol: {protocol}")

    metric_rows = [
        {key: value for key, value in row.items() if key not in {"image_id", "observer_id", "repeat"}}
        for row in per_observer
    ]
    summary = {
        "split": split,
        "num_images": len(rows),
        "num_observer_splits": len(per_observer),
        "scores": mean_scores(metric_rows),
        "protocol": protocol,
        "protocol_notes": (
            "leave-one-observer-out: prediction from other observers, evaluated on held-out observer map/fixations"
            if protocol == "leave_one_observer"
            else f"split-half: random observer halves, {repeats} repeats, evaluated on target half map/fixations"
        ),
    }

    if output_dir is not None:
        out = ensure_dir(output_dir)
        with open(out / f"{split}_human_upper_bound_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        if per_observer:
            with open(out / f"{split}_human_upper_bound_per_observer.csv", "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(per_observer[0].keys()))
                writer.writeheader()
                writer.writerows(per_observer)
    return summary
