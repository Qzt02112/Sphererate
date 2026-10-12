from __future__ import annotations

from collections import defaultdict
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from sphererate.data.conditioning import future_target_from_prefix_np, target_npz_to_dict
from sphererate.evaluation.metrics import mean_scores, saliency_scores
from sphererate.utils import ensure_dir, read_jsonl


def _load_prediction(pred_dir: Path, image_id: str) -> np.ndarray:
    npz_path = pred_dir / f"{image_id}.npz"
    npy_path = pred_dir / f"{image_id}.npy"
    if npz_path.exists():
        data = np.load(npz_path)
        if "final" in data:
            return data["final"].astype(np.float32).squeeze()
        first = data.files[0]
        return data[first].astype(np.float32).squeeze()
    if npy_path.exists():
        return np.load(npy_path).astype(np.float32).squeeze()
    raise FileNotFoundError(f"No prediction found for {image_id} in {pred_dir}")


def _row_dataset(row: dict[str, Any]) -> str:
    return str(row.get("dataset") or "unknown")


def _row_condition(row: dict[str, Any]) -> str | None:
    value = row.get("condition")
    if value is None or value == "":
        return None
    return str(value)


def _load_fixations(row: dict[str, Any]) -> np.ndarray | None:
    path = row.get("fixations")
    if not path:
        return None
    with np.load(path) as data:
        if "fixations" not in data:
            return None
        return data["fixations"].astype(np.float32)


def _load_target(row: dict[str, Any], condition_prefix_bins: int = 0) -> dict[str, np.ndarray]:
    with np.load(row["target"]) as target_data:
        target = target_npz_to_dict(target_data)
    if condition_prefix_bins > 0:
        target = future_target_from_prefix_np(
            target,
            condition_prefix_bins,
            fixations=_load_fixations(row),
        )
    return target


def evaluate_prediction_dir(
    processed_root: str | Path,
    pred_dir: str | Path,
    split: str = "val",
    output_dir: str | Path | None = None,
    image_ids: set[str] | None = None,
    condition_prefix_bins: int = 0,
) -> dict[str, Any]:
    processed_root = Path(processed_root)
    pred_dir = Path(pred_dir)
    rows = [row for row in read_jsonl(processed_root / "manifest.jsonl") if row["split"] == split]
    if image_ids is not None:
        rows = [row for row in rows if row["image_id"] in image_ids]
    per_image: list[dict[str, Any]] = []
    score_rows: list[dict[str, float]] = []
    score_rows_by_dataset: dict[str, list[dict[str, float]]] = defaultdict(list)
    score_rows_by_condition: dict[str, list[dict[str, float]]] = defaultdict(list)
    for row in rows:
        pred = _load_prediction(pred_dir, row["image_id"])
        target = _load_target(row, condition_prefix_bins=condition_prefix_bins)
        gt = target["final"].astype(np.float32)
        fixation = target["fixation"].astype(np.float32)
        area = target["area_weights"].astype(np.float32)
        scores = saliency_scores(pred, gt, fixation, area).as_dict()
        dataset_name = _row_dataset(row)
        condition_name = _row_condition(row)
        score_rows.append(scores)
        score_rows_by_dataset[dataset_name].append(scores)
        if condition_name is not None:
            score_rows_by_condition[condition_name].append(scores)
        image_row = {"dataset": dataset_name, "image_id": row["image_id"], **scores}
        if condition_name is not None:
            image_row["condition"] = condition_name
        per_image.append(image_row)

    dataset_summaries = {
        name: {"num_images": len(rows), "scores": mean_scores(rows)}
        for name, rows in sorted(score_rows_by_dataset.items())
    }
    macro_scores = mean_scores([data["scores"] for data in dataset_summaries.values()])
    condition_summaries = {
        name: {"num_images": len(rows), "scores": mean_scores(rows)}
        for name, rows in sorted(score_rows_by_condition.items())
    }
    summary = {
        "split": split,
        "num_images": len(per_image),
        "scores": mean_scores(score_rows),
        "datasets": dataset_summaries,
        "conditions": condition_summaries,
        "macro_scores": macro_scores,
    }
    if image_ids is not None:
        summary["image_id_filter"] = sorted(image_ids)
    if condition_prefix_bins > 0:
        summary["condition_prefix_bins"] = int(condition_prefix_bins)
    if output_dir is not None:
        out = ensure_dir(output_dir)
        with open(out / f"{split}_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        if per_image:
            with open(out / f"{split}_per_image.csv", "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(per_image[0].keys()))
                writer.writeheader()
                writer.writerows(per_image)
    return summary
