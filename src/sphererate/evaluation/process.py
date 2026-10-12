from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from sphererate.data.conditioning import future_target_from_prefix_np, target_npz_to_dict
from sphererate.evaluation.metrics import cc, kld, mean_scores, normalize_density, sim
from sphererate.utils import ensure_dir, read_jsonl


PROCESS_KEYS = ("temporal", "cumulative")


def _load_prediction_npz(pred_dir: Path, image_id: str) -> dict[str, np.ndarray]:
    path = pred_dir / f"{image_id}.npz"
    if not path.exists():
        raise FileNotFoundError(f"No npz prediction found for {image_id} in {pred_dir}")
    with np.load(path) as data:
        return {key: data[key].astype(np.float32).squeeze() for key in data.files}


def _as_stack(value: np.ndarray, key: str, image_id: str) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32).squeeze()
    if value.ndim != 3:
        raise ValueError(f"{image_id}/{key} must be a KxHxW stack, got shape={value.shape}")
    return value


def _bin_metric_means(prefix: str, pred: np.ndarray, gt: np.ndarray, area: np.ndarray) -> dict[str, float]:
    rows: list[dict[str, float]] = []
    for idx in range(pred.shape[0]):
        rows.append(
            {
                f"{prefix}_cc": cc(pred[idx], gt[idx]),
                f"{prefix}_sim": sim(pred[idx], gt[idx]),
                f"{prefix}_kld": kld(pred[idx], gt[idx]),
                f"{prefix}_cc_sph": cc(pred[idx], gt[idx], area),
                f"{prefix}_sim_sph": sim(pred[idx], gt[idx], area),
                f"{prefix}_kld_sph": kld(pred[idx], gt[idx], area),
            }
        )
    return mean_scores(rows)


def _step_tv(stack: np.ndarray, area: np.ndarray) -> float:
    if stack.shape[0] < 2:
        return float("nan")
    values: list[float] = []
    for idx in range(stack.shape[0] - 1):
        cur = normalize_density(stack[idx], area)
        nxt = normalize_density(stack[idx + 1], area)
        values.append(0.5 * float((np.abs(nxt - cur) * area).sum()))
    return float(np.nanmean(values))


def process_scores(prediction: dict[str, np.ndarray], target: dict[str, np.ndarray], image_id: str) -> dict[str, float]:
    area = np.asarray(target["area_weights"], dtype=np.float32)
    out: dict[str, float] = {}
    for key in PROCESS_KEYS:
        if key not in prediction:
            raise KeyError(f"{image_id} prediction is missing '{key}'")
        if key not in target:
            raise KeyError(f"{image_id} target is missing '{key}'")

        pred_stack = _as_stack(prediction[key], key, image_id)
        gt_stack = _as_stack(target[key], key, image_id)
        if pred_stack.shape != gt_stack.shape:
            raise ValueError(f"{image_id}/{key} shape mismatch: pred={pred_stack.shape}, gt={gt_stack.shape}")

        out.update(_bin_metric_means(key, pred_stack, gt_stack, area))
        pred_tv = _step_tv(pred_stack, area)
        gt_tv = _step_tv(gt_stack, area)
        out[f"{key}_step_tv_pred"] = pred_tv
        out[f"{key}_step_tv_gt"] = gt_tv
        out[f"{key}_step_tv_abs_gap"] = abs(pred_tv - gt_tv)
        out[f"{key}_step_tv_ratio"] = pred_tv / gt_tv if gt_tv > 1e-7 else float("nan")
    return out


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


def evaluate_process_prediction_dir(
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
    for row in rows:
        prediction = _load_prediction_npz(pred_dir, row["image_id"])
        target = _load_target(row, condition_prefix_bins=condition_prefix_bins)
        scores = process_scores(prediction, target, row["image_id"])
        score_rows.append(scores)
        per_image.append({"image_id": row["image_id"], **scores})

    summary = {"split": split, "num_images": len(per_image), "scores": mean_scores(score_rows)}
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
