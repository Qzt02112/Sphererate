from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image

from sphererate.data.dataset import ProcessedNextFixationDataset, coord_channels
from sphererate.data.maps import MapGeometry, erp_gaussian_map, fixation_binary_map
from sphererate.evaluation.metrics import mean_scores, saliency_scores
from sphererate.evaluation.nextfix import next_fixation_scores
from sphererate.utils import normalize_np


def peak_to_norm(pred: np.ndarray) -> tuple[float, float]:
    pred = np.asarray(pred, dtype=np.float32).squeeze()
    if pred.ndim != 2:
        raise ValueError(f"Expected 2D prediction map, got {pred.shape}")
    height, width = pred.shape
    y, x = np.unravel_index(int(np.nanargmax(pred)), pred.shape)
    return (float(x) + 0.5) / float(width), (float(y) + 0.5) / float(height)


def fast_single_point_map(
    point: tuple[float, float],
    height: int,
    width: int,
    area_weights: np.ndarray,
    sigma_px: float,
    area_weighted: bool,
) -> np.ndarray:
    x_norm, y_norm = point
    cx = int(np.floor(np.mod(x_norm, 1.0) * width))
    cy = int(np.floor(np.clip(y_norm, 0.0, np.nextafter(1.0, 0.0)) * height))
    xs = np.arange(width, dtype=np.float32)
    ys = np.arange(height, dtype=np.float32)
    dx = np.abs(xs - float(cx))
    dx = np.minimum(dx, float(width) - dx)
    dy = np.abs(ys - float(cy))
    sigma = max(float(sigma_px), 1e-3)
    gx = np.exp(-(dx * dx) / (2.0 * sigma * sigma))
    gy = np.exp(-(dy * dy) / (2.0 * sigma * sigma))
    out = (gy[:, None] * gx[None, :]).astype(np.float32)
    return normalize_np(out, weights=area_weights if area_weighted else None)


def points_to_map(
    points: Sequence[tuple[float, float]],
    height: int,
    width: int,
    area_weights: np.ndarray,
    sigma_px: float,
    area_weighted: bool,
) -> np.ndarray:
    if not points:
        return np.zeros((height, width), dtype=np.float32)
    if len(points) == 1:
        return fast_single_point_map(points[0], height, width, area_weights, sigma_px, area_weighted)
    pts = np.asarray(points, dtype=np.float32)
    weights = np.ones((len(pts),), dtype=np.float32)
    return erp_gaussian_map(
        pts[:, 0],
        pts[:, 1],
        height,
        width,
        sigma_px=sigma_px,
        weights=weights,
        area_weights=area_weights,
        area_weighted=area_weighted,
    )


def prefix_maps_from_history(
    history: Sequence[tuple[float, float]],
    prefix_length: int,
    height: int,
    width: int,
    area_weights: np.ndarray,
    sigma_px: float,
    area_weighted: bool,
) -> np.ndarray:
    empty_count = max(0, int(prefix_length) - len(history))
    recent = list(history)[-int(prefix_length) :]
    maps = [np.zeros((height, width), dtype=np.float32) for _ in range(empty_count)]
    maps.extend(points_to_map([point], height, width, area_weights, sigma_px, area_weighted) for point in recent)
    return np.stack(maps, axis=0).astype(np.float32)


def target_dataset_from_config(config: dict[str, Any], split: str, processed_root: Path) -> ProcessedNextFixationDataset:
    train_cfg = config["train"]
    sample_id_files = train_cfg.get("sample_id_files") or train_cfg.get("scanpath_sample_id_files") or {}
    sample_id_file = train_cfg.get(f"{split}_sample_ids") or sample_id_files.get(split)
    image_id_files = train_cfg.get("image_id_files") or {}
    image_id_file = train_cfg.get(f"{split}_image_ids") or image_id_files.get(split)
    max_key = "max_train_samples" if split == "train" else "max_val_samples"
    return ProcessedNextFixationDataset(
        processed_root,
        split=split,
        prefix_length=int(train_cfg.get("scanpath_prefix_length", 3)),
        prefix_mode="normal",
        prefix_encoding="sequence",
        augment_roll=False,
        max_samples=train_cfg.get(max_key),
        image_id_file=image_id_file,
        sample_id_file=sample_id_file,
        sigma_px=float(config.get("data", {}).get("sigma_px", 10.0)),
        area_weighted=bool(config.get("data", {}).get("area_weighted", True)),
        shuffle_seed=int(train_cfg.get("data_seed", config.get("project", {}).get("seed", 20260708))),
    )


def load_fixations(path: str | Path) -> np.ndarray:
    with np.load(path) as data:
        return data["fixations"].astype(np.float32)


@torch.no_grad()
def evaluate_first_gt_rollout(
    model: torch.nn.Module,
    config: dict[str, Any],
    split: str,
    processed_root: Path,
    device: torch.device,
    batch_size: int = 32,
    max_targets: int | None = None,
) -> dict[str, Any]:
    dataset = target_dataset_from_config(config, split, processed_root)
    target_rows = list(dataset.rows)
    if max_targets is not None and max_targets > 0:
        target_rows = target_rows[: int(max_targets)]
    prefix_length = int(config.get("train", {}).get("scanpath_prefix_length", 3))
    sigma_px = float(config.get("data", {}).get("sigma_px", 10.0))
    area_weighted = bool(config.get("data", {}).get("area_weighted", True))
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in target_rows:
        grouped[(str(row["image_id"]), int(row["observer_id"]))].append(row)

    states: list[dict[str, Any]] = []
    for (_, observer_id), group in grouped.items():
        group.sort(key=lambda row: int(row["target_index"]))
        first_row = group[0]
        with Image.open(first_row["image"]) as image:
            img = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        img_t = torch.from_numpy(img).permute(2, 0, 1).float()
        height, width = img_t.shape[-2:]
        area = MapGeometry(height, width).area_weights
        fix = load_fixations(first_row["fixations"])
        obs = fix[fix[:, 3].astype(np.int64) == observer_id]
        obs = np.asarray(sorted(obs, key=lambda item: (float(item[2]), float(item[5]))), dtype=np.float32)
        if len(obs) < 2:
            continue

        target_indices = {int(row["target_index"]) for row in group}
        max_target = min(max(target_indices), len(obs) - 1)
        base = torch.cat([img_t, coord_channels(height, width)], dim=0)
        states.append(
            {
                "obs": obs,
                "area": area,
                "height": height,
                "width": width,
                "target_indices": target_indices,
                "max_target": max_target,
                "history": [(float(obs[0, 0]), float(obs[0, 1]))],
                "dataset": str(first_row.get("dataset") or "unknown"),
                "base": base,
            }
        )

    rows: list[dict[str, float]] = []
    dataset_rows: dict[str, list[dict[str, float]]] = defaultdict(list)
    model.eval()
    max_global_target = max((int(state["max_target"]) for state in states), default=0)
    batch_size = max(1, int(batch_size))
    for target_index in range(1, max_global_target + 1):
        active = [state for state in states if target_index <= int(state["max_target"])]
        for start in range(0, len(active), batch_size):
            chunk = active[start : start + batch_size]
            inputs = []
            for state in chunk:
                height = int(state["height"])
                width = int(state["width"])
                prefix = torch.from_numpy(
                    prefix_maps_from_history(
                        state["history"],
                        prefix_length=prefix_length,
                        height=height,
                        width=width,
                        area_weights=state["area"],
                        sigma_px=sigma_px,
                        area_weighted=area_weighted,
                    )
                )
                step_value = float(target_index) / float(max(len(state["obs"]) - 1, 1))
                step = torch.full((1, height, width), step_value, dtype=torch.float32)
                inputs.append(torch.cat([state["base"], prefix, step], dim=0))
            batch = torch.stack(inputs, dim=0).to(device)
            preds = model(batch)["final"][:, 0].detach().cpu().numpy().astype(np.float32)
            for state, pred in zip(chunk, preds, strict=True):
                pred_point = peak_to_norm(pred)
                if target_index in state["target_indices"]:
                    obs = state["obs"]
                    height = int(state["height"])
                    width = int(state["width"])
                    area = state["area"]
                    target = obs[target_index]
                    gt = erp_gaussian_map(
                        target[None, 0],
                        target[None, 1],
                        height,
                        width,
                        sigma_px=sigma_px,
                        weights=np.ones((1,), dtype=np.float32),
                        area_weights=area,
                        area_weighted=area_weighted,
                    )
                    fixation = fixation_binary_map(target[None, 0], target[None, 1], height, width)
                    scores = saliency_scores(pred, gt, fixation, area).as_dict()
                    scores.update(next_fixation_scores(pred, fixation))
                    scores["rollout_target_index"] = float(target_index)
                    rows.append(scores)
                    dataset_rows[str(state["dataset"])].append(scores)
                state["history"].append(pred_point)

    mixed_scores = mean_scores(rows)
    datasets = {
        name: {"num_images": len(items), "scores": mean_scores(items)}
        for name, items in sorted(dataset_rows.items())
    }
    macro_scores: dict[str, float] = {}
    if datasets:
        metric_keys = list(next(iter(datasets.values()))["scores"].keys())
        for key in metric_keys:
            vals = np.asarray([data["scores"].get(key, np.nan) for data in datasets.values()], dtype=np.float32)
            macro_scores[key] = float(np.nanmean(vals))
    return {
        "split": split,
        "protocol": "first_gt_autoregressive_rollout",
        "max_targets": max_targets,
        "num_images": len(rows),
        "scores": mixed_scores,
        "datasets": datasets,
        "macro_scores": macro_scores,
    }
