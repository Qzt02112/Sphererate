#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Callable

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sphererate.config import load_config
from sphererate.data.dataset import ProcessedSaliencyDataset
from sphererate.evaluation.content_interaction import (
    density_mix,
    deterministic_wrong_image_indices,
    grouped_bootstrap_improvement,
    log_density_mix,
)
from sphererate.evaluation.metrics import kld, mean_scores, saliency_scores
from sphererate.train import load_model_for_inference
from sphererate.utils import ensure_dir, read_jsonl


ALL_METRICS = ("kld_sph", "sim_sph", "cc_sph", "nss", "auc_judd")


def _load_final(path: Path) -> np.ndarray:
    with np.load(path) as data:
        return np.asarray(data["final"], dtype=np.float32).squeeze()


def _load_target(row: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(row["target"]) as data:
        return (
            np.asarray(data["final"], dtype=np.float32).squeeze(),
            np.asarray(data["fixation"], dtype=np.float32).squeeze(),
            np.asarray(data["area_weights"], dtype=np.float32).squeeze(),
        )


def _conditioned_dataset(
    processed_root: Path,
    split: str,
    include_start: bool,
) -> ProcessedSaliencyDataset:
    return ProcessedSaliencyDataset(
        processed_root,
        split=split,
        augment_roll=False,
        condition_startpoint_channels=include_start,
        startpoint_condition_mode="actual",
        startpoint_prior_sigma_px=10.0,
    )


def _prediction_set_complete(out_dir: Path, rows: list[dict[str, Any]]) -> bool:
    return all((out_dir / f"{row['image_id']}.npz").exists() for row in rows)


@torch.no_grad()
def _infer_standard(
    model: torch.nn.Module,
    dataset: ProcessedSaliencyDataset,
    device: torch.device,
    out_dir: Path,
    batch_size: int,
    num_workers: int,
    reuse: bool,
) -> None:
    ensure_dir(out_dir)
    if reuse and _prediction_set_complete(out_dir, dataset.rows):
        return
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=True,
    )
    model.eval()
    for batch in loader:
        outputs = model(batch["image"].to(device, non_blocking=True))["final"].detach().cpu().numpy()
        for index, image_id in enumerate(batch["image_id"]):
            np.savez_compressed(out_dir / f"{image_id}.npz", final=outputs[index, 0].astype(np.float32))


@torch.no_grad()
def _infer_wrong_image(
    model: torch.nn.Module,
    dataset: ProcessedSaliencyDataset,
    donor_indices: list[int],
    device: torch.device,
    out_dir: Path,
    batch_size: int,
    reuse: bool,
) -> None:
    ensure_dir(out_dir)
    if reuse and _prediction_set_complete(out_dir, dataset.rows):
        return
    model.eval()
    for begin in range(0, len(dataset), int(batch_size)):
        target_indices = list(range(begin, min(begin + int(batch_size), len(dataset))))
        inputs: list[torch.Tensor] = []
        image_ids: list[str] = []
        for target_index in target_indices:
            target_item = dataset[target_index]
            donor_item = dataset[donor_indices[target_index]]
            target_input = target_item["image"].clone()
            target_input[:8] = donor_item["image"][:8]
            inputs.append(target_input)
            image_ids.append(str(target_item["image_id"]))
        batch_input = torch.stack(inputs, dim=0).to(device, non_blocking=True)
        outputs = model(batch_input)["final"].detach().cpu().numpy()
        for index, image_id in enumerate(image_ids):
            np.savez_compressed(out_dir / f"{image_id}.npz", final=outputs[index, 0].astype(np.float32))


def _fusion_function(mode: str) -> Callable[[np.ndarray, np.ndarray, float, np.ndarray], np.ndarray]:
    if mode == "density":
        return density_mix
    if mode == "log_density":
        return log_density_mix
    raise ValueError(f"Unsupported fusion mode: {mode}")


def _select_alpha(
    rows: list[dict[str, Any]],
    start_dir: Path,
    static_dir: Path,
    alpha_grid: list[float],
    mode: str,
) -> tuple[float, list[dict[str, float]]]:
    fusion = _fusion_function(mode)
    totals = np.zeros(len(alpha_grid), dtype=np.float64)
    for row in rows:
        start = _load_final(start_dir / f"{row['image_id']}.npz")
        static = _load_final(static_dir / f"{row['image_id']}.npz")
        gt, _, area = _load_target(row)
        for index, alpha in enumerate(alpha_grid):
            totals[index] += kld(fusion(start, static, float(alpha), area), gt, area)
    means = totals / max(len(rows), 1)
    curve = [
        {"alpha": float(alpha), "mean_kld_sph": float(means[index])}
        for index, alpha in enumerate(alpha_grid)
    ]
    best_index = min(range(len(alpha_grid)), key=lambda index: (means[index], -float(alpha_grid[index])))
    return float(alpha_grid[best_index]), curve


def _evaluate_prediction_dir(
    rows: list[dict[str, Any]],
    prediction_dir: Path,
) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    by_image: dict[str, dict[str, float]] = {}
    for row in rows:
        pred = _load_final(prediction_dir / f"{row['image_id']}.npz")
        gt, fixation, area = _load_target(row)
        by_image[str(row["image_id"])] = saliency_scores(pred, gt, fixation, area).as_dict()
    return by_image, mean_scores(list(by_image.values()))


def _evaluate_fusion(
    rows: list[dict[str, Any]],
    start_dir: Path,
    static_dir: Path,
    alpha: float,
    mode: str,
) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    fusion = _fusion_function(mode)
    by_image: dict[str, dict[str, float]] = {}
    for row in rows:
        start = _load_final(start_dir / f"{row['image_id']}.npz")
        static = _load_final(static_dir / f"{row['image_id']}.npz")
        gt, fixation, area = _load_target(row)
        pred = fusion(start, static, float(alpha), area)
        by_image[str(row["image_id"])] = saliency_scores(pred, gt, fixation, area).as_dict()
    return by_image, mean_scores(list(by_image.values()))


def _paired_comparison(
    candidate: dict[str, dict[str, float]],
    control: dict[str, dict[str, float]],
    groups: dict[str, str],
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    return {
        metric: grouped_bootstrap_improvement(
            candidate,
            control,
            groups,
            metric,
            iterations=iterations,
            seed=seed,
        )
        for metric in ALL_METRICS
    }


def _write_mapping(path: Path, rows: list[dict[str, Any]], donors: list[int]) -> str:
    records = [
        {
            "target_image_id": str(row["image_id"]),
            "target_group_id": str(row["group_id"]),
            "donor_image_id": str(rows[donors[index]]["image_id"]),
            "donor_group_id": str(rows[donors[index]]["group_id"]),
        }
        for index, row in enumerate(rows)
    ]
    canonical = json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(canonical).hexdigest()
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    return digest


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate R0 no-training content complementarity evaluation")
    parser.add_argument("--config", default="configs/gate_r0_content_interaction.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    if config.get("training") != "forbidden":
        raise ValueError("Gate R0 config must explicitly forbid training")
    if config["fusion"]["alpha_selection_split"] != "train" or config["fusion"]["evaluation_split"] != "val":
        raise ValueError("Gate R0 is locked to train-only alpha selection and val-only evaluation")

    paths = config["paths"]
    runtime = config["runtime"]
    statistics = config["statistics"]
    output_root = ensure_dir(paths["output_root"])
    prediction_root = ensure_dir(output_root / "predictions")
    processed_root = Path(paths["conditioned_root"])
    all_rows = read_jsonl(processed_root / "manifest.jsonl")
    rows_by_split = {
        split: sorted([row for row in all_rows if row["split"] == split], key=lambda row: row["image_id"])
        for split in ("train", "val")
    }

    start_model, _, device = load_model_for_inference(paths["start_only_checkpoint"], runtime["device"])
    static_model, _, _ = load_model_for_inference(paths["static_checkpoint"], runtime["device"])
    residual_model, _, _ = load_model_for_inference(paths["residual_checkpoint"], runtime["device"])
    batch_size = int(runtime["batch_size"])
    num_workers = int(runtime["num_workers"])
    reuse = bool(runtime.get("reuse_predictions", True))

    for split in ("train", "val"):
        start_dataset = _conditioned_dataset(processed_root, split, include_start=True)
        static_dataset = _conditioned_dataset(processed_root, split, include_start=False)
        _infer_standard(
            start_model,
            start_dataset,
            device,
            prediction_root / "start_only" / split,
            batch_size,
            num_workers,
            reuse,
        )
        _infer_standard(
            static_model,
            static_dataset,
            device,
            prediction_root / "static" / split,
            batch_size,
            num_workers,
            reuse,
        )

    val_dataset = _conditioned_dataset(processed_root, "val", include_start=True)
    val_rows = val_dataset.rows
    wrong_seed = int(config["wrong_image"]["seed"])
    donor_indices = deterministic_wrong_image_indices(val_rows, seed=wrong_seed)
    mapping_hash = _write_mapping(output_root / "wrong_image_mapping.csv", val_rows, donor_indices)
    _infer_standard(
        residual_model,
        val_dataset,
        device,
        prediction_root / "residual_actual" / "val",
        batch_size,
        num_workers,
        reuse,
    )
    _infer_wrong_image(
        residual_model,
        val_dataset,
        donor_indices,
        device,
        prediction_root / "residual_wrong_image" / "val",
        batch_size,
        reuse,
    )

    alpha_grid = [float(value) for value in config["fusion"]["alpha_grid"]]
    selected: dict[str, Any] = {}
    methods: dict[str, dict[str, dict[str, float]]] = {}
    summaries: dict[str, dict[str, float]] = {}
    train_rows = rows_by_split["train"]
    val_rows = rows_by_split["val"]
    start_train_dir = prediction_root / "start_only" / "train"
    static_train_dir = prediction_root / "static" / "train"
    start_val_dir = prediction_root / "start_only" / "val"
    static_val_dir = prediction_root / "static" / "val"

    for mode in config["fusion"]["modes"]:
        alpha, curve = _select_alpha(train_rows, start_train_dir, static_train_dir, alpha_grid, str(mode))
        method_name = f"{mode}_fusion"
        selected[str(mode)] = {"alpha": alpha, "train_selection_curve": curve}
        methods[method_name], summaries[method_name] = _evaluate_fusion(
            val_rows,
            start_val_dir,
            static_val_dir,
            alpha,
            str(mode),
        )

    for method_name, directory in {
        "start_only": start_val_dir,
        "static": static_val_dir,
        "residual_actual": prediction_root / "residual_actual" / "val",
        "residual_wrong_image": prediction_root / "residual_wrong_image" / "val",
    }.items():
        methods[method_name], summaries[method_name] = _evaluate_prediction_dir(val_rows, directory)

    groups = {str(row["image_id"]): str(row["base_image_id"]) for row in val_rows}
    iterations = int(statistics["bootstrap_iterations"])
    bootstrap_seed = int(statistics["bootstrap_seed"])
    comparisons: dict[str, Any] = {}
    for method_name in ("density_fusion", "log_density_fusion", "residual_actual", "static"):
        comparisons[f"{method_name}_vs_start_only"] = _paired_comparison(
            methods[method_name], methods["start_only"], groups, iterations, bootstrap_seed
        )
    comparisons["residual_actual_vs_wrong_image"] = _paired_comparison(
        methods["residual_actual"], methods["residual_wrong_image"], groups, iterations, bootstrap_seed
    )

    fusion_pass: dict[str, bool] = {}
    for method_name in ("density_fusion", "log_density_fusion"):
        comparison = comparisons[f"{method_name}_vs_start_only"]
        fusion_pass[method_name] = bool(
            comparison["kld_sph"]["ci95_low"] > 0.0
            and comparison["sim_sph"]["ci95_low"] > 0.0
            and comparison["cc_sph"]["ci95_high"] >= 0.0
        )
    wrong_comparison = comparisons["residual_actual_vs_wrong_image"]
    residual_wrong_image_pass = bool(
        wrong_comparison["kld_sph"]["ci95_low"] > 0.0
        and wrong_comparison["sim_sph"]["ci95_low"] > 0.0
    )
    gate_pass = bool(any(fusion_pass.values()) and residual_wrong_image_pass)

    result = {
        "protocol": config["protocol"],
        "gate": "R0",
        "decision": "PASS" if gate_pass else "FAIL",
        "training_performed": False,
        "selection_split": "train",
        "evaluation_split": "val",
        "row_counts": {split: len(rows) for split, rows in rows_by_split.items()},
        "selected_fusion": selected,
        "val_summaries": summaries,
        "paired_improvement": comparisons,
        "decision_components": {
            "fusion_pass": fusion_pass,
            "residual_actual_vs_wrong_image_pass": residual_wrong_image_pass,
            "gate_pass": gate_pass,
        },
        "wrong_image": {
            "seed": wrong_seed,
            "mapping_sha256": mapping_hash,
            "mapping_path": str(output_root / "wrong_image_mapping.csv"),
            "same_group_assignments": 0,
            "one_to_one": True,
        },
        "checkpoints": {
            "start_only": paths["start_only_checkpoint"],
            "static": paths["static_checkpoint"],
            "residual": paths["residual_checkpoint"],
        },
        "test_status": "frozen_not_evaluated",
    }
    (output_root / "results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
