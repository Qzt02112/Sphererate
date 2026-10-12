#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sphererate.data.maps import erp_gaussian_map
from sphererate.evaluation.evaluate import evaluate_prediction_dir
from sphererate.evaluation.metrics import cc, kld, sim
from sphererate.train import _loader, _move_batch, load_model_for_inference
from sphererate.utils import ensure_dir, read_jsonl


def _read_metric_rows(path: Path) -> dict[str, dict[str, float]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return {
            row["image_id"]: {key: float(value) for key, value in row.items() if key not in {"dataset", "image_id", "condition"}}
            for row in csv.DictReader(f)
        }


def _bootstrap_grouped_difference(
    actual: dict[str, dict[str, float]],
    control: dict[str, dict[str, float]],
    base_by_image: dict[str, str],
    metric: str,
    iterations: int,
    seed: int,
) -> dict[str, float | None]:
    oriented_by_base: dict[str, list[float]] = defaultdict(list)
    for image_id in sorted(actual):
        if metric.startswith("kld"):
            difference = control[image_id][metric] - actual[image_id][metric]
        else:
            difference = actual[image_id][metric] - control[image_id][metric]
        oriented_by_base[base_by_image[image_id]].append(float(difference))
    values = np.asarray([np.mean(items) for items in oriented_by_base.values()], dtype=np.float64)
    rng = np.random.default_rng(seed)
    samples = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        draw = rng.integers(0, len(values), size=len(values))
        samples[index] = values[draw].mean()
    return {
        "oriented_improvement_mean": float(values.mean()),
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
        "num_base_images": int(len(values)),
    }


def _make_start_prior_predictions(rows: list[dict[str, Any]], out_dir: Path, num_bins: int, sigma_px: float) -> None:
    ensure_dir(out_dir)
    for row in rows:
        with np.load(row["target"]) as target:
            area = target["area_weights"].astype(np.float32)
        height, width = area.shape
        x = ((float(row["start_lon_deg"]) + 180.0) / 360.0) % 1.0
        y = np.clip((90.0 - float(row.get("start_lat_deg", 0.0))) / 180.0, 0.0, np.nextafter(1.0, 0.0))
        pred = erp_gaussian_map(
            np.asarray([x], dtype=np.float32),
            np.asarray([y], dtype=np.float32),
            height,
            width,
            sigma_px=sigma_px,
            area_weights=area,
            area_weighted=True,
        )
        np.savez_compressed(out_dir / f"{row['image_id']}.npz", final=pred)


@torch.no_grad()
def _save_final_predictions(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    out_dir: Path,
) -> None:
    model.eval()
    ensure_dir(out_dir)
    for batch in loader:
        batch = _move_batch(batch, device)
        final = model(batch["image"])["final"].detach().cpu().numpy()
        for index, image_id in enumerate(batch["image_id"]):
            np.savez_compressed(out_dir / f"{image_id}.npz", final=final[index, 0].astype(np.float32))


@torch.no_grad()
def _residual_diagnostics(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict[str, float]:
    mean_abs: list[float] = []
    max_abs: list[float] = []
    gate_values: list[float] = []
    model.eval()
    for batch in loader:
        batch = _move_batch(batch, device)
        outputs = model(batch["image"])
        if "residual_logits" not in outputs:
            continue
        residual = outputs["residual_logits"].detach().float()
        mean_abs.extend(residual.abs().mean(dim=(1, 2, 3)).cpu().tolist())
        max_abs.extend(residual.abs().amax(dim=(1, 2, 3)).cpu().tolist())
        gate = outputs.get("residual_gate_value")
        if gate is not None:
            gate_values.append(float(gate.detach().cpu()))
    if not mean_abs:
        return {
            "mean_residual_abs": None,
            "mean_per_image_residual_max_abs": None,
            "max_residual_abs": None,
            "residual_gate_value": None,
        }
    return {
        "mean_residual_abs": float(np.mean(mean_abs)),
        "mean_per_image_residual_max_abs": float(np.mean(max_abs)),
        "max_residual_abs": float(np.max(max_abs)),
        "residual_gate_value": float(np.mean(gate_values)) if gate_values else float("nan"),
    }


def _condition_diagnostics(
    rows: list[dict[str, Any]],
    prediction_dirs: dict[str, Path],
) -> dict[str, Any]:
    rows_by_base: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    sensitivity: list[dict[str, float]] = []
    for row in rows:
        image_id = row["image_id"]
        with np.load(row["target"]) as target:
            area = target["area_weights"].astype(np.float32)
        with np.load(prediction_dirs["actual"] / f"{image_id}.npz") as data:
            actual = data["final"].astype(np.float32)
        with np.load(prediction_dirs["swapped"] / f"{image_id}.npz") as data:
            swapped = data["final"].astype(np.float32)
        with np.load(prediction_dirs["random"] / f"{image_id}.npz") as data:
            random = data["final"].astype(np.float32)
        sensitivity.append(
            {
                "actual_vs_swapped_one_minus_sim_sph": 1.0 - sim(actual, swapped, area),
                "actual_vs_random_one_minus_sim_sph": 1.0 - sim(actual, random, area),
                "actual_vs_swapped_kld_sph": kld(actual, swapped, area),
            }
        )
        rows_by_base[str(row["base_image_id"])][str(row["condition"])] = row

    pair_rows: list[dict[str, float]] = []
    for pair in rows_by_base.values():
        good_row, bad_row = pair["good"], pair["bad"]
        with np.load(good_row["target"]) as good_target, np.load(bad_row["target"]) as bad_target:
            good_gt = good_target["final"].astype(np.float64)
            bad_gt = bad_target["final"].astype(np.float64)
            area = good_target["area_weights"].astype(np.float64)
        with np.load(prediction_dirs["actual"] / f"{good_row['image_id']}.npz") as data:
            good_pred = data["final"].astype(np.float64)
        with np.load(prediction_dirs["actual"] / f"{bad_row['image_id']}.npz") as data:
            bad_pred = data["final"].astype(np.float64)
        target_delta = good_gt - bad_gt
        pred_delta = good_pred - bad_pred
        numerator = float((area * target_delta * pred_delta).sum())
        denominator = float(np.sqrt((area * np.square(target_delta)).sum() * (area * np.square(pred_delta)).sum()))
        pair_rows.append(
            {
                "target_pair_sim_sph": sim(good_gt, bad_gt, area),
                "prediction_pair_sim_sph": sim(good_pred, bad_pred, area),
                "delta_weighted_cosine": numerator / denominator if denominator > 0 else 0.0,
            }
        )

    return {
        "num_rows": len(sensitivity),
        "num_pairs": len(pair_rows),
        "mean_actual_vs_swapped_one_minus_sim_sph": float(
            np.mean([row["actual_vs_swapped_one_minus_sim_sph"] for row in sensitivity])
        ),
        "mean_actual_vs_random_one_minus_sim_sph": float(
            np.mean([row["actual_vs_random_one_minus_sim_sph"] for row in sensitivity])
        ),
        "mean_actual_vs_swapped_kld_sph": float(
            np.mean([row["actual_vs_swapped_kld_sph"] for row in sensitivity])
        ),
        "mean_target_pair_sim_sph": float(np.mean([row["target_pair_sim_sph"] for row in pair_rows])),
        "mean_prediction_pair_sim_sph": float(
            np.mean([row["prediction_pair_sim_sph"] for row in pair_rows])
        ),
        "mean_delta_weighted_cosine": float(np.mean([row["delta_weighted_cosine"] for row in pair_rows])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conditioned-checkpoint", required=True)
    parser.add_argument("--static-checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260710)
    args = parser.parse_args()

    output_root = ensure_dir(args.output_root)
    conditioned_model, conditioned_cfg, device = load_model_for_inference(args.conditioned_checkpoint, args.device)
    processed_root = Path(conditioned_cfg["paths"]["processed_root"]) / conditioned_cfg["data"]["processed_name"]
    val_rows = [row for row in read_jsonl(processed_root / "manifest.jsonl") if row["split"] == "val"]
    base_by_image = {row["image_id"]: row["base_image_id"] for row in val_rows}

    summaries: dict[str, Any] = {}
    prediction_dirs: dict[str, Path] = {}
    for mode in ("actual", "swapped", "random"):
        cfg = copy.deepcopy(conditioned_cfg)
        cfg["train"]["startpoint_condition_mode"] = mode
        cfg["train"]["max_val_samples"] = None
        loader = _loader(cfg, "val", processed_root, augment=False)
        pred_dir = Path(args.conditioned_checkpoint).parents[1] / "predictions" / f"val_{mode}"
        _save_final_predictions(conditioned_model, loader, device, pred_dir)
        prediction_dirs[mode] = pred_dir
        summaries[mode] = evaluate_prediction_dir(
            processed_root,
            pred_dir,
            split="val",
            output_dir=output_root / mode,
        )

    static_model, _, _ = load_model_for_inference(args.static_checkpoint, args.device)
    static_cfg = copy.deepcopy(conditioned_cfg)
    static_cfg["train"]["condition_startpoint_channels"] = False
    static_cfg["train"]["max_val_samples"] = None
    static_loader = _loader(static_cfg, "val", processed_root, augment=False)
    static_pred_dir = Path(args.conditioned_checkpoint).parents[1] / "predictions" / "val_static"
    _save_final_predictions(static_model, static_loader, device, static_pred_dir)
    prediction_dirs["static"] = static_pred_dir
    summaries["static"] = evaluate_prediction_dir(
        processed_root,
        static_pred_dir,
        split="val",
        output_dir=output_root / "static",
    )

    start_prior_dir = Path(args.conditioned_checkpoint).parents[1] / "predictions" / "val_start_prior"
    _make_start_prior_predictions(
        val_rows,
        start_prior_dir,
        num_bins=int(conditioned_cfg["data"]["num_bins"]),
        sigma_px=float(conditioned_cfg["train"]["startpoint_prior_sigma_px"]),
    )
    prediction_dirs["start_prior"] = start_prior_dir
    summaries["start_prior"] = evaluate_prediction_dir(
        processed_root,
        start_prior_dir,
        split="val",
        output_dir=output_root / "start_prior",
    )

    metric_rows = {
        mode: _read_metric_rows(output_root / mode / "val_per_image.csv")
        for mode in summaries
    }
    paired: dict[str, Any] = {}
    for control in ("static", "swapped", "random", "start_prior"):
        paired[control] = {
            metric: _bootstrap_grouped_difference(
                metric_rows["actual"],
                metric_rows[control],
                base_by_image,
                metric,
                iterations=int(args.bootstrap_iterations),
                seed=int(args.bootstrap_seed),
            )
            for metric in ("kld_sph", "sim_sph", "cc_sph")
        }

    result = {
        "protocol": "P1_start_conditioned",
        "split": "val",
        "num_rows": len(val_rows),
        "num_base_images": len(set(base_by_image.values())),
        "conditioned_checkpoint": args.conditioned_checkpoint,
        "static_checkpoint": args.static_checkpoint,
        "model": conditioned_cfg["train"].get("model"),
        "efficiency": {
            "total_parameters": int(sum(parameter.numel() for parameter in conditioned_model.parameters())),
            "trainable_parameters": int(
                sum(parameter.numel() for parameter in conditioned_model.parameters() if parameter.requires_grad)
            ),
        },
        "summaries": summaries,
        "paired_actual_improvement": paired,
        "condition_diagnostics": {
            **_condition_diagnostics(val_rows, prediction_dirs),
            **_residual_diagnostics(
                conditioned_model,
                _loader(conditioned_cfg, "val", processed_root, augment=False),
                device,
            ),
        },
        "test_status": "frozen_not_evaluated",
    }
    (output_root / "condition_signal_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
