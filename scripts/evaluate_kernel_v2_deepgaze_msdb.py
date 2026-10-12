#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from scripts.evaluate_eh6_scandmm import _macro, _score
from sphererate.baselines.deepgaze_msdb import (
    apply_log_linear_calibration,
    build_train_priors,
    fit_log_linear_calibration,
    spherical_area_weights,
)
from sphererate.baselines.reachability import (
    angular_distance_grid_deg,
    calibrate_reachability_radii,
    gate_final_map,
)
from sphererate.data.early_dataset import EarlyHorizonDataset
from sphererate.evaluation.early_model import compare_coverage_rows, compare_method_rows
from sphererate.utils import ensure_dir


HORIZONS = (0.5, 1.0, 3.0, 5.0)
START_Y_NORM = 0.5
EXPECTED_COUNTS = {"train": 50, "val": 9, "gate": 13, "test": 13}
SOURCE_COMMIT = "90c49a554ce334add49bd54825679f61d0ac32ba"


def _samples(root: Path, split: str, expected_count: int) -> list[dict[str, Any]]:
    dataset = EarlyHorizonDataset(root, split=split, target_mode="correct")
    values = [dataset[index] for index in range(len(dataset))]
    if len(values) != int(expected_count):
        raise ValueError(f"Expected {expected_count} {split} samples, got {len(values)}")
    return values


def _load_cache(cache_root: Path, split: str, image_ids: list[str]) -> dict[str, np.ndarray]:
    map_root = cache_root / split / "maps"
    expected = set(image_ids)
    found = {path.stem for path in map_root.glob("*.npy")}
    if found != expected:
        raise ValueError(f"MSDB {split} cache mismatch: missing={sorted(expected-found)}, extra={sorted(found-expected)}")
    values = {image_id: np.asarray(np.load(map_root / f"{image_id}.npy"), dtype=np.float32) for image_id in image_ids}
    if any(value.shape != (256, 512) or not np.all(np.isfinite(value)) for value in values.values()):
        raise ValueError(f"Invalid MSDB cache in {map_root}")
    return values


def _repeat(values: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {image_id: np.repeat(value[None], len(HORIZONS), axis=0).astype(np.float32) for image_id, value in values.items()}


def _reachable(
    calibrated: dict[str, np.ndarray],
    samples: list[dict[str, Any]],
    radii: dict[float, float],
    *,
    start_x_norm: float,
) -> dict[str, np.ndarray]:
    distance = angular_distance_grid_deg(
        256, 512, start_x_norm=float(start_x_norm), start_y_norm=START_Y_NORM
    )
    predictions = {}
    for sample in samples:
        image_id = str(sample["image_id"])
        predictions[image_id] = np.stack(
            [gate_final_map(calibrated[image_id], distance, radii[horizon]) for horizon in HORIZONS]
        ).astype(np.float32)
    return predictions


def _metrics(predictions: dict[str, np.ndarray], samples: list[dict[str, Any]]) -> dict[str, Any]:
    budgets = (5.0, 10.0, 15.0, 20.0)
    scores, coverages = _score(predictions, samples, budgets)
    return {
        "early_macro": _macro(scores, ("kld_sph", "sim_sph", "cc_sph")),
        "coverage_early_macro": {
            str(budget): _macro(coverages[str(budget)], ("coverage",))["coverage"] for budget in budgets
        },
        "score_rows": scores,
        "coverage_rows": coverages,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the frozen DeepGaze MSDB ERP adapter and reachable variant.")
    parser.add_argument("--split", choices=("val", "gate", "test"), default="val")
    parser.add_argument("--data-root", default="data/processed/early_horizon_salient360_spherical2deg_v2")
    parser.add_argument("--cache-root", default="runs/kernel_v2_deepgaze_msdb/raw")
    parser.add_argument("--output-dir")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--calibration-root")
    parser.add_argument("--calibration-cache-root")
    parser.add_argument("--expected-train", type=int, default=50)
    parser.add_argument("--expected-eval", type=int)
    parser.add_argument("--start-x-norm", type=float, default=0.75)
    parser.add_argument("--prediction-root", default="runs/kernel_v2_deepgaze_msdb/evaluated")
    args = parser.parse_args()

    data_root = ROOT / args.data_root
    evaluation_cache_root = ROOT / args.cache_root
    calibration_root = ROOT / (args.calibration_root or args.data_root)
    calibration_cache_root = ROOT / (args.calibration_cache_root or args.cache_root)
    expected_eval = int(
        args.expected_eval if args.expected_eval is not None else EXPECTED_COUNTS[args.split]
    )
    train = _samples(calibration_root, "train", int(args.expected_train))
    evaluation = _samples(data_root, args.split, expected_eval)
    train_ids = [str(sample["image_id"]) for sample in train]
    evaluation_ids = [str(sample["image_id"]) for sample in evaluation]
    train_raw = _load_cache(calibration_cache_root, "train", train_ids)
    evaluation_raw = _load_cache(evaluation_cache_root, args.split, evaluation_ids)

    area = spherical_area_weights(256, 512)
    final_targets = np.stack([np.asarray(sample["final"][0], dtype=np.float32) for sample in train])
    train_prior, loo_priors = build_train_priors(final_targets, area)
    calibration = fit_log_linear_calibration(
        np.stack([train_raw[image_id] for image_id in train_ids]),
        loo_priors,
        final_targets,
        area,
        device=args.device,
    )
    calibrated = {
        image_id: apply_log_linear_calibration(evaluation_raw[image_id], train_prior, area, calibration)
        for image_id in evaluation_ids
    }
    fixation_arrays = []
    for sample in train:
        with np.load(str(sample["fixations_path"])) as data:
            fixation_arrays.append(np.asarray(data["fixations"], dtype=np.float32))
    radii = calibrate_reachability_radii(
        fixation_arrays,
        HORIZONS,
        start_x_norm=float(args.start_x_norm),
        start_y_norm=START_Y_NORM,
        quantile=0.95,
    )
    raw_predictions = _repeat(evaluation_raw)
    calibrated_predictions = _repeat(calibrated)
    reachable_predictions = _reachable(
        calibrated, evaluation, radii, start_x_norm=float(args.start_x_norm)
    )
    raw_metrics = _metrics(raw_predictions, evaluation)
    calibrated_metrics = _metrics(calibrated_predictions, evaluation)
    reachable_metrics = _metrics(reachable_predictions, evaluation)
    template = {str(sample["image_id"]): sample["template"].numpy() for sample in evaluation}
    template_scores, template_coverage = _score(template, evaluation, (5.0, 10.0, 15.0, 20.0))

    train_hash = hashlib.sha256("\n".join(train_ids).encode("utf-8")).hexdigest()
    result = {
        "baseline": "DeepGaze MSDB (ICCV 2025) ERP adapter",
        "split": args.split,
        "tier": "recent_2d_fixation_saliency_with_explicit_erp_adapter",
        "training": "official pretrained model frozen; two scalar calibration parameters fitted on the source training split",
        "source": "https://github.com/matthias-k/deepgaze/tree/feature-deepgaze-msdb",
        "source_commit": SOURCE_COMMIT,
        "adapter": {
            "reference_ppd": 30.0,
            "physical_hmd_ppd_claimed": False,
            "yaw_roll_ensemble": [0.0, 0.25, 0.5, 0.75],
            "calibration": calibration.as_dict(),
            "calibration_split": "train",
            "train_content_count": len(train_ids),
            "train_content_hash": train_hash,
            "train_prior_for_fit": "leave-one-content-out",
            "train_prior_for_eval": f"all {len(train_ids)} source-train rows",
            "val_or_gate_used_for_calibration": False,
            "calibration_root": str(calibration_root),
            "evaluation_root": str(data_root),
        },
        "raw": raw_metrics,
        "train_calibrated": calibrated_metrics,
        "train_calibrated_reachable": reachable_metrics,
        "reachable": {
            "quantile": 0.95,
            "radii_deg": {str(key): value for key, value in radii.items()},
            "start_x_norm": float(args.start_x_norm),
            "source": "source-train-only reachability envelope",
        },
        "comparisons": {
            "calibrated_vs_raw": compare_method_rows(
                calibrated_metrics["score_rows"], raw_metrics["score_rows"], seed=20260712
            ),
            "reachable_vs_calibrated": compare_method_rows(
                reachable_metrics["score_rows"], calibrated_metrics["score_rows"], seed=20260712
            ),
            "reachable_vs_template": compare_method_rows(
                reachable_metrics["score_rows"], template_scores, seed=20260712
            ),
            "reachable_coverage_vs_template": {
                key: compare_coverage_rows(
                    reachable_metrics["coverage_rows"][key], template_coverage[key], seed=20260712
                )
                for key in reachable_metrics["coverage_rows"]
            },
        },
        "test_evaluated": bool(args.split == "test"),
        "test_used_for_selection": False,
    }
    output = ensure_dir(args.output_dir or f"reports/kernel_v2_deepgaze_msdb_{args.split}")
    (output / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    prediction_root = ensure_dir(ROOT / args.prediction_root / args.split)
    for image_id in evaluation_ids:
        np.savez_compressed(
            prediction_root / f"{image_id}.npz",
            raw=raw_predictions[image_id],
            calibrated=calibrated_predictions[image_id],
            reachable=reachable_predictions[image_id],
        )

    def row(label: str, metrics: dict[str, Any]) -> str:
        score = metrics["early_macro"]
        coverage = metrics["coverage_early_macro"]["10.0"]
        return f"| {label} | {score['kld_sph']:.6f} | {score['sim_sph']:.6f} | {score['cc_sph']:.6f} | {coverage:.6f} |"

    lines = [
        f"# DeepGaze MSDB (ICCV 2025) — {args.split}",
        "",
        f"The official pretrained CLIP+DINOv2 saliency model is frozen. A four-yaw ERP roll ensemble reduces seam dependence; two scalar parameters (content temperature and train-only static-prior weight) are fitted on {len(train_ids)} source-training rows. The reachable row uses the matching source-train-only 95% great-circle envelope.",
        "",
        "| Variant | KLD_sph ↓ | SIM_sph ↑ | CC_sph ↑ | 10% coverage ↑ |",
        "|---|---:|---:|---:|---:|",
        row("MSDB raw final-repeat", raw_metrics),
        row("MSDB train-calibrated final-repeat", calibrated_metrics),
        row("MSDB train-calibrated + reachable", reachable_metrics),
        "",
        "Caveat: MSDB is a planar-image model. `reference_ppd=30` is a frozen network-scale adapter, not a claim about physical HMD pixels per degree; this row must be labelled `ERP-adapted` in the paper.",
        "",
        f"Fitted scalars: temperature={calibration.content_temperature:.6f}, prior weight={calibration.prior_weight:.6f}; v2-test target loads=0.",
    ]
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(output / "summary.md"), "calibration": calibration.as_dict()}, indent=2))


if __name__ == "__main__":
    main()
