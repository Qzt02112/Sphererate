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

from scripts.evaluate_eh6_panoramic_static import _predict as _predict_panoramic
from scripts.evaluate_eh6_scandmm import _average_rows, _macro, _score
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
QUANTILE = 0.95


def _manifest_rows(data_root: Path, split: str) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in (data_root / "manifest.jsonl").read_text().splitlines()]
    selected = [row for row in rows if str(row["split"]) == split]
    return sorted(selected, key=lambda row: str(row["image_id"]))


def _calibrate(
    data_root: Path, *, start_x_norm: float, expected_train: int | None = None
) -> tuple[dict[float, float], dict[str, Any]]:
    rows = _manifest_rows(data_root, "train")
    if expected_train is not None and len(rows) != int(expected_train):
        raise ValueError(
            f"Reachability calibration requires {expected_train} train rows, got {len(rows)}"
        )
    fixation_arrays = []
    eye_labels = set()
    for row in rows:
        with np.load(str(row["fixations"])) as data:
            if "eye" in data.files:
                eye_labels.add(str(data["eye"].item()))
            fixation_arrays.append(np.asarray(data["fixations"], dtype=np.float32))
    if eye_labels and eye_labels != {"R"}:
        raise ValueError(f"Reachability calibration expected right-eye fixations, found {eye_labels}")
    radii = calibrate_reachability_radii(
        fixation_arrays,
        HORIZONS,
        start_x_norm=float(start_x_norm),
        start_y_norm=START_Y_NORM,
        quantile=QUANTILE,
    )
    ids = [str(row["canonical_content_id"]) for row in rows]
    audit = {
        "calibration_split": "train",
        "train_content_count": len(ids),
        "train_content_hash": hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest(),
        "primary_eye": "R" if eye_labels else "official_binocular_gaze",
        "start_x_norm": float(start_x_norm),
        "quantile": QUANTILE,
        "weighting": "clipped dwell duration",
        "radii_deg": {str(key): value for key, value in radii.items()},
        "val_or_gate_used_for_radius": False,
        "test_target_loads": 0,
    }
    return radii, audit


def _static_predictions(
    source: str,
    split: str,
    samples: list[dict[str, Any]],
    checkpoints: list[Path],
    device: torch.device,
    map_run_root: Path,
) -> tuple[list[dict[str, np.ndarray]], dict[str, Any]]:
    if source == "panoramic_cnn":
        if len(checkpoints) != 3:
            raise ValueError("Panoramic-CNN reachability baseline requires exactly three checkpoints")
        return (
            [_predict_panoramic(checkpoint, samples, device) for checkpoint in checkpoints],
            {"checkpoints": [str(path) for path in checkpoints], "training_policy": "same_split"},
        )
    if checkpoints:
        raise ValueError(f"{source} is deterministic and does not accept checkpoints")
    map_root = map_run_root / source / split / "maps"
    expected = {str(sample["image_id"]) for sample in samples}
    found = {path.stem for path in map_root.glob("*.npy")}
    if found != expected:
        raise ValueError(f"{source} cache mismatch: missing={sorted(expected-found)}, extra={sorted(found-expected)}")
    prediction = {}
    for sample in samples:
        image_id = str(sample["image_id"])
        final = np.asarray(np.load(map_root / f"{image_id}.npy"), dtype=np.float32)
        prediction[image_id] = np.repeat(final[None], len(HORIZONS), axis=0)
    return [prediction], {"map_root": str(map_root), "training_policy": "not_applicable"}


def _gate_predictions(
    predictions: dict[str, np.ndarray],
    samples: list[dict[str, Any]],
    radii: dict[float, float],
    start_x_norm: float,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    gated, fractions = {}, []
    for sample in samples:
        image_id = str(sample["image_id"])
        height, width = (int(value) for value in sample["target"].shape[-2:])
        distance = angular_distance_grid_deg(
            height,
            width,
            start_x_norm=float(start_x_norm),
            start_y_norm=START_Y_NORM,
        )
        area = sample["area_weights"].numpy()[0]
        maps = []
        for index, horizon in enumerate(sample["horizons_s"].numpy().tolist()):
            original = np.asarray(predictions[image_id][index], dtype=np.float32)
            current = gate_final_map(original, distance, radii[float(horizon)])
            maps.append(current)
            denominator = float((np.maximum(original, 0.0) * area).sum())
            fractions.append(
                {
                    "content_id": str(sample["canonical_content_id"]),
                    "horizon_s": float(horizon),
                    "retained_spherical_mass": float((current * area).sum() / max(denominator, 1e-12)),
                }
            )
        gated[image_id] = np.stack(maps).astype(np.float32)
    return gated, fractions


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate train-calibrated reachable-final baselines.")
    parser.add_argument("--source", choices=("panoramic_cnn", "bms360", "gbvs360"), required=True)
    parser.add_argument("--split", choices=("val", "gate", "test"), default="val")
    parser.add_argument("--checkpoint", action="append", default=[])
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--data-root", default="data/processed/early_horizon_salient360_spherical2deg_v2"
    )
    parser.add_argument("--output-dir")
    parser.add_argument("--calibration-root")
    parser.add_argument("--start-x-norm", type=float, default=0.75)
    parser.add_argument("--expected-train", type=int, default=50)
    parser.add_argument("--expected-eval", type=int)
    parser.add_argument("--map-run-root", default="runs/kernel_v2_bms_gbvs")
    parser.add_argument("--prediction-root", default="runs/kernel_v2_reachable_final")
    args = parser.parse_args()
    data_root = ROOT / args.data_root
    calibration_root = ROOT / (args.calibration_root or args.data_root)
    radii, calibration = _calibrate(
        calibration_root,
        start_x_norm=float(args.start_x_norm),
        expected_train=int(args.expected_train),
    )
    dataset = EarlyHorizonDataset(data_root, split=args.split, target_mode="correct")
    samples = [dataset[index] for index in range(len(dataset))]
    expected_eval = int(
        args.expected_eval if args.expected_eval is not None else {"val": 9, "gate": 13, "test": 13}[args.split]
    )
    if len(samples) != expected_eval:
        raise ValueError(f"Evaluation split count changed: {len(samples)} != {expected_eval}")
    seed_predictions, provenance = _static_predictions(
        args.source,
        args.split,
        samples,
        [ROOT / value for value in args.checkpoint],
        torch.device(args.device),
        ROOT / args.map_run_root,
    )
    budgets = (5.0, 10.0, 15.0, 20.0)
    original_score_seeds, gated_score_seeds = [], []
    original_coverage_seeds = {str(value): [] for value in budgets}
    gated_coverage_seeds = {str(value): [] for value in budgets}
    gated_prediction_seeds, retained = [], []
    for predictions in seed_predictions:
        gated, fractions = _gate_predictions(
            predictions, samples, radii, start_x_norm=float(args.start_x_norm)
        )
        original_scores, original_coverages = _score(predictions, samples, budgets)
        gated_scores, gated_coverages = _score(gated, samples, budgets)
        original_score_seeds.append(original_scores)
        gated_score_seeds.append(gated_scores)
        gated_prediction_seeds.append(gated)
        retained.extend(fractions)
        for budget in budgets:
            original_coverage_seeds[str(budget)].append(original_coverages[str(budget)])
            gated_coverage_seeds[str(budget)].append(gated_coverages[str(budget)])
    original_scores = _average_rows(original_score_seeds, ("kld_sph", "sim_sph", "cc_sph"))
    gated_scores = _average_rows(gated_score_seeds, ("kld_sph", "sim_sph", "cc_sph"))
    original_coverages = {
        key: _average_rows(value, ("coverage",)) for key, value in original_coverage_seeds.items()
    }
    gated_coverages = {
        key: _average_rows(value, ("coverage",)) for key, value in gated_coverage_seeds.items()
    }
    template = {str(sample["image_id"]): sample["template"].numpy() for sample in samples}
    template_scores, template_coverages = _score(template, samples, budgets)
    result = {
        "baseline": f"{args.source}+reachable-final",
        "split": args.split,
        "task": "start-conditioned kinematic filtering of a static final saliency prediction",
        "gate": "hard great-circle reachability support",
        "calibration": calibration,
        "source_provenance": provenance,
        "original_early_macro": _macro(original_scores, ("kld_sph", "sim_sph", "cc_sph")),
        "reachable_early_macro": _macro(gated_scores, ("kld_sph", "sim_sph", "cc_sph")),
        "original_coverage_early_macro": {
            key: _macro(rows, ("coverage",))["coverage"] for key, rows in original_coverages.items()
        },
        "reachable_coverage_early_macro": {
            key: _macro(rows, ("coverage",))["coverage"] for key, rows in gated_coverages.items()
        },
        "original_score_rows": original_scores,
        "reachable_score_rows": gated_scores,
        "original_coverage_rows": original_coverages,
        "reachable_coverage_rows": gated_coverages,
        "retained_mass_rows": retained,
        "comparisons": {
            "reachable_vs_original": compare_method_rows(gated_scores, original_scores),
            "reachable_vs_template": compare_method_rows(gated_scores, template_scores),
            "coverage_vs_original": {
                key: compare_coverage_rows(gated_coverages[key], original_coverages[key])
                for key in gated_coverages
            },
            "coverage_vs_template": {
                key: compare_coverage_rows(gated_coverages[key], template_coverages[key])
                for key in gated_coverages
            },
        },
        "test_evaluated": bool(args.split == "test"),
        "test_used_for_selection": False,
    }
    output = ensure_dir(
        args.output_dir or f"reports/kernel_v2_reachable_{args.source}_{args.split}"
    )
    (output / "results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    prediction_dir = ensure_dir(
        ROOT / args.prediction_root / args.source / args.split / "predictions"
    )
    for sample in samples:
        image_id = str(sample["image_id"])
        average = np.mean([values[image_id] for values in gated_prediction_seeds], axis=0)
        np.savez_compressed(prediction_dir / f"{image_id}.npz", cumulative=average.astype(np.float32))
    original = result["original_early_macro"]
    reachable = result["reachable_early_macro"]
    coverage = result["reachable_coverage_early_macro"]
    radii_text = ", ".join(f"{h:g}s={radii[h]:.2f}°" for h in HORIZONS)
    lines = [
        f"# Reachable-final: {args.source} — {args.split}",
        "",
        "The final saliency prediction is retained only inside a train-calibrated great-circle reachable region around the declared start. No network is trained or fine-tuned for this operation.",
        "",
        f"Frozen 95% dwell-mass radii: **{radii_text}**.",
        "",
        "| Variant | KLD_sph ↓ | SIM_sph ↑ | CC_sph ↑ | 10% coverage ↑ |",
        "|---|---:|---:|---:|---:|",
        f"| original final-repeat | {original['kld_sph']:.6f} | {original['sim_sph']:.6f} | {original['cc_sph']:.6f} | {result['original_coverage_early_macro']['10.0']:.6f} |",
        f"| reachable-final | {reachable['kld_sph']:.6f} | {reachable['sim_sph']:.6f} | {reachable['cc_sph']:.6f} | {coverage['10.0']:.6f} |",
        "",
        "Both rows remain static-content baselines; the reachable row adds only start and elapsed-time kinematics.",
    ]
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(output / "summary.md"), "radii_deg": calibration["radii_deg"]}, indent=2))


if __name__ == "__main__":
    main()
