#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sphererate.data.early_audit import audit_early_horizon_view
from sphererate.data.early_view import read_manifest
from sphererate.evaluation.early_reliability import evaluate_content_reliability
from sphererate.evaluation.metrics import cc, kld, sim
from sphererate.utils import ensure_dir


HORIZONS = (0.5, 1.0, 3.0, 5.0)
EARLY = {0.5, 1.0, 3.0}


def _percentiles(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"min": None, "p25": None, "median": None, "p75": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(array.min()),
        "p25": float(np.percentile(array, 25)),
        "median": float(np.median(array)),
        "p75": float(np.percentile(array, 75)),
        "max": float(array.max()),
    }


def _data_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    train = [row for row in rows if row["split"] == "train"]
    all_ready = [row for row in rows if row.get("fixations")]
    train_observers = [int(row["physical_observer_count"]) for row in train]
    all_observers = [int(row["physical_observer_count"]) for row in all_ready]
    return {
        "row_count": len(rows),
        "canonical_scene_count": len({str(row["canonical_content_id"]) for row in rows}),
        "train_condition_count": len(train),
        "train_scene_count": len({str(row["canonical_content_id"]) for row in train}),
        "train_observer_trials": int(sum(train_observers)),
        "all_ready_observer_trials": int(sum(all_observers)),
        "observer_count_per_train_target": _percentiles(train_observers),
        "observer_count_per_ready_target": _percentiles(all_observers),
        "train_targets_below_four_observers": int(sum(value < 4 for value in train_observers)),
        "train_targets_below_eight_observers": int(sum(value < 8 for value in train_observers)),
        "declared_start_count": len({float(row["declared_start_yaw_deg"]) for row in train}),
        "raw_declared_start_count": len(
            {float(row.get("raw_declared_start_yaw_deg", row["declared_start_yaw_deg"])) for row in train}
        ),
    }


def _reliability(
    name: str,
    rows: list[dict[str, Any]],
    *,
    height: int,
    width: int,
    observer_splits: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output: list[dict[str, Any]] = []
    train = [row for row in rows if row["split"] == "train"]
    skipped = 0
    for index, row in enumerate(train, start=1):
        with np.load(row["fixations"]) as data:
            fixations = np.asarray(data["fixations"], dtype=np.float32)
        observer_count = len(np.unique(fixations[:, 3]))
        if observer_count < 4:
            skipped += 1
            continue
        current = evaluate_content_reliability(
            content_id=str(row.get("condition_id", row["canonical_content_id"])),
            fixations=fixations,
            horizons_s=HORIZONS,
            height=height,
            width=width,
            sigma_px=None,
            renderer="sphere_chord",
            sigma_deg=2.0,
            observer_splits=observer_splits,
            seed=seed,
        )
        for item in current:
            item.update(
                {
                    "dataset": name,
                    "canonical_scene_id": str(row["canonical_content_id"]),
                    "condition_id": str(row.get("condition_id", row["canonical_content_id"])),
                }
            )
        output.extend(current)
        print(f"[{name}] reliability {index:02d}/{len(train)} {row['image_id']}", flush=True)

    # Four Sitzmann start conditions share one panorama. Average conditions
    # within a scene before averaging scenes to avoid pseudo-replication.
    grouped: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in output:
        grouped[(str(row["canonical_scene_id"]), float(row["horizon_s"]))].append(row)
    scene_rows = []
    for (scene, horizon), current in grouped.items():
        scene_rows.append(
            {
                "scene": scene,
                "horizon_s": horizon,
                **{
                    metric: float(np.mean([float(row[metric]) for row in current]))
                    for metric in ("same_kld", "same_sim", "same_cc")
                },
            }
        )
    per_horizon = {}
    for horizon in HORIZONS:
        current = [row for row in scene_rows if row["horizon_s"] == horizon]
        per_horizon[f"{horizon:g}"] = {
            metric: float(np.mean([row[metric] for row in current]))
            for metric in ("same_kld", "same_sim", "same_cc")
        }
    early_rows = [row for row in scene_rows if row["horizon_s"] in EARLY]
    summary = {
        "eligible_target_count": len({row["condition_id"] for row in output}),
        "skipped_below_four_observers": skipped,
        "effective_scene_count": len({row["scene"] for row in scene_rows}),
        "per_horizon": per_horizon,
        "early_macro": {
            metric: float(np.mean([row[metric] for row in early_rows]))
            for metric in ("same_kld", "same_sim", "same_cc")
        },
    }
    return output, summary


def _template_scores(rows: list[dict[str, Any]], root: Path) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for row in rows:
        if row["split"] not in {"val", "gate"}:
            continue
        with np.load(row["target"]) as data:
            target = np.asarray(data["cumulative"], dtype=np.float32)
            area = np.asarray(data["area_weights"], dtype=np.float32)
            horizons = np.asarray(data["horizons_s"], dtype=np.float32)
        with np.load(root / "baselines" / "early_template" / f"{row['image_id']}.npz") as data:
            prediction = np.asarray(data["cumulative"], dtype=np.float32)
        for index, horizon in enumerate(horizons):
            records.append(
                {
                    "split": str(row["split"]),
                    "scene": str(row["canonical_content_id"]),
                    "horizon_s": float(horizon),
                    "kld": kld(prediction[index], target[index], area),
                    "sim": sim(prediction[index], target[index], area),
                    "cc": cc(prediction[index], target[index], area),
                }
            )
    output = {}
    for split in ("val", "gate"):
        current = [row for row in records if row["split"] == split and row["horizon_s"] in EARLY]
        output[split] = {
            metric: float(np.mean([row[metric] for row in current])) for metric in ("kld", "sim", "cc")
        }
    return output


def _start_alignment(rows: list[dict[str, Any]]) -> dict[str, Any]:
    deviations: list[float] = []
    medians: list[float] = []
    for row in rows:
        if not row.get("fixations"):
            continue
        with np.load(row["fixations"]) as data:
            fixations = np.asarray(data["fixations"], dtype=np.float32)
        first = []
        for observer in np.unique(fixations[:, 3]):
            current = fixations[fixations[:, 3] == observer]
            first.append(float(current[np.argmin(current[:, 2]), 0]))
        if not first:
            continue
        medians.append(float(np.median(first)))
        delta = np.mod(np.asarray(first) - 0.5 + 0.5, 1.0) - 0.5
        deviations.extend((np.abs(delta) * 360.0).tolist())
    return {
        "target_x_norm": 0.5,
        "median_cell_first_fixation_x_norm": float(np.median(medians)),
        "absolute_first_fixation_error_deg": _percentiles(deviations),
    }


def _write_summary(result: dict[str, Any], path: Path) -> None:
    s = result["datasets"]["sitzmann"]
    a = result["datasets"]["salient360"]
    sr = s["reliability"]["early_macro"]
    ar = a["reliability"]["early_macro"]
    lines = [
        "# Sitzmann vs. Salient360 Training-Suitability Audit",
        "",
        f"Decision: **{result['decision']}**",
        "",
        "## Direct answer",
        "",
        result["direct_answer"],
        "",
        "## Supervision density and task fit",
        "",
        "| Dataset | Train scenes | Train condition maps | Observer trials | Median observers / target | Raw starts |",
        "|---|---:|---:|---:|---:|---:|",
        f"| Sitzmann | {s['stats']['train_scene_count']} | {s['stats']['train_condition_count']} | "
        f"{s['stats']['train_observer_trials']} | {s['stats']['observer_count_per_train_target']['median']:.1f} | "
        f"{s['stats']['raw_declared_start_count']} |",
        f"| Salient360 | {a['stats']['train_scene_count']} | {a['stats']['train_condition_count']} | "
        f"{a['stats']['train_observer_trials']} | {a['stats']['observer_count_per_train_target']['median']:.1f} | "
        f"{a['stats']['raw_declared_start_count']} |",
        "",
        "## Available-observer split-half agreement",
        "",
        "These values measure the actual noise of the supervision available to training. Four Sitzmann conditions from the same scene are averaged before the scene macro.",
        "",
        "| Dataset | early KLD ↓ | early SIM ↑ | early CC ↑ |",
        "|---|---:|---:|---:|",
        f"| Sitzmann | {sr['same_kld']:.4f} | {sr['same_sim']:.4f} | {sr['same_cc']:.4f} |",
        f"| Salient360 | {ar['same_kld']:.4f} | {ar['same_sim']:.4f} | {ar['same_cc']:.4f} |",
        "",
        "## Interpretation",
        "",
        "- Sitzmann is the cleaner **multi-start protocol**: four real declared orientations and correct start-centred reconstruction.",
        "- Salient360 is the stronger **population-map training source**: more unique scenes and many more observers behind each target.",
        "- Therefore total trajectory count is misleading. In Sitzmann, 780 trials are divided across 88 scene-start cells; after repeated-subject merging, the median target has only 8.5 physical observers.",
        "- The published Sitzmann test scenes are retained as a historically used gate, not claimed as a fresh test set.",
        "",
        "## Consequence",
        "",
        result["consequence"],
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Sitzmann and Salient360 as early-saliency training sources.")
    parser.add_argument("--sitzmann-root", default="data/processed/sitzmann_official_256x512_spherical2deg_v1")
    parser.add_argument("--salient360-root", default="data/processed/early_horizon_salient360_spherical2deg_v2")
    parser.add_argument("--output-dir", default="reports/sitzmann_training_suitability")
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--observer-splits", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260713)
    args = parser.parse_args()
    roots = {"sitzmann": Path(args.sitzmann_root), "salient360": Path(args.salient360_root)}
    datasets: dict[str, Any] = {}
    csv_rows: list[dict[str, Any]] = []
    for name, root in roots.items():
        rows = read_manifest(root / "manifest.jsonl")
        reliability_rows, reliability = _reliability(
            name,
            rows,
            height=int(args.height),
            width=int(args.width),
            observer_splits=int(args.observer_splits),
            seed=int(args.seed),
        )
        csv_rows.extend(reliability_rows)
        datasets[name] = {
            "root": str(root),
            "stats": _data_stats(rows),
            "view_audit": audit_early_horizon_view(root, expected_horizons_s=HORIZONS, integral_tolerance=2e-4),
            "reliability": reliability,
            "template_scores": _template_scores(rows, root),
        }
        if name == "sitzmann":
            datasets[name]["start_alignment"] = _start_alignment(rows)

    s = datasets["sitzmann"]
    a = datasets["salient360"]
    sole_replacement_checks = {
        "at_least_as_many_train_scenes": s["stats"]["train_scene_count"] >= a["stats"]["train_scene_count"],
        "at_least_half_as_many_observers_per_target": (
            s["stats"]["observer_count_per_train_target"]["median"]
            >= 0.5 * a["stats"]["observer_count_per_train_target"]["median"]
        ),
        "split_half_cc_not_worse": (
            s["reliability"]["early_macro"]["same_cc"]
            >= a["reliability"]["early_macro"]["same_cc"] - 0.02
        ),
        "has_real_multi_start": s["stats"]["raw_declared_start_count"] >= 4,
        "both_views_pass_integrity": s["view_audit"]["pass"] and a["view_audit"]["pass"],
    }
    replace = all(sole_replacement_checks.values())
    result = {
        "decision": "REPLACE_WITH_SITZMANN" if replace else "KEEP_SALIENT360_PRIMARY; USE_SITZMANN_AS_MULTI_START_COMPLEMENT",
        "direct_answer": (
            "The intuition is only partly correct: Sitzmann better matches the start-aware task, but the reconstructed labels are too sparse and the scene set too small to justify replacing Salient360 as the sole primary training set."
            if not replace
            else "The reconstructed Sitzmann data meets the frozen replacement checks and should become the primary training set."
        ),
        "consequence": (
            "Do not rerun the entire matched benchmark with Sitzmann-only training. The completed controlled comparison supports Salient360 as the primary source and Sitzmann as external multi-start evidence; fine-tuning is retained only as a secondary adaptation ablation."
            if not replace
            else "Retrain SphereHorizon and all trainable matched baselines on the Sitzmann content-disjoint split; keep Salient360 as frozen cross-dataset testing."
        ),
        "datasets": datasets,
        "sole_replacement_checks": sole_replacement_checks,
        "reliability_protocol": {
            "map_size": [int(args.height), int(args.width)],
            "renderer": "sphere_chord",
            "sigma_deg": 2.0,
            "observer_splits": int(args.observer_splits),
            "seed": int(args.seed),
            "aggregation": "condition-to-scene then scene macro",
        },
    }
    output = ensure_dir(args.output_dir)
    (output / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if csv_rows:
        with (output / "per_target_reliability.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
            writer.writeheader()
            writer.writerows(csv_rows)
    _write_summary(result, output / "summary.md")
    print(json.dumps({"decision": result["decision"], "summary": str(output / "summary.md")}, indent=2))


if __name__ == "__main__":
    main()
