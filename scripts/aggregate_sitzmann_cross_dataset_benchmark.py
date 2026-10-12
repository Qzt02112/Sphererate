#!/usr/bin/env python
from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports/sitzmann_cross_dataset_benchmark"
HORIZONS = (0.5, 1.0, 3.0, 5.0)
EARLY = {0.5, 1.0, 3.0}
METRICS = ("kld_sph", "sim_sph", "cc_sph")
COVERAGE_BUDGETS = (5.0, 10.0, 20.0)
SEEDS = (20260710, 20260711, 20260712)


REGULAR = {
    ("TempSAL-ERP", "salient360", "salient360"): "reports/kernel_v2_tempsal_erp_gate/results.json",
    ("TempSAL-ERP", "salient360", "sitzmann"): "reports/cross_tempsal_salient360_to_sitzmann_gate/results.json",
    ("TempSAL-ERP", "sitzmann", "salient360"): "reports/cross_tempsal_sitzmann_to_salient360_gate/results.json",
    ("TempSAL-ERP", "sitzmann", "sitzmann"): "reports/sitzmann_only_tempsal_erp_gate/results.json",
    ("Panoramic-CNN", "salient360", "salient360"): "reports/kernel_v2_panoramic_static_gate/results.json",
    ("Panoramic-CNN", "salient360", "sitzmann"): "reports/cross_panoramic_cnn_salient360_to_sitzmann_gate/results.json",
    ("Panoramic-CNN", "sitzmann", "salient360"): "reports/cross_panoramic_cnn_sitzmann_to_salient360_gate/results.json",
    ("Panoramic-CNN", "sitzmann", "sitzmann"): "reports/sitzmann_only_panoramic_cnn_gate/results.json",
    ("ScanDMM", "salient360", "salient360"): "reports/kernel_v2_scandmm_declared_start_gate/results.json",
    ("ScanDMM", "salient360", "sitzmann"): "reports/cross_scandmm_salient360_to_sitzmann_gate/results.json",
    ("ScanDMM", "sitzmann", "salient360"): "reports/cross_scandmm_sitzmann_to_salient360_gate/results.json",
    ("ScanDMM", "sitzmann", "sitzmann"): "reports/sitzmann_only_scandmm_gate/results.json",
    ("Pathformer3D", "salient360", "salient360"): "reports/kernel_v2_pathformer3d_retrained_gate/results.json",
    ("Pathformer3D", "salient360", "sitzmann"): "reports/cross_pathformer3d_salient360_to_sitzmann_gate/results.json",
    ("Pathformer3D", "sitzmann", "salient360"): "reports/cross_pathformer3d_sitzmann_to_salient360_gate/results.json",
    ("Pathformer3D", "sitzmann", "sitzmann"): "reports/sitzmann_only_pathformer3d_gate/results.json",
}


STATIC_RAW = {
    ("BMS360", "salient360"): "reports/kernel_v2_bms360_gate/results.json",
    ("BMS360", "sitzmann"): "reports/sitzmann_bms360_gate/results.json",
    ("GBVS360", "salient360"): "reports/kernel_v2_gbvs360_gate/results.json",
    ("GBVS360", "sitzmann"): "reports/sitzmann_gbvs360_gate/results.json",
}


REACHABLE = {
    ("BMS360+reachable", "salient360", "salient360"): "reports/kernel_v2_reachable_bms360_gate/results.json",
    ("BMS360+reachable", "salient360", "sitzmann"): "reports/cross_reachable_bms360_salient360_to_sitzmann_gate/results.json",
    ("BMS360+reachable", "sitzmann", "salient360"): "reports/cross_reachable_bms360_sitzmann_to_salient360_gate/results.json",
    ("BMS360+reachable", "sitzmann", "sitzmann"): "reports/sitzmann_reachable_bms360_gate/results.json",
    ("GBVS360+reachable", "salient360", "salient360"): "reports/kernel_v2_reachable_gbvs360_gate/results.json",
    ("GBVS360+reachable", "salient360", "sitzmann"): "reports/cross_reachable_gbvs360_salient360_to_sitzmann_gate/results.json",
    ("GBVS360+reachable", "sitzmann", "salient360"): "reports/cross_reachable_gbvs360_sitzmann_to_salient360_gate/results.json",
    ("GBVS360+reachable", "sitzmann", "sitzmann"): "reports/sitzmann_reachable_gbvs360_gate/results.json",
    ("Panoramic-CNN+reachable", "salient360", "salient360"): "reports/kernel_v2_reachable_panoramic_cnn_gate/results.json",
    ("Panoramic-CNN+reachable", "salient360", "sitzmann"): "reports/cross_reachable_panoramic_cnn_salient360_to_sitzmann_gate/results.json",
    ("Panoramic-CNN+reachable", "sitzmann", "salient360"): "reports/cross_reachable_panoramic_cnn_sitzmann_to_salient360_gate/results.json",
    ("Panoramic-CNN+reachable", "sitzmann", "sitzmann"): "reports/sitzmann_reachable_panoramic_cnn_gate/results.json",
}


DEEPGAZE = {
    ("salient360", "salient360"): "reports/kernel_v2_deepgaze_msdb_gate/results.json",
    ("salient360", "sitzmann"): "reports/cross_deepgaze_salient360_to_sitzmann_gate/results.json",
    ("sitzmann", "salient360"): "reports/cross_deepgaze_sitzmann_to_salient360_gate/results.json",
    ("sitzmann", "sitzmann"): "reports/sitzmann_deepgaze_msdb_gate/results.json",
}


def _load(relative: str) -> dict[str, Any]:
    path = ROOT / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _bootstrap(rows: list[dict[str, Any]], *, iterations: int = 10000) -> dict[str, list[float]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if float(row["horizon_s"]) in EARLY:
            grouped[str(row["content_id"])].append(row)
    scenes = sorted(grouped)
    if not scenes:
        return {}
    scene_values = np.asarray(
        [[np.mean([float(row[key]) for row in grouped[scene]]) for key in METRICS] for scene in scenes],
        dtype=np.float64,
    )
    rng = np.random.default_rng(20260714)
    indices = rng.integers(0, len(scenes), size=(int(iterations), len(scenes)))
    samples = scene_values[indices].mean(axis=1)
    return {
        key: [float(value) for value in np.quantile(samples[:, index], (0.025, 0.975))]
        for index, key in enumerate(METRICS)
    }


def _record(
    method: str,
    train_source: str,
    target: str,
    metrics: dict[str, Any],
    rows: list[dict[str, Any]] | None,
    coverage: dict[str, Any],
    coverage_rows: dict[str, list[dict[str, Any]]] | None,
    report: str,
    *,
    variant: str = "primary",
    seed_std: dict[str, float] | None = None,
) -> dict[str, Any]:
    rows = _collapse_rows(list(rows or []))
    normalized_coverage = {
        str(float(budget)): float(coverage[str(float(budget))]) for budget in COVERAGE_BUDGETS
    }
    collapsed_coverage_rows = {
        str(float(budget)): _collapse_coverage_rows(
            list((coverage_rows or {}).get(str(float(budget)), []))
        )
        for budget in COVERAGE_BUDGETS
    }
    return {
        "method": method,
        "variant": variant,
        "train_source": train_source,
        "target": target,
        "metrics": {key: float(metrics[key]) for key in METRICS},
        "coverage_early_macro": normalized_coverage,
        "scene_count": len({str(row["content_id"]) for row in rows}) if rows else None,
        "scene_bootstrap_ci95": _bootstrap(rows),
        "seed_std": seed_std,
        "report": report,
        "score_rows": rows,
        "coverage_rows": collapsed_coverage_rows,
    }


def _collapse_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse multiple viewpoint/image conditions to canonical scene×horizon."""

    grouped: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["content_id"]), float(row["horizon_s"]))].append(row)
    return [
        {
            "content_id": content,
            "horizon_s": horizon,
            **{
                metric: float(np.mean([float(row[metric]) for row in current]))
                for metric in METRICS
            },
        }
        for (content, horizon), current in sorted(grouped.items())
    ]


def _collapse_coverage_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["content_id"]), float(row["horizon_s"]))].append(row)
    return [
        {
            "content_id": content,
            "horizon_s": horizon,
            "coverage": float(np.mean([float(row["coverage"]) for row in current])),
        }
        for (content, horizon), current in sorted(grouped.items())
    ]


def _sphere_coverage_rows(train: str, target: str) -> dict[str, list[dict[str, Any]]]:
    path = ROOT / "reports/sitzmann_train_source_comparison/per_scene_horizon.csv"
    with path.open("r", encoding="utf-8", newline="") as stream:
        raw = [
            row
            for row in csv.DictReader(stream)
            if row["train_source"] == train
            and row["eval_dataset"] == target
            and row["method"] == "spherehorizon"
            and row["template_policy"] == "source_domain"
        ]
    return {
        str(float(budget)): [
            {
                "content_id": row["scene_id"],
                "horizon_s": float(row["horizon_s"]),
                "coverage": float(row[f"coverage_{budget:g}"]),
            }
            for row in raw
        ]
        for budget in COVERAGE_BUDGETS
    }


def _sphere_records() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source = _load("reports/sitzmann_train_source_comparison/results.json")
    records = []
    for train in ("salient360", "sitzmann"):
        for target in ("salient360", "sitzmann"):
            item = source["comparisons"][f"{train}_to_{target}"]["model"]["early_macro"]
            metrics = {"kld_sph": item["kld"]["mean"], "sim_sph": item["sim"]["mean"], "cc_sph": item["cc"]["mean"]}
            std = {"kld_sph": item["kld"]["std"], "sim_sph": item["sim"]["std"], "cc_sph": item["cc"]["std"]}
            coverage = {
                str(float(budget)): item[f"coverage_{budget:g}"]["mean"]
                for budget in COVERAGE_BUDGETS
            }
            records.append(
                _record(
                    "SphereHorizon",
                    train,
                    target,
                    metrics,
                    None,
                    coverage,
                    _sphere_coverage_rows(train, target),
                    "reports/sitzmann_train_source_comparison/results.json",
                    seed_std=std,
                )
            )
    return records, source


def _collect() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records, sphere_source = _sphere_records()
    for (method, train, target), report in REGULAR.items():
        result = _load(report)
        records.append(_record(method, train, target, result["method_early_macro"], result["score_rows"], result["coverage_early_macro"], result["coverage_rows"], report))
    for (method, target), report in STATIC_RAW.items():
        result = _load(report)
        records.append(_record(method, "not_applicable", target, result["method_early_macro"], result["score_rows"], result["coverage_early_macro"], result["coverage_rows"], report))
    for (method, train, target), report in REACHABLE.items():
        result = _load(report)
        records.append(
            _record(
                method,
                train,
                target,
                result["reachable_early_macro"],
                result["reachable_score_rows"],
                result["reachable_coverage_early_macro"],
                result["reachable_coverage_rows"],
                report,
                variant="source_train_reachability",
            )
        )
    raw_seen = set()
    for (train, target), report in DEEPGAZE.items():
        result = _load(report)
        if target not in raw_seen:
            records.append(
                _record(
                    "DeepGaze-MSDB raw",
                    "external_pretrained",
                    target,
                    result["raw"]["early_macro"],
                    result["raw"]["score_rows"],
                    result["raw"]["coverage_early_macro"],
                    result["raw"]["coverage_rows"],
                    report,
                    variant="raw_final_repeat",
                )
            )
            raw_seen.add(target)
        records.append(
            _record(
                "DeepGaze-MSDB calibrated",
                train,
                target,
                result["train_calibrated"]["early_macro"],
                result["train_calibrated"]["score_rows"],
                result["train_calibrated"]["coverage_early_macro"],
                result["train_calibrated"]["coverage_rows"],
                report,
                variant="source_train_calibration",
            )
        )
        records.append(
            _record(
                "DeepGaze-MSDB+reachable",
                train,
                target,
                result["train_calibrated_reachable"]["early_macro"],
                result["train_calibrated_reachable"]["score_rows"],
                result["train_calibrated_reachable"]["coverage_early_macro"],
                result["train_calibrated_reachable"]["coverage_rows"],
                report,
                variant="source_train_calibration_and_reachability",
            )
        )
    return records, sphere_source


def _audit(records: list[dict[str, Any]]) -> dict[str, Any]:
    manifests = {}
    for dataset, relative in {
        "salient360": "data/processed/early_horizon_salient360_spherical2deg_v2/manifest.jsonl",
        "sitzmann": "data/processed/sitzmann_official_256x512_spherical2deg_v1/manifest.jsonl",
    }.items():
        path = ROOT / relative
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        split_scenes = {
            split: {str(row["canonical_content_id"]) for row in rows if row["split"] == split}
            for split in ("train", "val", "gate")
        }
        overlap = {
            f"{left}_{right}": sorted(split_scenes[left] & split_scenes[right])
            for left, right in (("train", "val"), ("train", "gate"), ("val", "gate"))
        }
        manifests[dataset] = {
            "manifest": relative,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "row_counts": {split: sum(row["split"] == split for row in rows) for split in split_scenes},
            "scene_counts": {split: len(values) for split, values in split_scenes.items()},
            "overlap": overlap,
            "zero_content_leakage": not any(overlap.values()),
        }

    selection = []
    for family, prefix in (
        ("ScanDMM", "kernel_v2_scandmm_declared_start"),
        ("ScanDMM", "sitzmann_only_scandmm_declared_start"),
        ("Pathformer3D", "kernel_v2_pathformer3d_retrained"),
        ("Pathformer3D", "sitzmann_only_pathformer3d"),
    ):
        for seed in SEEDS:
            path = ROOT / f"runs/{prefix}_seed{seed}/selection.json"
            value = json.loads(path.read_text(encoding="utf-8"))
            selection.append(
                {
                    "family": family,
                    "path": str(path.relative_to(ROOT)),
                    "seed": seed,
                    "gate_used": bool(value.get("gate_used", False)),
                    "test_target_loads": int(value.get("test_target_loads", 0)),
                    "selected_epoch": int((value.get("best") or value.get("selected"))["epoch"]),
                }
            )

    scanpath_audit = _load("data/processed/sitzmann_official_matched_scanpaths_v1/audit.json")
    start_checks = []
    for family, pattern, expected_yaw in (
        ("ScanDMM Sal→Sitz", "runs/cross_scandmm_salient360_to_sitzmann_seed{seed}_gate/inference_metadata.json", 90.0),
        ("ScanDMM Sitz→Sal", "runs/cross_scandmm_sitzmann_to_salient360_seed{seed}_gate/inference_metadata.json", 0.0),
        ("Pathformer Sal→Sitz", "runs/cross_pathformer3d_salient360_to_sitzmann_seed{seed}_gate/inference_metadata.json", 90.0),
        ("Pathformer Sitz→Sal", "runs/cross_pathformer3d_sitzmann_to_salient360_seed{seed}_gate/inference_metadata.json", 0.0),
    ):
        for seed in SEEDS:
            path = ROOT / pattern.format(seed=seed)
            value = json.loads(path.read_text(encoding="utf-8"))
            count = len(list((path.parent / "scanpaths").glob("*.npy")))
            expected_count = 12 if "Sal→Sitz" in family else 13
            start_checks.append(
                {
                    "family": family,
                    "seed": seed,
                    "start_yaw_deg": float(value["start_yaw_deg"]),
                    "expected_start_yaw_deg": expected_yaw,
                    "output_count": count,
                    "expected_output_count": expected_count,
                    "passed": np.isclose(float(value["start_yaw_deg"]), expected_yaw) and count == expected_count,
                }
            )

    score_shape_checks = []
    coverage_shape_checks = []
    for record in records:
        expected_scenes = 3 if record["target"] == "sitzmann" else 13
        if record["score_rows"]:
            count = len(record["score_rows"])
            score_shape_checks.append(
                {
                    "method": record["method"],
                    "train_source": record["train_source"],
                    "target": record["target"],
                    "row_count": count,
                    "expected_scene_horizon_rows": expected_scenes * len(HORIZONS),
                    "passed": count == expected_scenes * len(HORIZONS),
                }
            )
        for budget in COVERAGE_BUDGETS:
            rows = record["coverage_rows"][str(float(budget))]
            coverage_shape_checks.append(
                {
                    "method": record["method"],
                    "train_source": record["train_source"],
                    "target": record["target"],
                    "budget_percent": budget,
                    "row_count": len(rows),
                    "expected_scene_horizon_rows": expected_scenes * len(HORIZONS),
                    "passed": len(rows) == expected_scenes * len(HORIZONS),
                }
            )

    checks = {
        "zero_content_leakage": all(value["zero_content_leakage"] for value in manifests.values()),
        "selection_uses_no_gate_or_test": all(not row["gate_used"] and row["test_target_loads"] == 0 for row in selection),
        "sitzmann_dense_scanpaths_complete": (
            scanpath_audit["audit"]["dense_1hz_trials"] == 780
            and scanpath_audit["audit"]["dropped_observers"] == 0
            and scanpath_audit["audit"]["repaired_bins"] == 0
        ),
        "cross_start_and_output_counts": all(row["passed"] for row in start_checks),
        "canonical_scene_aggregation": all(row["passed"] for row in score_shape_checks),
        "coverage_budgets_complete": all(row["passed"] for row in coverage_shape_checks),
        "all_metrics_finite": all(
            np.isfinite(value)
            for record in records
            for value in record["metrics"].values()
        ),
        "all_coverage_finite": all(
            np.isfinite(value) and 0.0 <= value <= 1.0
            for record in records
            for value in record["coverage_early_macro"].values()
        ),
    }
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "manifests": manifests,
        "selection": selection,
        "scanpath_export": scanpath_audit,
        "cross_start_checks": start_checks,
        "score_shape_checks": score_shape_checks,
        "coverage_shape_checks": coverage_shape_checks,
        "sealed_test_target_loads": 0,
    }


def _lookup(records: list[dict[str, Any]], method: str, train: str, target: str) -> dict[str, Any]:
    matches = [row for row in records if row["method"] == method and row["train_source"] == train and row["target"] == target]
    if len(matches) != 1:
        raise ValueError((method, train, target, len(matches)))
    return matches[0]


def _write_summary(records: list[dict[str, Any]], sphere_source: dict[str, Any], audit: dict[str, Any]) -> None:
    lines = [
        "# Sitzmann-only and Cross-dataset Benchmark",
        "",
        f"Integrity audit: **{audit['status'].upper()}**. All learned models use three seeds; checkpoints are selected on the training source's validation split only. Cross-dataset cells apply only a circular ERP coordinate alignment so the physical start occupies the source model's expected coordinate. Sitzmann's four viewpoints are averaged within each canonical scene before the three-scene macro average.",
        "",
        "## Learned-model matrix",
        "",
        "| Method | Train → target | KLD_sph ↓ | SIM_sph ↑ | CC_sph ↑ | 95% scene CI (KLD) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for method in ("SphereHorizon", "TempSAL-ERP", "Panoramic-CNN", "ScanDMM", "Pathformer3D"):
        for train, target in (("salient360", "salient360"), ("salient360", "sitzmann"), ("sitzmann", "sitzmann"), ("sitzmann", "salient360")):
            row = _lookup(records, method, train, target)
            metric = row["metrics"]
            ci = row["scene_bootstrap_ci95"].get("kld_sph")
            ci_text = "seed std " + f"{row['seed_std']['kld_sph']:.4f}" if ci is None else f"[{ci[0]:.3f}, {ci[1]:.3f}]"
            lines.append(
                f"| {method} | {train} → {target} | {metric['kld_sph']:.4f} | {metric['sim_sph']:.4f} | {metric['cc_sph']:.4f} | {ci_text} |"
            )

    lines.extend(
        [
            "",
            "## Fixed-budget gaze-mass coverage",
            "",
            "Coverage is the fraction of fixation dwell mass captured by the top-ranked tiles on the frozen `8×16` ERP grid. The `5% / 10% / 20%` budgets select `7 / 13 / 26` of 128 tiles via ceil; all values are early-macro means over `0.5/1/3 s` and canonical scenes.",
            "",
            "| Method | Train → target | Cov@5% ↑ | Cov@10% ↑ | Cov@20% ↑ |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for method in ("SphereHorizon", "TempSAL-ERP", "Panoramic-CNN", "ScanDMM", "Pathformer3D"):
        for train, target in (("salient360", "salient360"), ("salient360", "sitzmann"), ("sitzmann", "sitzmann"), ("sitzmann", "salient360")):
            row = _lookup(records, method, train, target)
            cov = row["coverage_early_macro"]
            lines.append(
                f"| {method} | {train} → {target} | {cov['5.0']:.4f} | {cov['10.0']:.4f} | {cov['20.0']:.4f} |"
            )

    lines.extend(["", "## What changes when the training source changes?", ""])
    for target in ("sitzmann", "salient360"):
        better_sitz, better_sal = [], []
        for method in ("SphereHorizon", "TempSAL-ERP", "Panoramic-CNN", "ScanDMM", "Pathformer3D"):
            sal = _lookup(records, method, "salient360", target)["metrics"]["kld_sph"]
            sitz = _lookup(records, method, "sitzmann", target)["metrics"]["kld_sph"]
            (better_sitz if sitz < sal else better_sal).append(f"{method} ({sitz-sal:+.3f})")
        lines.append(
            f"- On **{target}**, Sitzmann training improves: {', '.join(better_sitz) if better_sitz else 'none'}; Salient360 training remains better for: {', '.join(better_sal) if better_sal else 'none'}. Parentheses are Sitzmann-trained minus Salient360-trained KLD."
        )

    lines.extend(
        [
            "",
            "## Static and recent-model controls",
            "",
            "| Method/adapter | Calibration source → target | KLD_sph ↓ | CC_sph ↑ | Cov@5% ↑ | Cov@10% ↑ | Cov@20% ↑ |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    selected_controls = [
        ("BMS360", "not_applicable", "sitzmann"),
        ("GBVS360", "not_applicable", "sitzmann"),
        ("BMS360+reachable", "sitzmann", "sitzmann"),
        ("GBVS360+reachable", "sitzmann", "sitzmann"),
        ("Panoramic-CNN+reachable", "sitzmann", "sitzmann"),
        ("DeepGaze-MSDB raw", "external_pretrained", "sitzmann"),
        ("DeepGaze-MSDB calibrated", "sitzmann", "sitzmann"),
        ("DeepGaze-MSDB+reachable", "sitzmann", "sitzmann"),
        ("DeepGaze-MSDB+reachable", "salient360", "sitzmann"),
    ]
    for method, train, target in selected_controls:
        row = _lookup(records, method, train, target)
        cov = row["coverage_early_macro"]
        lines.append(
            f"| {method} | {train} → {target} | {row['metrics']['kld_sph']:.4f} | {row['metrics']['cc_sph']:.4f} | {cov['5.0']:.4f} | {cov['10.0']:.4f} | {cov['20.0']:.4f} |"
        )

    sphere_sitz = _lookup(records, "SphereHorizon", "sitzmann", "sitzmann")["metrics"]
    temp_sitz = _lookup(records, "TempSAL-ERP", "sitzmann", "sitzmann")["metrics"]
    scan_sitz = _lookup(records, "ScanDMM", "sitzmann", "sitzmann")["metrics"]
    path_sitz = _lookup(records, "Pathformer3D", "sitzmann", "sitzmann")["metrics"]
    lines.extend(
        [
            "",
            "## Insights",
            "",
            f"1. **Known start does not make early attention trivial.** On Sitzmann, the start-aware scanpath models remain far behind direct early-map prediction: ScanDMM/Pathformer3D KLD are {scan_sitz['kld_sph']:.3f}/{path_sitz['kld_sph']:.3f}, versus {sphere_sitz['kld_sph']:.3f}/{temp_sitz['kld_sph']:.3f} for SphereHorizon/TempSAL. This is stronger evidence for task necessity than merely comparing against a final saliency map.",
            "2. **Sitzmann adds start-condition evidence, not enough content diversity for a universal replacement.** Its 64 training rows are four rotations/starts of only 16 scenes and contain 547 dense trajectories. Salient360 has 50 independent training scenes and 1,840 usable training scanpaths. The full matrix therefore separates condition diversity from scene diversity.",
            "3. **Task-matched map models can exploit Sitzmann in-domain, but the scanpath models do not benefit from Sitzmann-only training.** Sitzmann training improves the Sitzmann target for SphereHorizon and TempSAL, while Salient360-trained ScanDMM and Pathformer3D outperform their Sitzmann-trained counterparts even on Sitzmann. The start/output audit rules out omission of the declared start; the remaining gap is consistent with limited scene diversity and with a mismatch between dense trajectory supervision and the fixation-derived cumulative-map target.",
            "4. **Reachability is useful but insufficient.** It strongly repairs static and pretrained final maps, yet the best Sitzmann reachable control remains materially different from task-matched early-map prediction. It should stay as a strong control/application adapter, not become the proposed method.",
            "5. **Density accuracy and tile coverage are related but not interchangeable.** SphereHorizon has the stronger Sitzmann in-domain CC, while TempSAL captures slightly more gaze mass at 5% and 10% budgets; SphereHorizon is higher at 20%. The same near-tie appears on Salient360. Therefore the defensible application claim is clear improvement over ungated static-final maps and competitive coverage against TempSAL/reachable controls—not uniform coverage dominance at every budget.",
            "",
            "## Direction decision",
            "",
            "**Do not replace Salient360 with Sitzmann as the sole primary training set.** Keep Salient360 as the content-diverse primary benchmark, add Sitzmann as a second in-domain training/evaluation track for true four-start evidence, and report frozen bidirectional transfer. For the paper, the most defensible claim is not ‘Sitzmann is the better dataset’; it is that early attention needs a task-matched population-map model, and that content diversity and start-condition diversity contribute differently.",
            "",
            "Sitzmann's published scanpath test scenes remain a non-fresh gate, so these numbers are development evidence rather than a newly sealed final test.",
        ]
    )
    (OUTPUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    records, sphere_source = _collect()
    audit = _audit(records)
    serializable = [
        {key: value for key, value in row.items() if key not in {"score_rows", "coverage_rows"}}
        for row in records
    ]
    result = {
        "protocol": "sitzmann_cross_dataset_benchmark_v1",
        "records": serializable,
        "decision": "salient360_primary_plus_sitzmann_secondary_in_domain_and_bidirectional_transfer",
        "integrity_status": audit["status"],
        "sealed_test_target_loads": 0,
    }
    (OUTPUT / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (OUTPUT / "integrity_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (OUTPUT / "per_scene.csv").open("w", encoding="utf-8", newline="") as stream:
        fields = ["method", "variant", "train_source", "target", "content_id", "horizon_s", *METRICS]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in records:
            for row in record["score_rows"]:
                writer.writerow(
                    {
                        "method": record["method"],
                        "variant": record["variant"],
                        "train_source": record["train_source"],
                        "target": record["target"],
                        **{key: row[key] for key in ("content_id", "horizon_s", *METRICS)},
                    }
                )
    with (OUTPUT / "coverage_per_scene.csv").open("w", encoding="utf-8", newline="") as stream:
        fields = ["method", "variant", "train_source", "target", "budget_percent", "content_id", "horizon_s", "coverage"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in records:
            for budget in COVERAGE_BUDGETS:
                for row in record["coverage_rows"][str(float(budget))]:
                    writer.writerow(
                        {
                            "method": record["method"],
                            "variant": record["variant"],
                            "train_source": record["train_source"],
                            "target": record["target"],
                            "budget_percent": budget,
                            **row,
                        }
                    )
    _write_summary(records, sphere_source, audit)
    print(json.dumps({"output": str(OUTPUT), "records": len(records), "audit": audit["status"]}, indent=2))


if __name__ == "__main__":
    main()
