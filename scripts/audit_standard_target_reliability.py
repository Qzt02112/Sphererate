#!/usr/bin/env python
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sphererate.data.early_horizon import audit_fixation_sequences
from sphererate.evaluation.early_reliability import (
    aggregate_reliability,
    evaluate_content_reliability,
)
from sphererate.utils import ensure_dir


DEFAULT_MANIFEST = (
    "data/processed/early_horizon_salient360_spherical2deg_standard_v3/manifest.jsonl"
)
DEFAULT_TEMPLATE_ROOT = (
    "data/processed/early_horizon_salient360_spherical2deg_standard_v3/"
    "baselines/early_template"
)
DEFAULT_OUTPUT = "reports/standard_split_v3_target_reliability"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _evaluate_one(payload: dict[str, Any]) -> dict[str, Any]:
    row = payload["row"]
    with np.load(str(row["fixations"])) as fixation_file:
        fixations = np.asarray(fixation_file["fixations"], dtype=np.float32)
    template_path = Path(payload["template_root"]) / f"{row['image_id']}.npz"
    with np.load(template_path) as template_file:
        template_horizons = np.asarray(template_file["horizons_s"], dtype=np.float64)
        template_maps = np.asarray(template_file["cumulative"], dtype=np.float32)
    templates = {
        float(horizon): template_maps[index]
        for index, horizon in enumerate(template_horizons.tolist())
    }
    rows = evaluate_content_reliability(
        content_id=str(row["canonical_content_id"]),
        fixations=fixations,
        horizons_s=tuple(payload["horizons_s"]),
        height=int(payload["height"]),
        width=int(payload["width"]),
        sigma_px=None,
        observer_splits=int(payload["observer_splits"]),
        seed=int(payload["seed"]),
        renderer="sphere_chord",
        sigma_deg=float(payload["sigma_deg"]),
        templates=templates,
        budgets_percent=tuple(payload["budgets_percent"]),
        tile_rows=int(payload["tile_rows"]),
        tile_cols=int(payload["tile_cols"]),
    )
    for current in rows:
        current.update({"split": str(row["split"]), "image_id": str(row["image_id"])})
    return {
        "image_id": str(row["image_id"]),
        "rows": rows,
        "audit": audit_fixation_sequences(fixations, horizons_s=payload["horizons_s"]),
    }


def _aggregate_audit(audits: list[dict[str, Any]]) -> dict[str, Any]:
    observer_counts = np.asarray([item["observer_count"] for item in audits], dtype=np.float64)
    horizon_keys = list(audits[0]["observers_with_mass_by_horizon"])
    return {
        "content_count": len(audits),
        "observer_count_per_content": {
            "min": int(observer_counts.min()),
            "median": float(np.median(observer_counts)),
            "max": int(observer_counts.max()),
        },
        "non_monotonic_observer_sequences": int(
            sum(int(item["non_monotonic_observer_count"]) for item in audits)
        ),
        "observer_mass_fraction_per_content": {
            key: {
                name: float(value)
                for name, value in zip(
                    ("min", "p25", "median", "p75", "max"),
                    np.percentile(
                        np.asarray(
                            [
                                item["observers_with_mass_by_horizon"][key]
                                / item["observer_count"]
                                for item in audits
                            ],
                            dtype=np.float64,
                        ),
                        (0, 25, 50, 75, 100),
                    ),
                )
            }
            for key in horizon_keys
        },
    }


def _ci(value: dict[str, Any], digits: int = 3) -> str:
    return (
        f"{value['mean']:.{digits}f} "
        f"[{value['ci95_low']:.{digits}f}, {value['ci95_high']:.{digits}f}]"
    )


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_summary(result: dict[str, Any], path: Path) -> None:
    aggregates = result["aggregates"]
    lines = [
        "# Salient360 V3 Target-Reliability Audit",
        "",
        "Status: **EVIDENCE COMPLETE**",
        "",
        "This is a human-to-human target audit; it is not a model comparison. No neural model was trained, and no standard-test fixation or target file was loaded.",
        "",
        "## Protocol",
        "",
        f"- contents: `{result['content_count']}` train+validation contents (`{result['split_counts']}`)",
        f"- physical observers/content: `{result['observer_audit']['observer_count_per_content']}`",
        f"- deterministic split-halves/content: `{result['observer_splits']}` with both A→B and B→A directions",
        f"- horizons: `{result['horizons_s']}` s; all four belong to the early window",
        f"- renderer: `256×512`, spherical chord Gaussian, sigma `{result['map']['sigma_deg']}°`, spherical-area normalization",
        "- primary eye: right; left/right eyes are never pooled as independent observers",
        "- confidence intervals: 10,000 bootstrap resamples grouped by canonical content",
        "",
        "## Absolute split-half agreement and temporal necessity",
        "",
        "Each cell is mean [95% content-bootstrap CI]. Positive deltas mean that a same-horizon observer half predicts the held-out half better than the source half's final map.",
        "",
        "| Horizon | Split-half KLD ↓ | Split-half SIM ↑ | Split-half CC ↑ | Full-cohort CC estimate ↑ | ΔKLD vs final ↑ | ΔSIM vs final ↑ | ΔCC vs final ↑ |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for horizon, metrics in aggregates["per_horizon"].items():
        lines.append(
            f"| {horizon}s | {_ci(metrics['same_kld'])} | {_ci(metrics['same_sim'])} | "
            f"{_ci(metrics['same_cc'])} | {_ci(metrics['full_cohort_cc_estimate'])} | "
            f"{_ci(metrics['kld_improvement'])} | {_ci(metrics['sim_improvement'])} | "
            f"{_ci(metrics['cc_improvement'])} |"
        )
    macro = aggregates["early_macro"]
    lines.append(
        f"| **0.5–5 s macro** | **{_ci(macro['same_kld'])}** | **{_ci(macro['same_sim'])}** | "
        f"**{_ci(macro['same_cc'])}** | **{_ci(macro['full_cohort_cc_estimate'])}** | "
        f"**{_ci(macro['kld_improvement'])}** | **{_ci(macro['sim_improvement'])}** | "
        f"**{_ci(macro['cc_improvement'])}** |"
    )
    lines.extend(
        [
            "",
            "## Separation from a train-only start template",
            "",
            "The template uses only training contents and leaves out the evaluated content when that content belongs to training. Positive values favor the content-specific human half-map.",
            "",
            "| Horizon | ΔKLD ↑ | ΔSIM ↑ | ΔCC ↑ |",
            "|---:|---:|---:|---:|",
        ]
    )
    for horizon, metrics in aggregates["per_horizon"].items():
        lines.append(
            f"| {horizon}s | {_ci(metrics['same_vs_template_kld_improvement'])} | "
            f"{_ci(metrics['same_vs_template_sim_improvement'])} | "
            f"{_ci(metrics['same_vs_template_cc_improvement'])} |"
        )
    lines.extend(
        [
            "",
            "## Raw-dwell tile-set reliability",
            "",
            "Top-k overlap compares the two observer-half tile sets. Normalized coverage divides held-out dwell captured by the source-half set by the held-out oracle coverage at the same budget.",
            "",
            "| Horizon | Top-10% overlap ↑ | Human-half normalized coverage ↑ | Start-template normalized coverage ↑ | Human − template raw coverage ↑ |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for horizon, metrics in aggregates["per_horizon"].items():
        lines.append(
            f"| {horizon}s | {_ci(metrics['tile_overlap_10'])} | "
            f"{_ci(metrics['tile_normalized_coverage_10'])} | "
            f"{_ci(metrics['tile_template_normalized_coverage_10'])} | "
            f"{_ci(metrics['tile_same_vs_template_coverage_10'])} |"
        )
    lines.extend(
        [
            "",
            "### Four-horizon tile macro",
            "",
            "| Budget | Top-k overlap ↑ | Human-half normalized coverage ↑ | Start-template normalized coverage ↑ |",
            "|---:|---:|---:|---:|",
        ]
    )
    for budget in result["budgets_percent"]:
        key = f"{float(budget):g}"
        lines.append(
            f"| {key}% | {_ci(macro[f'tile_overlap_{key}'])} | "
            f"{_ci(macro[f'tile_normalized_coverage_{key}'])} | "
            f"{_ci(macro[f'tile_template_normalized_coverage_{key}'])} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "The audit addresses label reproducibility and whether horizon-specific targets are needed. It does not demonstrate that SphereRate uses image content or outperforms a start prior; those claims require wrong-image, matched-template, benchmark, and codec evidence. Spearman–Brown CC is an approximate full-cohort spatial-map reliability estimate, not a behavioral test–retest coefficient.",
            "",
            "## Provenance",
            "",
            f"- manifest: `{result['manifest']['path']}` (`{result['manifest']['sha256']}`)",
            f"- test target/fixation files loaded: `{result['access_audit']['test_target_or_fixation_files_loaded']}`",
            f"- non-monotonic observer sequences: `{result['observer_audit']['non_monotonic_observer_sequences']}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit current spherical-2-degree Salient360 target reliability."
    )
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--template-root", default=DEFAULT_TEMPLATE_ROOT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--sigma-deg", type=float, default=2.0)
    parser.add_argument("--observer-splits", type=int, default=20)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--budgets", nargs="+", type=float, default=[5.0, 10.0, 20.0])
    parser.add_argument("--tile-rows", type=int, default=8)
    parser.add_argument("--tile-cols", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    manifest = Path(args.manifest)
    all_rows = _read_manifest(manifest)
    selected_splits = {str(value) for value in args.splits}
    rows = [row for row in all_rows if str(row["split"]) in selected_splits]
    if not rows:
        raise ValueError(f"No rows selected for splits {sorted(selected_splits)}")
    horizons = (0.5, 1.0, 3.0, 5.0)
    payloads = [
        {
            "row": row,
            "template_root": str(args.template_root),
            "horizons_s": horizons,
            "height": int(args.height),
            "width": int(args.width),
            "sigma_deg": float(args.sigma_deg),
            "observer_splits": int(args.observer_splits),
            "budgets_percent": tuple(float(value) for value in args.budgets),
            "tile_rows": int(args.tile_rows),
            "tile_cols": int(args.tile_cols),
            "seed": int(args.seed),
        }
        for row in rows
    ]

    evaluated: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
        futures = {executor.submit(_evaluate_one, payload): payload["row"] for payload in payloads}
        for index, future in enumerate(as_completed(futures), start=1):
            value = future.result()
            evaluated.append(value)
            print(f"[{index:02d}/{len(futures)}] {value['image_id']}", flush=True)
    evaluated.sort(key=lambda value: value["image_id"])
    metric_rows = [row for value in evaluated for row in value["rows"]]
    aggregates = aggregate_reliability(
        metric_rows,
        primary_horizons_s=horizons,
        iterations=int(args.bootstrap_iterations),
        seed=int(args.seed),
    )
    split_counts = {
        split: sum(str(row["split"]) == split for row in rows)
        for split in sorted(selected_splits)
    }
    result = {
        "protocol": "salient360_standard_v3_target_reliability_spherical2deg",
        "status": "EVIDENCE_COMPLETE",
        "content_count": len(rows),
        "split_counts": split_counts,
        "horizons_s": list(horizons),
        "budgets_percent": [float(value) for value in args.budgets],
        "observer_splits": int(args.observer_splits),
        "bootstrap_iterations": int(args.bootstrap_iterations),
        "seed": int(args.seed),
        "map": {
            "height": int(args.height),
            "width": int(args.width),
            "renderer": "sphere_chord",
            "sigma_deg": float(args.sigma_deg),
            "truncate_sigma": 4.0,
            "area_normalized": True,
        },
        "tile_grid": [int(args.tile_rows), int(args.tile_cols)],
        "eye_policy": "right_eye_primary; left_right_never_pooled_as_independent",
        "aggregates": aggregates,
        "observer_audit": _aggregate_audit([value["audit"] for value in evaluated]),
        "manifest": {"path": str(manifest), "sha256": _sha256(manifest)},
        "access_audit": {
            "manifest_rows_seen": len(all_rows),
            "selected_rows": len(rows),
            "test_metadata_rows_seen": sum(str(row["split"]) == "test" for row in all_rows),
            "test_target_or_fixation_files_loaded": 0,
        },
    }
    output = ensure_dir(args.output_dir)
    (output / "results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _write_csv(metric_rows, output / "per_content.csv")
    _write_summary(result, output / "summary.md")
    print(json.dumps({"status": result["status"], "report": str(output / 'summary.md')}, indent=2))


if __name__ == "__main__":
    main()
