#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sphererate.baselines.priors import _prior_map
from sphererate.data.early_horizon import build_cumulative_density, load_salient360_eye_fixations
from sphererate.data.maps import MapGeometry
from sphererate.evaluation.early_reliability import load_non_test_manifest
from sphererate.evaluation.tile_prefetch import (
    aggregate_tile_oracle,
    decide_eh1,
    evaluate_content_tile_oracle,
)
from sphererate.utils import ensure_dir, normalize_np


def _build_template_maps(
    rows: list[dict[str, Any]],
    fixations_by_content: dict[str, np.ndarray],
    *,
    horizons: tuple[float, ...],
    height: int,
    width: int,
    sigma_px: float,
) -> tuple[dict[str, dict[float, np.ndarray]], dict[str, Any]]:
    train_rows = [row for row in rows if row["split"] == "train"]
    if len(train_rows) != 50:
        raise ValueError(f"EH1 requires 50 train contents for templates, found {len(train_rows)}")
    area = MapGeometry(height, width).area_weights
    maps_by_content: dict[str, dict[float, np.ndarray]] = {}
    totals = {horizon: np.zeros((height, width), dtype=np.float64) for horizon in horizons}
    for row in train_rows:
        content = str(row["canonical_content_id"])
        fixations = fixations_by_content[content]
        observer_ids = np.unique(fixations[:, 3]).astype(int).tolist()
        maps_by_content[content] = {}
        for horizon in horizons:
            density = build_cumulative_density(
                fixations,
                observer_ids=observer_ids,
                horizon_s=horizon,
                height=height,
                width=width,
                sigma_px=sigma_px,
            )
            maps_by_content[content][horizon] = density
            totals[horizon] += density

    templates: dict[str, dict[float, np.ndarray]] = {}
    train_ids = set(maps_by_content)
    for row in rows:
        content = str(row["canonical_content_id"])
        templates[content] = {}
        for horizon in horizons:
            if content in train_ids:
                mean = (totals[horizon] - maps_by_content[content][horizon]) / float(len(train_rows) - 1)
                source_count = len(train_rows) - 1
            else:
                mean = totals[horizon] / float(len(train_rows))
                source_count = len(train_rows)
            templates[content][horizon] = normalize_np(mean.astype(np.float32), weights=area)
        templates[content]["source_count"] = source_count  # type: ignore[index]
    return templates, {
        "train_content_count": len(train_rows),
        "train_eval_policy": "leave_one_content_out_49",
        "non_train_eval_policy": "all_train_contents_50",
    }


def _coverage_means(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for budget in sorted({float(row["budget_percent"]) for row in rows}):
        out[f"{budget:g}"] = {}
        for horizon in sorted({float(row["horizon_s"]) for row in rows}):
            current = [
                row for row in rows
                if float(row["budget_percent"]) == budget and float(row["horizon_s"]) == horizon
            ]
            out[f"{budget:g}"][f"{horizon:g}"] = {
                name: float(np.mean([float(row[name]) for row in current]))
                for name in (
                    "same_coverage",
                    "static_coverage",
                    "template_coverage",
                    "equator_coverage",
                    "uniform_coverage",
                )
            }
    return out


def _format_ci(result: dict[str, Any]) -> str:
    return f"{result['mean']:.6f} [{result['ci95_low']:.6f}, {result['ci95_high']:.6f}]"


def _write_summary(result: dict[str, Any], path: Path) -> None:
    primary = result["eyes"][result["primary_eye"]]
    aggregate = primary["aggregates"]["primary"]
    coverage = primary["coverage_means"]["10"]
    lines = [
        "# Gate EH1: Tile-Prefetch Human-Oracle Headroom",
        "",
        f"Decision: **{result['decision']}**",
        "",
        "No neural-network training was performed. Outer-test target rows loaded: "
        f"**{result['access_audit']['outer_test_target_rows_loaded']}**.",
        "",
        "## Frozen application setting",
        "",
        f"- tile grid: `{result['tile_grid'][0]}×{result['tile_grid'][1]}`",
        f"- budgets: `{result['budgets_percent']}`%; primary: `10%`",
        "- tile counts use ceil: `7/13/26` of 128 tiles",
        "- ranking uses spherical probability mass; coverage uses raw clipped fixation dwell",
        "- right eye is primary; left eye is sensitivity; eyes are never pooled",
        "- train-content template evaluation is leave-one-content-out",
        "",
        "## Primary 10% budget coverage",
        "",
        "| Horizon | Same-horizon | Static-final | Early template | Same−static | Same−template |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for horizon, metrics in aggregate["per_horizon"].items():
        means = coverage[horizon]
        lines.append(
            f"| {horizon}s | {means['same_coverage']:.6f} | {means['static_coverage']:.6f} | "
            f"{means['template_coverage']:.6f} | {_format_ci(metrics['same_vs_static_improvement'])} | "
            f"{_format_ci(metrics['same_vs_template_improvement'])} |"
        )
    macro = aggregate["early_macro"]
    lines.extend(
        [
            "",
            "## Early-macro improvements",
            "",
            f"- same vs static: {_format_ci(macro['same_vs_static_improvement'])}",
            f"- same vs train-only early template: {_format_ci(macro['same_vs_template_improvement'])}",
            "",
            "## Gate checks",
            "",
        ]
    )
    lines.extend(f"- {name}: **{'PASS' if passed else 'FAIL'}**" for name, passed in result["checks"].items())
    left = result["eyes"][result["sensitivity_eye"]]["aggregates"]["primary"]["early_macro"]
    lines.extend(
        [
            "",
            "## Left-eye sensitivity",
            "",
            f"- same vs static: {_format_ci(left['same_vs_static_improvement'])}",
            f"- same vs template: {_format_ci(left['same_vs_template_improvement'])}",
            "",
            "## Decision consequence",
            "",
            "EH2 is authorized only if all checks pass. A failure closes the tile-prefetch application claim before model training.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate frozen EH1 tile-prefetch human oracle.")
    parser.add_argument("--manifest", default="data/processed/p2_salient360_temporal_aux_seed20260711/manifest.jsonl")
    parser.add_argument("--output-dir", default="reports/gate_eh1_tile_oracle")
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--sigma-px", type=float, default=2.5)
    parser.add_argument("--observer-splits", type=int, default=20)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260711)
    args = parser.parse_args()

    rows, access_audit = load_non_test_manifest(args.manifest)
    if len(rows) != 72:
        raise ValueError(f"EH1 requires 72 non-test contents, found {len(rows)}")
    horizons = (0.5, 1.0, 3.0, 5.0)
    budgets = (5.0, 10.0, 20.0)
    height, width = int(args.height), int(args.width)
    priors = {
        "uniform": _prior_map("uniform", height, width),
        "equator": _prior_map("equator", height, width),
    }
    eye_specs = {"R": "raw_scanpath_r", "L": "raw_scanpath_l"}
    eyes: dict[str, Any] = {}
    csv_rows: list[dict[str, Any]] = []

    for eye, path_key in eye_specs.items():
        fixations_by_content = {
            str(row["canonical_content_id"]): load_salient360_eye_fixations(row[path_key]) for row in rows
        }
        templates, template_audit = _build_template_maps(
            rows,
            fixations_by_content,
            horizons=horizons,
            height=height,
            width=width,
            sigma_px=float(args.sigma_px),
        )
        metrics: list[dict[str, Any]] = []
        for index, row in enumerate(rows, start=1):
            content = str(row["canonical_content_id"])
            content_templates = {horizon: templates[content][horizon] for horizon in horizons}
            current = evaluate_content_tile_oracle(
                content_id=content,
                fixations=fixations_by_content[content],
                templates=content_templates,
                priors=priors,
                horizons_s=horizons,
                budgets_percent=budgets,
                height=height,
                width=width,
                sigma_px=float(args.sigma_px),
                tile_rows=8,
                tile_cols=16,
                observer_splits=int(args.observer_splits),
                seed=int(args.seed),
            )
            for item in current:
                item.update({"eye": eye, "split": str(row["split"]), "image_id": str(row["image_id"])})
            metrics.extend(current)
            print(f"[{eye}] {index:02d}/{len(rows)} {row['image_id']}", flush=True)
        aggregates = aggregate_tile_oracle(
            metrics,
            primary_budget_percent=10.0,
            primary_horizons_s=(0.5, 1.0, 3.0),
            iterations=int(args.bootstrap_iterations),
            seed=int(args.seed),
        )
        eyes[eye] = {
            "aggregates": aggregates,
            "coverage_means": _coverage_means(metrics),
            "template_audit": template_audit,
        }
        csv_rows.extend(metrics)

    decision = decide_eh1(
        eyes["R"]["aggregates"],
        outer_test_used=bool(access_audit["outer_test_target_rows_loaded"]),
    )
    result = {
        "gate": "EH1-tile-prefetch-human-oracle",
        **decision,
        "primary_eye": "R",
        "sensitivity_eye": "L",
        "eyes": eyes,
        "horizons_s": list(horizons),
        "primary_horizons_s": [0.5, 1.0, 3.0],
        "tile_grid": [8, 16],
        "budgets_percent": list(budgets),
        "primary_budget_percent": 10.0,
        "budget_rounding": "ceil",
        "coverage": "raw_clipped_fixation_dwell_fraction",
        "observer_splits": int(args.observer_splits),
        "bootstrap_iterations": int(args.bootstrap_iterations),
        "seed": int(args.seed),
        "map": {"height": height, "width": width, "sigma_px": float(args.sigma_px)},
        "access_audit": access_audit,
        "outer_test_used": False,
    }
    output_dir = ensure_dir(args.output_dir)
    with (output_dir / "results.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    _write_csv(csv_rows, output_dir / "per_content.csv")
    _write_summary(result, output_dir / "summary.md")
    print(json.dumps({"decision": result["decision"], "report": str(output_dir / "summary.md")}, indent=2))


if __name__ == "__main__":
    main()
