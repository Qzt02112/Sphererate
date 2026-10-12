#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from scripts.aggregate_kernel_v2_matched_baselines import _split_methods
from sphererate.utils import ensure_dir


CONTROLS = (
    "Early template",
    "TempSAL-ERP",
    "GBVS360 + reachable",
    "Pathformer3D",
    "ScanDMM",
)
PRIMARY_HORIZONS = {0.5, 1.0, 3.0}


def _content_macro(rows: list[dict[str, Any]]) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        if float(row["horizon_s"]) not in PRIMARY_HORIZONS:
            continue
        grouped.setdefault(str(row["content_id"]), []).append(float(row["coverage"]))
    if not grouped or any(len(values) != 3 for values in grouped.values()):
        raise ValueError("Expected exactly three primary-horizon coverage rows per content")
    return {content: float(np.mean(values)) for content, values in grouped.items()}


def _paired_miss_reduction(
    candidate: dict[str, float],
    control: dict[str, float],
    *,
    seed: int = 20260711,
    iterations: int = 10000,
) -> dict[str, float | int]:
    contents = sorted(candidate)
    if set(contents) != set(control):
        raise ValueError("Candidate/control content IDs differ")
    cand = np.asarray([candidate[key] for key in contents], dtype=np.float64)
    base = np.asarray([control[key] for key in contents], dtype=np.float64)

    def statistic(indices: np.ndarray) -> float:
        cand_mean = float(cand[indices].mean())
        base_mean = float(base[indices].mean())
        return float((cand_mean - base_mean) / max(1.0 - base_mean, 1e-12))

    identity = np.arange(len(contents))
    point = statistic(identity)
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        bootstrap[index] = statistic(rng.integers(0, len(contents), size=len(contents)))
    return {
        "candidate_coverage": float(cand.mean()),
        "control_coverage": float(base.mean()),
        "absolute_coverage_gain": float(cand.mean() - base.mean()),
        "remaining_miss_mass_control": float(1.0 - base.mean()),
        "relative_miss_reduction": point,
        "ci95_low": float(np.quantile(bootstrap, 0.025)),
        "ci95_high": float(np.quantile(bootstrap, 0.975)),
        "contents": len(contents),
        "bootstrap_resamples": iterations,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Express fixed-budget gains as reduction of remaining miss mass.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="reports/kernel_v2_miss_reduction")
    args = parser.parse_args()
    methods, _ = _split_methods(
        "gate",
        "data/processed/early_horizon_salient360_spherical2deg_v2",
        torch.device(args.device),
    )
    candidate = _content_macro(methods["SphereHorizon"][1]["10.0"])
    comparisons = {
        control: _paired_miss_reduction(
            candidate,
            _content_macro(methods[control][1]["10.0"]),
        )
        for control in CONTROLS
    }
    result = {
        "metric": "relative reduction in uncovered gaze mass at 10% tile budget",
        "definition": "(coverage_SphereHorizon - coverage_control) / (1 - coverage_control)",
        "aggregation": "compute primary-horizon macro per content, then ratio of paired content means",
        "split": "historically_seen_gate",
        "comparisons": comparisons,
        "test_target_loads": 0,
    }
    output = ensure_dir(args.output_dir)
    (output / "results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    lines = [
        "# Remaining gaze-miss reduction at 10% tile budget",
        "",
        "This is a derived interpretation of the frozen coverage metric, not a replacement primary endpoint. It asks what fraction of the control's still-uncovered gaze mass is recovered by SphereHorizon.",
        "",
        "| Control | Control coverage | Absolute gain | Remaining miss | Relative miss reduction (95% CI) |",
        "|---|---:|---:|---:|---:|",
    ]
    for control, record in comparisons.items():
        lines.append(
            f"| {control} | {record['control_coverage']:.4f} | {record['absolute_coverage_gain']:.4f} | "
            f"{record['remaining_miss_mass_control']:.4f} | {100.0 * record['relative_miss_reduction']:.1f}% "
            f"[{100.0 * record['ci95_low']:.1f}%, {100.0 * record['ci95_high']:.1f}%] |"
        )
    lines.extend(
        [
            "",
            "Interpretation: the `+0.0173` absolute gain over the early template recovers a non-trivial fraction of the small residual miss mass, but it remains a development-gate result and must not be described as cross-dataset system-level savings.",
        ]
    )
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(output / "summary.md")}, indent=2))


if __name__ == "__main__":
    main()
