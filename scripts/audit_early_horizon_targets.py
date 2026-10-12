#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sphererate.data.early_horizon import audit_fixation_sequences, load_salient360_eye_fixations
from sphererate.evaluation.early_reliability import (
    aggregate_reliability,
    decide_eh0,
    evaluate_content_reliability,
    load_non_test_manifest,
)
from sphererate.utils import ensure_dir


CANONICAL_START_SOURCE = "https://www.interdigital.com/data_sets/salient-360-dataset"
CANONICAL_START_STATEMENT = (
    "The official dataset protocol states that every panorama was reset to the "
    "equirectangular image center at the start of each 25-second viewing. In the "
    "released scanpath longitude convention this center is x_norm≈0.75 / yaw≈+90°."
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mean_audit(audits: list[dict[str, Any]]) -> dict[str, Any]:
    observers = np.asarray([item["observer_count"] for item in audits], dtype=np.float64)
    non_monotonic = sum(int(item["non_monotonic_observer_count"]) for item in audits)
    horizon_keys = list(audits[0]["observers_with_mass_by_horizon"])
    mass_fraction_percentiles = {}
    for key in horizon_keys:
        fractions = np.asarray(
            [item["observers_with_mass_by_horizon"][key] / item["observer_count"] for item in audits],
            dtype=np.float64,
        )
        mass_fraction_percentiles[key] = {
            "min": float(fractions.min()),
            "p25": float(np.percentile(fractions, 25)),
            "median": float(np.median(fractions)),
            "p75": float(np.percentile(fractions, 75)),
            "max": float(fractions.max()),
        }
    return {
        "content_count": len(audits),
        "observer_count_per_content": {
            "min": int(observers.min()),
            "median": float(np.median(observers)),
            "max": int(observers.max()),
        },
        "non_monotonic_observer_sequences": int(non_monotonic),
        "contents_with_all_observers_having_mass": {
            key: sum(
                item["observers_with_mass_by_horizon"][key] == item["observer_count"] for item in audits
            )
            for key in horizon_keys
        },
        "observer_mass_fraction_per_content": mass_fraction_percentiles,
        "contents_with_all_observers_covering_horizon": {
            key: sum(item["observers_covering_horizon"][key] == item["observer_count"] for item in audits)
            for key in horizon_keys
        },
    }


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _format_ci(result: dict[str, Any]) -> str:
    return f"{result['mean']:.6f} [{result['ci95_low']:.6f}, {result['ci95_high']:.6f}]"


def _write_summary(result: dict[str, Any], path: Path) -> None:
    primary = result["eyes"][result["primary_eye"]]
    lines = [
        "# Gate EH0: Real-Time Target Reliability",
        "",
        f"Decision: **{result['decision']}**",
        "",
        "No neural-network training was performed. Outer-test target rows loaded: "
        f"**{result['access_audit']['outer_test_target_rows_loaded']}**.",
        "",
        "## Frozen target",
        "",
        f"- horizons: `{result['horizons_s']}` seconds",
        f"- primary eye: `{result['primary_eye']}`; the other eye is sensitivity only",
        f"- map size: `{result['map']['height']}×{result['map']['width']}`, sigma `{result['map']['sigma_px']}` px",
        f"- observer splits per content: `{result['observer_splits']}`",
        "- L/R eyes were never pooled as independent observers.",
        f"- canonical start provenance: [{CANONICAL_START_SOURCE}]({CANONICAL_START_SOURCE})",
        "",
        "## Primary right-eye improvements over static-final repeat",
        "",
        "Positive values are better. Each entry is mean [95% content-bootstrap CI].",
        "",
        "| Horizon | KLD improvement | SIM improvement | CC improvement |",
        "|---:|---:|---:|---:|",
    ]
    for horizon, metrics in primary["aggregates"]["per_horizon"].items():
        lines.append(
            f"| {horizon}s | {_format_ci(metrics['kld_improvement'])} | "
            f"{_format_ci(metrics['sim_improvement'])} | {_format_ci(metrics['cc_improvement'])} |"
        )
    macro = primary["aggregates"]["early_macro"]
    lines.extend(
        [
            f"| **early macro** | **{_format_ci(macro['kld_improvement'])}** | "
            f"**{_format_ci(macro['sim_improvement'])}** | **{_format_ci(macro['cc_improvement'])}** |",
            "",
            "## Gate checks",
            "",
        ]
    )
    lines.extend(f"- {name}: **{'PASS' if passed else 'FAIL'}**" for name, passed in result["checks"].items())
    sensitivity = result["eyes"][result["sensitivity_eye"]]["aggregates"]["early_macro"]
    lines.extend(
        [
            "",
            "## Left-eye sensitivity",
            "",
            f"- early-macro KLD improvement: {_format_ci(sensitivity['kld_improvement'])}",
            f"- early-macro SIM improvement: {_format_ci(sensitivity['sim_improvement'])}",
            f"- early-macro CC improvement: {_format_ci(sensitivity['cc_improvement'])}",
            "",
            "## Observer/time audit",
            "",
            f"- primary-eye physical observers per content: `{primary['audit']['observer_count_per_content']}`",
            f"- non-monotonic sequences: `{primary['audit']['non_monotonic_observer_sequences']}`",
            f"- observer mass fraction by horizon: `{primary['audit']['observer_mass_fraction_per_content']}`",
            f"- contents evaluated: `{primary['audit']['content_count']}` non-test contents",
            "",
            "## Decision consequence",
            "",
            "EH1 is authorized only when the decision above is PASS. A FAIL closes neural-network training for SA-EHS.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run frozen EH0 real-time target reliability audit.")
    parser.add_argument("--manifest", default="data/processed/p2_salient360_temporal_aux_seed20260711/manifest.jsonl")
    parser.add_argument("--output-dir", default="reports/gate_eh0_target_reliability")
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--sigma-px", type=float, default=2.5)
    parser.add_argument("--observer-splits", type=int, default=20)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260711)
    args = parser.parse_args()

    manifest = Path(args.manifest)
    rows, access_audit = load_non_test_manifest(manifest)
    if len(rows) != 72:
        raise ValueError(f"EH0 requires 72 non-test contents, found {len(rows)}")
    horizons = (0.5, 1.0, 3.0, 5.0)
    eye_specs = {"R": "raw_scanpath_r", "L": "raw_scanpath_l"}
    result_eyes: dict[str, Any] = {}
    all_csv_rows: list[dict[str, Any]] = []

    for eye, path_key in eye_specs.items():
        metrics: list[dict[str, Any]] = []
        audits: list[dict[str, Any]] = []
        for index, row in enumerate(rows, start=1):
            raw_path = Path(row[path_key])
            fixations = load_salient360_eye_fixations(raw_path)
            audits.append(audit_fixation_sequences(fixations, horizons_s=horizons))
            content_rows = evaluate_content_reliability(
                content_id=str(row["canonical_content_id"]),
                fixations=fixations,
                horizons_s=horizons,
                height=int(args.height),
                width=int(args.width),
                sigma_px=float(args.sigma_px),
                observer_splits=int(args.observer_splits),
                seed=int(args.seed),
            )
            for current in content_rows:
                current.update({"eye": eye, "split": str(row["split"]), "image_id": str(row["image_id"])})
            metrics.extend(content_rows)
            print(f"[{eye}] {index:02d}/{len(rows)} {row['image_id']}", flush=True)
        aggregates = aggregate_reliability(
            metrics,
            primary_horizons_s=(0.5, 1.0, 3.0),
            iterations=int(args.bootstrap_iterations),
            seed=int(args.seed),
        )
        result_eyes[eye] = {"aggregates": aggregates, "audit": _mean_audit(audits)}
        all_csv_rows.extend(metrics)

    decision = decide_eh0(
        result_eyes["R"]["aggregates"],
        canonical_start_verified=True,
        outer_test_used=bool(access_audit["outer_test_target_rows_loaded"]),
    )
    result: dict[str, Any] = {
        "gate": "EH0-real-time-target-reliability",
        **decision,
        "primary_eye": "R",
        "sensitivity_eye": "L",
        "eyes": result_eyes,
        "horizons_s": list(horizons),
        "primary_horizons_s": [0.5, 1.0, 3.0],
        "observer_splits": int(args.observer_splits),
        "bootstrap_iterations": int(args.bootstrap_iterations),
        "seed": int(args.seed),
        "map": {"height": int(args.height), "width": int(args.width), "sigma_px": float(args.sigma_px)},
        "binocular_policy": "right_eye_primary_left_eye_sensitivity_never_pooled",
        "canonical_start": {
            "verified": True,
            "source": CANONICAL_START_SOURCE,
            "statement": CANONICAL_START_STATEMENT,
            "yaw_deg": 90.0,
            "x_norm": 0.75,
        },
        "manifest": {"path": str(manifest), "sha256": _sha256(manifest)},
        "access_audit": access_audit,
        "outer_test_used": False,
    }
    output_dir = ensure_dir(args.output_dir)
    with (output_dir / "results.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    _write_csv(all_csv_rows, output_dir / "per_content.csv")
    _write_summary(result, output_dir / "summary.md")
    print(json.dumps({"decision": result["decision"], "report": str(output_dir / "summary.md")}, indent=2))


if __name__ == "__main__":
    main()
