#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sphererate.config import load_config
from sphererate.data.early_audit import audit_early_horizon_view
from sphererate.data.early_horizon import build_cumulative_density, load_salient360_eye_fixations
from sphererate.data.early_view import (
    build_early_horizon_view,
    build_train_only_baselines,
    read_manifest,
)
from sphererate.utils import ensure_dir


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _independent_target_check(
    source_rows: list[dict[str, Any]],
    view_rows: list[dict[str, Any]],
    *,
    horizons: tuple[float, ...],
    height: int,
    width: int,
    sigma_px: float | None,
    renderer: str,
    sigma_deg: float | None,
) -> dict[str, Any]:
    source_by_id = {str(row["image_id"]): row for row in source_rows}
    max_error = 0.0
    checked = 0
    for row in view_rows:
        if row["split"] == "test":
            continue
        source = source_by_id[str(row["image_id"])]
        fixations = load_salient360_eye_fixations(source["raw_scanpath_r"])
        with np.load(row["target"]) as target:
            saved = np.asarray(target["cumulative"], dtype=np.float32)
        for index, horizon in enumerate(horizons):
            reference = build_cumulative_density(
                fixations,
                observer_ids=np.unique(fixations[:, 3]).astype(int).tolist(),
                horizon_s=horizon,
                height=height,
                width=width,
                sigma_px=sigma_px,
                renderer=renderer,
                sigma_deg=sigma_deg,
            )
            max_error = max(max_error, float(np.max(np.abs(reference - saved[index]))))
        checked += 1
    return {"checked_non_test_contents": checked, "max_abs_error": max_error, "pass": max_error <= 1e-7}


def _write_summary(result: dict[str, Any], path: Path) -> None:
    audit = result["view_audit"]
    lines = [
        "# Gate EH2: Early-Horizon Data and Leakage Audit",
        "",
        f"Decision: **{result['decision']}**",
        "",
        "No candidate neural-network training was performed.",
        "",
        "## Data view",
        "",
        f"- rows: `{audit['row_count']}`; splits: `{audit['split_counts']}`",
        f"- non-test targets ready: `{audit['row_count'] - audit['split_counts'].get('test', 0)}`",
        f"- sealed outer-test rows: `{audit['split_counts'].get('test', 0)}`",
        f"- outer-test target files loaded: `{audit['outer_test_target_files_loaded']}`",
        f"- physical observers per content: `{audit['physical_observer_count']}`",
        "- binocular policy: right eye primary; left eye sensitivity only; never pooled",
        "",
        "## Integrity",
        "",
        f"- canonical-content zero leakage: `{audit['zero_leakage']}`",
        f"- invalid spherical integrals: `{audit['invalid_spherical_integrals']}`",
        f"- invalid horizon files: `{audit['invalid_horizon_files']}`",
        f"- independent target rebuild max error: `{result['independent_target_check']['max_abs_error']:.3e}`",
        "",
        "## Frozen baselines",
        "",
        "- train-only early template: leave-one-content-out on train contents; all 50 train contents elsewhere",
        "- uniform and equator priors",
        f"- strong static checkpoint status: `{result['strong_static_checkpoint']['status']}`",
        f"- strong static checkpoint: `{result['strong_static_checkpoint']['path']}`",
        f"- strong static SHA256: `{result['strong_static_checkpoint']['sha256']}`",
        "- baseline outer-test predictions: `0`",
        "",
        "## Gate checks",
        "",
    ]
    lines.extend(f"- {name}: **{'PASS' if passed else 'FAIL'}**" for name, passed in result["checks"].items())
    lines.extend(
        [
            "",
            "## Decision consequence",
            "",
            "EH3 single-seed minimal-model pilot is authorized only if every check passes.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and audit the frozen EH2 early-horizon view.")
    parser.add_argument("--config", default="configs/early_horizon_256x512.yaml")
    parser.add_argument("--output-dir", default="reports/gate_eh2_data_audit")
    args = parser.parse_args()

    config = load_config(args.config)
    data = config["data"]
    height, width = map(int, data["image_size"])
    horizons = tuple(float(value) for value in data["horizons_s"])
    source_manifest = Path(config["paths"]["source_manifest"])
    view_root = Path(config["paths"]["processed_root"]) / data["processed_name"]
    build_early_horizon_view(
        source_manifest=source_manifest,
        out_root=view_root,
        horizons_s=horizons,
        height=height,
        width=width,
        sigma_px=None if data.get("sigma_px") is None else float(data["sigma_px"]),
        renderer=str(data.get("renderer", "legacy_erp")),
        sigma_deg=None if data.get("sigma_deg") is None else float(data["sigma_deg"]),
        target_version=int(data.get("target_version", 1)),
        primary_eye=str(data["primary_eye"]),
    )
    baseline_root = build_train_only_baselines(view_root, view_root / "baselines")
    audit = audit_early_horizon_view(view_root, expected_horizons_s=horizons)
    source_rows = read_manifest(source_manifest)
    view_rows = read_manifest(view_root / "manifest.jsonl")
    independent = _independent_target_check(
        source_rows,
        view_rows,
        horizons=horizons,
        height=height,
        width=width,
        sigma_px=None if data.get("sigma_px") is None else float(data["sigma_px"]),
        renderer=str(data.get("renderer", "legacy_erp")),
        sigma_deg=None if data.get("sigma_deg") is None else float(data["sigma_deg"]),
    )
    baseline_metadata = json.loads((baseline_root / "metadata.json").read_text())
    checkpoint_value = config["baseline"].get("strong_static_checkpoint")
    checkpoint_status = str(config["baseline"].get("strong_static_status", "frozen"))
    checkpoint = None if checkpoint_value in {None, ""} else Path(str(checkpoint_value))
    if checkpoint_status == "frozen" and (checkpoint is None or not checkpoint.is_file()):
        raise FileNotFoundError(f"Missing frozen strong static checkpoint: {checkpoint}")
    if checkpoint_status not in {"frozen", "pending_v2_retrain"}:
        raise ValueError(f"Unknown strong_static_status: {checkpoint_status}")
    checks = {
        "view_integrity": bool(audit["pass"]),
        "content_zero_leakage": bool(audit["zero_leakage"]),
        "outer_test_targets_sealed": bool(audit["outer_test_target_files_loaded"] == 0),
        "independent_target_rebuild": bool(independent["pass"]),
        "train_only_baselines": bool(
            baseline_metadata["test_prediction_count"] == 0
            and baseline_metadata["train_template_policy"] == "leave_one_content_out"
        ),
        "strong_static_baseline_status_declared": bool(
            (checkpoint_status == "frozen" and checkpoint is not None and checkpoint.is_file())
            or checkpoint_status == "pending_v2_retrain"
        ),
    }
    result = {
        "gate": "EH2-data-view-and-leakage-audit",
        "decision": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "view_root": str(view_root),
        "view_audit": audit,
        "independent_target_check": independent,
        "baseline_metadata": baseline_metadata,
        "strong_static_checkpoint": {
            "status": checkpoint_status,
            "path": None if checkpoint is None else str(checkpoint),
            "sha256": None if checkpoint is None else _sha256(checkpoint),
        },
        "source_manifest": {"path": str(source_manifest), "sha256": _sha256(source_manifest)},
        "outer_test_used": False,
        "candidate_training_performed": False,
    }
    output_dir = ensure_dir(args.output_dir)
    (output_dir / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_summary(result, output_dir / "summary.md")
    print(json.dumps({"decision": result["decision"], "report": str(output_dir / "summary.md")}, indent=2))


if __name__ == "__main__":
    main()
