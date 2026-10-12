#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sphererate.config import load_config
from sphererate.data.early_view import build_early_horizon_view, build_train_only_baselines, read_manifest
from sphererate.data.p2_view import build_standard_salient360_view
from sphererate.utils import ensure_dir


EXPECTED_COUNTS = {"train": 59, "val": 13, "test": 13}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_projected_split(source: Path, destination: Path) -> None:
    payload = json.loads(source.read_text(encoding="utf-8"))
    assignments = {
        str(group): str(split)
        for group, split in payload["group_to_split"].items()
        if str(group).startswith("content:salient360:")
    }
    counts = Counter(assignments.values())
    actual = {split: int(counts.get(split, 0)) for split in EXPECTED_COUNTS}
    if actual != EXPECTED_COUNTS:
        raise ValueError(f"Canonical Salient360 split mismatch: {actual} != {EXPECTED_COUNTS}")
    projected = {
        "version": 3,
        "protocol": "salient360_standard_train59_val13_test13",
        "seed": payload.get("seed"),
        "source_split_file": str(source),
        "counts": actual,
        "split_roles": {
            "train": "fit",
            "val": "checkpoint_and_hyperparameter_selection",
            "test": "report",
        },
        "test_access_policy": "ordinary_repeatable_evaluation; never_model_selection",
        "group_to_split": dict(sorted(assignments.items())),
    }
    ensure_dir(destination.parent)
    destination.write_text(json.dumps(projected, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _audit(view_root: Path, horizons: tuple[float, ...]) -> dict[str, Any]:
    rows = read_manifest(view_root / "manifest.jsonl")
    counts = Counter(str(row["split"]) for row in rows)
    actual_counts = {split: int(counts.get(split, 0)) for split in EXPECTED_COUNTS}
    train_ids = {str(row["canonical_content_id"]) for row in rows if row["split"] == "train"}
    split_ids = {
        split: {str(row["canonical_content_id"]) for row in rows if row["split"] == split}
        for split in EXPECTED_COUNTS
    }
    zero_leakage = all(
        split_ids[left].isdisjoint(split_ids[right])
        for index, left in enumerate(split_ids)
        for right in list(split_ids)[index + 1 :]
    )
    missing: list[str] = []
    invalid_integrals: list[str] = []
    invalid_horizons: list[str] = []
    invalid_templates: list[str] = []
    observer_counts: list[int] = []
    expected_horizons = np.asarray(horizons, dtype=np.float32)
    template_root = view_root / "baselines" / "early_template"

    for row in rows:
        image_id = str(row["image_id"])
        for key in ("image", "target", "fixations"):
            value = row.get(key)
            if not value or not Path(str(value)).is_file():
                missing.append(f"{image_id}:{key}")
        if missing and any(value.startswith(f"{image_id}:") for value in missing):
            continue
        with np.load(str(row["target"])) as target:
            cumulative = np.asarray(target["cumulative"], dtype=np.float64)
            final = np.asarray(target["final"], dtype=np.float64)
            area = np.asarray(target["area_weights"], dtype=np.float64)
            saved_horizons = np.asarray(target["horizons_s"], dtype=np.float32)
        if saved_horizons.shape != expected_horizons.shape or not np.allclose(saved_horizons, expected_horizons):
            invalid_horizons.append(image_id)
        integrals = np.concatenate(((cumulative * area).sum(axis=(1, 2)), [(final * area).sum()]))
        if np.any(np.abs(integrals - 1.0) > 1e-4):
            invalid_integrals.append(image_id)
        with np.load(str(row["fixations"])) as fixation_file:
            observer_counts.append(len(np.unique(fixation_file["fixations"][:, 3])))

        template_path = template_root / f"{image_id}.npz"
        if not template_path.is_file():
            invalid_templates.append(f"{image_id}:missing")
            continue
        with np.load(template_path) as template:
            source_ids = {str(value) for value in template["source_content_ids"].tolist()}
            source_count = int(template["source_count"])
        expected_sources = train_ids - ({str(row["canonical_content_id"])} if row["split"] == "train" else set())
        if source_ids != expected_sources or source_count != len(expected_sources):
            invalid_templates.append(image_id)

    checks = {
        "counts_59_13_13": actual_counts == EXPECTED_COUNTS,
        "all_85_contents_present": len(rows) == 85 and len(set().union(*split_ids.values())) == 85,
        "canonical_content_zero_leakage": zero_leakage,
        "all_images_targets_fixations_readable": not missing,
        "all_targets_use_declared_horizons": not invalid_horizons,
        "all_spherical_density_integrals_valid": not invalid_integrals,
        "all_templates_train_only": not invalid_templates,
        "test_targets_materialized": all(row.get("target") and not row.get("target_sealed") for row in rows if row["split"] == "test"),
    }
    return {
        "decision": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "split_counts": actual_counts,
        "row_count": len(rows),
        "train_content_count": len(train_ids),
        "missing": missing,
        "invalid_horizon_contents": invalid_horizons,
        "invalid_integral_contents": invalid_integrals,
        "invalid_template_contents": invalid_templates,
        "observer_count": {
            "min": min(observer_counts) if observer_counts else None,
            "median": float(np.median(observer_counts)) if observer_counts else None,
            "max": max(observer_counts) if observer_counts else None,
        },
    }


def _write_summary(result: dict[str, Any], path: Path) -> None:
    audit = result["audit"]
    lines = [
        "# Salient360 V3 Standard-Split Data Audit",
        "",
        f"Decision: **{audit['decision']}**",
        "",
        "## Protocol",
        "",
        "- Train: former Train + former Val, 59 contents.",
        "- Validation: former Gate, 13 contents; used for checkpoint and hyperparameter selection.",
        "- Test: unchanged former Test, 13 contents; ordinary repeatable evaluation, never model selection.",
        "- Target: right-eye, timestamp-aware cumulative saliency, spherical 2-degree kernel.",
        "- Template: train-only; leave-one-content-out for train rows and all 59 train contents for validation/test.",
        "",
        "## Audit",
        "",
        f"- rows/splits: `{audit['row_count']}` / `{audit['split_counts']}`",
        f"- physical observers per content: `{audit['observer_count']}`",
    ]
    lines.extend(f"- {name}: **{'PASS' if passed else 'FAIL'}**" for name, passed in audit["checks"].items())
    lines.extend(
        [
            "",
            "## Provenance",
            "",
            f"- canonical split: `{result['files']['standard_split']['path']}` (`{result['files']['standard_split']['sha256']}`)",
            f"- source manifest: `{result['files']['source_manifest']['path']}` (`{result['files']['source_manifest']['sha256']}`)",
            f"- V3 manifest: `{result['files']['view_manifest']['path']}` (`{result['files']['view_manifest']['sha256']}`)",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and audit the standard Salient360 59/13/13 V3 view.")
    parser.add_argument("--config", default="configs/early_horizon_salient360_standard_v3.yaml")
    parser.add_argument("--output-dir", default="reports/standard_split_data_audit")
    args = parser.parse_args()

    config = load_config(args.config)
    paths = config["paths"]
    data = config["data"]
    raw_source = Path(paths["raw_source_manifest"])
    canonical_source = Path(paths["canonical_split_source"])
    standard_split = Path(paths["standard_split_file"])
    source_root = Path(paths["source_view_root"])
    view_root = Path(paths["processed_root"])
    horizons = tuple(float(value) for value in data["horizons_s"])

    _write_projected_split(canonical_source, standard_split)
    source_manifest = build_standard_salient360_view(raw_source, standard_split, source_root)
    build_early_horizon_view(
        source_manifest=source_manifest,
        out_root=view_root,
        horizons_s=horizons,
        height=int(data["image_size"][0]),
        width=int(data["image_size"][1]),
        sigma_px=None if data.get("sigma_px") is None else float(data["sigma_px"]),
        renderer=str(data["renderer"]),
        sigma_deg=None if data.get("sigma_deg") is None else float(data["sigma_deg"]),
        target_version=int(data["target_version"]),
        primary_eye=str(data["primary_eye"]),
        seal_test_targets=False,
    )
    build_train_only_baselines(view_root, view_root / "baselines")
    audit = _audit(view_root, horizons)
    result = {
        "protocol": "salient360_standard_train59_val13_test13",
        "audit": audit,
        "files": {
            "standard_split": {"path": str(standard_split), "sha256": _sha256(standard_split)},
            "source_manifest": {"path": str(source_manifest), "sha256": _sha256(source_manifest)},
            "view_manifest": {"path": str(view_root / 'manifest.jsonl'), "sha256": _sha256(view_root / 'manifest.jsonl')},
        },
    }
    output_dir = ensure_dir(args.output_dir)
    (output_dir / "audit.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_summary(result, output_dir / "summary.md")
    print(json.dumps({"decision": audit["decision"], "report": str(output_dir / 'summary.md')}, indent=2))
    if audit["decision"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
