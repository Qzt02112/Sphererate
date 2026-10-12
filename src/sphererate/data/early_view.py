from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from sphererate.baselines.priors import _prior_map
from sphererate.data.early_horizon import build_cumulative_density, load_salient360_eye_fixations
from sphererate.data.maps import MapGeometry, fixation_binary_map
from sphererate.utils import ensure_dir, normalize_np, write_jsonl


def read_manifest(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def build_early_horizon_view(
    *,
    source_manifest: str | Path,
    out_root: str | Path,
    horizons_s: Sequence[float],
    height: int,
    width: int,
    sigma_px: float | None = None,
    renderer: str = "legacy_erp",
    sigma_deg: float | None = None,
    target_version: int = 1,
    primary_eye: str = "R",
    seal_test_targets: bool = False,
) -> Path:
    source_rows = read_manifest(source_manifest)
    out_root = ensure_dir(out_root)
    targets_dir = ensure_dir(out_root / "targets")
    fixations_dir = ensure_dir(out_root / "fixations")
    geometry = MapGeometry(int(height), int(width))
    eye = str(primary_eye).upper()
    if eye not in {"R", "L"}:
        raise ValueError("primary_eye must be R or L")
    path_key = "raw_scanpath_r" if eye == "R" else "raw_scanpath_l"
    horizons = tuple(float(value) for value in horizons_s)
    output_rows: list[dict[str, Any]] = []
    ready = 0
    sealed = 0

    for source in source_rows:
        row = {
            "dataset": "salient360_early_horizon",
            "image_id": str(source["image_id"]),
            "canonical_content_id": str(source.get("canonical_content_id") or f"salient360:{source['image_id']}"),
            "group_id": str(source.get("group_id") or f"content:{source.get('canonical_content_id', source['image_id'])}"),
            "split": str(source["split"]),
            "outer_split": source.get("outer_split"),
            "image": str(source["image"]),
            "declared_start_yaw_deg": 90.0,
            "declared_start_source": "Salient360 official centered ERP reset",
            "primary_eye": eye,
            "horizons_s": list(horizons),
            "target_provenance": (
                f"raw_single_eye_scanpaths_real_timestamp_clipped_dwell_{str(renderer).lower()}"
            ),
        }
        if row["split"] == "test" and bool(seal_test_targets):
            row.update({"target": None, "fixations": None, "target_sealed": True})
            sealed += 1
            output_rows.append(row)
            continue

        fixations = load_salient360_eye_fixations(source[path_key])
        observer_ids = np.unique(fixations[:, 3]).astype(int).tolist()
        cumulative = np.stack(
            [
                build_cumulative_density(
                    fixations,
                    observer_ids=observer_ids,
                    horizon_s=horizon,
                    height=height,
                    width=width,
                    sigma_px=sigma_px,
                    renderer=renderer,
                    sigma_deg=sigma_deg,
                )
                for horizon in horizons
            ],
            axis=0,
        ).astype(np.float32)
        final = build_cumulative_density(
            fixations,
            observer_ids=observer_ids,
            horizon_s=None,
            height=height,
            width=width,
            sigma_px=sigma_px,
            renderer=renderer,
            sigma_deg=sigma_deg,
        )
        fixation = fixation_binary_map(fixations[:, 0], fixations[:, 1], height, width)
        target_path = targets_dir / f"{row['image_id']}.npz"
        fixation_path = fixations_dir / f"{row['image_id']}.npz"
        np.savez_compressed(
            target_path,
            cumulative=cumulative,
            final=final.astype(np.float32),
            fixation=fixation.astype(np.float32),
            area_weights=geometry.area_weights,
            horizons_s=np.asarray(horizons, dtype=np.float32),
        )
        np.savez_compressed(
            fixation_path,
            fixations=fixations.astype(np.float32),
            columns=np.array(["x_norm", "y_norm", "timestamp_s", "physical_observer_id", "duration_s", "legacy_bin"]),
            eye=np.array(eye),
        )
        row.update(
            {
                "target": str(target_path),
                "fixations": str(fixation_path),
                "target_sealed": False,
                "physical_observer_count": len(observer_ids),
            }
        )
        ready += 1
        output_rows.append(row)

    write_jsonl(output_rows, out_root / "manifest.jsonl")
    info = {
        "dataset": "salient360_early_horizon",
        "version": int(target_version),
        "source_manifest": str(source_manifest),
        "image_size": [int(height), int(width)],
        "renderer": str(renderer),
        "sigma_px": None if sigma_px is None else float(sigma_px),
        "sigma_deg": None if sigma_deg is None else float(sigma_deg),
        "kernel_distance": "unit_sphere_chord" if str(renderer).lower() in {"sphere_chord", "spherical_chord"} else "erp_pixel",
        "kernel_truncate_sigma": 4.0 if str(renderer).lower() in {"sphere_chord", "spherical_chord"} else 3.0,
        "horizons_s": list(horizons),
        "primary_eye": eye,
        "binocular_policy": "single_eye_primary; other_eye_sensitivity_only; never_pool_as_independent",
        "declared_start_yaw_deg": 90.0,
        "coordinate_note": "Official ERP image center maps to +90 degrees in the released Salient360 scanpath longitude convention (x_norm approximately 0.75).",
        "row_count": len(output_rows),
        "ready_target_count": ready,
        "sealed_outer_test_count": sealed,
        "split_counts": {
            split: sum(row["split"] == split for row in output_rows)
            for split in ("train", "val", "gate", "test")
        },
        "outer_test_target_files_loaded": sum(
            row["split"] == "test" and row.get("target") is not None for row in output_rows
        ),
        "test_access_policy": (
            "legacy_sealed" if bool(seal_test_targets) else "ordinary_repeatable_evaluation; never_model_selection"
        ),
    }
    (out_root / "dataset_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out_root


def build_train_only_baselines(
    view_root: str | Path,
    out_root: str | Path,
    *,
    include_outer_test: bool | None = None,
    authorization: str | None = None,
) -> Path:
    view_root = Path(view_root)
    out_root = ensure_dir(out_root)
    rows = read_manifest(view_root / "manifest.jsonl")
    del authorization  # Retained only for compatibility with historical callers.
    train = [row for row in rows if row["split"] == "train"]
    if include_outer_test is None:
        prediction_rows = [row for row in rows if row.get("target") is not None]
    elif bool(include_outer_test):
        prediction_rows = rows
    else:
        prediction_rows = [row for row in rows if row["split"] != "test"]
    if len(train) < 2:
        raise ValueError("At least two train contents are required for leave-one-out templates")
    loaded: dict[str, list[dict[str, np.ndarray]]] = {}
    for row in train:
        with np.load(row["target"]) as target:
            loaded.setdefault(str(row["canonical_content_id"]), []).append(
                {
                    "cumulative": np.asarray(target["cumulative"], dtype=np.float32),
                    "horizons_s": np.asarray(target["horizons_s"], dtype=np.float32),
                    "area_weights": np.asarray(target["area_weights"], dtype=np.float32),
                }
            )
    reference = next(iter(loaded.values()))[0]
    area = reference["area_weights"]
    horizons = reference["horizons_s"]
    total = np.sum(
        [item["cumulative"] for items in loaded.values() for item in items],
        axis=0,
        dtype=np.float64,
    )
    total_count = sum(len(items) for items in loaded.values())
    train_content_ids = sorted(loaded)
    template_dir = ensure_dir(out_root / "early_template")
    uniform_dir = ensure_dir(out_root / "uniform")
    equator_dir = ensure_dir(out_root / "equator")
    uniform = _prior_map("uniform", *area.shape)
    equator = _prior_map("equator", *area.shape)
    test_predictions = 0

    for row in prediction_rows:
        if row.get("target") is None:
            raise ValueError(f"Cannot build baseline for sealed target {row['image_id']}")
        content = str(row["canonical_content_id"])
        if content in loaded:
            held_out = np.sum(
                [item["cumulative"] for item in loaded[content]], axis=0, dtype=np.float64
            )
            source_count = total_count - len(loaded[content])
            if source_count <= 0:
                raise ValueError("Leave-one-content-out template has no source rows")
            template = (total - held_out) / float(source_count)
            source_content_ids = [value for value in train_content_ids if value != content]
        else:
            template = total / float(total_count)
            source_count = total_count
            source_content_ids = train_content_ids
        template = np.stack(
            [normalize_np(current.astype(np.float32), weights=area) for current in template], axis=0
        )
        common = {"horizons_s": horizons, "area_weights": area}
        np.savez_compressed(
            template_dir / f"{row['image_id']}.npz",
            cumulative=template,
            source_count=source_count,
            source_content_ids=np.asarray(source_content_ids),
            held_out_content_id=np.asarray(content if content in loaded else ""),
            **common,
        )
        np.savez_compressed(
            uniform_dir / f"{row['image_id']}.npz",
            cumulative=np.repeat(uniform[None], len(horizons), axis=0),
            **common,
        )
        test_predictions += int(row["split"] == "test")
        np.savez_compressed(
            equator_dir / f"{row['image_id']}.npz",
            cumulative=np.repeat(equator[None], len(horizons), axis=0),
            **common,
        )

    metadata = {
        "train_content_count": len(loaded),
        "train_condition_count": len(train),
        "prediction_count_per_baseline": len(prediction_rows),
        "test_prediction_count": test_predictions,
        "train_template_policy": "leave_one_content_out",
        "multi_condition_policy": "remove_all_rows_of_the_held_out_canonical_content",
        "non_train_template_policy": "all_train_contents",
        "horizons_s": horizons.tolist(),
        "baselines": ["early_template", "uniform", "equator"],
    }
    (out_root / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out_root


def materialize_eh5_outer_test(
    *,
    source_manifest: str | Path,
    view_root: str | Path,
    authorization: str,
) -> dict[str, Any]:
    if authorization != "EH5_OUTER_TEST_ONCE":
        raise PermissionError("Outer-test materialization requires EH5_OUTER_TEST_ONCE")
    view_root = Path(view_root)
    source_rows = {str(row["image_id"]): row for row in read_manifest(source_manifest)}
    rows = read_manifest(view_root / "manifest.jsonl")
    info_path = view_root / "dataset_info.json"
    info = json.loads(info_path.read_text())
    height, width = map(int, info["image_size"])
    renderer = str(info.get("renderer", "legacy_erp"))
    sigma_px = None if info.get("sigma_px") is None else float(info["sigma_px"])
    sigma_deg = None if info.get("sigma_deg") is None else float(info["sigma_deg"])
    horizons = tuple(float(value) for value in info["horizons_s"])
    geometry = MapGeometry(height, width)
    materialized = 0
    for row in rows:
        if row["split"] != "test":
            continue
        if not row.get("target_sealed"):
            raise RuntimeError(f"Outer-test row was already materialized: {row['image_id']}")
        source = source_rows[str(row["image_id"])]
        fixations = load_salient360_eye_fixations(source["raw_scanpath_r"])
        observer_ids = np.unique(fixations[:, 3]).astype(int).tolist()
        cumulative = np.stack(
            [
                build_cumulative_density(
                    fixations,
                    observer_ids=observer_ids,
                    horizon_s=horizon,
                    height=height,
                    width=width,
                    sigma_px=sigma_px,
                    renderer=renderer,
                    sigma_deg=sigma_deg,
                )
                for horizon in horizons
            ],
            axis=0,
        ).astype(np.float32)
        final = build_cumulative_density(
            fixations,
            observer_ids=observer_ids,
            horizon_s=None,
            height=height,
            width=width,
            sigma_px=sigma_px,
            renderer=renderer,
            sigma_deg=sigma_deg,
        )
        fixation = fixation_binary_map(fixations[:, 0], fixations[:, 1], height, width)
        target_path = view_root / "targets" / f"{row['image_id']}.npz"
        fixation_path = view_root / "fixations" / f"{row['image_id']}.npz"
        np.savez_compressed(
            target_path,
            cumulative=cumulative,
            final=final.astype(np.float32),
            fixation=fixation.astype(np.float32),
            area_weights=geometry.area_weights,
            horizons_s=np.asarray(horizons, dtype=np.float32),
        )
        np.savez_compressed(
            fixation_path,
            fixations=fixations.astype(np.float32),
            columns=np.array(["x_norm", "y_norm", "timestamp_s", "physical_observer_id", "duration_s", "legacy_bin"]),
            eye=np.array("R"),
        )
        row.update(
            {
                "target": str(target_path),
                "fixations": str(fixation_path),
                "target_sealed": False,
                "physical_observer_count": len(observer_ids),
                "materialized_stage": "EH5",
            }
        )
        materialized += 1
    if materialized != 13:
        raise ValueError(f"EH5 must materialize exactly 13 outer-test contents, found {materialized}")
    write_jsonl(rows, view_root / "manifest.jsonl")
    info.update(
        {
            "sealed_outer_test_count": 0,
            "materialized_outer_test_count": materialized,
            "outer_test_target_files_loaded": materialized,
            "outer_test_status": "materialized_once_at_EH5",
        }
    )
    info_path.write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {"materialized_outer_test_count": materialized, "outer_test_used": True}
