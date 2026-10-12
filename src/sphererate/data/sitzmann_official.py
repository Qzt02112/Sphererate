from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

from sphererate.data.early_horizon import build_cumulative_density
from sphererate.data.early_view import build_train_only_baselines
from sphererate.data.maps import MapGeometry, fixation_binary_map
from sphererate.utils import ensure_dir, write_jsonl


VIEWPOINT_YAW_DEG = {0: 0.0, 1: 90.0, 2: 180.0, 3: 270.0}
OFFICIAL_SCANPATH_TEST_SCENES = (0, 6, 9)


def alignment_shift_px(
    viewpoint_idx: int,
    width: int,
    *,
    reference_yaw_deg: float = 180.0,
) -> int:
    if int(viewpoint_idx) not in VIEWPOINT_YAW_DEG:
        raise ValueError(f"Unknown Sitzmann viewpoint_idx={viewpoint_idx}")
    delta = float(reference_yaw_deg) - VIEWPOINT_YAW_DEG[int(viewpoint_idx)]
    return int(round(delta / 360.0 * int(width)))


def align_fixations_to_reference(
    fixations: np.ndarray,
    viewpoint_idx: int,
    *,
    width: int,
    reference_yaw_deg: float = 180.0,
) -> np.ndarray:
    output = np.asarray(fixations, dtype=np.float32).copy()
    shift = alignment_shift_px(viewpoint_idx, width, reference_yaw_deg=reference_yaw_deg)
    output[:, 0] = np.mod(output[:, 0] + float(shift) / float(width), 1.0)
    return output


def make_content_split(
    scene_ids: Sequence[int],
    *,
    seed: int = 20260713,
    val_count: int = 3,
    official_test_scenes: Sequence[int] = OFFICIAL_SCANPATH_TEST_SCENES,
) -> dict[str, Any]:
    scenes = sorted({int(value) for value in scene_ids})
    official = sorted({int(value) for value in official_test_scenes})
    if not set(official).issubset(scenes):
        raise ValueError("Official test scenes are missing from the source data")
    candidates = np.asarray([value for value in scenes if value not in set(official)], dtype=np.int64)
    if int(val_count) <= 0 or int(val_count) >= len(candidates):
        raise ValueError("val_count must leave at least one training scene")
    order = np.random.default_rng(int(seed)).permutation(candidates)
    validation = sorted(int(value) for value in order[: int(val_count)])
    training = sorted(int(value) for value in order[int(val_count) :])
    return {
        "dataset": "sitzmann_official_vr",
        "seed": int(seed),
        "unit": "canonical_scene",
        "train": training,
        "val": validation,
        "gate": official,
        "gate_source_role": "published_scanpath_test_scenes; not a fresh holdout",
        "zero_scene_overlap": True,
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _scene_split_map(split: dict[str, Any]) -> dict[int, str]:
    output: dict[int, str] = {}
    report_roles = [role for role in ("test", "gate") if role in split]
    if len(report_roles) != 1:
        raise ValueError("Sitzmann split must define exactly one of test or gate")
    for role in ("train", "val", report_roles[0]):
        for scene in split[role]:
            scene_id = int(scene)
            if scene_id in output:
                raise ValueError(f"Scene {scene_id} occurs in multiple splits")
            output[scene_id] = role
    return output


def build_sitzmann_early_horizon_view(
    *,
    interim_root: str | Path,
    out_root: str | Path,
    split_file: str | Path,
    horizons_s: Sequence[float] = (0.5, 1.0, 3.0, 5.0),
    height: int = 256,
    width: int = 512,
    sigma_deg: float = 2.0,
    reference_yaw_deg: float = 180.0,
    final_horizon_s: float = 30.0,
) -> Path:
    interim_root = Path(interim_root)
    out_root = ensure_dir(out_root)
    split_file = Path(split_file)
    source_rows = _read_jsonl(interim_root / "cells.jsonl")
    source_info = json.loads((interim_root / "dataset_info.json").read_text(encoding="utf-8"))
    scenes = sorted({int(row["scene_id"]) for row in source_rows})
    if split_file.exists():
        split = json.loads(split_file.read_text(encoding="utf-8"))
    else:
        split = make_content_split(scenes)
        split_file.parent.mkdir(parents=True, exist_ok=True)
        split_file.write_text(json.dumps(split, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    role_by_scene = _scene_split_map(split)
    report_role = "test" if "test" in split else "gate"
    if set(role_by_scene) != set(scenes):
        raise ValueError("Split file does not cover every Sitzmann scene exactly once")

    images_dir = ensure_dir(out_root / "images")
    targets_dir = ensure_dir(out_root / "targets")
    fixations_dir = ensure_dir(out_root / "fixations")
    scanpaths_dir = ensure_dir(out_root / "scanpaths")
    geometry = MapGeometry(int(height), int(width))
    horizons = tuple(float(value) for value in horizons_s)
    output_rows: list[dict[str, Any]] = []

    for source in sorted(source_rows, key=lambda row: (int(row["scene_id"]), int(row["viewpoint_idx"]))):
        scene = int(source["scene_id"])
        viewpoint = int(source["viewpoint_idx"])
        image_id = f"sitzmann_cubemap_{scene:04d}_vp{viewpoint}"
        shift = alignment_shift_px(viewpoint, width, reference_yaw_deg=reference_yaw_deg)
        with Image.open(interim_root / source["image"]) as image:
            rgb = np.asarray(image.convert("RGB").resize((int(width), int(height))), dtype=np.uint8)
        aligned_rgb = np.roll(rgb, shift=shift, axis=1)
        image_path = images_dir / f"{image_id}.png"
        Image.fromarray(aligned_rgb, mode="RGB").save(image_path)

        with np.load(interim_root / source["fixations"]) as data:
            raw_fixations = np.asarray(data["fixations"], dtype=np.float32)
            subject_ids = np.asarray(data["subject_ids"])
            raw_scanpaths = np.asarray(data["scanpaths_1hz"], dtype=np.float32)
            scanpath_subject_ids = np.asarray(data["scanpath_subject_ids"])
            scanpath_step_s = float(np.asarray(data["scanpath_step_s"]))
        fixations = align_fixations_to_reference(
            raw_fixations,
            viewpoint,
            width=width,
            reference_yaw_deg=reference_yaw_deg,
        )
        observer_ids = np.unique(fixations[:, 3]).astype(int).tolist()
        scanpaths = raw_scanpaths.copy()
        scanpaths[..., 0] = np.mod(
            scanpaths[..., 0] + float(alignment_shift_px(viewpoint, width, reference_yaw_deg=reference_yaw_deg)) / float(width),
            1.0,
        )
        cumulative = np.stack(
            [
                build_cumulative_density(
                    fixations,
                    observer_ids=observer_ids,
                    horizon_s=horizon,
                    height=height,
                    width=width,
                    renderer="sphere_chord",
                    sigma_deg=sigma_deg,
                )
                for horizon in horizons
            ],
            axis=0,
        ).astype(np.float32)
        final = build_cumulative_density(
            fixations,
            observer_ids=observer_ids,
            horizon_s=float(final_horizon_s),
            height=height,
            width=width,
            renderer="sphere_chord",
            sigma_deg=sigma_deg,
        ).astype(np.float32)
        fixation = fixation_binary_map(fixations[:, 0], fixations[:, 1], height, width).astype(np.float32)
        target_path = targets_dir / f"{image_id}.npz"
        fixation_path = fixations_dir / f"{image_id}.npz"
        scanpath_path = scanpaths_dir / f"{image_id}.npz"
        np.savez_compressed(
            target_path,
            cumulative=cumulative,
            final=final,
            fixation=fixation,
            area_weights=geometry.area_weights,
            horizons_s=np.asarray(horizons, dtype=np.float32),
        )
        np.savez_compressed(
            fixation_path,
            fixations=fixations,
            columns=np.asarray(
                ["x_norm", "y_norm", "timestamp_s", "trial_observer_id", "duration_s", "fixation_id"]
            ),
            subject_ids=subject_ids,
            viewpoint_idx=np.asarray(viewpoint, dtype=np.int64),
        )
        np.savez_compressed(
            scanpath_path,
            scanpaths_yx=np.stack((scanpaths[..., 1], scanpaths[..., 0]), axis=-1).astype(np.float32),
            subject_ids=scanpath_subject_ids,
            step_s=np.asarray(scanpath_step_s, dtype=np.float32),
            viewpoint_idx=np.asarray(viewpoint, dtype=np.int64),
        )
        output_rows.append(
            {
                "dataset": "sitzmann_official_early_horizon",
                "image_id": image_id,
                "canonical_content_id": f"sitzmann:cubemap_{scene:04d}",
                "condition_id": f"sitzmann:cubemap_{scene:04d}:vp{viewpoint}",
                "group_id": f"content:sitzmann:cubemap_{scene:04d}",
                "scene_id": scene,
                "viewpoint_idx": viewpoint,
                "source_split": "published_test" if scene in OFFICIAL_SCANPATH_TEST_SCENES else "published_train",
                "split": role_by_scene[scene],
                "image": str(image_path),
                "target": str(target_path),
                "fixations": str(fixation_path),
                "scanpaths_1hz": str(scanpath_path),
                "target_sealed": False,
                "raw_declared_start_yaw_deg": VIEWPOINT_YAW_DEG[viewpoint],
                "declared_start_yaw_deg": float(reference_yaw_deg),
                # RGB, fixations, and targets have already been rolled so the
                # physical start is at ERP x=0.5.  EarlyHorizonDataset uses the
                # conventional model grid longitude x=0.5 -> 0 degrees; keep
                # this explicit model-space value separate from the historical
                # Sitzmann longitude convention above.
                "model_start_x_norm": 0.5,
                "model_declared_start_yaw_deg": 0.0,
                "alignment_shift_px": shift,
                "horizons_s": list(horizons),
                "trial_count": int(source["trial_count"]),
                "unique_subject_count": int(source["unique_subject_count"]),
                "physical_observer_count": len(observer_ids),
                "scanpath_trial_count": int(len(scanpaths)),
                "target_provenance": "official_vr_pickle_fixations_clipped_dwell_sphere_chord_2deg",
            }
        )

    write_jsonl(output_rows, out_root / "manifest.jsonl")
    by_scene: dict[int, set[str]] = {}
    for row in output_rows:
        by_scene.setdefault(int(row["scene_id"]), set()).add(str(row["split"]))
    invalid_integrals = 0
    for row in output_rows:
        with np.load(row["target"]) as data:
            maps = np.concatenate((np.asarray(data["cumulative"]), np.asarray(data["final"])[None]), axis=0)
            area = np.asarray(data["area_weights"])
        invalid_integrals += sum(not np.isclose(float((current * area).sum()), 1.0, atol=2e-4) for current in maps)
    info = {
        "dataset": "sitzmann_official_early_horizon",
        "source_interim_root": str(interim_root),
        "source_archive": source_info.get("source_archive"),
        "source_archive_sha256": source_info.get("source_archive_sha256"),
        "split_file": str(split_file),
        "image_size": [int(height), int(width)],
        "horizons_s": list(horizons),
        "final_horizon_s": float(final_horizon_s),
        "renderer": "sphere_chord",
        "sigma_deg": float(sigma_deg),
        "reference_yaw_deg": float(reference_yaw_deg),
        "viewpoint_yaw_deg": VIEWPOINT_YAW_DEG,
        "scene_count": len(scenes),
        "condition_count": len(output_rows),
        "trial_count": int(sum(int(row["trial_count"]) for row in output_rows)),
        "observer_cell_count": int(sum(int(row["physical_observer_count"]) for row in output_rows)),
        "duplicate_subject_trials_within_cell": int(
            sum(int(row["trial_count"]) - int(row["physical_observer_count"]) for row in output_rows)
        ),
        "unique_subject_count_across_dataset": int(source_info.get("unique_subject_count", 0)),
        "split_scene_counts": {
            role: len({int(row["scene_id"]) for row in output_rows if row["split"] == role})
            for role in ("train", "val", report_role)
        },
        "split_condition_counts": {
            role: sum(row["split"] == role for row in output_rows)
            for role in ("train", "val", report_role)
        },
        "zero_scene_leakage": all(len(roles) == 1 for roles in by_scene.values()),
        "complete_four_viewpoints": all(
            {int(row["viewpoint_idx"]) for row in output_rows if int(row["scene_id"]) == scene}
            == {0, 1, 2, 3}
            for scene in scenes
        ),
        "invalid_spherical_integrals": int(invalid_integrals),
        "published_test_is_not_fresh": True,
    }
    (out_root / "dataset_info.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    build_train_only_baselines(out_root, out_root / "baselines")
    return out_root
