#!/usr/bin/env python
"""Prepare a matched Salient360 split for the official HAT implementation.

The exported trajectory contains an explicit declared-start token followed by
the same twenty one-second gaze samples used by the ScanDMM and Pathformer3D
comparators. Validation/evaluation images are copied for inference, while only
training trajectories are used to fit the model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sphererate.baselines.scanpath_export import xyz_to_erp_yx


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare matched HAT trajectories and ERP images.")
    parser.add_argument(
        "--scanpath-manifest",
        default="data/processed/kernel_v2_matched_scanpaths/manifest.jsonl",
    )
    parser.add_argument(
        "--saliency-manifest",
        default="data/processed/early_horizon_salient360_spherical2deg_v2/manifest.jsonl",
    )
    parser.add_argument(
        "--output-root",
        default="data/processed/kernel_v2_matched_scanpaths/hat",
    )
    parser.add_argument("--expected-train-images", type=int, default=50)
    parser.add_argument("--expected-val-images", type=int, default=9)
    parser.add_argument("--expected-eval-images", type=int, default=13)
    parser.add_argument("--eval-split", choices=("gate", "test"), default="gate")
    parser.add_argument("--expected-train-paths", type=int)
    parser.add_argument("--expected-val-paths", type=int)
    args = parser.parse_args()
    scanpath_manifest = (ROOT / args.scanpath_manifest).resolve()
    saliency_manifest = (ROOT / args.saliency_manifest).resolve()
    output = (ROOT / args.output_root).resolve()
    image_dir = output / "images"

    saliency_rows = [
        json.loads(line)
        for line in saliency_manifest.read_text(encoding="utf-8").splitlines()
        if line
    ]
    expected_images = {
        "train": int(args.expected_train_images),
        "val": int(args.expected_val_images),
        str(args.eval_split): int(args.expected_eval_images),
    }
    allowed = [row for row in saliency_rows if str(row["split"]) in expected_images]
    by_id = {str(row["image_id"]): row for row in allowed}
    image_counts = {
        split: sum(str(row["split"]) == split for row in allowed) for split in expected_images
    }
    if image_counts != expected_images:
        raise ValueError(f"Image counts changed: {image_counts} != {expected_images}")

    images: dict[str, list[dict[str, Any]]] = {split: [] for split in expected_images}
    for image_id, row in sorted(by_id.items()):
        source = (ROOT / str(row["image"])).resolve()
        destination = image_dir / f"{image_id}.png"
        _link_or_copy(source, destination)
        images[str(row["split"])].append(
            {
                "image_id": image_id,
                "canonical_content_id": str(row["canonical_content_id"]),
                "file": destination.name,
                "declared_start_yaw_deg": float(row["declared_start_yaw_deg"]),
            }
        )

    trajectory_rows = [
        json.loads(line)
        for line in scanpath_manifest.read_text(encoding="utf-8").splitlines()
        if line
    ]
    trajectories: dict[str, list[dict[str, Any]]] = {"train": [], "val": []}
    for row in trajectory_rows:
        split = str(row["split"])
        if split not in trajectories:
            continue
        with np.load(ROOT / str(row["path"])) as data:
            xyz = np.asarray(data["scanpath_xyz"], dtype=np.float32)
            start_xyz = np.asarray(data["initial_xyz"], dtype=np.float32)
        if xyz.shape != (20, 3) or start_xyz.shape != (3,):
            raise ValueError(f"Unexpected HAT source trajectory shape: {row['path']}")
        yx = xyz_to_erp_yx(xyz)
        start_yx = xyz_to_erp_yx(start_xyz[None])[0]
        points_yx = np.concatenate((start_yx[None], yx), axis=0)
        if not np.isfinite(points_yx).all():
            raise ValueError(f"Non-finite HAT trajectory: {row['path']}")
        trajectories[split].append(
            {
                "image_id": str(row["image_id"]),
                "canonical_content_id": str(row["canonical_content_id"]),
                "observer_id": int(row["observer_id"]),
                "source_observer_label": str(
                    row.get("source_observer_label", row["observer_id"])
                ),
                "points_yx": points_yx.tolist(),
                "step_s": 1.0,
            }
        )

    path_counts = {split: len(rows) for split, rows in trajectories.items()}
    expected_paths = {
        "train": int(args.expected_train_paths) if args.expected_train_paths is not None else path_counts["train"],
        "val": int(args.expected_val_paths) if args.expected_val_paths is not None else path_counts["val"],
    }
    if path_counts != expected_paths:
        raise ValueError(f"Trajectory counts changed: {path_counts} != {expected_paths}")
    train_ids = {row["image_id"] for row in images["train"]}
    val_ids = {row["image_id"] for row in images["val"]}
    eval_ids = {row["image_id"] for row in images[str(args.eval_split)]}
    if train_ids & val_ids or train_ids & eval_ids or val_ids & eval_ids:
        raise RuntimeError("Content leakage in the HAT image split")

    payload = {
        "protocol": "kernel_v2_hat_declared_start_v1",
        "coordinate_order": "normalized_yx",
        "trajectory_semantics": "declared start token + twenty one-second gaze targets",
        "images": images,
        "trajectories": trajectories,
        "source": {
            "scanpath_manifest": str(scanpath_manifest),
            "scanpath_manifest_sha256": _sha256(scanpath_manifest),
            "saliency_manifest": str(saliency_manifest),
            "saliency_manifest_sha256": _sha256(saliency_manifest),
        },
        "audit": {
            "image_counts": image_counts,
            "trajectory_counts": path_counts,
            "train_val_overlap": sorted(train_ids & val_ids),
            "train_eval_overlap": sorted(train_ids & eval_ids),
            "val_eval_overlap": sorted(val_ids & eval_ids),
            "test_target_loads": int(image_counts.get("test", 0)),
            "test_used_for_training": False,
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    dataset_path = output / "dataset.json"
    dataset_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"dataset": str(dataset_path), **payload["audit"]}, indent=2))


if __name__ == "__main__":
    main()
