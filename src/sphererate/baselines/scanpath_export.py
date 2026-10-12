from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import pickle
from typing import Any, Iterable

import numpy as np
from PIL import Image
import torch


@dataclass(frozen=True)
class SampledScanpath:
    xyz: np.ndarray
    repaired_bins: tuple[int, ...]


def erp_xy_to_xyz(x_norm: np.ndarray, y_norm: np.ndarray) -> np.ndarray:
    """Convert normalized ERP coordinates to ScanDMM/Pathformer unit vectors.

    Both external repositories use ``lat=(y-.5)*180`` and
    ``lon=(x-.5)*360``.  This flips the physical z convention relative to the
    project's saliency renderer but preserves the ERP y coordinate on round
    trip, which is what their image feature extractors and output adapters use.
    """

    x = np.mod(np.asarray(x_norm, dtype=np.float64), 1.0)
    y = np.clip(np.asarray(y_norm, dtype=np.float64), 0.0, 1.0)
    if x.shape != y.shape:
        raise ValueError(f"x/y shapes differ: {x.shape} vs {y.shape}")
    latitude = (y - 0.5) * np.pi
    longitude = (x - 0.5) * 2.0 * np.pi
    cos_latitude = np.cos(latitude)
    return np.stack(
        (cos_latitude * np.cos(longitude), cos_latitude * np.sin(longitude), np.sin(latitude)), axis=-1
    ).astype(np.float32)


def xyz_to_lat_lon_deg(xyz: np.ndarray) -> np.ndarray:
    values = _normalize_vectors(np.asarray(xyz, dtype=np.float64))
    longitude = np.arctan2(values[..., 1], values[..., 0])
    latitude = np.arctan2(values[..., 2], np.sqrt(values[..., 0] ** 2 + values[..., 1] ** 2))
    return np.stack((np.rad2deg(latitude), np.rad2deg(longitude)), axis=-1).astype(np.float32)


def xyz_to_erp_yx(xyz: np.ndarray) -> np.ndarray:
    lat_lon = xyz_to_lat_lon_deg(xyz)
    y = lat_lon[..., 0] / 180.0 + 0.5
    x = np.mod(lat_lon[..., 1] / 360.0 + 0.5, 1.0)
    return np.stack((y, x), axis=-1).astype(np.float32)


def declared_start_xyz(pitch_deg: float = 0.0, yaw_deg: float = 90.0) -> np.ndarray:
    y = float(pitch_deg) / 180.0 + 0.5
    x = float(yaw_deg) / 360.0 + 0.5
    return erp_xy_to_xyz(np.asarray([x]), np.asarray([y]))[0]


def sample_observer_seconds(
    fixations: np.ndarray,
    observer_id: int,
    *,
    length: int = 20,
    step_s: float = 1.0,
    max_missing_run: int = 1,
) -> SampledScanpath | None:
    """Sample the first active fixation in each fixed one-second interval.

    Missing isolated intervals follow ScanDMM's published preprocessing:
    leading/trailing bins use the nearest valid point and an interior bin is
    interpolated.  Longer missing runs are rejected instead of silently
    inventing a trajectory.
    """

    array = np.asarray(fixations, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] < 5:
        raise ValueError(f"Expected fixation rows [N, >=5], got {array.shape}")
    if length <= 0 or step_s <= 0.0:
        raise ValueError("length and step_s must be positive")
    selected = array[array[:, 3].astype(np.int64) == int(observer_id)]
    selected = selected[np.argsort(selected[:, 2], kind="stable")]
    if not len(selected):
        return None
    points = np.full((int(length), 3), np.nan, dtype=np.float64)
    for index in range(int(length)):
        begin = index * float(step_s)
        end = begin + float(step_s)
        starts = selected[:, 2]
        ends = starts + np.maximum(selected[:, 4], 0.0)
        active = np.flatnonzero(np.minimum(ends, end) - np.maximum(starts, begin) > 0.0)
        if active.size:
            row = selected[int(active[0])]
            points[index] = erp_xy_to_xyz(np.asarray([row[0]]), np.asarray([row[1]]))[0]
    missing = np.flatnonzero(~np.isfinite(points).all(axis=1))
    if missing.size:
        runs = _consecutive_runs(missing)
        if max(len(run) for run in runs) > int(max_missing_run):
            return None
        valid = np.flatnonzero(np.isfinite(points).all(axis=1))
        if valid.size == 0:
            return None
        for index in missing.tolist():
            before = valid[valid < index]
            after = valid[valid > index]
            if before.size == 0:
                points[index] = points[int(after[0])]
            elif after.size == 0:
                points[index] = points[int(before[-1])]
            else:
                left, right = int(before[-1]), int(after[0])
                fraction = (index - left) / float(right - left)
                points[index] = _normalized_lerp(points[left], points[right], fraction)
    points = _normalize_vectors(points).astype(np.float32)
    if not np.isfinite(points).all():
        raise RuntimeError("Non-finite scanpath after missing-bin repair")
    return SampledScanpath(xyz=points, repaired_bins=tuple(int(value) for value in missing.tolist()))


def read_non_test_rows(manifest: str | Path, allowed_splits: Iterable[str] = ("train", "val", "gate")) -> list[dict[str, Any]]:
    allowed = {str(value) for value in allowed_splits}
    rows = []
    for line in Path(manifest).read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        row = json.loads(line)
        split = str(row["split"])
        if split not in allowed:
            continue
        if row.get("fixations") is None:
            raise ValueError(f"Non-test row lacks fixations: {row.get('image_id')}")
        rows.append(row)
    return rows


def export_matched_scanpaths(
    manifest: str | Path,
    output_root: str | Path,
    *,
    length: int = 20,
    step_s: float = 1.0,
    start_pitch_deg: float = 0.0,
    start_yaw_deg: float = 90.0,
    allowed_splits: Iterable[str] = ("train", "val", "gate"),
) -> dict[str, Any]:
    output = Path(output_root)
    canonical_root = output / "canonical"
    pathformer_root = output / "pathformer3d"
    canonical_root.mkdir(parents=True, exist_ok=True)
    pathformer_root.mkdir(parents=True, exist_ok=True)
    allowed = tuple(str(value) for value in allowed_splits)
    rows = read_non_test_rows(manifest, allowed_splits=allowed)
    start_xyz = declared_start_xyz(start_pitch_deg, start_yaw_deg)
    records: list[dict[str, Any]] = []
    scandmm: dict[str, Any] = {"train": {}, "test": {}, "info": {}}
    pathformer_records: dict[str, list[dict[str, Any]]] = {split: [] for split in allowed}
    audit = {
        "dropped_observers": 0,
        "repaired_bins": 0,
        "test_target_rows_loaded": 0,
        "dense_1hz_trials": 0,
        "fixation_sampled_observers": 0,
    }
    for row in rows:
        split = str(row["split"])
        audit["test_target_rows_loaded"] += int(split == "test")
        image_id = str(row["image_id"])
        paths, repaired, observer_ids = [], [], []
        source_observer_labels: list[str] = []
        if row.get("scanpaths_1hz"):
            with np.load(str(row["scanpaths_1hz"])) as data:
                scanpaths_yx = np.asarray(data["scanpaths_yx"], dtype=np.float32)
                source_step_s = float(np.asarray(data["step_s"]))
                labels = np.asarray(data.get("subject_ids", np.arange(len(scanpaths_yx)))).astype(str)
            if scanpaths_yx.ndim != 3 or scanpaths_yx.shape[-1] != 2:
                raise ValueError(f"Invalid dense scanpaths for {image_id}: {scanpaths_yx.shape}")
            if scanpaths_yx.shape[1] < int(length) or not np.isclose(source_step_s, float(step_s)):
                raise ValueError(
                    f"Dense scanpath time contract mismatch for {image_id}: "
                    f"shape={scanpaths_yx.shape}, step={source_step_s}"
                )
            for observer_id, path_yx in enumerate(scanpaths_yx[:, : int(length)]):
                paths.append(erp_xy_to_xyz(path_yx[:, 1], path_yx[:, 0]))
                repaired.append(tuple())
                observer_ids.append(observer_id)
                source_observer_labels.append(str(labels[observer_id]))
            audit["dense_1hz_trials"] += len(paths)
        else:
            with np.load(str(row["fixations"])) as data:
                fixations = np.asarray(data["fixations"], dtype=np.float32)
            for observer_id in sorted(np.unique(fixations[:, 3]).astype(int).tolist()):
                sampled = sample_observer_seconds(fixations, observer_id, length=length, step_s=step_s)
                if sampled is None:
                    audit["dropped_observers"] += 1
                    continue
                paths.append(sampled.xyz)
                repaired.append(sampled.repaired_bins)
                observer_ids.append(observer_id)
                source_observer_labels.append(str(observer_id))
                audit["repaired_bins"] += len(sampled.repaired_bins)
                audit["fixation_sampled_observers"] += 1
        for path_xyz, repaired_bins, observer_id, source_label in zip(
            paths, repaired, observer_ids, source_observer_labels
        ):
            destination = canonical_root / split / f"{image_id}_observer{observer_id:03d}.npz"
            destination.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                destination,
                scanpath_xyz=path_xyz,
                initial_xyz=start_xyz,
                observer_id=np.asarray(observer_id, dtype=np.int64),
                source_observer_label=np.asarray(source_label),
                image_id=np.asarray(image_id),
                split=np.asarray(split),
                step_s=np.asarray(step_s, dtype=np.float32),
            )
            records.append(
                {
                    "image_id": image_id,
                    "canonical_content_id": str(row["canonical_content_id"]),
                    "split": split,
                    "observer_id": observer_id,
                    "path": str(destination),
                    "source_observer_label": source_label,
                    "repaired_bins": list(repaired_bins),
                }
            )
        if not paths:
            raise RuntimeError(f"No valid scanpaths for {split}/{image_id}")
        path_array = np.stack(paths).astype(np.float32)
        lat_lon = xyz_to_lat_lon_deg(path_array)
        tensor_path = pathformer_root / "scanpaths" / f"{image_id}.pt"
        tensor_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(torch.from_numpy(lat_lon), tensor_path)
        if split == "train":
            for observer_index in range(len(path_array)):
                pathformer_records[split].append(
                    {"image_path": str(row["image"]), "scanpaths": str(tensor_path), "index": observer_index}
                )
        else:
            pathformer_records[split].append(
                {"image_path": str(row["image"]), "scanpaths": str(tensor_path), "index": 0}
            )
        scandmm_split = "train" if split == "train" else "test"
        if split in {"train", "val"}:
            scandmm[scandmm_split][image_id] = {
                "image": _load_scandmm_image(row["image"]),
                "scanpaths": path_array,
            }
    for split_name in ("train", "test"):
        contents = scandmm[split_name]
        num_scanpaths = sum(len(value["scanpaths"]) for value in contents.values())
        scandmm["info"][split_name] = {
            "num_image": len(contents),
            "num_scanpath": num_scanpaths,
            "scanpath_length": int(length),
            "max_scan_length": int(length),
        }
    output.mkdir(parents=True, exist_ok=True)
    with (output / "scandmm_train_val.pkl").open("wb") as handle:
        pickle.dump(scandmm, handle, protocol=pickle.HIGHEST_PROTOCOL)
    (pathformer_root / "dataset.json").write_text(
        json.dumps(pathformer_records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    manifest_path = output / "manifest.jsonl"
    manifest_path.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8")
    content_ids = {
        split: sorted({record["image_id"] for record in records if record["split"] == split})
        for split in allowed
    }
    result = {
        "source_manifest": str(manifest),
        "source_manifest_sha256": hashlib.sha256(Path(manifest).read_bytes()).hexdigest(),
        "content_ids": content_ids,
        "content_counts": {key: len(value) for key, value in content_ids.items()},
        "scanpath_counts": {split: sum(record["split"] == split for record in records) for split in content_ids},
        "length": int(length),
        "step_s": float(step_s),
        "initial_gaze": {"pitch_deg": float(start_pitch_deg), "yaw_deg": float(start_yaw_deg), "xyz": start_xyz.tolist()},
        "audit": audit,
    }
    (output / "audit.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def _load_scandmm_image(path: str | Path) -> torch.Tensor:
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB").resize((256, 128), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(rgb).permute(2, 0, 1)
    return (tensor - 0.5) / 0.5


def _normalize_vectors(values: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(values, axis=-1, keepdims=True)
    if np.any(~np.isfinite(norm)) or np.any(norm <= 1e-12):
        raise ValueError("Cannot normalize a zero or non-finite vector")
    return values / norm


def _normalized_lerp(left: np.ndarray, right: np.ndarray, fraction: float) -> np.ndarray:
    mixed = (1.0 - float(fraction)) * np.asarray(left) + float(fraction) * np.asarray(right)
    return _normalize_vectors(mixed)


def _consecutive_runs(indices: np.ndarray) -> list[np.ndarray]:
    if not len(indices):
        return []
    boundaries = np.flatnonzero(np.diff(indices) > 1) + 1
    return [part for part in np.split(indices, boundaries) if len(part)]
