#!/usr/bin/env python
"""Extract compact official standing-VR records from the legacy pandas pickles.

Run this script with the pinned Python 3.7 / NumPy 1.x environment documented
by ``scripts/prepare_sitzmann_official.py``. It intentionally has no dependency
on the modern sphererate package.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import pickle
from zipfile import ZipFile

import numpy as np
from PIL import Image


def _compat_reconstructor(cls, base, state):
    if base is object:
        return cls.__new__(cls)
    obj = base.__new__(cls, state)
    if base.__init__ is not object.__init__:
        base.__init__(obj, state)
    return obj


class _CompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module in ("copy_reg", "copyreg") and name == "_reconstructor":
            return _compat_reconstructor
        return super().find_class(module, name)


def _fixations(data, observer_id, final_horizon_s=30.0):
    rows = []
    for fallback_id, fixation in enumerate(data["gaze_fixations_raw"]):
        x_norm, y_norm = fixation["norm_pos"]
        start = max(float(fixation["start_time"]), 0.0)
        end = min(float(fixation["end_time"]), float(final_horizon_s))
        duration = max(end - start, 0.0)
        values = (x_norm, y_norm, start, duration)
        if not all(np.isfinite(value) for value in values) or duration <= 0.0:
            continue
        rows.append(
            [
                float(x_norm) % 1.0,
                float(np.clip(y_norm, 0.0, 1.0)),
                start,
                float(observer_id),
                duration,
                float(fixation.get("id", fallback_id)),
            ]
        )
    return rows


def _sample_gaze_seconds(data, length=30):
    """Sample the official dense gaze trace at integer seconds.

    Scanpath baselines require one gaze direction per second.  Using detected
    fixations for this purpose discards saccades and tracking intervals, so the
    dense ``gaze_lat_lon``/``timestamp`` stream is the authoritative source.
    """

    timestamp = np.asarray(data["timestamp"], dtype=np.float64).reshape(-1)
    gaze = np.asarray(data["gaze_lat_lon"], dtype=np.float64)
    if timestamp.ndim != 1 or gaze.shape != (len(timestamp), 2):
        raise ValueError("Invalid official dense gaze trace")
    targets = np.arange(int(length), dtype=np.float64)
    right = np.searchsorted(timestamp, targets, side="left")
    right = np.clip(right, 0, len(timestamp) - 1)
    left = np.clip(right - 1, 0, len(timestamp) - 1)
    choose_left = np.abs(timestamp[left] - targets) <= np.abs(timestamp[right] - targets)
    indices = np.where(choose_left, left, right)
    sampled = gaze[indices]
    # Official latitude is colatitude [0,180] and longitude is [0,360).
    xy = np.stack((np.mod(sampled[:, 1] / 360.0, 1.0), np.clip(sampled[:, 0] / 180.0, 0.0, 1.0)), axis=-1)
    if not np.isfinite(xy).all():
        raise ValueError("Non-finite official gaze sample")
    return xy.astype(np.float32), float(np.max(np.abs(timestamp[indices] - targets)))


def _circular_delta_deg(value, reference):
    return (float(value) - float(reference) + 180.0) % 360.0 - 180.0


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Extract official Sitzmann standing-VR trials.")
    parser.add_argument("--archive", default="data/sitzmann_raw_data.zip")
    parser.add_argument("--out-root", default="data/interim/sitzmann_official_vr_v1")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=512)
    args = parser.parse_args()
    out_root = Path(args.out_root)
    images_dir = out_root / "images"
    cells_dir = out_root / "cells"
    images_dir.mkdir(parents=True, exist_ok=True)
    cells_dir.mkdir(parents=True, exist_ok=True)
    cell_rows = []
    all_trials = []

    with ZipFile(args.archive) as archive:
        for scene in range(22):
            pickle_name = "data/vr/cubemap_%04d.pck" % scene
            with archive.open(pickle_name) as handle:
                frame = _CompatUnpickler(handle, encoding="latin1").load()
            image_name = "data/img_files/cubemap_%04d.png" % scene
            with archive.open(image_name) as handle:
                image = Image.open(handle).convert("RGB")
                resampling = getattr(Image, "Resampling", Image)
                image = image.resize((int(args.width), int(args.height)), resampling.LANCZOS)
                image.save(images_dir / ("cubemap_%04d.png" % scene))

            cells = defaultdict(list)
            scanpaths = defaultdict(list)
            scanpath_subjects = defaultdict(list)
            metadata = defaultdict(list)
            observer_id_by_subject = defaultdict(dict)
            for trial_index, row in frame.iterrows():
                viewpoint = int(row["viewpoint_idx"])
                data = row["data"]
                subject = str(row["experiment_id"])
                subject_ids = observer_id_by_subject[viewpoint]
                if subject not in subject_ids:
                    subject_ids[subject] = len(subject_ids)
                observer_id = subject_ids[subject]
                current = _fixations(data, observer_id)
                sampled_scanpath, scanpath_max_time_error_s = _sample_gaze_seconds(data, length=30)
                cells[viewpoint].extend(current)
                scanpaths[viewpoint].append(sampled_scanpath)
                scanpath_subjects[viewpoint].append(subject)
                start_yaw = 90.0 * viewpoint
                trial = {
                    "scene_id": scene,
                    "trial_index": int(trial_index),
                    "subject_id": subject,
                    "cell_observer_id": observer_id,
                    "viewpoint_idx": viewpoint,
                    "declared_start_yaw_deg": start_yaw,
                    "duration_s": float(data["timestamp"][-1] - data["timestamp"][0]),
                    "fixation_count": len(current),
                    "scanpath_length": int(len(sampled_scanpath)),
                    "scanpath_max_time_error_s": scanpath_max_time_error_s,
                    "initial_gaze_yaw_error_deg": _circular_delta_deg(data["gaze_lat_lon"][0, 1], start_yaw),
                    "initial_head_yaw_error_deg": _circular_delta_deg(data["headLatLon"][0, 1], start_yaw),
                }
                metadata[viewpoint].append(trial)
                all_trials.append(trial)
            for viewpoint in range(4):
                if not metadata[viewpoint]:
                    raise RuntimeError("Missing viewpoint %d for scene %d" % (viewpoint, scene))
                fixation_array = np.asarray(cells[viewpoint], dtype=np.float32)
                subjects = np.asarray([item["subject_id"] for item in metadata[viewpoint]])
                relative = Path("cells") / ("cubemap_%04d_vp%d.npz" % (scene, viewpoint))
                np.savez_compressed(
                    out_root / relative,
                    fixations=fixation_array,
                    subject_ids=subjects,
                    scanpaths_1hz=np.stack(scanpaths[viewpoint]).astype(np.float32),
                    scanpath_subject_ids=np.asarray(scanpath_subjects[viewpoint]),
                    scanpath_step_s=np.asarray(1.0, dtype=np.float32),
                    columns=np.asarray(
                        ["x_norm", "y_norm", "timestamp_s", "trial_observer_id", "duration_s", "fixation_id"]
                    ),
                )
                durations = np.asarray([item["duration_s"] for item in metadata[viewpoint]], dtype=np.float64)
                gaze_errors = np.asarray(
                    [item["initial_gaze_yaw_error_deg"] for item in metadata[viewpoint]], dtype=np.float64
                )
                head_errors = np.asarray(
                    [item["initial_head_yaw_error_deg"] for item in metadata[viewpoint]], dtype=np.float64
                )
                cell_rows.append(
                    {
                        "scene_id": scene,
                        "viewpoint_idx": viewpoint,
                        "declared_start_yaw_deg": 90.0 * viewpoint,
                        "image": "images/cubemap_%04d.png" % scene,
                        "fixations": str(relative),
                        "trial_count": len(metadata[viewpoint]),
                        "unique_subject_count": len(set(subjects.tolist())),
                        "duplicate_subject_trials": len(subjects) - len(set(subjects.tolist())),
                        "scanpath_trial_count": len(scanpaths[viewpoint]),
                        "scanpath_length": 30,
                        "duration_s_min": float(durations.min()),
                        "duration_s_median": float(np.median(durations)),
                        "initial_gaze_yaw_error_median_deg": float(np.median(gaze_errors)),
                        "initial_head_yaw_error_median_deg": float(np.median(head_errors)),
                    }
                )
            print("scene %02d: %d trials" % (scene, len(frame)), flush=True)

    (out_root / "cells.jsonl").write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in cell_rows) + "\n", encoding="utf-8"
    )
    (out_root / "trials.jsonl").write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in all_trials) + "\n", encoding="utf-8"
    )
    info = {
        "dataset": "sitzmann_official_standing_vr",
        "source_archive": str(args.archive),
        "source_archive_sha256": _sha256(args.archive),
        "source_pickle_family": "data/vr/cubemap_*.pck",
        "excluded_conditions": ["browser", "seated_vr"],
        "scene_count": 22,
        "viewpoint_count": 4,
        "cell_count": len(cell_rows),
        "trial_count": len(all_trials),
        "unique_subject_count": len(set(row["subject_id"] for row in all_trials)),
        "image_size": [int(args.height), int(args.width)],
        "coordinate_semantics": "fixation norm_pos=(longitude/360, latitude_colatitude/180)",
        "viewpoint_yaw_deg": {"0": 0.0, "1": 90.0, "2": 180.0, "3": 270.0},
    }
    (out_root / "dataset_info.json").write_text(
        json.dumps(info, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
