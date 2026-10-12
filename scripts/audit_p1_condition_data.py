#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sphererate.data.dataset import ProcessedSaliencyDataset, _startpoint_relative_channels
from sphererate.utils import ensure_dir, read_jsonl


def normalize_longitude_deg(value: float) -> float:
    return float((float(value) + 180.0) % 360.0 - 180.0)


def circular_distance_deg(left: float, right: float) -> float:
    return float(abs((float(left) - float(right) + 180.0) % 360.0 - 180.0))


def fixed_longitude_sector(value: float) -> int:
    longitude = normalize_longitude_deg(value)
    if longitude < -60.0:
        return 0
    if longitude < 60.0:
        return 1
    return 2


def load_startpoints_xls(path: Path) -> dict[str, tuple[float, float]]:
    import xlrd

    book = xlrd.open_workbook(str(path))
    sheet = book.sheet_by_index(0)
    rows: dict[str, tuple[float, float]] = {}
    for row_index in range(1, sheet.nrows):
        values = sheet.row_values(row_index)
        if len(values) >= 3 and values[0]:
            rows[str(values[0])] = (float(values[1]), float(values[2]))
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_conditioned_rows(
    rows: list[dict[str, Any]],
    raw_startpoints: dict[str, tuple[float, float]],
    check_artifacts: bool = True,
) -> dict[str, Any]:
    by_base: dict[str, list[dict[str, Any]]] = defaultdict(list)
    provenance_mismatches = 0
    missing_startpoint_rows = 0
    missing_artifacts: list[str] = []
    nonfinite_coordinates = 0
    for row in rows:
        by_base[str(row["base_image_id"])].append(row)
        source_key = f"{row['source_image_id']}.png"
        raw = raw_startpoints.get(source_key)
        if raw is None:
            missing_startpoint_rows += 1
        else:
            expected = raw[0] if str(row["condition"]) == "good" else raw[1]
            if circular_distance_deg(float(row["start_lon_deg"]), expected) > 1e-6:
                provenance_mismatches += 1
        if not np.isfinite(float(row.get("start_lon_deg", float("nan")))) or not np.isfinite(
            float(row.get("start_lat_deg", float("nan")))
        ):
            nonfinite_coordinates += 1
        if check_artifacts:
            for key in ("image", "target", "fixations"):
                path = Path(str(row.get(key, "")))
                if not path.exists():
                    missing_artifacts.append(f"{row.get('image_id')}:{key}:{path}")

    complete_pairs = 0
    split_group_mismatches = 0
    image_mismatches = 0
    reciprocal_coordinate_mismatches = 0
    pair_condition_mismatches = 0
    for pair in by_base.values():
        by_condition = {str(row.get("condition")): row for row in pair}
        if len(pair) != 2 or set(by_condition) != {"good", "bad"}:
            pair_condition_mismatches += 1
            continue
        complete_pairs += 1
        good, bad = by_condition["good"], by_condition["bad"]
        if good.get("split") != bad.get("split") or good.get("group_id") != bad.get("group_id"):
            split_group_mismatches += 1
        good_image, bad_image = Path(str(good["image"])), Path(str(bad["image"]))
        if good_image != bad_image:
            if not good_image.exists() or not bad_image.exists() or _sha256(good_image) != _sha256(bad_image):
                image_mismatches += 1
        if (
            circular_distance_deg(good["start_lon_deg"], bad["counterpart_start_lon_deg"]) > 1e-6
            or circular_distance_deg(bad["start_lon_deg"], good["counterpart_start_lon_deg"]) > 1e-6
            or abs(float(good.get("start_lat_deg", 0.0)) - float(bad.get("counterpart_start_lat_deg", 0.0))) > 1e-6
            or abs(float(bad.get("start_lat_deg", 0.0)) - float(good.get("counterpart_start_lat_deg", 0.0))) > 1e-6
        ):
            reciprocal_coordinate_mismatches += 1

    return {
        "row_count": len(rows),
        "pair_count": len(by_base),
        "complete_pair_count": complete_pairs,
        "incomplete_or_duplicate_pair_count": pair_condition_mismatches,
        "coordinate_provenance_mismatches": provenance_mismatches,
        "missing_raw_startpoint_rows": missing_startpoint_rows,
        "nonfinite_coordinate_rows": nonfinite_coordinates,
        "pair_split_group_mismatches": split_group_mismatches,
        "pair_image_mismatches": image_mismatches,
        "reciprocal_coordinate_mismatches": reciprocal_coordinate_mismatches,
        "missing_artifact_count": len(missing_artifacts),
        "missing_artifact_examples": missing_artifacts[:20],
    }


def _nearest_train_label(
    longitude: float,
    train_rows: list[dict[str, Any]],
) -> str:
    distances = [circular_distance_deg(longitude, float(row["start_lon_deg"])) for row in train_rows]
    minimum = min(distances)
    labels = [str(row["condition"]) for row, distance in zip(train_rows, distances) if abs(distance - minimum) < 1e-9]
    counts = Counter(labels)
    return sorted(counts, key=lambda label: (-counts[label], label))[0]


def condition_coordinate_summary(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    train_rows = [row for row in rows if row["split"] == "train"]
    train_coordinates = {
        (round(normalize_longitude_deg(float(row["start_lon_deg"])), 8), round(float(row["start_lat_deg"]), 8))
        for row in train_rows
    }
    coordinate_rows: list[dict[str, Any]] = []
    for row in rows:
        longitude = normalize_longitude_deg(float(row["start_lon_deg"]))
        latitude = float(row["start_lat_deg"])
        nearest = min(circular_distance_deg(longitude, float(train["start_lon_deg"])) for train in train_rows)
        coordinate_rows.append(
            {
                "image_id": row.get("image_id", ""),
                "base_image_id": row.get("base_image_id", ""),
                "split": row["split"],
                "condition": row["condition"],
                "start_lon_deg": longitude,
                "start_lat_deg": latitude,
                "exact_coordinate_seen_in_train": (round(longitude, 8), round(latitude, 8)) in train_coordinates,
                "nearest_train_longitude_distance_deg": float(nearest),
                "longitude_sector": fixed_longitude_sector(longitude),
            }
        )

    split_summaries: dict[str, Any] = {}
    for split in ("train", "val", "test"):
        split_rows = [row for row in coordinate_rows if row["split"] == split]
        if not split_rows:
            continue
        longitudes = [float(row["start_lon_deg"]) for row in split_rows]
        latitudes = [float(row["start_lat_deg"]) for row in split_rows]
        unseen = [not bool(row["exact_coordinate_seen_in_train"]) for row in split_rows]
        distances = [float(row["nearest_train_longitude_distance_deg"]) for row in split_rows]
        accuracy = None
        if split != "train":
            predicted = [_nearest_train_label(float(row["start_lon_deg"]), train_rows) for row in split_rows]
            accuracy = float(np.mean([pred == row["condition"] for pred, row in zip(predicted, split_rows)]))
        split_summaries[split] = {
            "row_count": len(split_rows),
            "unique_coordinate_count": len({(row["start_lon_deg"], row["start_lat_deg"]) for row in split_rows}),
            "unique_longitude_count": len(set(longitudes)),
            "longitude_min_deg": min(longitudes),
            "longitude_max_deg": max(longitudes),
            "latitude_min_deg": min(latitudes),
            "latitude_max_deg": max(latitudes),
            "condition_counts": dict(Counter(str(row["condition"]) for row in split_rows)),
            "sector_counts": dict(Counter(str(row["longitude_sector"]) for row in split_rows)),
            "unseen_exact_coordinate_fraction": float(np.mean(unseen)),
            "nearest_train_longitude_distance_deg": {
                "mean": float(np.mean(distances)),
                "median": float(np.median(distances)),
                "p90": float(np.percentile(distances, 90)),
                "max": float(np.max(distances)),
            },
            "longitude_only_1nn_condition_accuracy": accuracy,
        }
    return {
        "all_latitudes_zero": all(abs(float(row["start_lat_deg"])) < 1e-9 for row in rows),
        "scope": "continuous_equatorial_longitude" if all(abs(float(row["start_lat_deg"])) < 1e-9 for row in rows) else "continuous_2d_start",
        "splits": split_summaries,
    }, coordinate_rows


def _percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "median": 0.0, "mean": 0.0, "p90": 0.0, "max": 0.0}
    return {
        "min": float(np.min(values)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "p90": float(np.percentile(values, 90)),
        "max": float(np.max(values)),
    }


def external_condition_feasibility(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for dataset in ("sitzmann", "salient360"):
        dataset_rows = [row for row in rows if row.get("dataset") == dataset and row.get("split") in {"train", "val"}]
        observer_counts: list[float] = []
        trajectory_lengths: list[float] = []
        min_sector_counts: list[float] = []
        all_sector_three = 0
        missing_or_invalid = 0
        first_points = 0
        for row in dataset_rows:
            path = Path(str(row.get("fixations", "")))
            if not path.exists():
                missing_or_invalid += 1
                continue
            with np.load(path) as data:
                fixations = np.asarray(data["fixations"], dtype=np.float32)
            if fixations.ndim != 2 or fixations.shape[1] < 6:
                missing_or_invalid += 1
                continue
            sectors = [0, 0, 0]
            observer_ids = np.unique(fixations[:, 3].astype(np.int64))
            observer_counts.append(float(len(observer_ids)))
            for observer_id in observer_ids:
                trajectory = fixations[fixations[:, 3].astype(np.int64) == observer_id]
                trajectory = trajectory[np.argsort(trajectory[:, 2], kind="stable")]
                if not len(trajectory):
                    continue
                trajectory_lengths.append(float(len(trajectory)))
                longitude = float(trajectory[0, 0]) * 360.0 - 180.0
                sectors[fixed_longitude_sector(longitude)] += 1
                first_points += 1
            min_sector_counts.append(float(min(sectors)))
            if min(sectors) >= 3:
                all_sector_three += 1
        result[dataset] = {
            "image_count": len(dataset_rows),
            "missing_or_invalid_fixation_rows": missing_or_invalid,
            "observer_count_per_image": _percentiles(observer_counts),
            "trajectory_length": _percentiles(trajectory_lengths),
            "first_fixation_count": first_points,
            "minimum_fixed_sector_observers_per_image": _percentiles(min_sector_counts),
            "images_with_all_three_sectors_at_least_3_observers": all_sector_three,
            "fraction_images_with_all_three_sectors_at_least_3_observers": float(all_sector_three / max(len(dataset_rows), 1)),
            "condition_target_requirement": "exclude the first/conditioning fixation from the predicted target",
        }
    return result


def _spherical_angular_distance_deg(lon_a: float, lat_a: float, lon_b: float, lat_b: float) -> float:
    lon_a_r, lat_a_r = np.deg2rad([lon_a, lat_a])
    lon_b_r, lat_b_r = np.deg2rad([lon_b, lat_b])
    cosine = (
        np.sin(lat_a_r) * np.sin(lat_b_r)
        + np.cos(lat_a_r) * np.cos(lat_b_r) * np.cos(lon_a_r - lon_b_r)
    )
    return float(np.rad2deg(np.arccos(np.clip(cosine, -1.0, 1.0))))


def jufe_condition_target_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Audit conditioning-fixation inclusion without opening the frozen test targets."""

    opened_splits: set[str] = set()
    first_distances: list[float] = []
    trajectory_lengths: list[float] = []
    first_mass_fractions: list[float] = []
    row_count = 0
    observer_count = 0
    for row in rows:
        split = str(row.get("split"))
        if split not in {"train", "val"}:
            continue
        opened_splits.add(split)
        path = Path(str(row.get("fixations", "")))
        if not path.exists():
            continue
        with np.load(path) as data:
            fixations = np.asarray(data["fixations"], dtype=np.float32)
        row_count += 1
        row_observers = np.unique(fixations[:, 3].astype(np.int64))
        if len(fixations):
            first_mass_fractions.append(float(len(row_observers) / len(fixations)))
        for current_observer in row_observers:
            trajectory = fixations[fixations[:, 3].astype(np.int64) == current_observer]
            trajectory = trajectory[np.argsort(trajectory[:, 2], kind="stable")]
            if not len(trajectory):
                continue
            observer_count += 1
            trajectory_lengths.append(float(len(trajectory)))
            first_lon = float(trajectory[0, 0]) * 360.0 - 180.0
            first_lat = 90.0 - float(trajectory[0, 1]) * 180.0
            first_distances.append(
                _spherical_angular_distance_deg(
                    first_lon,
                    first_lat,
                    float(row["start_lon_deg"]),
                    float(row.get("start_lat_deg", 0.0)),
                )
            )
    return {
        "opened_splits": sorted(opened_splits),
        "test_target_opened": False,
        "conditioned_rows_opened": row_count,
        "observer_trajectories": observer_count,
        "trajectory_length": _percentiles(trajectory_lengths),
        "first_fixation_angular_distance_deg": _percentiles(first_distances),
        "first_fixation_within_5deg_fraction": float(
            np.mean([distance <= 5.0 for distance in first_distances]) if first_distances else 0.0
        ),
        "conditioning_fixation_mass_fraction_mean": float(np.mean(first_mass_fractions) if first_mass_fractions else 0.0),
        "target_includes_conditioning_fixation": True,
        "interpretation": (
            "The conditioned target includes the initial fixation. Report its mass share and require external "
            "diagnostic targets to exclude the conditioning fixation to avoid a tautological peak."
        ),
    }


def runtime_input_audit(processed_root: Path) -> dict[str, Any]:
    samples: dict[str, Any] = {}
    for split in ("train", "val"):
        dataset = ProcessedSaliencyDataset(
            processed_root,
            split=split,
            augment_roll=False,
            condition_startpoint_channels=True,
            startpoint_condition_mode="actual",
            startpoint_prior_sigma_px=10.0,
        )
        item = dataset[0]
        row = dataset.rows[0]
        expected = _startpoint_relative_channels(
            row,
            int(item["image"].shape[-2]),
            int(item["image"].shape[-1]),
            item["area_weights"][0].numpy(),
            sigma_px=10.0,
        )
        samples[split] = {
            "input_channels": int(item["image"].shape[0]),
            "dataset_item_keys": sorted(item),
            "relative_start_channel_max_abs_error": float(
                np.max(np.abs(item["image"][8:13].numpy() - expected))
            ),
            "categorical_condition_tensor_present": any(
                key in item for key in ("condition", "condition_id", "good_bad", "observer_id")
            ),
        }
    return {
        "expected_channel_decomposition": "RGB3 + absolute_ERP_geometry5 + continuous_relative_start5",
        "samples": samples,
        "test_target_opened": False,
    }


def raw_hm_provenance(rows: list[dict[str, Any]], raw_hm_root: Path) -> dict[str, Any]:
    counts: list[int] = []
    missing = 0
    seen: set[tuple[str, str]] = set()
    for row in rows:
        key = (str(row["condition"]), str(row["source_image_id"]))
        if key in seen:
            continue
        seen.add(key)
        count = len(list((raw_hm_root / key[0]).glob(f"*/*/{key[1]}.csv")))
        counts.append(count)
        if count == 0:
            missing += 1
    return {
        "condition_source_pairs": len(seen),
        "missing_condition_source_pairs": missing,
        "observer_file_count": _percentiles([float(value) for value in counts]),
        "all_pairs_have_raw_files": missing == 0,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit P1 continuous-start provenance and feasibility without test evaluation.")
    parser.add_argument(
        "--conditioned-root",
        default="data/processed/jufe_start_conditioned_256x512_k6_order_groupdisjoint_v1",
    )
    parser.add_argument(
        "--multi360-root",
        default="data/processed/multi360_sitzmann_salient360_jufe_256x512_k6_order_groupdisjoint_v1",
    )
    parser.add_argument("--raw-jufe-root", default="${SCANDMM_ROOT}/data/raw/jufe")
    parser.add_argument("--output-root", default="reports/p1_condition_audit")
    args = parser.parse_args()

    conditioned_root = Path(args.conditioned_root)
    multi360_root = Path(args.multi360_root)
    raw_jufe_root = Path(args.raw_jufe_root)
    output_root = ensure_dir(args.output_root)
    rows = read_jsonl(conditioned_root / "manifest.jsonl")
    startpoint_path = raw_jufe_root / "startpoint.xls"
    startpoints = load_startpoints_xls(startpoint_path)

    pair_audit = audit_conditioned_rows(rows, startpoints, check_artifacts=True)
    coordinate_summary, coordinate_rows = condition_coordinate_summary(rows)
    external = external_condition_feasibility(read_jsonl(multi360_root / "manifest.jsonl"))
    runtime = runtime_input_audit(conditioned_root)
    raw_hm = raw_hm_provenance(rows, raw_jufe_root / "HMData_revised_final")
    target_audit = jufe_condition_target_audit(rows)
    base_pairs = defaultdict(dict)
    for row in rows:
        base_pairs[str(row["base_image_id"])][str(row["condition"])] = row
    pair_deltas = [
        circular_distance_deg(pair["good"]["start_lon_deg"], pair["bad"]["start_lon_deg"])
        for pair in base_pairs.values()
        if set(pair) == {"good", "bad"}
    ]

    hard_failures = {
        "incomplete_pairs": pair_audit["complete_pair_count"] != pair_audit["pair_count"],
        "coordinate_provenance": pair_audit["coordinate_provenance_mismatches"] > 0,
        "missing_raw_startpoints": pair_audit["missing_raw_startpoint_rows"] > 0,
        "nonfinite_coordinates": pair_audit["nonfinite_coordinate_rows"] > 0,
        "split_or_group_mismatch": pair_audit["pair_split_group_mismatches"] > 0,
        "pair_image_mismatch": pair_audit["pair_image_mismatches"] > 0,
        "reciprocal_coordinate_mismatch": pair_audit["reciprocal_coordinate_mismatches"] > 0,
        "missing_artifacts": pair_audit["missing_artifact_count"] > 0,
        "missing_raw_hm": not raw_hm["all_pairs_have_raw_files"],
        "runtime_input_shape": any(sample["input_channels"] != 13 for sample in runtime["samples"].values()),
        "runtime_label_tensor": any(
            sample["categorical_condition_tensor_present"] for sample in runtime["samples"].values()
        ),
        "runtime_channel_mismatch": any(
            sample["relative_start_channel_max_abs_error"] > 1e-6 for sample in runtime["samples"].values()
        ),
    }
    result = {
        "protocol": "P1_start_conditioned",
        "gate": "P1-A",
        "conditioned_manifest": str(conditioned_root / "manifest.jsonl"),
        "raw_startpoint_table": str(startpoint_path),
        "raw_startpoint_table_sha256": _sha256(startpoint_path),
        "pair_and_provenance_audit": pair_audit,
        "raw_hm_provenance": raw_hm,
        "runtime_input_audit": runtime,
        "conditioned_target_audit": target_audit,
        "coordinate_summary": coordinate_summary,
        "pair_start_angular_distance_deg": _percentiles(pair_deltas),
        "external_feasibility": external,
        "hard_failures": hard_failures,
        "data_audit_pass": not any(hard_failures.values()),
        "scope_limit": "JUFE provides continuous equatorial longitude only; latitude generalization is not supported.",
        "test_status": "metadata_integrity_only; no test target or prediction opened",
    }
    (output_root / "condition_audit.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output_root / "external_feasibility.json").write_text(
        json.dumps(external, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _write_csv(output_root / "coordinate_rows.csv", coordinate_rows)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
