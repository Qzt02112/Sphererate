#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def tangent_coordinates(x_norm: np.ndarray, y_norm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    longitude = 2.0 * np.pi * np.asarray(x_norm, dtype=np.float64) - np.pi
    latitude = np.pi / 2.0 - np.pi * np.asarray(y_norm, dtype=np.float64)
    distance = np.arccos(np.clip(np.cos(latitude) * np.cos(longitude), -1.0, 1.0))
    scale = np.divide(
        distance,
        np.sin(distance),
        out=np.ones_like(distance),
        where=np.abs(np.sin(distance)) > 1e-8,
    )
    east = np.cos(latitude) * np.sin(longitude) * scale
    north = np.sin(latitude) * scale
    return east, north


def clipped_overlap(timestamp: np.ndarray, duration: np.ndarray, start: float, end: float) -> np.ndarray:
    return np.maximum(
        0.0,
        np.minimum(timestamp + duration, float(end)) - np.maximum(timestamp, float(start)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit train-only Sitzmann tangent-plane dwell dynamics.")
    parser.add_argument(
        "--data-root",
        default="data/processed/sitzmann_official_256x512_spherical2deg_standard_v2",
    )
    parser.add_argument("--output-dir", default="reports/sitzmann_standard_v2/tangent_calibration")
    args = parser.parse_args()
    root = Path(args.data_root)
    rows = [
        json.loads(line)
        for line in (root / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    train = [row for row in rows if row["split"] == "train"]
    if not train or any(row.get("model_declared_start_yaw_deg") != 0.0 for row in train):
        raise ValueError("Calibration requires the aligned Sitzmann train split with model start at 0 degrees")
    intervals = ((0.0, 0.5), (0.5, 1.0), (1.0, 3.0), (3.0, 5.0))
    records = []
    for start, end in intervals:
        east_parts, north_parts, weight_parts = [], [], []
        for row in train:
            with np.load(row["fixations"], allow_pickle=False) as data:
                fixations = np.asarray(data["fixations"], dtype=np.float64)
            weights = clipped_overlap(fixations[:, 2], fixations[:, 4], start, end)
            keep = weights > 0.0
            east, north = tangent_coordinates(fixations[keep, 0], fixations[keep, 1])
            east_parts.append(east)
            north_parts.append(north)
            weight_parts.append(weights[keep])
        east = np.concatenate(east_parts)
        north = np.concatenate(north_parts)
        weights = np.concatenate(weight_parts)
        total = float(weights.sum())
        normalized = weights / total
        mean_east = float(np.sum(normalized * east))
        mean_north = float(np.sum(normalized * north))
        variance_east = float(np.sum(normalized * np.square(east - mean_east)))
        variance_north = float(np.sum(normalized * np.square(north - mean_north)))
        records.append(
            {
                "interval_s": [start, end],
                "positive_fixation_count": int(len(weights)),
                "clipped_dwell_s": total,
                "offset_deg": np.rad2deg([mean_east, mean_north]).tolist(),
                "std_deg": np.rad2deg(np.sqrt([variance_east, variance_north])).tolist(),
                "beta": [1.0 / (2.0 * variance_east), 1.0 / (2.0 * variance_north)],
            }
        )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    result = {
        "data_root": str(root),
        "calibration_split": "train",
        "train_condition_count": len(train),
        "train_scene_count": len({row["canonical_content_id"] for row in train}),
        "val_target_loads": 0,
        "test_target_loads": 0,
        "estimator": "clipped-dwell weighted tangent-plane Gaussian; beta=1/(2 variance)",
        "intervals": records,
    }
    (output / "results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    lines = [
        "# Sitzmann train-only tangent calibration",
        "",
        "Only the 16 training scenes are read; validation/test target loads are zero.",
        "",
        "| Interval | East/North offset (deg) | East/North std (deg) | East/North beta |",
        "|---|---:|---:|---:|",
    ]
    for record in records:
        a, b = record["interval_s"]
        oe, on = record["offset_deg"]
        se, sn = record["std_deg"]
        be, bn = record["beta"]
        lines.append(
            f"| {a:g}--{b:g} s | {oe:.3f} / {on:.3f} | {se:.3f} / {sn:.3f} | {be:.4f} / {bn:.4f} |"
        )
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(output / "summary.md")}, indent=2))


if __name__ == "__main__":
    main()

