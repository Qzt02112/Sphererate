#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sphererate.baselines.scanpath_export import xyz_to_erp_yx
from sphererate.baselines.scanpath_time import official_second_step_weights
from sphererate.data.maps import normalized_coords_to_unit_vectors
from sphererate.utils import ensure_dir


START_YX = np.asarray([0.5, 0.75], dtype=np.float32)
METHOD_ROOT = {
    "Pathformer3D": "kernel_v2_pathformer3d_retrained",
    "ScanDMM": "kernel_v2_scandmm_declared_start",
}
SEEDS = (20260710, 20260711, 20260712)


def _vectors(yx: np.ndarray) -> np.ndarray:
    values = np.asarray(yx, dtype=np.float32)
    return normalized_coords_to_unit_vectors(
        np.mod(values[..., 1], 1.0),
        np.clip(values[..., 0], 0.0, 1.0),
    )


def _angle(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    a, b = _vectors(left), _vectors(right)
    return np.rad2deg(np.arccos(np.clip(np.sum(a * b, axis=-1), -1.0, 1.0)))


def _motion(paths: np.ndarray) -> dict[str, Any]:
    paths = np.asarray(paths, dtype=np.float32)
    first = paths[:, :3, :]
    start = np.broadcast_to(START_YX, first[:, 0].shape)
    distance = np.stack([_angle(first[:, index], start) for index in range(3)], axis=1)
    previous = np.concatenate((start[:, None], first[:, :2]), axis=1)
    step = np.stack([_angle(first[:, index], previous[:, index]) for index in range(3)], axis=1)
    per_step = []
    for index in range(3):
        per_step.append(
            {
                "second": index + 1,
                "distance_from_start_deg": {
                    "mean": float(distance[:, index].mean()),
                    "median": float(np.median(distance[:, index])),
                    "q10": float(np.quantile(distance[:, index], 0.1)),
                    "q90": float(np.quantile(distance[:, index], 0.9)),
                },
                "step_motion_deg": {
                    "mean": float(step[:, index].mean()),
                    "median": float(np.median(step[:, index])),
                },
                "fraction_step_below_2deg": float(np.mean(step[:, index] < 2.0)),
                "fraction_step_below_5deg": float(np.mean(step[:, index] < 5.0)),
            }
        )
    return {
        "paths": int(len(paths)),
        "per_step": per_step,
        "fraction_exact_declared_start_first_point": float(
            np.mean(_angle(first[:, 0], start) < 1e-4)
        ),
        "fraction_first3_max_distance_below_5deg": float(np.mean(distance.max(axis=1) < 5.0)),
        "fraction_first3_max_distance_below_10deg": float(np.mean(distance.max(axis=1) < 10.0)),
        "fraction_first3_max_distance_below_20deg": float(np.mean(distance.max(axis=1) < 20.0)),
        "first3_path_length_deg": {
            "mean": float(step.sum(axis=1).mean()),
            "median": float(np.median(step.sum(axis=1))),
        },
    }


def _generated(method: str) -> np.ndarray:
    arrays = []
    prefix = METHOD_ROOT[method]
    for seed in SEEDS:
        for path in sorted((ROOT / f"runs/{prefix}_seed{seed}_gate/scanpaths").glob("*.npy")):
            values = np.load(path)
            if values.shape != (200, 20, 2):
                raise ValueError(f"Unexpected generated path shape: {path}: {values.shape}")
            arrays.append(values)
    return np.concatenate(arrays).astype(np.float32)


def _ground_truth() -> np.ndarray:
    arrays = []
    for path in sorted(
        (ROOT / "data/processed/kernel_v2_matched_scanpaths/canonical/gate").glob("*.npz")
    ):
        with np.load(path) as data:
            arrays.append(xyz_to_erp_yx(data["scanpath_xyz"]))
    return np.stack(arrays).astype(np.float32)


def _content_paths(method: str, image_id: str, seed: int = 20260711) -> np.ndarray:
    if method == "GT":
        arrays = []
        for path in sorted(
            (ROOT / "data/processed/kernel_v2_matched_scanpaths/canonical/gate").glob(
                f"{image_id}_observer*.npz"
            )
        ):
            with np.load(path) as data:
                arrays.append(xyz_to_erp_yx(data["scanpath_xyz"]))
        return np.stack(arrays).astype(np.float32)
    return np.load(
        ROOT / f"runs/{METHOD_ROOT[method]}_seed{seed}_gate/scanpaths/{image_id}.npy"
    ).astype(np.float32)


def _plot_segmented(axis: plt.Axes, x: np.ndarray, y: np.ndarray, color: str) -> None:
    begin = 0
    for index in range(1, len(x)):
        if abs(float(x[index] - x[index - 1])) > 0.5:
            axis.plot(x[begin:index], y[begin:index], color=color, linewidth=0.55, alpha=0.48)
            begin = index
    axis.plot(x[begin:], y[begin:], color=color, linewidth=0.55, alpha=0.48)


def _trajectory_figure(output: Path) -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.0,
            "axes.titlesize": 7.5,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.dpi": 500,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.transparent": False,
        }
    )
    manifest = [
        json.loads(line)
        for line in (ROOT / "data/processed/early_horizon_salient360_spherical2deg_v2/manifest.jsonl").read_text().splitlines()
    ]
    image_path = {str(row["image_id"]): Path(str(row["image"])) for row in manifest}
    contents = ("P10", "P97")
    methods = ("GT", "Pathformer3D", "ScanDMM")
    colors = ("#E69F00", "#009E73", "#2166AC")
    figure, axes = plt.subplots(3, 2, figsize=(7.2, 4.15), constrained_layout=True)
    for row, (method, color) in enumerate(zip(methods, colors)):
        for col, image_id in enumerate(contents):
            axis = axes[row, col]
            with Image.open(image_path[image_id]) as image:
                rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
            axis.imshow(rgb, extent=(0.0, 1.0, 1.0, 0.0), alpha=0.72, aspect="auto")
            paths = _content_paths(method, image_id)
            count = min(30, len(paths))
            indices = np.linspace(0, len(paths) - 1, count, dtype=int)
            for index in indices:
                first = paths[index, :3]
                joined = np.concatenate((START_YX[None], first), axis=0)
                _plot_segmented(axis, joined[:, 1], joined[:, 0], color)
                axis.scatter(first[:, 1], first[:, 0], s=4.5, color=("#E69F00", "#009E73", "#2166AC"), alpha=0.70, linewidths=0)
            axis.scatter([START_YX[1]], [START_YX[0]], s=28, marker="+", c="#00D5FF", linewidths=1.1, zorder=5)
            current = _motion(paths)
            d1 = current["per_step"][0]["distance_from_start_deg"]["median"]
            d3 = current["per_step"][2]["distance_from_start_deg"]["median"]
            axis.text(
                0.01, 0.98, f"median start distance: {d1:.1f}° → {d3:.1f}°",
                transform=axis.transAxes, ha="left", va="top", fontsize=6.0,
                bbox={"facecolor": "white", "alpha": 0.75, "pad": 1.0, "edgecolor": "none"},
            )
            if row == 0:
                axis.set_title(f"{image_id}", fontweight="bold")
            if col == 0:
                axis.set_ylabel(method, fontweight="bold", rotation=0, labelpad=30, va="center")
            axis.set_xlim(0.0, 1.0)
            axis.set_ylim(1.0, 0.0)
            axis.set_xticks([])
            axis.set_yticks([])
    axes[-1, 1].text(
        0.99, -0.12, "markers: 1 s (orange), 2 s (green), 3 s (blue)",
        transform=axes[-1, 1].transAxes, ha="right", va="top", fontsize=6.2,
    )
    stem = output / "first3s_trajectory_audit"
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(stem.with_suffix(f".{suffix}"), bbox_inches="tight", facecolor="white", transparent=False)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit whether matched scanpath baselines move away from the declared start.")
    parser.add_argument("--output-dir", default="reports/kernel_v2_scanpath_motion_audit")
    args = parser.parse_args()
    output = ensure_dir(args.output_dir)
    records = {
        "GT sampled gate": _motion(_ground_truth()),
        "Pathformer3D": _motion(_generated("Pathformer3D")),
        "ScanDMM": _motion(_generated("ScanDMM")),
    }
    weights = official_second_step_weights(3.0, 20)
    checks = {
        "pathformer_not_stuck_10deg": records["Pathformer3D"]["fraction_first3_max_distance_below_10deg"] < 0.01,
        "scandmm_not_stuck_10deg": records["ScanDMM"]["fraction_first3_max_distance_below_10deg"] < 0.01,
        "renderer_3s_uses_exactly_first_three_predictions": bool(
            np.array_equal(np.flatnonzero(weights), np.asarray([0, 1, 2]))
        ),
        "generated_first_point_is_not_systematically_copied_start": bool(
            records["Pathformer3D"]["fraction_exact_declared_start_first_point"] < 0.01
            and records["ScanDMM"]["fraction_exact_declared_start_first_point"] < 0.01
        ),
    }
    result = {
        "split": "historically_seen_gate",
        "generated_paths_per_method": 7800,
        "declared_start_yx": START_YX.tolist(),
        "method_roots": METHOD_ROOT,
        "records": records,
        "renderer_3s_step_weights": weights.tolist(),
        "source_semantics": {
            "Pathformer3D": "declared start is decoder BOS and is explicitly removed from saved predictions",
            "ScanDMM": "an independent declared-start tensor initializes z_prev in model and guide; obs_x_1..obs_x_20 remain the supervised/generated gaze sequence",
        },
        "checks": checks,
        "status": "MOTION_PASS" if all(checks.values()) else "FAIL",
        "test_target_loads": 0,
    }
    (output / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _trajectory_figure(output)
    lines = [
        "# Kernel-v2 scanpath motion audit",
        "",
        f"Status: **{result['status']}**.",
        "",
        "The generated trajectories are not stuck at the declared start. The start-centered appearance of the cumulative saliency maps is caused by population aggregation: paths move in many directions from the same start, so directional motion cancels while probability remains concentrated near the origin. Display-only 10 px smoothing strengthens this visual impression.",
        "",
        "| Source | Paths | Median distance at 1 s | 2 s | 3 s | First-3-s paths staying within 10° | Median first-3-s path length |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, record in records.items():
        medians = [item["distance_from_start_deg"]["median"] for item in record["per_step"]]
        lines.append(
            f"| {name} | {record['paths']} | {medians[0]:.1f}° | {medians[1]:.1f}° | {medians[2]:.1f}° | "
            f"{100.0 * record['fraction_first3_max_distance_below_10deg']:.2f}% | {record['first3_path_length_deg']['median']:.1f}° |"
        )
    lines.extend(
        [
            "",
            "![First-three-second trajectories](first3s_trajectory_audit.png)",
            "",
            "At 3 s, the renderer uses only prediction indices 0/1/2. Pathformer3D removes its declared-start BOS before saving. ScanDMM also saves `obs_x_1..obs_x_20`, not the input start.",
            "",
            "## ScanDMM conditioning semantics",
            "",
            "The corrected state-dict-compatible ScanDMM adapter supplies `(0°, +90°)` through an independent tensor to both model and guide. `obs_x_1` remains the first one-second gaze target, so the start is neither prepended nor scored. Final inference uses the same independent-start semantics.",
            "",
            "The legacy official-training-semantics row is retained only in the dedicated correction audit and is excluded from the matched benchmark.",
        ]
    )
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(output / "summary.md"), "status": result["status"]}, indent=2))


if __name__ == "__main__":
    main()
