#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sphererate.baselines.scanpath_time import second_sampled_scanpaths_to_cumulative_maps
from sphererate.data.early_dataset import EarlyHorizonDataset
from sphererate.evaluation.metrics import kld, sim
from sphererate.utils import ensure_dir


SEEDS = (20260710, 20260711, 20260712)
COUNTS_PER_SEED = (10, 20, 50, 100, 200)
REPLICATES = 5
METHOD_ROOT = {
    "Pathformer3D": "kernel_v2_pathformer3d_retrained",
    "ScanDMM": "kernel_v2_scandmm_declared_start",
}


def _paths(method: str, seed: int, image_id: str) -> np.ndarray:
    path = ROOT / f"runs/{METHOD_ROOT[method]}_seed{seed}_gate/scanpaths/{image_id}.npy"
    values = np.load(path).astype(np.float32)
    if values.shape != (200, 20, 2):
        raise ValueError(f"Unexpected scanpath cache: {path}: {values.shape}")
    return values


def _render(paths: np.ndarray, sample: dict[str, Any]) -> np.ndarray:
    return second_sampled_scanpaths_to_cumulative_maps(
        paths,
        horizons_s=(3.0,),
        height=int(sample["target"].shape[-2]),
        width=int(sample["target"].shape[-1]),
        renderer="sphere_chord",
        sigma_deg=2.0,
        coordinate_order="yx",
    )[0]


def _subset(values: np.ndarray, count: int, *, key: str) -> np.ndarray:
    if count == len(values):
        return values
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    return values[rng.choice(len(values), size=int(count), replace=False)]


def _maps_for_count(
    method: str,
    sample: dict[str, Any],
    count_per_seed: int,
    replicate: int,
) -> np.ndarray:
    image_id = str(sample["image_id"])
    maps = []
    for seed in SEEDS:
        values = _paths(method, seed, image_id)
        selected = _subset(
            values,
            count_per_seed,
            key=f"{method}:{image_id}:{seed}:{count_per_seed}:{replicate}",
        )
        maps.append(_render(selected, sample))
    return np.mean(maps, axis=0).astype(np.float32)


def _plot(output: Path, display_maps: dict[str, dict[int, np.ndarray]], sample: dict[str, Any]) -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.0,
            "axes.titlesize": 7.3,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.dpi": 500,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.transparent": False,
        }
    )
    rgb = np.clip(sample["image"][:3].numpy().transpose(1, 2, 0), 0.0, 1.0)
    figure, axes = plt.subplots(2, len(COUNTS_PER_SEED), figsize=(7.2, 2.45), constrained_layout=True)
    for row, method in enumerate(METHOD_ROOT):
        reference = display_maps[method][600]
        area = sample["area_weights"].numpy()[0]
        for col, per_seed in enumerate(COUNTS_PER_SEED):
            total = per_seed * len(SEEDS)
            current = display_maps[method][total]
            axis = axes[row, col]
            axis.imshow(rgb, aspect="equal")
            axis.imshow(current, cmap="jet", alpha=0.42, aspect="equal")
            axis.scatter([0.75 * rgb.shape[1]], [0.5 * rgb.shape[0]], s=10, marker="+", c="#00D5FF", linewidths=0.9)
            axis.set_box_aspect(0.5)
            axis.set_xticks([])
            axis.set_yticks([])
            axis.set_title(f"{total} paths\nSIM to 600: {sim(current, reference, area):.2f}")
            if col == 0:
                axis.set_ylabel(method, rotation=0, labelpad=30, va="center", fontweight="bold")
    stem = output / "p10_population_count_sensitivity"
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(stem.with_suffix(f".{suffix}"), bbox_inches="tight", facecolor="white", transparent=False)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit whether 3x200 scanpaths over-smooth qualitative population density.")
    parser.add_argument("--output-dir", default="reports/kernel_v2_scanpath_sample_count_audit")
    args = parser.parse_args()
    output = ensure_dir(args.output_dir)
    dataset = EarlyHorizonDataset(
        "data/processed/early_horizon_salient360_spherical2deg_v2",
        split="gate",
        target_mode="correct",
    )
    samples = [dataset[index] for index in range(len(dataset))]
    rows = []
    display_maps: dict[str, dict[int, np.ndarray]] = {method: {} for method in METHOD_ROOT}
    display_sample = next(sample for sample in samples if str(sample["image_id"]) == "P10")
    for method in METHOD_ROOT:
        references = {
            str(sample["image_id"]): _maps_for_count(method, sample, 200, 0)
            for sample in samples
        }
        display_maps[method][600] = references["P10"]
        for count_per_seed in COUNTS_PER_SEED:
            total = count_per_seed * len(SEEDS)
            repetitions = 1 if count_per_seed == 200 else REPLICATES
            for replicate in range(repetitions):
                for sample in samples:
                    image_id = str(sample["image_id"])
                    current = (
                        references[image_id]
                        if count_per_seed == 200
                        else _maps_for_count(method, sample, count_per_seed, replicate)
                    )
                    if image_id == "P10" and replicate == 0:
                        display_maps[method][total] = current
                    area = sample["area_weights"].numpy()[0]
                    target = sample["target"].numpy()[2]
                    rows.append(
                        {
                            "method": method,
                            "paths_per_seed": count_per_seed,
                            "total_paths": total,
                            "replicate": replicate,
                            "content_id": str(sample["canonical_content_id"]),
                            "sim_to_600": sim(current, references[image_id], area),
                            "kld_to_600": kld(current, references[image_id], area),
                            "kld_to_gt": kld(current, target, area),
                        }
                    )
    summaries = []
    for method in METHOD_ROOT:
        for count_per_seed in COUNTS_PER_SEED:
            current = [
                row for row in rows
                if row["method"] == method and row["paths_per_seed"] == count_per_seed
            ]
            reference_kld = np.mean(
                [
                    row["kld_to_gt"] for row in rows
                    if row["method"] == method and row["paths_per_seed"] == 200
                ]
            )
            summaries.append(
                {
                    "method": method,
                    "paths_per_seed": count_per_seed,
                    "total_paths": count_per_seed * len(SEEDS),
                    "sim_to_600_mean": float(np.mean([row["sim_to_600"] for row in current])),
                    "sim_to_600_std": float(np.std([row["sim_to_600"] for row in current])),
                    "kld_to_600_mean": float(np.mean([row["kld_to_600"] for row in current])),
                    "kld_to_gt_mean": float(np.mean([row["kld_to_gt"] for row in current])),
                    "kld_to_gt_difference_from_600": float(
                        np.mean([row["kld_to_gt"] for row in current]) - reference_kld
                    ),
                    "content_replicate_rows": len(current),
                }
            )
    result = {
        "split": "historically_seen_gate",
        "horizon_s": 3.0,
        "reference": "three-seed mean of 200 paths/seed (600 paths total)",
        "subset_replicates": REPLICATES,
        "summaries": summaries,
        "rows": rows,
        "test_target_loads": 0,
    }
    (output / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _plot(output, display_maps, display_sample)
    lines = [
        "# Scanpath population-density sample-count audit",
        "",
        "This audit separates Monte Carlo population-density stability from trajectory visualization. The 600-path map is a three-seed ensemble reference, not an individual model output. Smaller counts use deterministic, without-replacement subsets and five repeats.",
        "",
        "| Method | Total paths | SIM to 600 ↑ | KLD to 600 ↓ | KLD-to-GT change vs 600 |",
        "|---|---:|---:|---:|---:|",
    ]
    for item in summaries:
        lines.append(
            f"| {item['method']} | {item['total_paths']} | {item['sim_to_600_mean']:.4f} ± {item['sim_to_600_std']:.4f} | "
            f"{item['kld_to_600_mean']:.4f} | {item['kld_to_gt_difference_from_600']:+.4f} |"
        )
    lines.extend(
        [
            "",
            "![P10 sample-count sensitivity](p10_population_count_sensitivity.png)",
            "",
            "The figure uses native 2° densities without the extra 10 px display smoothing. Use 200 paths from the fixed central seed for representative qualitative population maps, 30 paths for trajectory motion, and three-seed metric averaging for formal quantitative results. Do not use a 600-path map ensemble as if it were a representative single-model qualitative output.",
        ]
    )
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(output / "summary.md"), "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
