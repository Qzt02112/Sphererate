#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from scripts.evaluate_eh3_gate import _predict_learned
from scripts.evaluate_kernel_v2_temporal_baseline import _predict as _predict_tempsal
from scripts.plot_kernel_v2_linear_visualization import _linear_relative, _sitzmann_display_smooth
from sphererate.baselines.scanpath_time import second_sampled_scanpaths_to_cumulative_maps
from sphererate.data.early_dataset import EarlyHorizonDataset
from sphererate.evaluation.tile_prefetch import gaze_mass_coverage, rank_tiles
from sphererate.utils import ensure_dir


HORIZON_INDEX = 2
HORIZON_S = 3.0
TILE_ROWS = 8
TILE_COLS = 16
BUDGET_PERCENT = 10.0
EXAMPLES = (
    ("P95", "gain"),
    ("P37", "near-oracle"),
    ("P10", "failure"),
)
METHODS = (
    "GT ranking",
    "Early template",
    "GBVS360-R",
    "MSDB-25-R",
    "Pathformer3D",
    "ScanDMM",
    "HAT",
    "TempSAL-ERP",
    "SphereHorizon",
)
METHOD_LABELS = {
    "GT ranking": "GT ranking",
    "Early template": "Early template",
    "GBVS360-R": r"GBVS360$^{\dagger}$",
    "MSDB-25-R": r"MSDB-25$^{\dagger}$",
    "Pathformer3D": "Pathformer3D",
    "ScanDMM": "ScanDMM",
    "HAT": "HAT",
    "TempSAL-ERP": "TempSAL-ERP",
    "SphereHorizon": "SphereHorizon",
}


def _style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 6.4,
            "axes.titlesize": 7.0,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.dpi": 500,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.transparent": False,
        }
    )


def _tile_rect(tile_id: int, height: int, width: int) -> tuple[float, float, float, float]:
    row, col = divmod(int(tile_id), TILE_COLS)
    return (
        col * width / TILE_COLS,
        row * height / TILE_ROWS,
        width / TILE_COLS,
        height / TILE_ROWS,
    )


def _draw_panel(
    axis: plt.Axes,
    rgb: np.ndarray,
    density: np.ndarray,
    selected: set[int],
    oracle: set[int],
    *,
    coverage: float,
    show_start: bool = True,
) -> None:
    display = _linear_relative(_sitzmann_display_smooth(density, 10.0))
    # Keep the panorama as spatial context, but make the tile decision the
    # dominant visual layer. This display treatment never enters evaluation.
    luminance = np.tensordot(rgb[..., :3], np.array([0.2126, 0.7152, 0.0722]), axes=1)
    desaturated = 0.15 * rgb[..., :3] + 0.85 * luminance[..., None]
    muted_rgb = np.clip(0.48 * desaturated + 0.52, 0.0, 1.0)
    axis.imshow(muted_rgb, aspect="equal")
    axis.imshow(display, cmap="jet", vmin=0.0, vmax=1.0, alpha=0.30, aspect="equal")
    height, width = density.shape

    # The dashed oracle boundary stays fixed across method panels.
    for tile_id in sorted(oracle):
        x, y, w, h = _tile_rect(tile_id, height, width)
        axis.add_patch(
            Rectangle(
                (x, y),
                w,
                h,
                fill=False,
                edgecolor="#56B4E9",
                linestyle=(0, (2, 1.5)),
                linewidth=0.85,
            )
        )

    for tile_id in sorted(selected):
        x, y, w, h = _tile_rect(tile_id, height, width)
        matched = tile_id in oracle
        color = "#0072B2" if matched else "#E69F00"
        axis.add_patch(
            Rectangle(
                (x, y),
                w,
                h,
                facecolor=mpl.colors.to_rgba(color, 0.34),
                edgecolor=color,
                linewidth=0.95,
            )
        )

    for row in range(1, TILE_ROWS):
        axis.axhline(row * height / TILE_ROWS, color="white", linewidth=0.34, alpha=0.92)
    for col in range(1, TILE_COLS):
        axis.axvline(col * width / TILE_COLS, color="white", linewidth=0.34, alpha=0.92)

    if show_start:
        axis.scatter(
            [0.75 * width],
            [0.5 * height],
            s=14,
            marker="+",
            c="#00D5FF",
            linewidths=1.15,
        )
    overlap = len(selected & oracle)
    axis.text(
        0.018,
        0.035,
        f"{coverage:.3f} | {overlap}/{len(oracle)}",
        transform=axis.transAxes,
        color="white",
        fontsize=5.0,
        bbox={"facecolor": "black", "alpha": 0.58, "pad": 1.2, "edgecolor": "none"},
    )
    axis.set_box_aspect(0.5)
    axis.set_xticks([])
    axis.set_yticks([])


def _scanpath_map(prefix: str, image_id: str, sample: dict[str, Any]) -> np.ndarray:
    paths = np.load(
        ROOT / f"runs/{prefix}_seed20260711_gate/scanpaths/{image_id}.npy"
    )
    return second_sampled_scanpaths_to_cumulative_maps(
        paths,
        horizons_s=sample["horizons_s"].numpy().tolist(),
        height=int(sample["target"].shape[-2]),
        width=int(sample["target"].shape[-1]),
        renderer="sphere_chord",
        sigma_deg=2.0,
        coordinate_order="yx",
    )[HORIZON_INDEX]


def _coverage(sample: dict[str, Any], selected: set[int]) -> float:
    with np.load(str(sample["fixations_path"])) as data:
        fixations = np.asarray(data["fixations"], dtype=np.float32)
    observer_ids = np.unique(fixations[:, 3]).astype(int).tolist()
    return gaze_mass_coverage(
        fixations,
        observer_ids=observer_ids,
        horizon_s=HORIZON_S,
        selected_tile_ids=selected,
        tile_rows=TILE_ROWS,
        tile_cols=TILE_COLS,
    )


def main() -> None:
    _style()
    output = ensure_dir("reports/kernel_v2_tile_decisions")
    dataset = EarlyHorizonDataset(
        "data/processed/early_horizon_salient360_spherical2deg_v2",
        split="gate",
        target_mode="correct",
    )
    samples = [dataset[index] for index in range(len(dataset))]
    by_id = {str(sample["image_id"]): sample for sample in samples}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sphere_seeds: list[dict[str, np.ndarray]] = []
    tempsal_seeds: list[dict[str, np.ndarray]] = []
    for seed in (20260710, 20260711, 20260712):
        sphere, _ = _predict_learned(
            ROOT / f"runs/kernel_v2_candidate_correct_seed{seed}/checkpoints/best.pt",
            samples,
            device,
        )
        tempsal, _ = _predict_tempsal(
            ROOT / f"runs/kernel_v2_tempsal_erp_seed{seed}/checkpoints/best.pt",
            samples,
            device,
        )
        sphere_seeds.append(sphere)
        tempsal_seeds.append(tempsal)

    figure, axes = plt.subplots(
        len(EXAMPLES),
        len(METHODS),
        figsize=(7.12, 2.30),
        constrained_layout=False,
        squeeze=False,
    )
    figure.subplots_adjust(
        left=0.052,
        right=0.995,
        top=0.865,
        bottom=0.125,
        wspace=0.045,
        hspace=0.075,
    )
    records: list[dict[str, Any]] = []
    for row, (image_id, role) in enumerate(EXAMPLES):
        sample = by_id[image_id]
        rgb = np.clip(sample["image"][:3].numpy().transpose(1, 2, 0), 0.0, 1.0)
        area = sample["area_weights"].numpy()[0]
        with np.load(
            ROOT / f"runs/kernel_v2_reachable_final/gbvs360/gate/predictions/{image_id}.npz"
        ) as data:
            gbvs_reachable = np.asarray(data["cumulative"][HORIZON_INDEX], dtype=np.float32)
        with np.load(
            ROOT / f"runs/kernel_v2_deepgaze_msdb/evaluated/gate/{image_id}.npz"
        ) as data:
            msdb_reachable = np.asarray(data["reachable"][HORIZON_INDEX], dtype=np.float32)
        maps = {
            "GT ranking": sample["target"].numpy()[HORIZON_INDEX],
            "Early template": sample["template"].numpy()[HORIZON_INDEX],
            "GBVS360-R": gbvs_reachable,
            "MSDB-25-R": msdb_reachable,
            "Pathformer3D": _scanpath_map(
                "kernel_v2_pathformer3d_retrained", image_id, sample
            ),
            "ScanDMM": _scanpath_map(
                "kernel_v2_scandmm_declared_start", image_id, sample
            ),
            "HAT": _scanpath_map("kernel_v2_hat", image_id, sample),
            "TempSAL-ERP": np.mean(
                [prediction[image_id][HORIZON_INDEX] for prediction in tempsal_seeds], axis=0
            ).astype(np.float32),
            "SphereHorizon": np.mean(
                [prediction[image_id][HORIZON_INDEX] for prediction in sphere_seeds], axis=0
            ).astype(np.float32),
        }
        selected = {
            method: set(
                rank_tiles(
                    density,
                    area,
                    tile_rows=TILE_ROWS,
                    tile_cols=TILE_COLS,
                    budget_percent=BUDGET_PERCENT,
                ).tolist()
            )
            for method, density in maps.items()
        }
        oracle = selected["GT ranking"]
        for col, method in enumerate(METHODS):
            coverage = _coverage(sample, selected[method])
            _draw_panel(
                axes[row, col],
                rgb,
                maps[method],
                selected[method],
                oracle,
                coverage=coverage,
            )
            records.append(
                {
                    "image_id": image_id,
                    "role": role,
                    "method": method,
                    "coverage": coverage,
                    "oracle_overlap": len(selected[method] & oracle),
                    "selected_tile_ids": sorted(selected[method]),
                }
            )
            if row == 0:
                axes[row, col].set_title(
                    METHOD_LABELS[method],
                    fontsize=6.2,
                    fontweight="bold" if method in {"GT ranking", "SphereHorizon"} else "normal",
                    color="#1769AA" if method == "SphereHorizon" else "black",
                    pad=2.0,
                )
            if col == 0:
                axes[row, col].set_ylabel(
                    image_id,
                    rotation=0,
                    labelpad=5,
                    va="center",
                    ha="right",
                    fontsize=6.4,
                    fontweight="bold",
                )
            if method == "SphereHorizon":
                for spine in axes[row, col].spines.values():
                    spine.set_color("#1769AA")
                    spine.set_linewidth(0.9)

    figure.text(
        0.52,
        0.018,
        "dashed cyan: GT set   blue: hit   orange: outside GT",
        ha="center",
        fontsize=5.8,
        color="#4F5963",
    )
    stem = output / "matched_tile_decisions_3s"
    for suffix in ("pdf", "svg", "png"):
        figure.savefig(stem.with_suffix(f".{suffix}"), bbox_inches="tight", facecolor="white", transparent=False)
    paper_stem = ensure_dir("paper/figures") / "matched_tile_decisions_3s"
    figure.savefig(
        paper_stem.with_suffix(".pdf"),
        bbox_inches="tight",
        facecolor="white",
        transparent=False,
    )
    plt.close(figure)

    result = {
        "split": "historically_seen_gate",
        "horizon_s": HORIZON_S,
        "tile_grid": [TILE_ROWS, TILE_COLS],
        "budget_percent": BUDGET_PERCENT,
        "selected_tile_count": 13,
        "examples": [dict(image_id=image_id, role=role) for image_id, role in EXAMPLES],
        "selection_note": "Illustrative positive and failure cases; quantitative claims use all 13 gate contents.",
        "gt_note": "GT ranking is an in-sample qualitative oracle and is not a generalization result.",
        "display_smoothing_sigma_px": 10.0,
        "metrics_affected": False,
        "scanpath_qualitative_sampling": {
            "seed": 20260711,
            "paths_per_content": 200,
            "selection": "fixed central seed; no 600-path three-seed ensemble",
        },
        "outer_test_used": False,
        "records": records,
        "figure": str(stem.with_suffix(".png")),
    }
    (output / "results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (output / "index.md").write_text(
        "# Matched tile decisions\n\n![Tile decisions](matched_tile_decisions_3s.png)\n\n"
        "All panels use the same 3-s horizon and 13-of-128 tile budget. "
        "Rows are contents and columns are methods in a full-width landscape layout; no separate ERP-only column is used. "
        "GT ranking is a qualitative in-sample oracle. P95/P37 show positive decision changes; "
        "P10 shows that improved density accuracy need not improve the top-budget decision. "
        "Pathformer3D and ScanDMM use 200 paths from the fixed central seed 20260711.\n",
        encoding="utf-8",
    )
    print(json.dumps({"figure": result["figure"], "index": str(output / "index.md")}, indent=2))


if __name__ == "__main__":
    main()
