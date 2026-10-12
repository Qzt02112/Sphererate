#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from scripts.plot_kernel_v2_linear_visualization import _linear_relative, _sitzmann_display_smooth
from sphererate.baselines.scanpath_time import second_sampled_scanpaths_to_cumulative_maps
from sphererate.data.early_dataset import EarlyHorizonDataset
from sphererate.evaluation.metrics import kld
from sphererate.utils import ensure_dir


HORIZON_INDEX = 2
HORIZON_S = 3.0
CONTENT_ROLES = (("P55", "prior-close"), ("P10", "content case"), ("P97", "hard case"))


def _style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 6.5,
            "axes.titlesize": 7.0,
            "axes.linewidth": 0.5,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.dpi": 500,
        }
    )


def _scanpath_map(run: Path, image_id: str, sample: dict) -> np.ndarray:
    paths = np.load(run / "scanpaths" / f"{image_id}.npy")
    return second_sampled_scanpaths_to_cumulative_maps(
        paths,
        horizons_s=sample["horizons_s"].numpy().tolist(),
        height=int(sample["target"].shape[-2]),
        width=int(sample["target"].shape[-1]),
        renderer="sphere_chord",
        sigma_deg=2.0,
        coordinate_order="yx",
    )[HORIZON_INDEX]


def _representative_scanpath_map(prefix: str, image_id: str, sample: dict) -> np.ndarray:
    """Use the protocol's central seed for qualitative display, never a 3-seed ensemble."""
    return _scanpath_map(
        ROOT / f"runs/{prefix}_seed20260711_gate",
        image_id,
        sample,
    )


def _overlay(ax: plt.Axes, rgb: np.ndarray, density: np.ndarray, sigma_px: float) -> None:
    display = _linear_relative(_sitzmann_display_smooth(density, sigma_px))
    ax.imshow(rgb, aspect="equal")
    ax.imshow(display, cmap="jet", vmin=0.0, vmax=1.0, alpha=0.42, aspect="equal")
    ax.scatter([0.75 * rgb.shape[1]], [0.5 * rgb.shape[0]], s=10, marker="+", c="#00D5FF", linewidths=0.9)
    ax.set_box_aspect(0.5)
    ax.set_xticks([])
    ax.set_yticks([])


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot publication qualitative comparisons for matched baselines.")
    parser.add_argument("--output-dir", default="reports/kernel_v2_matched_qualitative")
    parser.add_argument("--display-sigma-px", type=float, default=10.0)
    args = parser.parse_args()
    _style()
    output = ensure_dir(args.output_dir)
    dataset = EarlyHorizonDataset(
        "data/processed/early_horizon_salient360_spherical2deg_v2",
        split="gate",
        target_mode="correct",
    )
    samples = [dataset[index] for index in range(len(dataset))]
    by_id = {str(sample["image_id"]): sample for sample in samples}
    missing = [image_id for image_id, _ in CONTENT_ROLES if image_id not in by_id]
    if missing:
        raise ValueError(f"Missing frozen qualitative contents: {missing}")
    sphere_predictions = []
    for seed in (20260710, 20260711, 20260712):
        predictions = {}
        cache = ROOT / f"runs/kernel_v2_candidate_correct_seed{seed}/predictions/gate"
        for sample in samples:
            image_id = str(sample["image_id"])
            with np.load(cache / f"{image_id}.npz") as data:
                predictions[image_id] = np.asarray(data["cumulative"], dtype=np.float32)
        sphere_predictions.append(predictions)

    payloads = []
    for image_id, role in CONTENT_ROLES:
        sample = by_id[image_id]
        rgb = np.clip(sample["image"][:3].numpy().transpose(1, 2, 0), 0.0, 1.0)
        target = sample["target"].numpy()[HORIZON_INDEX]
        template = sample["template"].numpy()[HORIZON_INDEX]
        gbvs = np.load(ROOT / f"runs/kernel_v2_bms_gbvs/gbvs360/gate/maps/{image_id}.npy")
        with np.load(
            ROOT / f"runs/kernel_v2_reachable_final/gbvs360/gate/predictions/{image_id}.npz"
        ) as data:
            reachable = np.asarray(data["cumulative"][HORIZON_INDEX], dtype=np.float32)
        with np.load(ROOT / f"runs/kernel_v2_deepgaze_msdb/evaluated/gate/{image_id}.npz") as data:
            msdb_reachable = np.asarray(data["reachable"][HORIZON_INDEX], dtype=np.float32)
        pathformer = _representative_scanpath_map("kernel_v2_pathformer3d_retrained", image_id, sample)
        scandmm = _representative_scanpath_map(
            "kernel_v2_scandmm_declared_start", image_id, sample
        )
        sphere = np.mean(
            [prediction[image_id][HORIZON_INDEX] for prediction in sphere_predictions], axis=0
        ).astype(np.float32)
        payloads.append(
            {
                "image_id": image_id,
                "role": role,
                "sample": sample,
                "rgb": rgb,
                "maps": {
                    "GT 3 s": target,
                    "Early template": template,
                    "GBVS360 final": gbvs,
                    "GBVS + reachable": reachable,
                    "MSDB-25 ERP\n+ reachable": msdb_reachable,
                    "Pathformer3D\npopulation (200)": pathformer,
                    "ScanDMM\npopulation (200)": scandmm,
                    "SphereHorizon": sphere,
                },
            }
        )

    columns = (
        "ERP image",
        "GT 3 s",
        "Early template",
        "GBVS360 final",
        "GBVS + reachable",
        "MSDB-25 ERP\n+ reachable",
        "Pathformer3D\npopulation (200)",
        "ScanDMM\npopulation (200)",
        "SphereHorizon",
    )
    fig, axes = plt.subplots(len(payloads), len(columns), figsize=(12.8, 3.05), constrained_layout=True)
    records = []
    for row, payload in enumerate(payloads):
        area = payload["sample"]["area_weights"].numpy()[0]
        target = payload["maps"]["GT 3 s"]
        for col, title in enumerate(columns):
            ax = axes[row, col]
            if row == 0:
                ax.set_title(
                    title,
                    fontweight="bold"
                    if title in {"GT 3 s", "GBVS + reachable", "MSDB-25 ERP\n+ reachable", "SphereHorizon"}
                    else "normal",
                )
            if col == 0:
                ax.imshow(payload["rgb"], aspect="equal")
                ax.scatter([0.75 * payload["rgb"].shape[1]], [0.5 * payload["rgb"].shape[0]], s=12, marker="+", c="#00D5FF", linewidths=1.0)
                ax.set_box_aspect(0.5)
                ax.set_ylabel(
                    f"{payload['image_id']}\n{payload['role']}",
                    rotation=0,
                    labelpad=26,
                    va="center",
                    fontweight="bold",
                )
            else:
                density = payload["maps"][title]
                _overlay(ax, payload["rgb"], density, float(args.display_sigma_px))
                value = kld(density, target, area)
                ax.text(
                    0.02,
                    0.04,
                    "GT" if title == "GT 3 s" else f"KLD {value:.2f}",
                    transform=ax.transAxes,
                    color="white",
                    fontsize=5.8,
                    bbox={"facecolor": "black", "alpha": 0.55, "pad": 1.0, "edgecolor": "none"},
                )
                records.append({"image_id": payload["image_id"], "method": title, "kld_sph": value})
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.5)
            if col == len(columns) - 1:
                for spine in ax.spines.values():
                    spine.set_color("#1769AA")
                    spine.set_linewidth(1.0)
    stem = output / "matched_qualitative_3s"
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(stem.with_suffix(f".{suffix}"), bbox_inches="tight")
    plt.close(fig)

    # A submission-scale version with fewer columns and at least 9-pt labels.
    # The expanded nine-column figure remains the evidence-package view.
    mpl.rcParams.update({"font.size": 9.0, "axes.titlesize": 9.0})
    compact_columns = (
        "ERP image",
        "GT 3 s",
        "Early template",
        "GBVS + reachable",
        "SphereHorizon",
    )
    compact, compact_axes = plt.subplots(
        len(payloads), len(compact_columns), figsize=(7.2, 3.0), constrained_layout=True
    )
    for row, payload in enumerate(payloads):
        for col, title in enumerate(compact_columns):
            ax = compact_axes[row, col]
            if row == 0:
                ax.set_title(title, fontweight="bold" if title in {"GT 3 s", "GBVS + reachable", "SphereHorizon"} else "normal")
            if col == 0:
                ax.imshow(payload["rgb"], aspect="equal")
                ax.scatter(
                    [0.75 * payload["rgb"].shape[1]],
                    [0.5 * payload["rgb"].shape[0]],
                    s=12,
                    marker="+",
                    c="#00D5FF",
                    linewidths=1.0,
                )
                ax.set_box_aspect(0.5)
                ax.set_ylabel(
                    f"{payload['image_id']}\n{payload['role']}",
                    rotation=0,
                    labelpad=28,
                    va="center",
                    fontweight="bold",
                    fontsize=9.0,
                )
            else:
                _overlay(ax, payload["rgb"], payload["maps"][title], float(args.display_sigma_px))
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.5)
            if col == len(compact_columns) - 1:
                for spine in ax.spines.values():
                    spine.set_color("#1769AA")
                    spine.set_linewidth(1.0)
    compact_stem = output / "matched_qualitative_3s_compact"
    for suffix in ("png", "pdf", "svg"):
        compact.savefig(compact_stem.with_suffix(f".{suffix}"), bbox_inches="tight")
    plt.close(compact)
    result = {
        "split": "historically_seen_gate",
        "horizon_s": HORIZON_S,
        "contents": [dict(image_id=image_id, role=role) for image_id, role in CONTENT_ROLES],
        "display": {
            "overlay": "Sitzmann-style linear per-panel normalization, jet alpha=0.42",
            "smoothing_sigma_y_px": float(args.display_sigma_px),
            "metrics_affected": False,
        },
        "start_marker": "cyan plus at declared x=0.75, y=0.5",
        "scanpath_qualitative_sampling": {
            "seed": 20260711,
            "paths_per_content": 200,
            "selection": "protocol central seed fixed before qualitative inspection; no three-seed map ensemble",
        },
        "records": records,
        "test_target_loads": 0,
        "figure": str(stem.with_suffix(".png")),
        "compact_figure": str(compact_stem.with_suffix(".png")),
    }
    (output / "results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    lines = [
        "# Matched qualitative comparison",
        "",
        "Takeaway: **reachability filtering is a strong simple correction, but it cannot replace content-conditioned early-horizon prediction.**",
        "",
        f"![Matched qualitative comparison]({stem.name}.png)",
        "",
        "All maps use the frozen 3 s arrays for the printed KLD values. The extra 10 px latitude-adaptive smoothing and per-panel color normalization are display-only. The cyan plus marks the declared start. P55 is a case where reachability is nearly sufficient, P10 highlights content re-ranking, and P97 is a difficult case for static reachability. MSDB-25 is the official ICCV 2025 model with the frozen ERP roll/calibration adapter and the same reachable mask.",
        "The Pathformer3D/ScanDMM panels use the protocol's fixed central seed 20260711 and 200 paths. They are population densities, not individual trajectory plots; no three-seed map ensemble is used. Their start-centered appearance must not be interpreted as stationary paths.",
        "",
        "Individual-motion companion (30 paths from one fixed seed): [scanpath motion audit](../kernel_v2_scanpath_motion_audit/summary.md).",
        "Corrected ScanDMM uses an independent declared-start tensor in both model and guide while retaining the first one-second gaze as `obs_x_1`; the legacy first-target-initialized row is excluded.",
    ]
    (output / "index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"figure": result["figure"], "index": str(output / "index.md")}, indent=2))


if __name__ == "__main__":
    main()
