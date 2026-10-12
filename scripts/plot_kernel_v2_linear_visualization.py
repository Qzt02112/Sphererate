#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
from scipy.ndimage import gaussian_filter1d
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from sphererate.data.early_dataset import EarlyHorizonDataset
from sphererate.evaluation.metrics import kld
from sphererate.utils import ensure_dir
from scripts.evaluate_eh3_gate import _predict_learned


HORIZON = 3.0
HORIZON_INDEX = 2
BLUE = "#1769AA"


def _style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.0,
            "axes.titlesize": 7.5,
            "axes.linewidth": 0.55,
            "savefig.dpi": 400,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def _fixations(sample: dict) -> np.ndarray:
    with np.load(str(sample["fixations_path"])) as data:
        return np.asarray(data["fixations"], dtype=np.float32)


def _linear_relative(density: np.ndarray) -> np.ndarray:
    """Linear per-map normalization, equivalent to imshow's auto range for non-negative maps."""

    values = np.asarray(density, dtype=np.float64)
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    if maximum <= minimum:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - minimum) / (maximum - minimum)).astype(np.float32)


def _sitzmann_display_smooth(density: np.ndarray, sigma_y_px: float) -> np.ndarray:
    """Display-only smoothing using Sitzmann's latitude-adaptive ERP filter.

    This never enters training or metric computation.  It follows the public
    notebook: vertical wrap filtering with a fixed sigma, then horizontal wrap
    filtering with ``sigma_x = sigma_y / cos(latitude)`` for each ERP row.
    """

    values = np.asarray(density, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"Expected a 2-D saliency map, got {values.shape}")
    sigma_y = float(sigma_y_px)
    if not np.isfinite(sigma_y) or sigma_y <= 0.0:
        raise ValueError("display sigma must be finite and positive")
    smoothed = gaussian_filter1d(values, sigma=sigma_y, mode="wrap", axis=0)
    height = smoothed.shape[0]
    for row in range(height):
        angle = (row / float(height) - 0.5) * np.pi
        sigma_x = sigma_y / (np.cos(angle) + 1e-3)
        smoothed[row] = gaussian_filter1d(smoothed[row], sigma=sigma_x, mode="wrap")
    return smoothed.astype(np.float32)


def _overlay_sitzmann(ax: plt.Axes, rgb: np.ndarray, density: np.ndarray) -> plt.AxesImage:
    """Match Sitzmann's ``overlay_image_salmap`` display convention."""

    ax.imshow(rgb, aspect="equal")
    overlay = ax.imshow(
        _linear_relative(density),
        cmap=plt.cm.jet,
        vmin=0.0,
        vmax=1.0,
        alpha=0.4,
        aspect="equal",
    )
    ax.set_box_aspect(0.5)
    return overlay


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot the kernel-v2 audit with linear saliency-map display.")
    parser.add_argument("--output-dir", default="reports/kernel_v2_content_audit")
    parser.add_argument("--device", default="cuda")
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
    checkpoint = Path("runs/kernel_v2_candidate_correct_seed20260711/checkpoints/best.pt")
    device = torch.device(args.device)
    correct, provenance = _predict_learned(checkpoint, samples, device)
    wrong, _ = _predict_learned(checkpoint, samples, device, counterfactual="wrong_image")

    records = []
    for sample in samples:
        image_id = str(sample["image_id"])
        area = sample["area_weights"].numpy()[0]
        target = sample["target"].numpy()[HORIZON_INDEX]
        template = sample["template"].numpy()[HORIZON_INDEX]
        records.append(
            {
                "image_id": image_id,
                "gain_vs_template": kld(template, target, area) - kld(correct[image_id][HORIZON_INDEX], target, area),
                "gain_vs_wrong": kld(wrong[image_id][HORIZON_INDEX], target, area)
                - kld(correct[image_id][HORIZON_INDEX], target, area),
            }
        )
    ordered = sorted(records, key=lambda row: (row["gain_vs_wrong"], row["image_id"]))
    selected_records = [ordered[0], ordered[len(ordered) // 2], ordered[-1]]
    roles = ("weakest", "median", "strongest")
    by_id = {str(sample["image_id"]): sample for sample in samples}
    selected = []
    for role, record in zip(roles, selected_records):
        sample = by_id[record["image_id"]]
        image_id = str(sample["image_id"])
        selected.append(
            {
                "role": role,
                "record": record,
                "sample": sample,
                "rgb": np.clip(sample["image"][:3].numpy().transpose(1, 2, 0), 0.0, 1.0),
                "target": sample["target"].numpy()[HORIZON_INDEX],
                "template": sample["template"].numpy()[HORIZON_INDEX],
                "correct": correct[image_id][HORIZON_INDEX],
                "wrong": wrong[image_id][HORIZON_INDEX],
            }
        )

    display_sigma_px = float(args.display_sigma_px)
    for item in selected:
        for key in ("target", "template", "correct", "wrong"):
            item[f"display_{key}"] = _sitzmann_display_smooth(item[key], display_sigma_px)

    residuals = [
        _linear_relative(item["display_correct"]) - _linear_relative(item["display_template"])
        for item in selected
    ]
    residual_limit = float(np.quantile(np.abs(np.concatenate([value.ravel() for value in residuals])), 0.995))
    residual_limit = max(residual_limit, 1e-8)

    fig = plt.figure(figsize=(11.4, 4.25), constrained_layout=True)
    grid = fig.add_gridspec(3, 7)
    titles = (
        "ERP + observed gaze",
        "Raw GT\nspherical 2°",
        f"Displayed GT\nσy = {display_sigma_px:g} px",
        f"Displayed template\nσy = {display_sigma_px:g} px",
        "Displayed correct pred.",
        "Displayed wrong pred.",
        "Displayed residual\nvs template",
    )
    saliency_axes: list[plt.Axes] = []
    residual_axes: list[plt.Axes] = []
    saliency_image = None
    residual_image = None
    for row_index, (payload, residual) in enumerate(zip(selected, residuals)):
        fixations = _fixations(payload["sample"])
        current = fixations[fixations[:, 2] < HORIZON]
        for col_index, title in enumerate(titles):
            ax = fig.add_subplot(grid[row_index, col_index])
            if row_index == 0:
                ax.set_title(title, fontweight="bold" if col_index in {0, 2, 4, 6} else "normal")
            if col_index == 0:
                ax.imshow(payload["rgb"], aspect="equal")
                ax.set_box_aspect(0.5)
                if len(current):
                    indices = np.linspace(0, len(current) - 1, min(220, len(current)), dtype=int)
                    ax.scatter(
                        np.mod(current[indices, 0], 1.0) * 512,
                        np.clip(current[indices, 1], 0.0, 1.0) * 256,
                        s=2.2,
                        c="white",
                        edgecolors="black",
                        linewidths=0.15,
                        alpha=0.62,
                    )
                ax.axvline(0.75 * 512, color="#00B4D8", linestyle="--", linewidth=0.75)
                ax.set_ylabel(
                    f"{payload['role']}\n{payload['record']['image_id']}\n"
                    f"ΔKLDwrong={payload['record']['gain_vs_wrong']:+.3f}",
                    rotation=0,
                    labelpad=31,
                    va="center",
                    fontweight="bold",
                )
            elif col_index in {1, 2, 3}:
                key = {1: "target", 2: "display_target", 3: "display_template"}[col_index]
                saliency_image = _overlay_sitzmann(ax, payload["rgb"], payload[key])
                saliency_axes.append(ax)
            elif col_index in {4, 5}:
                _overlay_sitzmann(
                    ax,
                    payload["rgb"],
                    payload["display_correct" if col_index == 4 else "display_wrong"],
                )
                if col_index == 4:
                    for spine in ax.spines.values():
                        spine.set_color(BLUE)
                        spine.set_linewidth(1.0)
            else:
                ax.imshow(payload["rgb"], aspect="equal")
                strength = np.clip(np.abs(residual) / residual_limit, 0.0, 1.0)
                residual_image = ax.imshow(
                    residual,
                    cmap="coolwarm",
                    norm=Normalize(-residual_limit, residual_limit),
                    alpha=0.12 + 0.72 * strength,
                    aspect="equal",
                )
                ax.set_box_aspect(0.5)
                residual_axes.append(ax)
            ax.set_xticks([])
            ax.set_yticks([])

    assert saliency_image is not None and residual_image is not None
    saliency_bar = fig.colorbar(saliency_image, ax=saliency_axes, location="bottom", shrink=0.48, pad=0.035, aspect=35)
    saliency_bar.set_label("linear saliency / panel maximum (display smoothing only where labelled)")
    residual_bar = fig.colorbar(residual_image, ax=residual_axes, location="bottom", shrink=0.8, pad=0.035, aspect=18)
    residual_bar.set_label("linear relative-saliency residual")
    fig.suptitle("Display-only smoothing improves readability without changing metrics", fontsize=9.4, fontweight="bold")
    sigma_tag = f"{display_sigma_px:g}".replace(".", "p")
    stem = output / f"kernel_v2_content_attribution_3s_display_smoothed_{sigma_tag}px"
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(stem.with_suffix(f".{suffix}"), bbox_inches="tight")
    plt.close(fig)

    summary = {
        "split": "historically_seen_gate",
        "outer_test_used": False,
        "horizon_s": HORIZON,
        "checkpoint": provenance,
        "selection": [dict(role=role, **record) for role, record in zip(roles, selected_records)],
        "display": "sitzmann_overlay_linear_per_map_minmax_jet_alpha_0.4",
        "display_smoothing": {
            "family": "sitzmann_latitude_adaptive_erp",
            "sigma_y_px": display_sigma_px,
            "equatorial_sigma_deg_at_width_512": display_sigma_px * 360.0 / 512.0,
            "applied_uniformly_to": ["target", "template", "correct_prediction", "wrong_image_prediction"],
            "training_or_metrics_affected": False,
        },
        "target_values_changed": False,
        "figure": str(stem.with_suffix(".png")),
    }
    (output / f"display_smoothed_{sigma_tag}px_results.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"figure": summary["figure"], "target_values_changed": False}, indent=2))


if __name__ == "__main__":
    main()
