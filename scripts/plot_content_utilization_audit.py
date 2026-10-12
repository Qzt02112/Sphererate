#!/usr/bin/env python
"""Plot the frozen content-utilization audit prepared by the evaluation script."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Patch, Rectangle
import numpy as np


ROOT = Path("reports/content_utilization_audit")
DATA = ROOT / "data"
HORIZONS = (0.5, 1.0, 3.0, 5.0)
BLUE = "#1769AA"
ORANGE = "#E07A1F"
PURPLE = "#7A5195"
GRAY = "#6B7280"
SIGNED_CMAP = LinearSegmentedColormap.from_list(
    "content_signed", ["#2166AC", "#F7F7F7", "#D55E00"]
)


def _style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.2,
            "axes.titlesize": 8.0,
            "axes.labelsize": 7.4,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "legend.fontsize": 6.8,
            "axes.linewidth": 0.65,
            "lines.linewidth": 1.25,
            "savefig.dpi": 400,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def _curve(payload: dict, key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = [payload[f"{horizon:g}"][key] for horizon in HORIZONS]
    return (
        np.asarray([value["mean"] for value in values]),
        np.asarray([value["ci95_low"] for value in values]),
        np.asarray([value["ci95_high"] for value in values]),
    )


def plot_quantitative(result: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.15, 4.95))
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.09, top=0.84, hspace=0.42, wspace=0.28)

    mean, low, high = _curve(result["candidate_vs_template_seed_average"]["per_horizon"], "kld_sph")
    ax = axes[0, 0]
    ax.axhline(0, color="#9CA3AF", linewidth=0.7)
    ax.plot(HORIZONS, mean, color=BLUE, marker="o", markersize=3.8)
    ax.fill_between(HORIZONS, low, high, color=BLUE, alpha=0.14, linewidth=0)
    ax.set_ylabel("KLD gain vs template ↑")
    ax.text(-0.13, 1.05, "(a)", transform=ax.transAxes, fontweight="bold", fontsize=8.3)

    mean, low, high = _curve(result["candidate_vs_wrong_image_seed20260711"]["per_horizon"], "kld_sph")
    ax = axes[0, 1]
    ax.axhline(0, color="#9CA3AF", linewidth=0.7)
    ax.plot(HORIZONS, mean, color=ORANGE, marker="s", markersize=3.8)
    ax.fill_between(HORIZONS, low, high, color=ORANGE, alpha=0.14, linewidth=0)
    ax.set_ylabel("KLD gain vs wrong image ↑")
    ax.text(-0.13, 1.05, "(b)", transform=ax.transAxes, fontweight="bold", fontsize=8.3)

    l1 = result["spherical_l1_correct_vs_wrong_image"]
    mean = np.asarray([l1[f"{h:g}"]["mean"] for h in HORIZONS])
    low = np.asarray([l1[f"{h:g}"]["ci95_low"] for h in HORIZONS])
    high = np.asarray([l1[f"{h:g}"]["ci95_high"] for h in HORIZONS])
    ax = axes[1, 0]
    ax.plot(HORIZONS, mean, color=PURPLE, marker="D", markersize=3.5)
    ax.fill_between(HORIZONS, low, high, color=PURPLE, alpha=0.14, linewidth=0)
    ax.set_ylabel("Spherical L1 map change ↑")
    ax.set_ylim(bottom=0)
    ax.text(-0.13, 1.05, "(c)", transform=ax.transAxes, fontweight="bold", fontsize=8.3)

    fractions = result["content_win_fractions"]
    template_fraction = [fractions[f"{h:g}"]["correct_better_than_template"] for h in HORIZONS]
    wrong_fraction = [fractions[f"{h:g}"]["correct_better_than_wrong_image"] for h in HORIZONS]
    ax = axes[1, 1]
    ax.plot(HORIZONS, template_fraction, color=BLUE, marker="o", markersize=3.8, label="Beats template")
    ax.plot(HORIZONS, wrong_fraction, color=ORANGE, marker="s", markersize=3.8, label="Beats wrong image")
    ax.set_ylabel("Fraction of 13 contents")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, loc="lower left")
    ax.text(-0.13, 1.05, "(d)", transform=ax.transAxes, fontweight="bold", fontsize=8.3)

    for ax in axes.ravel():
        ax.set_xlabel("Horizon (s)")
        ax.set_xticks(HORIZONS)
        ax.grid(axis="y", color="#D1D5DB", linewidth=0.5, alpha=0.7)
    fig.suptitle(
        "Image dependence grows with horizon, but useful gains peak at 3 s",
        y=0.965,
        fontsize=9.3,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.895,
        "Frozen test; shaded bands are 95% content-bootstrap CIs. Panel (a): three-seed mean; (b–d): seed 20260711.",
        ha="center",
        va="center",
        fontsize=6.8,
        color="#4B5563",
    )
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(ROOT / f"content_utilization_curves.{suffix}", bbox_inches="tight")
    plt.close(fig)


def _load_selection(selection: list[dict]) -> list[tuple[dict, dict[str, np.ndarray]]]:
    loaded = []
    for row in selection:
        with np.load(DATA / f"{row['role']}_{row['image_id']}.npz") as data:
            loaded.append((row, {key: np.asarray(data[key]) for key in data.files}))
    return loaded


def plot_qualitative(selection: list[dict], horizon: float) -> None:
    loaded = _load_selection(selection)
    horizon_index = HORIZONS.index(float(horizon))
    density_values = []
    difference_values = []
    for _, data in loaded:
        for key in ("gt", "template", "correct", "wrong_image"):
            density_values.append(np.log10(np.maximum(data[key][horizon_index], 1e-8)).ravel())
        difference_values.append(
            np.log10(np.maximum(np.abs(data["correct"][horizon_index] - data["wrong_image"][horizon_index]), 1e-8)).ravel()
        )
    density_flat = np.concatenate(density_values)
    difference_flat = np.concatenate(difference_values)
    density_min = float(np.quantile(density_flat, 0.04))
    density_max = float(np.quantile(density_flat, 0.997))
    difference_min = float(np.quantile(difference_flat, 0.04))
    difference_max = float(np.quantile(difference_flat, 0.997))

    fig = plt.figure(figsize=(11.0, 4.25), constrained_layout=True)
    grid = fig.add_gridspec(3, 7, width_ratios=[1, 1, 1, 1, 1, 1, 1])
    titles = ("Correct ERP", "Wrong ERP", "GT", "Template", "Correct pred.", "Wrong-image pred.", "|Correct − wrong|")
    density_axes: list[plt.Axes] = []
    difference_axes: list[plt.Axes] = []
    density_image = None
    difference_image = None
    for row_index, (metadata, data) in enumerate(loaded):
        for col_index, title in enumerate(titles):
            ax = fig.add_subplot(grid[row_index, col_index])
            if row_index == 0:
                ax.set_title(title, fontweight="bold" if col_index in {0, 4, 6} else "normal")
            if col_index == 0:
                ax.imshow(data["rgb"], aspect="equal")
                ax.set_box_aspect(0.5)
                ax.set_ylabel(
                    f"{metadata['role'].capitalize()}\n{metadata['image_id']}\nΔKLD={metadata['selection_value']:+.3f}",
                    rotation=0,
                    labelpad=33,
                    va="center",
                    fontweight="bold",
                )
            elif col_index == 1:
                ax.imshow(data["wrong_rgb"], aspect="equal")
                ax.set_box_aspect(0.5)
                ax.text(
                    0.98,
                    0.04,
                    metadata["wrong_image_id"],
                    transform=ax.transAxes,
                    ha="right",
                    va="bottom",
                    fontsize=6.5,
                    color="white",
                    bbox={"facecolor": "black", "alpha": 0.55, "pad": 1.2, "edgecolor": "none"},
                )
            elif col_index == 6:
                difference_image = ax.imshow(
                    np.log10(
                        np.maximum(
                            np.abs(data["correct"][horizon_index] - data["wrong_image"][horizon_index]),
                            1e-8,
                        )
                    ),
                    cmap="cividis",
                    vmin=difference_min,
                    vmax=difference_max,
                    aspect="equal",
                )
                ax.set_box_aspect(0.5)
                difference_axes.append(ax)
            else:
                key = {2: "gt", 3: "template", 4: "correct", 5: "wrong_image"}[col_index]
                density_image = ax.imshow(
                    np.log10(np.maximum(data[key][horizon_index], 1e-8)),
                    cmap="magma",
                    vmin=density_min,
                    vmax=density_max,
                    aspect="equal",
                )
                ax.set_box_aspect(0.5)
                density_axes.append(ax)
                for spine in ax.spines.values():
                    spine.set_linewidth(1.05 if col_index == 4 else 0.45)
                    spine.set_edgecolor(BLUE if col_index == 4 else (ORANGE if col_index == 5 else "#9CA3AF"))
            ax.set_xticks([])
            ax.set_yticks([])

    assert density_image is not None and difference_image is not None
    density_bar = fig.colorbar(density_image, ax=density_axes, location="bottom", shrink=0.48, pad=0.045, aspect=32)
    density_bar.set_label("log$_{10}$ spherical density (shared scale)")
    difference_bar = fig.colorbar(difference_image, ax=difference_axes, location="bottom", shrink=0.72, pad=0.045, aspect=20)
    difference_bar.set_label("log$_{10}$ absolute density change")
    fig.suptitle(
        f"Correct image content changes frozen predictions at {horizon:g} s",
        fontsize=9.3,
        fontweight="bold",
    )
    stem = ROOT / f"content_attribution_{horizon:g}s"
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(stem.with_suffix(f".{suffix}"), bbox_inches="tight")
    plt.close(fig)


def _manifest_by_id() -> dict[str, dict]:
    output = {}
    path = Path("data/processed/early_horizon_salient360_v1/manifest.jsonl")
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                output[str(row["image_id"])] = row
    return output


def _area_weights(height: int, width: int) -> np.ndarray:
    y = np.arange(height, dtype=np.float64) + 0.5
    latitude = np.pi / 2.0 - np.pi * y / float(height)
    return np.repeat(np.cos(latitude).clip(min=1e-4)[:, None], width, axis=1)


def _rank_tiles(density: np.ndarray, area: np.ndarray, budget_percent: float = 10.0) -> np.ndarray:
    tile_rows, tile_cols = 8, 16
    height, width = density.shape
    tile_h, tile_w = height // tile_rows, width // tile_cols
    mass = (density * area).reshape(tile_rows, tile_h, tile_cols, tile_w).sum(axis=(1, 3)).ravel()
    count = int(np.ceil(tile_rows * tile_cols * budget_percent / 100.0))
    ids = np.arange(len(mass), dtype=np.int64)
    return np.lexsort((ids, -mass))[:count]


def _gaze_at_horizon(fixations: np.ndarray, horizon: float) -> tuple[np.ndarray, np.ndarray]:
    starts = fixations[:, 2].astype(np.float64)
    durations = np.maximum(fixations[:, 4].astype(np.float64), 0.0)
    weights = np.maximum(np.minimum(starts + durations, horizon) - np.maximum(starts, 0.0), 0.0)
    valid = weights > 0.0
    return fixations[valid], weights[valid]


def _coverage(fixations: np.ndarray, weights: np.ndarray, selected: np.ndarray) -> float:
    x_tile = np.floor(np.mod(fixations[:, 0], 1.0) * 16).astype(np.int64)
    y_tile = np.floor(np.clip(fixations[:, 1], 0.0, np.nextafter(1.0, 0.0)) * 8).astype(np.int64)
    ids = y_tile * 16 + x_tile
    return float(weights[np.isin(ids, selected)].sum() / weights.sum())


def _overlay_signed(ax: plt.Axes, rgb: np.ndarray, signed: np.ndarray, support: np.ndarray, limit: float) -> None:
    ax.imshow(rgb, aspect="equal")
    threshold = float(np.quantile(support, 0.55))
    relevance = support >= threshold
    strength = np.clip(np.abs(signed) / limit, 0.0, 1.0)
    alpha = np.where(relevance, 0.18 + 0.62 * strength, 0.0)
    ax.imshow(signed, cmap=SIGNED_CMAP, vmin=-limit, vmax=limit, alpha=alpha, aspect="equal")
    ax.set_box_aspect(0.5)


def _draw_gt_contours(ax: plt.Axes, gt: np.ndarray) -> None:
    positive = gt[gt > 1e-8]
    if len(positive):
        levels = np.unique(np.quantile(positive, [0.72, 0.92]))
        if len(levels):
            ax.contour(gt, levels=levels, colors=["white", "#00B4D8"][: len(levels)], linewidths=[0.65, 0.9][: len(levels)])


def _draw_tile_decision(
    ax: plt.Axes,
    rgb: np.ndarray,
    template_tiles: np.ndarray,
    candidate_tiles: np.ndarray,
    fixations: np.ndarray,
) -> None:
    ax.imshow(rgb, aspect="equal")
    ax.set_box_aspect(0.5)
    template_set = set(int(value) for value in template_tiles)
    candidate_set = set(int(value) for value in candidate_tiles)
    for tile_id in sorted(template_set | candidate_set):
        row, col = divmod(tile_id, 16)
        if tile_id in candidate_set and tile_id not in template_set:
            color, alpha, linewidth = BLUE, 0.30, 0.9
        elif tile_id in template_set and tile_id not in candidate_set:
            color, alpha, linewidth = ORANGE, 0.28, 0.9
        else:
            color, alpha, linewidth = GRAY, 0.08, 0.45
        ax.add_patch(
            Rectangle((col * 32, row * 32), 32, 32, facecolor=color, edgecolor=color, alpha=alpha, linewidth=linewidth)
        )
    if len(fixations):
        indices = np.linspace(0, len(fixations) - 1, min(len(fixations), 180), dtype=int)
        ax.scatter(
            np.mod(fixations[indices, 0], 1.0) * 512,
            np.clip(fixations[indices, 1], 0.0, 1.0) * 256,
            s=3.0,
            facecolor="white",
            edgecolor="black",
            linewidth=0.18,
            alpha=0.65,
        )


def plot_content_reallocation(result: dict, horizon: float = 3.0) -> None:
    loaded = _load_selection(result["selection"])
    manifest = _manifest_by_id()
    horizon_index = HORIZONS.index(float(horizon))
    residuals = []
    contrasts = []
    density_values = []
    for _, data in loaded:
        correct = data["correct"][horizon_index]
        template = data["template"][horizon_index]
        wrong = data["wrong_image"][horizon_index]
        residuals.append(np.log(correct + 1e-8) - np.log(template + 1e-8))
        contrasts.append(np.log(correct + 1e-8) - np.log(wrong + 1e-8))
        density_values.extend(
            [np.log10(np.maximum(template, 1e-8)).ravel(), np.log10(np.maximum(correct, 1e-8)).ravel()]
        )
    signed_values = np.concatenate([np.abs(value).ravel() for value in residuals + contrasts])
    signed_limit = float(np.quantile(signed_values, 0.985))
    density_flat = np.concatenate(density_values)
    density_min = float(np.quantile(density_flat, 0.04))
    density_max = float(np.quantile(density_flat, 0.997))

    fig = plt.figure(figsize=(11.2, 4.65), constrained_layout=True)
    grid = fig.add_gridspec(3, 6)
    titles = (
        "RGB + GT contours",
        "Template",
        "Correct prediction",
        "Signed residual\nvs template",
        "Signed contrast\nvs wrong image",
        "10% tile decision + gaze",
    )
    density_axes: list[plt.Axes] = []
    signed_axes: list[plt.Axes] = []
    density_image = None
    coverage_rows = []
    for row_index, ((metadata, data), residual, contrast) in enumerate(zip(loaded, residuals, contrasts)):
        image_id = str(metadata["image_id"])
        with np.load(manifest[image_id]["fixations"]) as fixation_data:
            all_fixations = np.asarray(fixation_data["fixations"], dtype=np.float32)
        current_fixations, dwell_weights = _gaze_at_horizon(all_fixations, horizon)
        template = data["template"][horizon_index]
        correct = data["correct"][horizon_index]
        area = _area_weights(*template.shape)
        template_tiles = _rank_tiles(template, area)
        candidate_tiles = _rank_tiles(correct, area)
        template_coverage = _coverage(current_fixations, dwell_weights, template_tiles)
        candidate_coverage = _coverage(current_fixations, dwell_weights, candidate_tiles)
        coverage_rows.append(candidate_coverage - template_coverage)
        for col_index, title in enumerate(titles):
            ax = fig.add_subplot(grid[row_index, col_index])
            if row_index == 0:
                ax.set_title(title, fontweight="bold" if col_index in {0, 3, 4, 5} else "normal")
            if col_index == 0:
                ax.imshow(data["rgb"], aspect="equal")
                ax.set_box_aspect(0.5)
                _draw_gt_contours(ax, data["gt"][horizon_index])
                ax.set_ylabel(
                    f"{metadata['role'].capitalize()}\n{image_id}\nΔKLD={metadata['selection_value']:+.3f}",
                    rotation=0,
                    labelpad=31,
                    va="center",
                    fontweight="bold",
                )
            elif col_index in {1, 2}:
                current = template if col_index == 1 else correct
                density_image = ax.imshow(
                    np.log10(np.maximum(current, 1e-8)),
                    cmap="magma",
                    vmin=density_min,
                    vmax=density_max,
                    aspect="equal",
                )
                ax.set_box_aspect(0.5)
                density_axes.append(ax)
                if col_index == 2:
                    for spine in ax.spines.values():
                        spine.set_linewidth(1.0)
                        spine.set_edgecolor(BLUE)
            elif col_index == 3:
                _overlay_signed(ax, data["rgb"], residual, np.maximum(correct, template), signed_limit)
                signed_axes.append(ax)
            elif col_index == 4:
                wrong = data["wrong_image"][horizon_index]
                _overlay_signed(ax, data["rgb"], contrast, np.maximum(correct, wrong), signed_limit)
                signed_axes.append(ax)
            else:
                _draw_tile_decision(ax, data["rgb"], template_tiles, candidate_tiles, current_fixations)
                ax.text(
                    0.98,
                    0.04,
                    f"Δcov {candidate_coverage-template_coverage:+.3f}",
                    transform=ax.transAxes,
                    ha="right",
                    va="bottom",
                    color="white",
                    fontsize=6.4,
                    bbox={"facecolor": "black", "alpha": 0.55, "pad": 1.1, "edgecolor": "none"},
                )
            ax.set_xticks([])
            ax.set_yticks([])

    assert density_image is not None
    density_bar = fig.colorbar(density_image, ax=density_axes, location="bottom", shrink=0.52, pad=0.045, aspect=26)
    density_bar.set_label("log$_{10}$ spherical density")
    signed_scalar = mpl.cm.ScalarMappable(norm=Normalize(-signed_limit, signed_limit), cmap=SIGNED_CMAP)
    signed_bar = fig.colorbar(signed_scalar, ax=signed_axes, location="bottom", shrink=0.52, pad=0.045, aspect=28)
    signed_bar.set_label("signed log-density ratio (blue removes / orange adds)")
    fig.legend(
        handles=[
            Patch(facecolor=BLUE, alpha=0.30, label="candidate-only tile"),
            Patch(facecolor=ORANGE, alpha=0.28, label="template-only tile"),
            Patch(facecolor=GRAY, alpha=0.15, label="shared tile"),
        ],
        loc="lower right",
        bbox_to_anchor=(0.992, 0.015),
        ncol=1,
        frameon=False,
    )
    fig.suptitle(
        f"Image evidence reallocates probability beyond the start prior at {horizon:g} s",
        fontsize=9.4,
        fontweight="bold",
    )
    stem = ROOT / f"content_reallocation_{horizon:g}s"
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(stem.with_suffix(f".{suffix}"), bbox_inches="tight")
    plt.close(fig)


def plot_per_content_3s() -> None:
    rows = []
    with (ROOT / "per_content.csv").open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if float(row["horizon_s"]) == 3.0:
                rows.append(row)
    rows.sort(key=lambda row: float(row["kld_gain_vs_template"]))
    y = np.arange(len(rows))
    template = np.asarray([float(row["kld_gain_vs_template"]) for row in rows])
    wrong = np.asarray([float(row["kld_gain_vs_wrong_image"]) for row in rows])
    labels = [str(row["image_id"]) for row in rows]
    fig, ax = plt.subplots(figsize=(5.2, 4.1))
    for index in range(len(rows)):
        ax.plot([template[index], wrong[index]], [index, index], color="#CBD5E1", linewidth=0.8, zorder=1)
    ax.scatter(template, y, s=25, color=BLUE, label="vs template", zorder=3)
    ax.scatter(wrong, y, s=25, marker="s", color=ORANGE, label="vs wrong image", zorder=3)
    ax.axvline(0, color="#6B7280", linewidth=0.8)
    ax.set_yticks(y, labels)
    ax.set_xlabel("Per-content KLD improvement at 3 s ↑")
    ax.set_ylabel("Frozen test content")
    ax.grid(axis="x", color="#D1D5DB", linewidth=0.5, alpha=0.7)
    ax.legend(frameon=False, loc="lower right")
    ax.set_title("Image use is consistent; gains over the template vary by content", fontweight="bold")
    fig.tight_layout()
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(ROOT / f"content_gain_by_content_3s.{suffix}", bbox_inches="tight")
    plt.close(fig)


def write_summary(result: dict) -> None:
    selection = result["selection"]
    lines = [
        "# Frozen Content-Utilization Audit",
        "",
        "Status: **diagnostic complete; no training, checkpoint change, or post-test tuning was performed.**",
        "",
        "## Content-specific probability reallocation",
        "",
        "![Signed content reallocation at 3 s](content_reallocation_3s.png)",
        "",
        "Blue regions lose probability and orange regions gain probability relative to the stated control. Signed residuals are overlaid on the original ERP so their relationship to scene content is visible. The final column shows how the image residual changes the 10% tile-prefetch decision; white points are observed gaze events up to 3 s.",
        "",
        "![Per-content 3 s gains](content_gain_by_content_3s.png)",
        "",
        "The correct image consistently outperforms a wrong image, but improvement over the start-aligned template is heterogeneous across contents.",
        "",
        "## Horizon-level attribution",
        "",
        "![Content-utilization curves](content_utilization_curves.png)",
        "",
        "![1 s attribution](content_attribution_1s.png)",
        "",
        "![3 s attribution](content_attribution_3s.png)",
        "",
        "## Quantitative result",
        "",
        "| Horizon | KLD gain vs template | KLD gain vs wrong image | Spherical L1 change | Beats template | Beats wrong image |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for horizon in HORIZONS:
        key = f"{horizon:g}"
        template = result["candidate_vs_template_seed_average"]["per_horizon"][key]["kld_sph"]
        wrong = result["candidate_vs_wrong_image_seed20260711"]["per_horizon"][key]["kld_sph"]
        l1 = result["spherical_l1_correct_vs_wrong_image"][key]
        fraction = result["content_win_fractions"][key]
        lines.append(
            f"| {key} s | {template['mean']:+.4f} [{template['ci95_low']:+.4f}, {template['ci95_high']:+.4f}] | "
            f"{wrong['mean']:+.4f} [{wrong['ci95_low']:+.4f}, {wrong['ci95_high']:+.4f}] | "
            f"{l1['mean']:.4f} [{l1['ci95_low']:.4f}, {l1['ci95_high']:.4f}] | "
            f"{fraction['correct_better_than_template']:.0%} | {fraction['correct_better_than_wrong_image']:.0%} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation lock",
            "",
            "- The model is not image-independent: replacing only the ERP image changes the output increasingly with horizon and degrades KLD on nearly every content.",
            "- Image dependence is not identical to usefulness. The correct image beats the template on all contents at 1 s and 77% at 3 s, but only 31% at 5 s.",
            "- The defensible claim is therefore **template-dominated early prediction with modest image-specific corrections, strongest around 3 s**, not a fully content-driven transition.",
            "- Wrong-image inference keeps target, time, start and train-only template fixed; only the ERP image is deterministically replaced by the next frozen-test image.",
            "- Raw probability maps remain supplementary evidence. The primary qualitative evidence is the signed probability reallocation on RGB plus its downstream tile-decision change.",
            "",
            "## Transparent qualitative selection",
            "",
            "Rows are the weakest, median and strongest contents ranked by 3 s correct-vs-wrong-image KLD improvement; selection is for visualization only and never changes the model.",
            "",
        ]
    )
    for row in selection:
        lines.append(
            f"- {row['role']}: {row['image_id']} ← wrong {row['wrong_image_id']}; "
            f"wrong-image ΔKLD `{row['selection_value']:+.4f}`, template ΔKLD `{row['candidate_vs_template_3s']:+.4f}`."
        )
    (ROOT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    _style()
    result = json.loads((ROOT / "results.json").read_text())
    plot_quantitative(result)
    plot_qualitative(result["selection"], 1.0)
    plot_qualitative(result["selection"], 3.0)
    plot_content_reallocation(result, 3.0)
    plot_per_content_3s()
    write_summary(result)
    print(json.dumps({"summary": str(ROOT / "summary.md")}, indent=2))


if __name__ == "__main__":
    main()
