#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Rectangle
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from scripts.evaluate_eh3_gate import _predict_learned
from scripts.plot_kernel_v2_linear_visualization import _linear_relative, _sitzmann_display_smooth
from sphererate.data.early_dataset import EarlyHorizonDataset
from sphererate.evaluation.tile_prefetch import rank_tiles
from sphererate.utils import ensure_dir


def _style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.0,
            "axes.titlesize": 7.5,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.dpi": 500,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.transparent": False,
        }
    )


def _overlay(axis: plt.Axes, rgb: np.ndarray, density: np.ndarray, *, start: bool = True) -> None:
    display = _linear_relative(_sitzmann_display_smooth(density, 10.0))
    axis.imshow(rgb, aspect="equal")
    axis.imshow(display, cmap="jet", vmin=0.0, vmax=1.0, alpha=0.42, aspect="equal")
    if start:
        axis.scatter([0.75 * rgb.shape[1]], [0.5 * rgb.shape[0]], s=11, marker="+", c="#00D5FF", linewidths=1.0)
    axis.set_box_aspect(0.5)
    axis.set_xticks([])
    axis.set_yticks([])


def _box(axis: plt.Axes, x: float, y: float, w: float, h: float, text: str, color: str) -> None:
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=0.8,
        edgecolor="#444444",
        facecolor=color,
        transform=axis.transAxes,
    )
    axis.add_patch(patch)
    axis.text(x + w / 2, y + h / 2, text, ha="center", va="center", transform=axis.transAxes, fontsize=6.0)


def _arrow(axis: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    axis.annotate(
        "",
        xy=end,
        xytext=start,
        xycoords=axis.transAxes,
        textcoords=axis.transAxes,
        arrowprops={"arrowstyle": "-|>", "lw": 0.85, "color": "#555555", "shrinkA": 2, "shrinkB": 2},
    )


def _condition_arrow(axis: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    axis.annotate(
        "",
        xy=end,
        xytext=start,
        xycoords=axis.transAxes,
        textcoords=axis.transAxes,
        arrowprops={
            "arrowstyle": "-|>",
            "lw": 0.75,
            "linestyle": (0, (3, 2)),
            "color": "#6B7280",
            "shrinkA": 2,
            "shrinkB": 2,
        },
    )


def _image_inset(
    axis: plt.Axes,
    rect: tuple[float, float, float, float],
    rgb: np.ndarray,
    *,
    density: np.ndarray | None = None,
    title: str,
    show_start: bool = True,
) -> None:
    inset = axis.inset_axes(rect)
    if density is None:
        inset.imshow(rgb, aspect="equal")
        if show_start:
            inset.scatter(
                [0.75 * rgb.shape[1]],
                [0.5 * rgb.shape[0]],
                s=10,
                marker="+",
                c="#00D5FF",
                linewidths=1.0,
            )
    else:
        _overlay(inset, rgb, density, start=show_start)
    inset.set_xticks([])
    inset.set_yticks([])
    for spine in inset.spines.values():
        spine.set_linewidth(0.65)
        spine.set_color("#42464D")
    axis.text(
        rect[0] + rect[2] / 2,
        rect[1] + rect[3] + 0.035,
        title,
        transform=axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=6.4,
        fontweight="bold",
    )


def _tensor_stack(
    axis: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    color: str,
    *,
    depth: int = 3,
) -> None:
    for index in reversed(range(depth)):
        offset = 0.005 * index
        axis.add_patch(
            Rectangle(
                (x + offset, y + offset),
                w,
                h,
                transform=axis.transAxes,
                facecolor=color,
                edgecolor="#4B5159",
                linewidth=0.65,
                alpha=0.82 + 0.06 * (depth - index - 1),
            )
        )


def _circle_node(axis: plt.Axes, x: float, y: float, radius: float, text: str, color: str) -> None:
    axis.add_patch(
        Circle(
            (x, y),
            radius,
            transform=axis.transAxes,
            facecolor=color,
            edgecolor="#464B52",
            linewidth=0.8,
        )
    )
    axis.text(x, y, text, transform=axis.transAxes, ha="center", va="center", fontsize=6.4)


def _diamond_node(axis: plt.Axes, x: float, y: float, w: float, h: float, text: str, color: str) -> None:
    axis.add_patch(
        Polygon(
            [(x, y + h / 2), (x + w / 2, y), (x, y - h / 2), (x - w / 2, y)],
            closed=True,
            transform=axis.transAxes,
            facecolor=color,
            edgecolor="#464B52",
            linewidth=0.8,
        )
    )
    axis.text(x, y, text, transform=axis.transAxes, ha="center", va="center", fontsize=5.8)


def _density_inset(
    axis: plt.Axes,
    rect: tuple[float, float, float, float],
    density: np.ndarray,
    title: str,
) -> None:
    inset = axis.inset_axes(rect)
    inset.imshow(_linear_relative(density), cmap="cividis", vmin=0.0, vmax=1.0, aspect="equal")
    inset.set_xticks([])
    inset.set_yticks([])
    for spine in inset.spines.values():
        spine.set_linewidth(0.65)
        spine.set_color("#42464D")
    axis.text(
        rect[0] + rect[2] / 2,
        rect[1] + rect[3] + 0.018,
        title,
        transform=axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=6.2,
        fontweight="bold",
    )


def _framework(axis: plt.Axes, rgb: np.ndarray, prediction: np.ndarray, template: np.ndarray) -> None:
    axis.set_axis_off()
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)

    # Image endpoints and coordinate alignment.
    _image_inset(axis, (0.008, 0.570, 0.115, 0.180), rgb, title="ERP + geometry")
    _image_inset(
        axis,
        (0.900, 0.570, 0.092, 0.180),
        rgb,
        density=prediction,
        title="Early map $\\widehat C_t$",
    )
    axis.text(0.066, 0.515, "$g_0$", transform=axis.transAxes, ha="center", fontsize=6.2, color="#5B6470")
    _circle_node(axis, 0.155, 0.660, 0.028, "$R_{g_0}$", "#E9EEF4")
    axis.text(0.155, 0.605, "align", transform=axis.transAxes, ha="center", fontsize=5.4, color="#5B6470")

    # Dual-path encoder represented by feature geometry rather than prose boxes.
    encoder = FancyBboxPatch(
        (0.195, 0.395),
        0.245,
        0.500,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=0.8,
        linestyle=(0, (4, 2)),
        edgecolor="#76808C",
        facecolor="#F8FAFC",
        transform=axis.transAxes,
    )
    axis.add_patch(encoder)
    axis.text(0.3175, 0.858, "Dual-path encoder", transform=axis.transAxes, ha="center", fontsize=6.5, fontweight="bold")

    # Tiny U-Net icon: spatial scale contracts and expands.
    unet_x = [0.215, 0.252, 0.289, 0.326, 0.363]
    unet_h = [0.120, 0.095, 0.070, 0.095, 0.120]
    for x, h in zip(unet_x, unet_h):
        _tensor_stack(axis, x, 0.705 - h / 2, 0.022, h, "#8FC1DE", depth=2)
    axis.plot([x + 0.022 for x in unet_x], [0.705] * len(unet_x), transform=axis.transAxes, color="#5B6470", lw=0.7)
    axis.text(0.300, 0.790, "Tiny U-Net", transform=axis.transAxes, ha="center", fontsize=5.8)

    # Frozen ResNet icon: progressive semantic stages.
    for index, (w, h) in enumerate(((0.030, 0.105), (0.027, 0.090), (0.024, 0.075), (0.021, 0.060))):
        _tensor_stack(axis, 0.225 + 0.041 * index, 0.490 - h / 2, w, h, "#B9C2D8", depth=3)
    axis.text(0.290, 0.420, "ResNet-18 (frozen)", transform=axis.transAxes, ha="center", fontsize=5.8)

    _circle_node(axis, 0.415, 0.635, 0.025, "$\\oplus$", "#D3E5F1")
    axis.text(0.415, 0.585, "fuse", transform=axis.transAxes, ha="center", fontsize=5.2, color="#5B6470")

    # Compact content head.
    _diamond_node(axis, 0.490, 0.635, 0.060, 0.140, "FiLM$_t$", "#D8E4F3")
    for index, label in enumerate(("3$\\times$3", "1$\\times$1")):
        x = 0.545 + 0.038 * index
        _tensor_stack(axis, x, 0.585, 0.028, 0.100, "#8FD0BD", depth=2)
        axis.text(x + 0.014, 0.555, label, transform=axis.transAxes, ha="center", fontsize=4.8, color="#4F5963")
    axis.text(0.570, 0.745, "Residual head", transform=axis.transAxes, ha="center", fontsize=5.8)
    axis.text(0.615, 0.635, "$\\Delta_t$", transform=axis.transAxes, ha="center", va="center", fontsize=6.6)

    # Minimal multiplicative/additive density fusion.
    _circle_node(axis, 0.655, 0.635, 0.022, "$\\times$", "#D9EEE7")
    _circle_node(axis, 0.725, 0.635, 0.024, "$+$", "#F6E6C8")
    _box(axis, 0.765, 0.570, 0.060, 0.130, "Sph.\nnorm", "#F6E6C8")
    _circle_node(axis, 0.855, 0.635, 0.026, "$R_{g_0}^{-1}$", "#E9EEF4")

    # Horizon, gate, and population template use symbols/mini-plots.
    _circle_node(axis, 0.485, 0.175, 0.025, "$t$", "#F4F4F4")
    _box(axis, 0.525, 0.120, 0.050, 0.110, "MLP", "#EEEAF6")
    gate = FancyBboxPatch(
        (0.605, 0.115),
        0.065,
        0.125,
        boxstyle="round,pad=0.008,rounding_size=0.012",
        linewidth=0.7,
        edgecolor="#555555",
        facecolor="#F3F3F3",
        transform=axis.transAxes,
    )
    axis.add_patch(gate)
    gate_x = np.linspace(0.617, 0.658, 4)
    gate_y = np.asarray([0.135, 0.150, 0.185, 0.222])
    axis.plot(gate_x, gate_y, transform=axis.transAxes, color="#4E8F7B", marker="o", markersize=2.2, lw=0.8)
    axis.text(0.6375, 0.255, "$\\beta_t$", transform=axis.transAxes, ha="center", fontsize=6.2)
    _density_inset(axis, (0.690, 0.115, 0.075, 0.120), template, "$T_t$")
    axis.text(0.7275, 0.085, "train-only", transform=axis.transAxes, ha="center", fontsize=5.0, color="#5B6470")

    # Main feature/density flow.
    _arrow(axis, (0.123, 0.660), (0.127, 0.660))
    _arrow(axis, (0.183, 0.660), (0.215, 0.705))
    _arrow(axis, (0.183, 0.640), (0.225, 0.490))
    _arrow(axis, (0.385, 0.705), (0.390, 0.650))
    _arrow(axis, (0.375, 0.490), (0.390, 0.620))
    _arrow(axis, (0.440, 0.635), (0.460, 0.635))
    _arrow(axis, (0.520, 0.635), (0.545, 0.635))
    _arrow(axis, (0.607, 0.635), (0.633, 0.635))
    _arrow(axis, (0.677, 0.635), (0.701, 0.635))
    _arrow(axis, (0.749, 0.635), (0.765, 0.635))
    _arrow(axis, (0.825, 0.635), (0.829, 0.635))
    _arrow(axis, (0.881, 0.635), (0.900, 0.635))

    # Horizon conditioning and prior flow.
    _condition_arrow(axis, (0.510, 0.175), (0.525, 0.175))
    _condition_arrow(axis, (0.550, 0.230), (0.490, 0.565))
    _condition_arrow(axis, (0.510, 0.160), (0.605, 0.175))
    _condition_arrow(axis, (0.510, 0.150), (0.690, 0.175))
    _arrow(axis, (0.6375, 0.240), (0.655, 0.613))
    _arrow(axis, (0.7275, 0.235), (0.725, 0.611))
    axis.text(0.695, 0.410, "$\\log T_t$", transform=axis.transAxes, ha="center", fontsize=5.7, color="#5B6470")

    axis.text(0.480, 0.020, "feature / density", transform=axis.transAxes, fontsize=5.2, color="#555555")
    axis.plot([0.452, 0.474], [0.032, 0.032], transform=axis.transAxes, color="#555555", lw=0.85)
    axis.text(0.625, 0.020, "conditioning", transform=axis.transAxes, fontsize=5.2, color="#6B7280")
    axis.plot([0.596, 0.618], [0.032, 0.032], transform=axis.transAxes, color="#6B7280", lw=0.75, linestyle=(0, (3, 2)))


def _tiles(axis: plt.Axes, rgb: np.ndarray, density: np.ndarray, area: np.ndarray) -> None:
    _overlay(axis, rgb, density)
    selected = set(rank_tiles(density, area, tile_rows=8, tile_cols=16, budget_percent=10.0).tolist())
    height, width = density.shape
    tile_h, tile_w = height / 8.0, width / 16.0
    for row in range(8):
        for col in range(16):
            tile_id = row * 16 + col
            axis.add_patch(
                Rectangle(
                    (col * tile_w, row * tile_h),
                    tile_w,
                    tile_h,
                    fill=tile_id in selected,
                    facecolor="#00D5FF" if tile_id in selected else "none",
                    alpha=0.18 if tile_id in selected else 1.0,
                    edgecolor="#00D5FF" if tile_id in selected else "white",
                    linewidth=0.75 if tile_id in selected else 0.20,
                )
            )
    axis.set_title("(c) Prefetch top-$B$ tiles", loc="left", fontweight="bold", pad=2)


def main() -> None:
    _style()
    output = ensure_dir("reports/spherehorizon_overview")
    dataset = EarlyHorizonDataset(
        "data/processed/early_horizon_salient360_spherical2deg_v2",
        split="gate",
        target_mode="correct",
    )
    samples = [dataset[index] for index in range(len(dataset))]
    sample = next(item for item in samples if str(item["image_id"]) == "P10")
    rgb = np.clip(sample["image"][:3].numpy().transpose(1, 2, 0), 0.0, 1.0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictions = []
    for seed in (20260710, 20260711, 20260712):
        current, _ = _predict_learned(
            ROOT / f"runs/kernel_v2_candidate_correct_seed{seed}/checkpoints/best.pt",
            [sample],
            device,
        )
        predictions.append(current["P10"])
    prediction = np.mean(predictions, axis=0).astype(np.float32)

    figure, framework = plt.subplots(figsize=(7.2, 2.30), constrained_layout=True)
    _framework(framework, rgb, prediction[2], sample["template"].numpy()[2])

    stem = output / "spherehorizon_overview"
    for suffix in ("pdf", "svg", "png"):
        figure.savefig(stem.with_suffix(f".{suffix}"), bbox_inches="tight", facecolor="white", transparent=False)
    plt.close(figure)
    caption = (
        "SphereHorizon network architecture. The start-aligned ERP is encoded by an ERP-local Tiny U-Net and a frozen semantic ResNet-18, followed by feature fusion and affine horizon modulation. "
        "A residual head predicts the content log-density ratio, while a train-only horizon template and monotonic gate provide the population base measure. "
        "Log-density fusion, spherical softmax, and inverse yaw restoration produce the early cumulative map. The displayed input/output example is P10 at 3 s; display smoothing affects only the visualization."
    )
    result = {
        "content": "P10",
        "split": "historically_seen_gate",
        "display_smoothing_sigma_px": 10.0,
        "metrics_affected": False,
        "test_target_loads": 0,
        "caption": caption,
        "figure": str(stem.with_suffix(".png")),
    }
    (output / "results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (output / "index.md").write_text(
        "# SphereHorizon overview\n\n![Overview](spherehorizon_overview.png)\n\n"
        f"**Caption.** {caption}\n",
        encoding="utf-8",
    )
    print(json.dumps({"figure": result["figure"], "index": str(output / "index.md")}, indent=2))


if __name__ == "__main__":
    main()
