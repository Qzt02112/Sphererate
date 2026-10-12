#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from scripts.aggregate_v12c_dwell_rate import _model_from_checkpoint
from sphererate.data.early_dataset import EarlyHorizonDataset
from sphererate.evaluation.tile_prefetch import rank_tiles


DATA_ROOT = ROOT / "data/processed/early_horizon_salient360_spherical2deg_standard_v3"
CHECKPOINT = ROOT / "runs/standard_split_v3/v12c_hybrid_dwell_rate_seed20260711/checkpoints/best.pt"
OUTPUT = ROOT / "reports/sphererate_overview"
PAPER_FIGURE = ROOT / "paper/figures/sphererate_overview.pdf"
IMAGE_ID = "P17"

INK = "#24313A"
MUTED = "#5F6E78"
BLUE = "#276B89"
BLUE_LIGHT = "#EAF3F8"
TEAL = "#277A5B"
TEAL_LIGHT = "#EAF5F0"
ORANGE = "#A15C16"
ORANGE_LIGHT = "#FBF1E5"
GRAY_LIGHT = "#F4F6F7"


def _style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Nimbus Roman",
            "font.size": 7.0,
            "mathtext.fontset": "stix",
            "text.color": INK,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.dpi": 500,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.transparent": False,
        }
    )


def _relative(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    low, high = np.quantile(values, [0.01, 0.995])
    return np.clip((values - low) / max(high - low, 1e-12), 0.0, 1.0)


def _mute(rgb: np.ndarray) -> np.ndarray:
    luminance = np.tensordot(rgb[..., :3], np.asarray([0.2126, 0.7152, 0.0722]), axes=1)
    gray = np.repeat(luminance[..., None], 3, axis=-1)
    return np.clip(0.20 * rgb + 0.80 * gray, 0.0, 1.0)


def _panel(
    axis: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    title: str,
    *,
    face: str,
    edge: str,
    title_color: str,
) -> None:
    axis.add_patch(
        FancyBboxPatch(
            xy,
            width,
            height,
            boxstyle="round,pad=0.004,rounding_size=0.010",
            facecolor=face,
            edgecolor=edge,
            linewidth=0.7,
            transform=axis.transAxes,
            zorder=0,
        )
    )
    axis.text(
        xy[0] + 0.012,
        xy[1] + height - 0.033,
        title,
        transform=axis.transAxes,
        ha="left",
        va="center",
        fontsize=7.2,
        fontweight="bold",
        color=title_color,
    )


def _box(
    axis: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    face: str,
    *,
    edge: str = MUTED,
    fontsize: float = 6.2,
    fontweight: str = "normal",
) -> None:
    axis.add_patch(
        FancyBboxPatch(
            xy,
            width,
            height,
            boxstyle="round,pad=0.006,rounding_size=0.010",
            facecolor=face,
            edgecolor=edge,
            linewidth=0.75,
            transform=axis.transAxes,
            zorder=2,
        )
    )
    axis.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        transform=axis.transAxes,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=fontweight,
        zorder=3,
    )


def _arrow(
    axis: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    dashed: bool = False,
    color: str = MUTED,
    label: str | None = None,
    label_xy: tuple[float, float] | None = None,
) -> None:
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            transform=axis.transAxes,
            arrowstyle="-|>",
            mutation_scale=8,
            linewidth=0.85,
            linestyle=(0, (3, 2)) if dashed else "solid",
            color=color,
            zorder=1,
        )
    )
    if label is not None and label_xy is not None:
        axis.text(
            label_xy[0],
            label_xy[1],
            label,
            transform=axis.transAxes,
            ha="center",
            va="center",
            fontsize=5.2,
            color=color,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.6},
            zorder=4,
        )


def _map_axis(
    figure: plt.Figure,
    rect: tuple[float, float, float, float],
    rgb: np.ndarray,
    values: np.ndarray | None,
    title: str,
    *,
    cmap: str = "magma",
    start_xy: tuple[float, float] | None = None,
    tiles: set[int] | None = None,
    title_color: str = INK,
    title_size: float = 5.8,
) -> plt.Axes:
    axis = figure.add_axes(rect, zorder=5)
    if values is None:
        axis.imshow(rgb, aspect="equal")
    else:
        axis.imshow(_mute(rgb), aspect="equal")
        axis.imshow(_relative(values), cmap=cmap, vmin=0.0, vmax=1.0, alpha=0.68, aspect="equal")
    height, width = rgb.shape[:2]
    if start_xy is not None:
        axis.scatter(
            [start_xy[0]],
            [start_xy[1]],
            marker="+",
            s=22,
            c="#00D5FF",
            linewidths=1.4,
            zorder=10,
        )
    if tiles is not None:
        for row in range(8):
            for col in range(16):
                tile_id = row * 16 + col
                axis.add_patch(
                    Rectangle(
                        (col * width / 16, row * height / 8),
                        width / 16,
                        height / 8,
                        facecolor="#00A6D6" if tile_id in tiles else "none",
                        alpha=0.30 if tile_id in tiles else 1.0,
                        edgecolor="#00A6D6" if tile_id in tiles else "white",
                        linewidth=0.70 if tile_id in tiles else 0.16,
                    )
                )
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_box_aspect(0.5)
    for spine in axis.spines.values():
        spine.set_color(MUTED)
        spine.set_linewidth(0.65)
    axis.set_title(title, fontsize=title_size, pad=1.6, fontweight="bold", color=title_color)
    return axis


@torch.no_grad()
def main() -> None:
    _style()
    dataset = EarlyHorizonDataset(DATA_ROOT, split="test")
    sample = next(item for item in dataset if str(item["image_id"]) == IMAGE_ID)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model = _model_from_checkpoint(checkpoint, device)
    output = model(
        sample["image"].unsqueeze(0).to(device),
        start_geometry=sample["start_geometry"].unsqueeze(0).to(device),
    )

    rgb = np.clip(sample["image"][:3].numpy().transpose(1, 2, 0), 0.0, 1.0)
    geometry = sample["start_geometry"].numpy()
    start_y, start_x = np.unravel_index(np.argmax(geometry[1]), geometry[1].shape)
    start_xy = (float(start_x), float(start_y))
    semantic = output["semantic_map"][0, 0].float().cpu().numpy()
    rates = output["interval_rates"][0].float().cpu().numpy()
    cumulative = output["cumulative"][0].float().cpu().numpy()
    betas_e = output["reachability_betas"][0].float().cpu().numpy()
    betas_n = output["reachability_north_betas"][0].float().cpu().numpy()
    offsets = model.reachability_offsets().float().cpu().numpy()
    east = geometry[2][None] - offsets[:, 0, None, None]
    north = geometry[3][None] - offsets[:, 1, None, None]
    costs = betas_e[:, None, None] * east**2 + betas_n[:, None, None] * north**2
    area = sample["area_weights"][0].numpy()
    output_index = 2  # The flowchart illustrates the 3-s query.
    selected = set(
        rank_tiles(
            cumulative[output_index],
            area,
            tile_rows=8,
            tile_cols=16,
            budget_percent=10.0,
        ).tolist()
    )

    figure = plt.figure(figsize=(7.12, 2.62))
    canvas = figure.add_axes((0, 0, 1, 1), zorder=0)
    canvas.set_axis_off()

    # Three outer regions make inference inputs and the two outputs explicit.
    _panel(canvas, (0.008, 0.055), 0.157, 0.895, "Inference Inputs", face=GRAY_LIGHT, edge="#CDD5DA", title_color=INK)
    _panel(canvas, (0.178, 0.055), 0.642, 0.895, "SphereRate", face="#FBFCFD", edge="#B8C6CE", title_color=BLUE)
    _panel(canvas, (0.833, 0.055), 0.159, 0.895, "Outputs", face="#F3F8F6", edge="#BED3C8", title_color=TEAL)

    # Inputs.
    _map_axis(
        figure,
        (0.022, 0.620, 0.129, 0.165),
        rgb,
        None,
        r"ERP Panorama  $I$",
        start_xy=start_xy,
        title_size=5.6,
    )
    canvas.text(0.086, 0.590, "RGB + ERP Coordinates", transform=canvas.transAxes, ha="center", va="center", fontsize=5.1, color=MUTED)
    _box(canvas, (0.035, 0.350), 0.104, 0.125, "Declared Start\n" + r"$s=(\mathrm{yaw},\mathrm{pitch})$", BLUE_LIGHT, edge=BLUE)
    _box(canvas, (0.035, 0.135), 0.104, 0.115, "Query Horizon\n" + r"$t\in\{0.5,1,3,5\}\,\mathrm{s}$", ORANGE_LIGHT, edge=ORANGE)

    # Cue branches.
    _box(canvas, (0.190, 0.640), 0.080, 0.130, "Frozen\nSemantic\nEncoder", BLUE_LIGHT, edge=BLUE, fontsize=5.3, fontweight="bold")
    _map_axis(
        figure,
        (0.286, 0.605, 0.112, 0.170),
        rgb,
        semantic,
        r"Utility  $U(I)$",
        title_color=BLUE,
    )
    _box(canvas, (0.190, 0.310), 0.080, 0.130, "Start-Relative\nSpherical\nGeometry", ORANGE_LIGHT, edge=ORANGE, fontsize=5.1, fontweight="bold")
    _map_axis(
        figure,
        (0.286, 0.275, 0.112, 0.170),
        rgb,
        -costs[output_index],
        r"Reachability  $-R_i(\cdot;s)$",
        cmap="magma",
        start_xy=start_xy,
        title_color=ORANGE,
    )

    # The cue competition is the first method-specific operation.
    _box(
        canvas,
        (0.420, 0.365),
        0.116,
        0.285,
        "Cue Competition\n\n" + r"$z_i=\alpha_i[U+E_i]-R_i$" + "\n\n" + r"$r_i=\mathrm{Softmax}_{\mathrm{sph}}(z_i)$",
        "#F0F5F7",
        edge=BLUE,
        fontsize=6.2,
        fontweight="bold",
    )
    canvas.text(0.478, 0.322, r"$\alpha_i\uparrow,\ \beta_i\downarrow$ Over Time", transform=canvas.transAxes, ha="center", va="center", fontsize=5.1, color=MUTED)

    # Four actual non-negative interval-rate maps, shown as a compact stack.
    canvas.text(0.608, 0.835, "Non-negative Interval Rates", transform=canvas.transAxes, ha="center", va="center", fontsize=6.3, color=TEAL, fontweight="bold")
    rate_titles = ("$r_1$  0--0.5 s", "$r_2$  0.5--1 s", "$r_3$  1--3 s", "$r_4$  3--5 s")
    for index, (y, title) in enumerate(zip((0.675, 0.545, 0.415, 0.285), rate_titles)):
        _map_axis(
            figure,
            (0.558, y, 0.108, 0.090),
            rgb,
            rates[index],
            title,
            title_color=TEAL,
            title_size=5.0,
        )
    canvas.text(0.608, 0.246, r"$r_i(\mathbf{p})\geq0$", transform=canvas.transAxes, ha="center", va="center", fontsize=5.7, color=TEAL, fontweight="bold")

    # Accumulation is performed before spherical normalization.
    _box(
        canvas,
        (0.690, 0.365),
        0.108,
        0.285,
        "Raw-Mass\nAccumulation\n\n" + r"$M_k=\sum_{i\leq k}\Delta t_i r_i$" + "\n\n" + r"$\widehat C_t=\mathrm{Norm}_{\mathrm{sph}}(M_k)$",
        TEAL_LIGHT,
        edge=TEAL,
        fontsize=6.0,
        fontweight="bold",
    )
    _box(canvas, (0.704, 0.135), 0.080, 0.090, r"Select $k(t)$", ORANGE_LIGHT, edge=ORANGE, fontsize=5.7, fontweight="bold")

    # Prediction and deployment outputs use the same frozen 3-s map.
    _map_axis(
        figure,
        (0.848, 0.610, 0.129, 0.170),
        rgb,
        cumulative[output_index],
        r"1  Cumulative Map  $\widehat C_t$",
        title_color=TEAL,
        title_size=5.5,
    )
    canvas.text(0.912, 0.578, r"$H\!\times\!2H$ Normalized Density", transform=canvas.transAxes, ha="center", va="center", fontsize=5.1, color=MUTED)
    _map_axis(
        figure,
        (0.848, 0.270, 0.129, 0.170),
        rgb,
        cumulative[output_index],
        r"2  Top-$B$ Tile Set  $\mathcal{S}_t^B$",
        tiles=selected,
        title_color=TEAL,
        title_size=5.4,
    )
    canvas.text(0.912, 0.238, "Top-$B$ Tiles from the Same Map", transform=canvas.transAxes, ha="center", va="center", fontsize=5.1, color=MUTED)

    # Solid arrows are data flow; the dashed arrow is the horizon/index query.
    _arrow(canvas, (0.151, 0.695), (0.188, 0.695))
    _arrow(canvas, (0.270, 0.705), (0.284, 0.705))
    _arrow(canvas, (0.139, 0.412), (0.188, 0.377))
    _arrow(canvas, (0.270, 0.377), (0.284, 0.360))
    _arrow(canvas, (0.398, 0.690), (0.418, 0.565), color=BLUE)
    _arrow(canvas, (0.398, 0.360), (0.418, 0.455), color=ORANGE)
    _arrow(canvas, (0.536, 0.508), (0.555, 0.508), color=TEAL)
    _arrow(canvas, (0.667, 0.508), (0.688, 0.508), color=TEAL)
    _arrow(canvas, (0.798, 0.508), (0.845, 0.695), color=TEAL)
    _arrow(canvas, (0.981, 0.595), (0.981, 0.445), color=TEAL, label="Rank Tile Mass", label_xy=(0.912, 0.510))
    _arrow(canvas, (0.139, 0.190), (0.702, 0.180), dashed=True, color=ORANGE, label="Horizon Control", label_xy=(0.420, 0.198))
    _arrow(canvas, (0.744, 0.225), (0.744, 0.363), dashed=True, color=ORANGE)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    PAPER_FIGURE.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "svg", "png"):
        figure.savefig(OUTPUT / f"sphererate_overview.{suffix}", bbox_inches="tight", pad_inches=0.02)
    figure.savefig(PAPER_FIGURE, bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)

    (OUTPUT / "index.md").write_text(
        "# SphereRate flowchart\n\n![SphereRate flowchart](sphererate_overview.png)\n\n"
        "The flowchart makes the inference inputs, interval-rate mechanism, cumulative saliency output, "
        "and fixed-budget tile output explicit. All ERP maps are actual frozen seed-20260711 outputs "
        f"for standard-test content {IMAGE_ID}; heatmap scaling is display-only.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
